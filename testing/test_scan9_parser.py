"""
Regression tests for the parser defects found by the ninth release-readiness scan: the coverage gate that
rejected every site sampled by the target-site counter above ploidy two, the sampling pass aborting on a contig
missing from the FASTA, the two-SFS extrapolation leaving the divergence row and column empty and
double-counting the monomorphic sites an all-sites input already provides, the per-pass state of the components
that the sampling pass overwrote, and ``max_sites=0`` parsing the whole input. Kept fast and unmarked so they
run in the default suite.
"""

import gzip

import numpy as np
import pytest

import sfsutils as su
from sfsutils.filtration import SNPFiltration
from sfsutils.parser import _snapshot_state, _restore_state
from sfsutils.settings import Settings

Settings.disable_pbar = True

HEADER_INFO = (
    "##fileformat=VCFv4.2\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)


def _write_vcf(path, contigs, rows, samples):
    """
    Write a minimal VCF holding the given data rows.

    :param path: The path to write to.
    :param contigs: The contigs, as ``(name, length)`` pairs.
    :param rows: The data rows, each a sequence of the nine fixed columns followed by the genotypes.
    :param samples: The sample names.
    :return: The path as a string.
    """
    with open(path, "w") as fh:
        fh.write(HEADER_INFO)

        for name, length in contigs:
            fh.write(f"##contig=<ID={name},length={length}>\n")

        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")

        for row in rows:
            fh.write("\t".join(str(x) for x in row) + "\n")

    return str(path)


def _write_fasta(path, sequences):
    """
    Write a gzipped FASTA file.

    :param path: The path to write to.
    :param sequences: Mapping of contig name to sequence.
    :return: The path as a string.
    """
    with gzip.open(path, "wt") as fh:
        for name, seq in sequences.items():
            fh.write(f">{name}\n")

            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")

    return str(path)


def _polyploid_input(tmp_path, ploidy, n_individuals=5, n_snps=99, length=1000):
    """
    Write a SNP-only VCF of the given ploidy together with the reference it was drawn from.

    :param tmp_path: The directory to write to.
    :param ploidy: The ploidy of each sample.
    :param n_individuals: The number of samples.
    :param n_snps: The number of segregating sites.
    :param length: The length of the contig.
    :return: The VCF path and the FASTA path.
    """
    rng = np.random.default_rng(0)
    n_hap = n_individuals * ploidy
    ref = "".join(rng.choice(list("ACGT"), size=length))
    samples = [f"s{i}" for i in range(n_individuals)]
    rows = []

    for pos in sorted(rng.choice(np.arange(1, length + 1), size=n_snps, replace=False)):
        haplotypes = np.array([1] * int(rng.integers(1, n_hap)) + [0] * n_hap)[:n_hap]
        rng.shuffle(haplotypes)
        base = ref[pos - 1]
        alt = next(b for b in "ACGT" if b != base)
        genotypes = ["|".join(map(str, g)) for g in haplotypes.reshape(-1, ploidy)]
        rows.append(["1", pos, ".", base, alt, ".", "PASS", f"AA={base}", "GT"] + genotypes)

    return (_write_vcf(tmp_path / "poly.vcf", [("1", length)], rows, samples),
            _write_fasta(tmp_path / "poly.fasta.gz", {"1": ref}))


@pytest.mark.parametrize("ploidy", [2, 4, 6])
def test_target_site_counter_extrapolates_above_ploidy_two(tmp_path, ploidy):
    """The sites the counter samples from the FASTA stand for fully covered sites, so the coverage gate must not
    reject them when the requested sample size exceeds twice the number of samples, which would silently turn the
    extrapolation into a no-op."""
    vcf, fasta = _polyploid_input(tmp_path, ploidy)

    n_target_sites = 1000
    spectra = su.Parser(
        source=vcf,
        fasta=fasta,
        n=5 * ploidy,
        skip_non_polarized=False,
        filtrations=[SNPFiltration()],
        target_site_counter=su.TargetSiteCounter(n_target_sites=n_target_sites, n_samples=500),
    ).parse()

    data = spectra.data

    assert float(data.values.sum()) == pytest.approx(n_target_sites)
    assert float(data.iloc[0].sum()) > 0


def test_target_site_counter_skips_contig_missing_from_fasta(tmp_path):
    """A contig the FASTA does not cover contributes no target sites, rather than aborting the run with a
    ``LookupError`` and discarding the whole first pass."""
    samples = ["s0", "s1", "s2", "s3", "s4"]
    rng = np.random.default_rng(1)
    ref = "".join(rng.choice(list("ACGT"), size=2000))
    rows = []

    for contig in ("1", "2"):
        for pos in range(10, 1000, 10):
            base = ref[pos - 1] if contig == "1" else "A"
            alt = next(b for b in "ACGT" if b != base)
            rows.append([contig, pos, ".", base, alt, ".", "PASS", f"AA={base}", "GT",
                         "0|1", "0|0", "0|0", "0|0", "0|0"])

    vcf = _write_vcf(tmp_path / "two_contigs.vcf", [("1", 2000), ("2", 2000)], rows, samples)
    fasta = _write_fasta(tmp_path / "one_contig.fasta.gz", {"1": ref})

    n_target_sites = 100_000
    spectra = su.Parser(
        source=vcf,
        fasta=fasta,
        n=10,
        skip_non_polarized=False,
        filtrations=[SNPFiltration()],
        target_site_counter=su.TargetSiteCounter(n_target_sites=n_target_sites, n_samples=2000),
    ).parse()

    assert float(spectra.data.values.sum()) == pytest.approx(n_target_sites)


def test_two_sfs_extrapolation_populates_the_divergence_row():
    """A divergence site (bin ``n``) pairs with the sites missing from the input just as a segregating site does,
    so the divergence row and column must be populated rather than zeroed and their mass booked into ``(0, 0)``."""
    marginal = np.array([0.0, 10.0, 5.0, 20.0])
    n_target_sites, region_length, distance = 1000, 1000.0, 10

    extrapolated = su.TargetSiteCounter(n_target_sites=n_target_sites)._extrapolate_two_sfs(
        np.zeros((4, 4)), marginal, region_length, distance)

    # each observed site pairs with the missing ones, which number the target sites less the observed ones
    rho = (n_target_sites - marginal.sum()) / region_length

    np.testing.assert_allclose(extrapolated[0, 1:], marginal[1:] * rho * distance)
    np.testing.assert_allclose(extrapolated[1:, 0], marginal[1:] * rho * distance)
    assert extrapolated[0, -1] > 0
    assert extrapolated[0, 0] == pytest.approx((n_target_sites - marginal.sum()) * rho * distance)


def test_two_sfs_extrapolation_does_not_double_count_observed_monomorphic_sites():
    """Only the sites missing from the input are extrapolated: an input whose monomorphic sites are all present
    leaves the two-SFS untouched, rather than adding a second copy of their pairs."""
    marginal = np.array([900.0, 10.0, 5.0, 85.0])
    observed = np.full((4, 4), 7.0)

    extrapolated = su.TargetSiteCounter(n_target_sites=int(marginal.sum()))._extrapolate_two_sfs(
        observed.copy(), marginal, region_length=1000.0, distance=10)

    np.testing.assert_allclose(extrapolated, observed)


def test_two_sfs_extrapolation_matches_all_sites_ground_truth_with_divergence(tmp_path):
    """Ground truth: parse an all-sites VCF, which counts every pair for real, and compare against the SNP-only
    projection of the same data parsed with a target-site counter. Divergence sites (fixed for the derived
    allele) are present in both inputs, so their extrapolated row and column must reproduce the real ones."""
    length, n_hap, distance = 4000, 6, 50
    rng = np.random.default_rng(1)

    # 3% segregating, 2% fixed for the derived allele, the rest all-ancestral and absent from the SNP-only input
    u = rng.random(length)
    derived = np.where(u < 0.03, rng.integers(1, n_hap, size=length), np.where(u < 0.05, n_hap, 0))

    samples = [f"s{i}" for i in range(n_hap // 2)]
    all_rows, snp_rows = [], []

    for pos, k in enumerate(derived, start=1):
        haplotypes = np.array([1] * int(k) + [0] * (n_hap - int(k)))
        rng.shuffle(haplotypes)
        row = ["1", pos, ".", "A", "T" if k else ".", ".", "PASS", "AA=A", "GT"] + \
              [f"{a}|{b}" for a, b in haplotypes.reshape(-1, 2)]
        all_rows.append(row)

        if k:
            snp_rows.append(row)

    contigs = [("1", length)]
    all_sites = _write_vcf(tmp_path / "all.vcf", contigs, all_rows, samples)
    snps = _write_vcf(tmp_path / "snp.vcf", contigs, snp_rows, samples)

    kw = dict(n=n_hap, two_sfs=True, d=distance, skip_non_polarized=False, subsample_mode="random")
    truth = np.asarray(su.Parser(source=all_sites, **kw).parse()["all"].data)
    extrapolated = np.asarray(su.Parser(source=snps, **kw, target_site_counter=su.TargetSiteCounter(
        n_target_sites=length)).parse()["all"].data)

    # the pairs among observed sites are counted directly and must match exactly
    np.testing.assert_allclose(extrapolated[1:, 1:], truth[1:, 1:])

    # the sites near the contig edges have fewer partners than the uniform density assumes, hence the tolerance
    assert extrapolated[0, 0] == pytest.approx(truth[0, 0], rel=0.03)
    assert extrapolated[0, -1] == pytest.approx(truth[0, -1], rel=0.03)
    assert extrapolated[-1, 0] == pytest.approx(truth[-1, 0], rel=0.03)
    assert extrapolated[0, 1:-1].sum() == pytest.approx(truth[0, 1:-1].sum(), rel=0.03)


def test_two_sfs_with_counter_leaves_all_sites_input_unchanged():
    """An all-sites input carries its own monomorphic pairs, so adding a target-site counter that does not exceed
    the observed sites must not scale the ``(0, 0)`` corner, which broke the correlation matrix."""
    source = "resources/msprime/two_sfs_kingman.all.vcf.gz"

    if not __import__("os").path.exists(source):
        pytest.skip("the all-sites msprime fixture is absent")

    kw = dict(source=source, n=10, two_sfs=True, d=1000, skip_non_polarized=False)

    plain = su.Parser(**kw).parse()["all"]
    with_counter = su.Parser(**kw, target_site_counter=su.TargetSiteCounter(n_target_sites=600_000)).parse()["all"]

    np.testing.assert_allclose(np.asarray(with_counter.data), np.asarray(plain.data))
    np.testing.assert_allclose(np.diag(with_counter.corr())[1:-1], 1.0)


def test_snapshot_state_covers_counters_and_copies_lists():
    """The per-pass state is discovered by type rather than by a name list, so a component gaining a counter does
    not fall out of the snapshot, and the list diagnostics are copied rather than aliased."""

    class Component:
        def __init__(self):
            self.n_valid = 3
            self.n_filtered = 4
            self.mismatches = ["a"]
            self.use_parser = True
            self.label = "x"

    component = Component()
    state = _snapshot_state(component)

    assert state == {"n_valid": 3, "n_filtered": 4, "mismatches": ["a"]}

    component.n_valid, component.n_filtered = 99, 99
    component.mismatches.append("b")

    _restore_state(component, state)

    assert (component.n_valid, component.n_filtered, component.mismatches) == (3, 4, ["a"])


def test_target_site_counter_restores_component_state():
    """The sampling pass re-runs every component, whose counters must keep describing the sites that produced the
    spectra rather than the sampled ones."""
    vcf, fasta = "resources/msprime/two_epoch.vcf", "resources/msprime/two_epoch.ref.fasta.gz"

    if not (__import__("os").path.exists(vcf) and __import__("os").path.exists(fasta)):
        pytest.skip("the msprime VCF / reference FASTA fixtures are absent")

    def parse(counter):
        stratification = su.AncestralBaseStratification()
        filtration = SNPFiltration()

        su.Parser(source=vcf, fasta=fasta, n=10, skip_non_polarized=False,
                  stratifications=[stratification], filtrations=[filtration],
                  target_site_counter=counter).parse()

        return stratification.n_valid, filtration.n_filtered

    without = parse(None)
    with_counter = parse(su.TargetSiteCounter(n_target_sites=500_000, n_samples=2000))

    assert with_counter == without


def test_max_sites_is_positive():
    """A cap of zero parsed the whole input, as the site count it is compared against is itself capped at it."""
    with pytest.raises(ValueError, match="max_sites"):
        su.Parser(source="resources/msprime/two_epoch.vcf", n=10, max_sites=0)


def test_max_sites_caps_the_number_of_parsed_sites():
    """A positive cap stops the parse at that many sites."""
    if not __import__("os").path.exists("resources/msprime/two_epoch.vcf"):
        pytest.skip("the msprime VCF fixture is absent")

    parser = su.Parser(source="resources/msprime/two_epoch.vcf", n=10, max_sites=3, skip_non_polarized=False)

    assert float(parser.parse().data.values.sum()) == pytest.approx(3)


def test_no_samples_mask_installed_without_a_restriction():
    """Without a sample restriction no mask is installed, so no site copies the whole genotype array; a
    restriction still installs one, and the spectrum is the same either way."""
    if not __import__("os").path.exists("resources/msprime/two_epoch.vcf"):
        pytest.skip("the msprime VCF fixture is absent")

    kw = dict(source="resources/msprime/two_epoch.vcf", n=6, skip_non_polarized=False)

    unrestricted = su.Parser(**kw)
    unrestricted._setup()
    assert unrestricted._samples_mask is None

    restricted = su.Parser(**kw, include_samples=[f"tsk_{i}" for i in range(3)])
    restricted._setup()
    assert restricted._samples_mask is not None and restricted._samples_mask.sum() == 3

    # every sample selected explicitly reaches the same spectrum as no restriction at all
    everything = su.Parser(**kw, include_samples=[f"tsk_{i}" for i in range(10)])

    np.testing.assert_allclose(su.Parser(**kw).parse().data.values, everything.parse().data.values)
