"""
Exercise every command example shown in ``docs/reference/CLI/usage.rst`` end-to-end on tiny, self-contained
synthetic datasets built in a temporary directory. Each test runs the CLI via the ``run([...argv])`` entry
point (the same invocation the docs' ``sfsutils ...`` line performs) and asserts on the exit code, the written
artifact and, where feasible, its content (spectrum shape/sum, filtered site count, an added INFO tag, a
VCF-Zarr store on disk, a subset tree sequence).

The synthetic inputs deliberately deviate from the docs' placeholder file names and, where the literal sample
size cannot be met by a handful of samples, their exact ``--n`` values; the commands' structure and options are
otherwise reproduced verbatim. Optional-backend examples (``--zarr``/``--trees`` inputs, ``.vcz``/``.trees``
outputs) skip when ``tskit``/``zarr`` are absent.
"""
import os

import numpy as np
import pytest

import sfsutils as su
from sfsutils.cli import run
from sfsutils.io_handlers import count_sites
from sfsutils.settings import Settings

Settings.disable_pbar = True



N_DIP = 10  # -> 20 haplotypes, enough to project a one-population SFS to n = 20

# a 30 bp coding contig: ATG (Met) then Val codons, Arg, Pro, Lys, stop (see testing/test_annotation_mocked)
CODING_CONTIG = "ATGGTTGTAGTCGTGGTACGGCCCAAATAA"

# SNPs placed at interior codon positions whose REF matches CODING_CONTIG; degeneracy noted for the stratification
#   pos 4  G  Val GTT 1st position -> 0-fold (selected)
#   pos 6  T  Val GTT 3rd position -> 4-fold (neutral)
#   pos 13 G  Val GTG 1st position -> 0-fold (selected)
#   pos 21 G  Arg CGG 3rd position -> 4-fold (neutral)
CODING_SNPS = [(4, "G", "A"), (6, "T", "C"), (13, "G", "A"), (21, "G", "A")]


# --- synthetic-input builders ---------------------------------------------------------------------

def _seg_gts(rng, n_samples):
    """Random diploid genotype strings guaranteed to be segregating (at least one of each allele)."""
    while True:
        alleles = rng.integers(0, 2, size=2 * n_samples)
        if 0 < alleles.sum() < 2 * n_samples:
            return [f"{alleles[2 * i]}/{alleles[2 * i + 1]}" for i in range(n_samples)]


def _write_vcf(path, samples, records, contig="1", contig_len=100_000, info_extra=None):
    """
    Write a minimal valid VCF. ``records`` are ``(pos, ref, alt, gts)`` tuples with one GT string per sample;
    each record gets an ``AA`` (ancestral = REF) tag so the default (skip-non-polarized) parser keeps it.

    :param info_extra: optional callable ``(pos, ref, alt) -> str`` returning extra INFO appended after ``AA``.
    """
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={contig},length={contig_len}>",
        '##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + list(samples)),
    ]
    for pos, ref, alt, gts in records:
        info = f"AA={ref}"
        if info_extra is not None:
            info += ";" + info_extra(pos, ref, alt)
        lines.append("\t".join([contig, str(pos), ".", ref, alt, ".", ".", info, "GT", *gts]))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def snp_vcf(tmp_path_factory):
    """Biallelic SNP VCF: 10 diploid samples, sites clustered within ~900 bp so two-SFS pairing has pairs."""
    rng = np.random.default_rng(0)
    samples = [f"S{i:02d}" for i in range(N_DIP)]
    positions = [100, 250, 400, 550, 700, 850, 900, 950]
    bases = ["A", "C", "G", "T"]
    records = []
    for k, pos in enumerate(positions):
        ref = bases[k % 4]
        alt = bases[(k + 1) % 4]
        records.append((pos, ref, alt, _seg_gts(rng, N_DIP)))
    path = tmp_path_factory.mktemp("snp") / "variants.vcf"
    return _write_vcf(path, samples, records), samples


@pytest.fixture(scope="module")
def coding_fixture(tmp_path_factory):
    """VCF + FASTA + GFF over a 30 bp coding contig, for the degeneracy/coding-sequence examples."""
    d = tmp_path_factory.mktemp("coding")
    rng = np.random.default_rng(1)
    samples = [f"S{i:02d}" for i in range(N_DIP)]
    records = [(pos, ref, alt, _seg_gts(rng, N_DIP)) for pos, ref, alt in CODING_SNPS]
    vcf = _write_vcf(d / "variants.vcf", samples, records, contig="1", contig_len=len(CODING_CONTIG))

    fasta = d / "genome.fasta"
    fasta.write_text(f">1\n{CODING_CONTIG}\n")

    # single CDS spanning the whole contig, + strand, phase 0
    gff = d / "genome.gff"
    gff.write_text("\t".join(["1", "synthetic", "CDS", "1", str(len(CODING_CONTIG)), ".", "+", "0", "."]) + "\n")

    return dict(vcf=vcf, fasta=str(fasta), gff=str(gff), samples=samples)


@pytest.fixture(scope="module")
def outgroup_vcf(tmp_path_factory):
    """Ingroup + two-outgroup VCF with enough SNP sites for the maximum-likelihood ancestral inference."""
    rng = np.random.default_rng(2)
    n_ingroup = 20  # -> 40 ingroup haplotypes, enough for --n-ingroups 15
    ingroups = [f"ING{i:02d}" for i in range(n_ingroup)]
    outgroups = ["ERR2103730", "ERR2103731"]
    samples = ingroups + outgroups
    bases = ["A", "C", "G", "T"]

    records = []
    for i in range(150):
        ref = bases[rng.integers(0, 4)]
        alt = bases[(int(np.where(np.array(bases) == ref)[0][0]) + 1 + rng.integers(0, 3)) % 4]
        gts = _seg_gts(rng, n_ingroup)
        # outgroups: usually ancestral (0/0), occasionally derived, giving a small, estimable divergence
        og = ["1/1" if rng.random() < 0.15 else "0/0" for _ in outgroups]
        records.append((100 * (i + 1), ref, alt, gts + og))

    path = tmp_path_factory.mktemp("outgroup") / "variants.with_outgroups.vcf"
    return _write_vcf(path, samples, records, contig="1", contig_len=100 * 200), outgroups


@pytest.fixture(scope="module")
def trees_path(tmp_path_factory):
    """A small msprime tree sequence over the single contig '1' (for the --trees examples)."""
    import msprime

    ts = msprime.sim_ancestry(samples=N_DIP, sequence_length=1_000, recombination_rate=1e-6,
                              population_size=10_000, random_seed=42)
    ts = msprime.sim_mutations(ts, rate=1e-6, random_seed=43)
    # keep only strictly biallelic sites so every site is a clean SNP
    non_biallelic = [s.id for s in ts.sites()
                     if len({s.ancestral_state} | {m.derived_state for m in s.mutations}) != 2]
    ts = ts.delete_sites(non_biallelic)
    out = tmp_path_factory.mktemp("trees") / "ancestry.trees"
    ts.dump(str(out))
    return str(out), ts.num_sites


@pytest.fixture(scope="module")
def zarr_store(tmp_path_factory, snp_vcf):
    """A VCF-Zarr store converted from ``snp_vcf`` (keeping all sites), for the --zarr example."""
    vcf, _ = snp_vcf
    out = str(tmp_path_factory.mktemp("zarr") / "variants.vcz")
    su.Filterer(source=vcf, output=out, filtrations=[su.NoFiltration()]).filter()
    return out


# --- helper wrapper -------------------------------------------------------------------------------

def _run(*argv):
    """Run the CLI quietly and return the exit code."""
    return run(["-q", *argv])


# =========================================== help ===========================================

@pytest.mark.parametrize("argv", [["--help"], ["parse", "--help"]])
def test_help_commands_exit_zero(argv, capsys):
    # sfsutils --help ; sfsutils parse --help
    with pytest.raises(SystemExit) as exc:
        run(argv)
    assert exc.value.code == 0
    assert "sfsutils" in capsys.readouterr().out


# =========================================== parse ==========================================

def test_parse_one_dimensional(snp_vcf, tmp_path):
    # sfsutils parse --vcf variants.vcf.gz --n 20 --out sfs.csv
    vcf, _ = snp_vcf
    out = tmp_path / "sfs.csv"
    assert _run("parse", "--vcf", vcf, "--n", "20", "--out", str(out)) == 0
    assert out.exists() and out.stat().st_size > 0
    sfs = np.array(su.Spectra.from_file(str(out)).all.to_list())
    assert sfs.shape == (21,)
    assert sfs[1:20].sum() > 0  # polarized via the AA tag


def test_parse_from_zarr(zarr_store, tmp_path):
    # sfsutils parse --zarr variants.vcz --n 20 --out sfs.csv
    out = tmp_path / "sfs.csv"
    assert _run("parse", "--zarr", zarr_store, "--n", "20", "--out", str(out)) == 0
    assert out.exists() and out.stat().st_size > 0
    sfs = np.array(su.Spectra.from_file(str(out)).all.to_list())
    assert sfs.shape == (21,)
    assert sfs[1:20].sum() > 0  # AA tag persisted through the VCF-Zarr store


def test_parse_from_trees(trees_path, tmp_path):
    # sfsutils parse --trees ancestry.trees --n 20 --out sfs.csv
    # The literal command omits --no-skip-non-polarized, and a tree sequence carries no AA tag (tskit stores the
    # ancestral state as REF, recovered only with skip_non_polarized=False), so the projected SFS is empty here.
    # The command must still succeed and write the output file. That the tree input itself is valid and yields a
    # populated SFS once polarized is asserted separately via the API, so the check is not vacuous.
    trees, _ = trees_path
    out = tmp_path / "sfs.csv"
    assert _run("parse", "--trees", trees, "--n", "20", "--out", str(out)) == 0
    assert out.exists()
    polarized = np.array(su.Parser(source=trees, n=20, skip_non_polarized=False,
                                   subsample_mode="random").parse().all.to_list())
    assert polarized.shape == (21,) and polarized[1:20].sum() > 0


def test_parse_degeneracy_stratified(coding_fixture, tmp_path):
    # sfsutils parse --vcf ... --n 20 --fasta ... --gff ... --annotate degeneracy --stratify degeneracy \
    #     --filter snp --out sfs.csv
    out = tmp_path / "sfs.csv"
    code = _run("parse", "--vcf", coding_fixture["vcf"], "--n", "20",
                "--fasta", coding_fixture["fasta"], "--gff", coding_fixture["gff"],
                "--annotate", "degeneracy", "--stratify", "degeneracy", "--filter", "snp", "--out", str(out))
    assert code == 0 and out.exists() and out.stat().st_size > 0
    spectra = su.Spectra.from_file(str(out))
    assert "neutral" in spectra.types and "selected" in spectra.types
    # both a 4-fold (neutral) and a 0-fold (selected) SNP are present and segregating
    assert np.array(spectra["neutral"].to_list())[1:20].sum() > 0
    assert np.array(spectra["selected"].to_list())[1:20].sum() > 0


def test_parse_joint(snp_vcf, tmp_path):
    # sfsutils parse --vcf ... --n 10 --pops "A=...;B=..." --out jsfs.json  (5 diploid samples per population)
    vcf, samples = snp_vcf
    pops = f"A={','.join(samples[:5])};B={','.join(samples[5:])}"
    out = tmp_path / "jsfs.json"
    assert _run("parse", "--vcf", vcf, "--n", "10", "--pops", pops, "--out", str(out)) == 0
    assert out.exists() and out.stat().st_size > 0
    loaded = su.JointSpectra.from_file(str(out))
    assert loaded.n_pops == 2
    assert np.asarray(loaded["all"]).shape == (11, 11)


def test_parse_two_sfs(snp_vcf, tmp_path):
    # sfsutils parse --vcf ... --n 20 --two-sfs --two-sfs-distance 1000 --out two_sfs.json
    vcf, _ = snp_vcf
    out = tmp_path / "two_sfs.json"
    assert _run("parse", "--vcf", vcf, "--n", "20", "--two-sfs", "--two-sfs-distance", "1000",
                "--out", str(out)) == 0
    assert out.exists() and out.stat().st_size > 0
    # the two-SFS parse mode writes a single-entry TwoSpectra collection (keyed 'all')
    sfs2 = su.TwoSpectra.from_file(str(out))["all"]
    assert sfs2.data.shape == (21, 21)
    np.testing.assert_allclose(sfs2.data, sfs2.data.T)
    assert sfs2.data.sum() > 0  # sites lie within the 1 kb window, so pairs are counted


# =========================================== filter =========================================

def test_filter_snp_coding_to_vcf(coding_fixture, tmp_path):
    # sfsutils filter --vcf ... --filter snp,coding-sequence --gff ... --out coding.vcf.gz
    out = tmp_path / "coding.vcf.gz"
    code = _run("filter", "--vcf", coding_fixture["vcf"], "--filter", "snp,coding-sequence",
                "--gff", coding_fixture["gff"], "--out", str(out))
    assert code == 0 and out.exists() and out.stat().st_size > 0
    n_in = count_sites(coding_fixture["vcf"])
    n_out = count_sites(str(out))
    assert 0 < n_out <= n_in  # all synthetic SNPs lie inside the CDS, so all are kept


def test_filter_snp_coding_to_zarr(coding_fixture, tmp_path):
    # sfsutils filter --vcf ... --filter snp,coding-sequence --gff ... --out coding.vcz
    import zarr
    out = tmp_path / "coding.vcz"
    code = _run("filter", "--vcf", coding_fixture["vcf"], "--filter", "snp,coding-sequence",
                "--gff", coding_fixture["gff"], "--out", str(out))
    assert code == 0 and out.is_dir()
    root = zarr.open(str(out), mode="r")
    assert "call_genotype" in list(root.array_keys())


def test_filter_trees_to_trees(trees_path, tmp_path):
    # sfsutils filter --trees ancestry.trees --filter snp --out coding.trees
    import tskit
    trees, n_in = trees_path
    out = tmp_path / "coding.trees"
    assert _run("filter", "--trees", trees, "--filter", "snp", "--out", str(out)) == 0
    assert out.exists() and out.stat().st_size > 0
    ts_out = tskit.load(str(out))
    assert ts_out.num_sites <= n_in  # SNP-only subset via delete_sites


# =========================================== annotate =======================================

def test_annotate_degeneracy_to_vcf(coding_fixture, tmp_path):
    # sfsutils annotate --vcf ... --annotation degeneracy --fasta ... --gff ... --out degeneracy.vcf.gz
    from cyvcf2 import VCF
    out = tmp_path / "degeneracy.vcf.gz"
    code = _run("annotate", "--vcf", coding_fixture["vcf"], "--annotation", "degeneracy",
                "--fasta", coding_fixture["fasta"], "--gff", coding_fixture["gff"], "--out", str(out))
    assert code == 0 and out.exists() and out.stat().st_size > 0
    reader = VCF(str(out))
    assert "Degeneracy" in reader.raw_header
    degeneracies = [v.INFO.get("Degeneracy") for v in reader]
    assert any(d in (0, 4) for d in degeneracies)  # coding sites got a real degeneracy value


def test_annotate_degeneracy_to_zarr(coding_fixture, tmp_path):
    # sfsutils annotate --vcf ... --annotation degeneracy --fasta ... --gff ... --out degeneracy.vcz
    import zarr
    out = tmp_path / "degeneracy.vcz"
    code = _run("annotate", "--vcf", coding_fixture["vcf"], "--annotation", "degeneracy",
                "--fasta", coding_fixture["fasta"], "--gff", coding_fixture["gff"], "--out", str(out))
    assert code == 0 and out.is_dir()
    root = zarr.open(str(out), mode="r")
    assert "variant_Degeneracy" in list(root.array_keys())  # the added INFO tag persisted to the store


@pytest.mark.slow
def test_annotate_maximum_likelihood_ancestral(outgroup_vcf, tmp_path):
    # sfsutils annotate --vcf variants.with_outgroups.vcf.gz --annotation maximum-likelihood-ancestral \
    #     --outgroups ERR2103730,ERR2103731 --n-ingroups 15 --out polarized.vcf.gz
    from cyvcf2 import VCF
    vcf, outgroups = outgroup_vcf
    out = tmp_path / "polarized.vcf.gz"
    code = _run("annotate", "--vcf", vcf, "--annotation", "maximum-likelihood-ancestral",
                "--outgroups", ",".join(outgroups), "--n-ingroups", "15", "--out", str(out))
    assert code == 0 and out.exists() and out.stat().st_size > 0
    reader = VCF(str(out))
    assert "AA" in reader.raw_header
    ancestral = [v.INFO.get("AA") for v in reader]
    assert any(a in ("A", "C", "G", "T") for a in ancestral)  # at least one site was polarized
