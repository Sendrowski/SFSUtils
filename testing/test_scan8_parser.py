"""
Regression tests for the parser defects found by the eighth release-readiness scan: the joint spectrum
snapshot taken before the target-site sampling pass, fixed-derived invariant sites booked as ancestral,
the reader that was not rewound on entry to :meth:`~sfsutils.parser.Parser.parse`, the random
stratification that continued its stream across passes, the SNP filtration left suspended by a raising
sampling pass, the sampling pass running off the end of a short FASTA record, and the quadratic chunk
lookup. Kept fast and unmarked so they run in the default suite.
"""

import random
import time

import numpy as np
import pytest

import sfsutils as su
from sfsutils.filtration import Filtration, SNPFiltration
from sfsutils.settings import Settings

HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length=5000>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##INFO=<ID=AA_prob,Number=1,Type=Float,Description="ancestral allele probability">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)

SAMPLES = ("s0", "s1", "s2", "s3", "s4")

COMPLEMENT = {"A": "G", "G": "A", "C": "T", "T": "C"}


def _write_vcf(path, rows, samples=SAMPLES):
    """
    Write a minimal VCF holding the given data rows.

    :param path: The path to write to.
    :param rows: The data rows, each a sequence of the nine fixed columns followed by the genotypes.
    :param samples: The sample names.
    :return: The path as a string.
    """
    columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples]

    path.write_text(HEADER + "#" + "\t".join(columns) + "\n" + "".join("\t".join(r) + "\n" for r in rows))

    return str(path)


def _write_fasta(path, seq, contig="1"):
    """
    Write a single-contig FASTA.

    :param path: The path to write to.
    :param seq: The sequence.
    :param contig: The contig name.
    :return: The path as a string.
    """
    path.write_text(f">{contig}\n{seq}\n")

    return str(path)


def _snp_rows(positions, seq=None):
    """
    Build one bi-allelic row per position, singleton in the first sample and polarized to the reference.

    :param positions: The positions.
    :param seq: The reference sequence the positions index into, or ``None`` for an all-``A`` reference.
    :return: The rows.
    """
    rows = []

    for pos in positions:
        ref = seq[pos - 1] if seq is not None else "A"
        rows.append(["1", str(pos), ".", ref, COMPLEMENT[ref], ".", ".", f"AA={ref}", "GT",
                     "0|1", "0|0", "0|0", "0|0", "0|0"])

    return rows


class _RaisingFiltration(Filtration):
    """
    Filtration that raises once it has seen a given number of sites, to interrupt a pass part-way through.
    """

    def __init__(self, at: int):
        """
        Create instance.

        :param at: The number of the site at which to raise.
        """
        super().__init__()

        self.at: int = at
        self.n_seen: int = 0

    def filter_site(self, variant) -> bool:
        """
        Pass every site until the configured one.

        :param variant: The site.
        :return: Always ``True``.
        """
        self.n_seen += 1

        if self.n_seen == self.at:
            raise RuntimeError("interrupted")

        return True


def test_joint_target_sites_keep_types_first_seen_when_sampling(tmp_path):
    """A stratification type that only the sampled monomorphic sites carry must appear in the joint
    spectra, and the joint total must match the one-dimensional total."""
    Settings.disable_pbar = True

    seq = "".join(random.Random(1).choice("ACGT") for _ in range(5000))
    fasta = _write_fasta(tmp_path / "ref.fasta", seq)

    # the variants sit in a short stretch, so most base contexts appear only among the sampled sites
    vcf = _write_vcf(tmp_path / "joint_types.vcf", _snp_rows(list(range(2, 60)) + [4900], seq))

    def parse(pops):
        kwargs = dict(
            source=vcf,
            n=4,
            fasta=fasta,
            seed=7,
            stratifications=[su.BaseContextStratification(fasta=fasta, n_flanking=1)],
            filtrations=[SNPFiltration()],
            target_site_counter=su.TargetSiteCounter(n_target_sites=40000, n_samples=2000),
        )

        if pops:
            kwargs['pops'] = {"p1": list(SAMPLES[:3]), "p2": list(SAMPLES[3:])}

        return su.Parser(**kwargs).parse()

    spectra = parse(False)
    joint = parse(True)

    assert set(joint.types) == set(spectra.types)
    assert sum(float(np.asarray(joint[t]).sum()) for t in joint.types) == pytest.approx(40000)


def test_fixed_derived_invariant_site_lands_in_divergence_bin(tmp_path):
    """A site without an alternate allele whose ancestral allele differs from the reference is a fixed
    difference, so its mass belongs in the divergence bin rather than the monomorphic one."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "fixed_derived_invariant.vcf", [
        ["1", "1", ".", "A", ".", ".", ".", "AA=C", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
        ["1", "2", ".", "A", ".", ".", ".", "AA=A", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
        ["1", "3", ".", "A", "G", ".", ".", "AA=A", "GT", "0|1", "0|0", "0|0", "0|0", "0|0"],
    ])

    sfs = su.Parser(source=vcf, n=4, skip_non_polarized=False).parse()["all"]

    assert sfs.to_list() == pytest.approx([1.6, 0.4, 0.0, 0.0, 1.0])
    assert sfs.n_div == pytest.approx(1.0)


def test_fixed_derived_invariant_site_matches_alt_encoding(tmp_path):
    """The two encodings of a fixed difference, all haplotypes carrying the alternate allele and no
    alternate allele at all, describe the same biology and must give the same spectrum."""
    Settings.disable_pbar = True

    no_alt = _write_vcf(tmp_path / "no_alt.vcf", [
        ["1", "1", ".", "A", ".", ".", ".", "AA=C", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
    ])

    with_alt = _write_vcf(tmp_path / "with_alt.vcf", [
        ["1", "1", ".", "C", "A", ".", ".", "AA=C", "GT", "1|1", "1|1", "1|1", "1|1", "1|1"],
    ])

    left = su.Parser(source=no_alt, n=4, skip_non_polarized=False).parse()["all"]
    right = su.Parser(source=with_alt, n=4, skip_non_polarized=False).parse()["all"]

    assert left.to_list() == pytest.approx(right.to_list())


def test_fixed_derived_invariant_site_polarized_probabilistically(tmp_path):
    """The ancestral allele probability splits the mass of an invariant site between the two
    monomorphic bins, as it does at a fixed-derived site carrying an alternate allele."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "prob_invariant.vcf", [
        ["1", "1", ".", "A", ".", ".", ".", "AA=C;AA_prob=0.7", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
        ["1", "2", ".", "A", ".", ".", ".", "AA=A;AA_prob=0.7", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
    ])

    sfs = su.Parser(source=vcf, n=4, polarize_probabilistically=True, skip_non_polarized=False).parse()["all"]

    assert sfs.to_list() == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0])


def test_fixed_derived_invariant_site_joint(tmp_path):
    """In joint mode a fixed difference without an alternate allele belongs in the all-derived corner,
    where every population carries the derived allele."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "fixed_derived_joint.vcf", [
        ["1", "1", ".", "A", ".", ".", ".", "AA=C", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
        ["1", "2", ".", "A", ".", ".", ".", "AA=A", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
    ])

    sfs = np.asarray(su.Parser(
        source=vcf,
        n=2,
        pops={"p1": list(SAMPLES[:3]), "p2": list(SAMPLES[3:])},
        skip_non_polarized=False,
    ).parse()["all"])

    assert sfs[-1, -1] == pytest.approx(1.0)
    assert sfs[0, 0] == pytest.approx(1.0)
    assert sfs.sum() == pytest.approx(2.0)


def test_invariant_site_without_ancestral_allele_stays_monomorphic(tmp_path):
    """An invariant site carrying no ancestral allele has no polarization information of its own, so the
    reference allele remains the ancestral one and the site keeps counting as a target site."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "no_aa.vcf", [
        ["1", "1", ".", "A", ".", ".", ".", ".", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
        ["1", "2", ".", "A", ".", ".", ".", "AA=N", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
    ])

    for skip in [True, False]:
        sfs = su.Parser(source=vcf, n=4, skip_non_polarized=skip).parse()["all"]

        assert sfs.to_list() == pytest.approx([2.0, 0.0, 0.0, 0.0, 0.0])


def test_parse_rewinds_the_reader_after_an_interrupted_pass(tmp_path):
    """A parse that raises part-way through leaves the reader mid-input, so the next parse must rewind
    it and see every record again."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "rewind.vcf", _snp_rows(range(1, 201)))

    parser = su.Parser(source=vcf, n=4, filtrations=[_RaisingFiltration(at=100)])

    with pytest.raises(RuntimeError):
        parser.parse()

    parser.filtrations = []

    retried = parser.parse()["all"].to_list()
    fresh = su.Parser(source=vcf, n=4).parse()["all"].to_list()

    assert retried == pytest.approx(fresh)


def test_random_stratification_is_reproducible_across_passes(tmp_path):
    """The random stratification is re-seeded per pass, so parsing the same input twice with the same
    parser assigns the same sites to the same bins."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "random_strat.vcf", _snp_rows(range(1, 301)))

    parser = su.Parser(source=vcf, n=4, stratifications=[su.RandomStratification(n_bins=3, seed=42)])

    first = parser.parse().data.sum().to_dict()
    second = parser.parse().data.sum().to_dict()

    assert first == second

    # and the same as a parser that never saw the input before
    other = su.Parser(source=vcf, n=4, stratifications=[su.RandomStratification(n_bins=3, seed=42)])

    assert other.parse().data.sum().to_dict() == first


def test_target_site_counter_restores_filtrations_when_sampling_raises(tmp_path):
    """The counter suspends the SNP filtration while sampling, and must hand it back to the parser even
    when the sampling pass raises."""
    Settings.disable_pbar = True

    seq = "".join(random.Random(2).choice("ACGT") for _ in range(1000))
    fasta = _write_fasta(tmp_path / "ref.fasta", seq)
    vcf = _write_vcf(tmp_path / "restore.vcf", _snp_rows(range(1, 51), seq))

    # the filtration survives the variant pass and raises in the sampling pass that follows it
    parser = su.Parser(
        source=vcf,
        n=4,
        fasta=fasta,
        filtrations=[SNPFiltration(), _RaisingFiltration(at=80)],
        target_site_counter=su.TargetSiteCounter(n_target_sites=1000, n_samples=200),
    )

    with pytest.raises(RuntimeError):
        parser.parse()

    assert [type(f) for f in parser.filtrations] == [SNPFiltration, _RaisingFiltration]


def test_target_site_counter_handles_fasta_shorter_than_variants(tmp_path):
    """A FASTA record that does not span the parsed variants must not send the sampling pass past its
    end; the sampling is confined to the part backed by the reference."""
    Settings.disable_pbar = True

    seq = "".join(random.Random(3).choice("ACGT") for _ in range(500))
    fasta = _write_fasta(tmp_path / "short.fasta", seq)
    vcf = _write_vcf(tmp_path / "short_ref.vcf", _snp_rows([1, 50, 300, 900]))

    sfs = su.Parser(
        source=vcf,
        n=4,
        fasta=fasta,
        filtrations=[SNPFiltration()],
        target_site_counter=su.TargetSiteCounter(n_target_sites=1000, n_samples=50),
    ).parse()["all"]

    assert sfs.data.sum() == pytest.approx(1000)


def test_chunked_stratification_scales_with_the_number_of_chunks(tmp_path):
    """Locating a site's chunk is a lookup among the cumulative boundaries, so many chunks cost no more
    than few."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "chunks.vcf", _snp_rows(range(1, 20001)))

    def parse(n_chunks):
        strat = [su.ChunkedStratification(n_chunks=n_chunks)] if n_chunks else []

        start = time.process_time()
        spectra = su.Parser(source=vcf, n=4, stratifications=strat).parse()

        return time.process_time() - start, spectra

    few, spectra_few = parse(10)
    many, spectra_many = parse(1000)

    # the totals are unaffected by how the sites are split up
    assert spectra_many.data.to_numpy().sum() == pytest.approx(spectra_few.data.to_numpy().sum())

    # a hundredfold more chunks used to cost close to a hundredfold more time
    assert many < 3 * few


def test_chunked_stratification_assigns_the_same_chunks_as_a_running_total():
    """The cumulative boundaries locate the same chunk as summing the sizes site by site does."""
    strat = su.ChunkedStratification(n_chunks=4)
    strat.chunk_sizes = [3, 2, 0, 2]

    types = [strat.get_type(su.io_handlers.DummyVariant("A", i + 1, "1")) for i in range(9)]

    assert types == ['chunk0', 'chunk0', 'chunk0', 'chunk1', 'chunk1', 'chunk3', 'chunk3', 'chunk3', 'chunk3']
