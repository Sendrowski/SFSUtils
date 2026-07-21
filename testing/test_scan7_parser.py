"""
Regression tests for the parser defects found by the seventh release-readiness scan: probabilistic
polarization at fixed-derived sites, the joint target-site accounting, the chunk assignment of the
target-site sampling pass, and the reader that was reopened after being closed. Kept fast and unmarked
so they run in the default suite.
"""

import types

import numpy as np
import pytest

import sfsutils as su
from sfsutils.io_handlers import VCFHandler
from sfsutils.settings import Settings
from sfsutils.spectrum import Spectra

HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length=1000>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##INFO=<ID=AA_prob,Number=1,Type=Float,Description="ancestral allele probability">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)


def _write_vcf(path, rows, samples=("s1", "s2")):
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


def _write_fasta(path, length=1000, contig="1"):
    """
    Write a single-contig FASTA of a constant base.

    :param path: The path to write to.
    :param length: The contig length.
    :param contig: The contig name.
    :return: The path as a string.
    """
    path.write_text(f">{contig}\n" + "A" * length + "\n")

    return str(path)


def test_probabilistic_polarization_applies_to_fixed_derived_sites(tmp_path):
    """A site fixed for the derived allele shows a single observed base but is bi-allelic, so its mass
    must be split by the ancestral allele probability like any other site."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "fixed_derived.vcf", [
        ["1", "10", ".", "A", "T", ".", ".", "AA=A;AA_prob=0.7", "GT", "1|1", "1|1"],
        ["1", "20", ".", "A", "T", ".", ".", "AA=A;AA_prob=0.7", "GT", "0|1", "1|1"],
    ])

    for mode in ["random", "probabilistic"]:
        sfs = su.Parser(source=vcf, n=4, polarize_probabilistically=True, subsample_mode=mode).parse()["all"]

        # the fixed-derived site contributes 0.7 to the divergence bin and 0.3 to the monomorphic bin
        assert sfs.to_list() == pytest.approx([0.3, 0.3, 0.0, 0.7, 0.7])
        assert sfs.n_div == pytest.approx(0.7)


def test_probabilistic_polarization_applies_to_fixed_derived_sites_joint(tmp_path):
    """The joint projection reflects a fixed-derived site on all axes, moving mass off the all-derived
    corner and onto the origin."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "fixed_derived_joint.vcf", [
        ["1", "10", ".", "A", "T", ".", ".", "AA=A;AA_prob=0.7", "GT", "1|1", "1|1"],
    ], samples=("s1", "s2"))

    sfs = np.asarray(su.Parser(
        source=vcf,
        pops={"A": ["s1"], "B": ["s2"]},
        n={"A": 2, "B": 2},
        polarize_probabilistically=True,
        subsample_mode="probabilistic",
    ).parse()["all"])

    assert sfs[2, 2] == pytest.approx(0.7)
    assert sfs[0, 0] == pytest.approx(0.3)


def test_subsample_modes_agree_at_fixed_derived_site(tmp_path):
    """Both subsample modes must place the same mass at a site with a single observed base, as the
    down-projection there is deterministic."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "modes.vcf", [
        ["1", "10", ".", "A", "T", ".", ".", "AA=A;AA_prob=0.8", "GT", "1|1", "1|1"],
        ["1", "20", ".", "A", "T", ".", ".", "AA=A;AA_prob=0.8", "GT", "0|0", "0|0"],
    ])

    def parse(mode):
        return su.Parser(source=vcf, n=4, polarize_probabilistically=True,
                         subsample_mode=mode).parse()["all"].to_list()

    assert parse("random") == pytest.approx(parse("probabilistic"))


def _joint_counter(n_target_sites, sampled, polymorphic, shape=(4, 4)):
    """
    Build a target site counter primed with a synthetic joint spectrum.

    :param n_target_sites: The number of target sites.
    :param sampled: The monomorphic mass sampled from the FASTA file per type.
    :param polymorphic: The polymorphic mass per type.
    :param shape: The shape of the joint spectrum.
    :return: The counter and the per-type joint SFS after sampling.
    """
    counter = su.TargetSiteCounter(n_samples=int(sum(sampled.values())), n_target_sites=n_target_sites)
    counter.parser = types.SimpleNamespace(_joint_shape=shape)

    before = {}
    for t, n in polymorphic.items():
        arr = np.zeros(shape)
        arr[1, 1] = n
        before[t] = arr

    counter._sfs_polymorphic = before

    sfs = {}
    for t, arr in before.items():
        after = arr.copy()
        after[(0,) * len(shape)] += sampled[t]
        sfs[t] = after

    return counter, sfs


def _one_dimensional_totals(n_target_sites, sampled, polymorphic):
    """
    Run the one-dimensional target-site accounting on the same numbers.

    :param n_target_sites: The number of target sites.
    :param sampled: The monomorphic mass sampled from the FASTA file per type.
    :param polymorphic: The polymorphic mass per type.
    :return: The total number of sites per type.
    """
    counter = su.TargetSiteCounter(n_samples=int(sum(sampled.values())), n_target_sites=n_target_sites)
    counter._sfs_polymorphic = Spectra({t: [0.0, polymorphic[t], 0.0, 0.0, 0.0] for t in polymorphic})
    after = Spectra({t: [sampled[t], polymorphic[t], 0.0, 0.0, 0.0] for t in polymorphic})

    return counter._update_target_sites(after).data.sum().to_dict()


@pytest.mark.parametrize("n_target_sites", [1000000, 20000])
def test_joint_target_sites_match_one_dimensional_accounting(n_target_sites):
    """The sites sampled from the FASTA file estimate the composition of all sites, not of the monomorphic
    ones, so each type is scaled to its share of the target sites just as in the one-dimensional path."""
    sampled = {"a": 3000.0, "b": 7000.0}
    polymorphic = {"a": 6000.0, "b": 4000.0}

    counter, sfs = _joint_counter(n_target_sites, sampled, polymorphic)
    joint = {t: float(arr.sum()) for t, arr in counter._update_target_sites_joint(sfs).items()}

    assert joint == pytest.approx(_one_dimensional_totals(n_target_sites, sampled, polymorphic))
    assert sum(joint.values()) == pytest.approx(n_target_sites)

    # the shares follow the sampled composition, not the composition of the monomorphic sites alone
    assert joint["a"] / joint["b"] == pytest.approx(sampled["a"] / sampled["b"])


def test_joint_target_sites_clip_negative_monomorphic_counts():
    """A type whose observed sites outnumber its share of the target sites is clipped to zero monomorphic
    sites rather than left with a negative mutational opportunity."""
    sampled = {"a": 1000.0, "b": 9000.0}
    polymorphic = {"a": 8000.0, "b": 1000.0}

    counter, sfs = _joint_counter(20000, sampled, polymorphic)
    updated = counter._update_target_sites_joint(sfs)

    # type 'a' is entitled to 2000 sites but was observed at 8000
    assert updated["a"][0, 0] == 0
    assert updated["a"][1, 1] == 8000
    assert updated["b"][0, 0] == pytest.approx(9000 * 2 - 1000)


def test_joint_target_sites_preserve_divergence_corner():
    """The fixed-derived corner is monomorphic but observed, so it survives the update and consumes
    target-site budget."""
    counter = su.TargetSiteCounter(n_samples=100, n_target_sites=10000)
    counter.parser = types.SimpleNamespace(_joint_shape=(3, 3))

    before = np.zeros((3, 3))
    before[1, 1] = 400.0
    before[2, 2] = 100.0
    counter._sfs_polymorphic = {"all": before}

    after = before.copy()
    after[0, 0] = 100.0

    updated = counter._update_target_sites_joint({"all": after})["all"]

    assert updated[2, 2] == 100.0
    assert updated[0, 0] == pytest.approx(10000 - 500)
    assert updated.sum() == pytest.approx(10000)


def test_chunked_stratification_assigns_sampled_sites_by_position(tmp_path):
    """The target-site sampling pass visits more sites than the first pass, so chunks are assigned by
    genomic position; the monomorphic mass then follows each chunk's genomic span."""
    Settings.disable_pbar = True

    # ten dense variants followed by ten sparse ones, so the two chunks span very different lengths
    positions = list(range(1, 11)) + list(range(500, 1000, 50))
    rows = [["1", str(pos), ".", "A", "T", ".", ".", ".", "GT", "0|1", "0|0"] for pos in positions]

    vcf = _write_vcf(tmp_path / "chunks.vcf", rows)
    fasta = _write_fasta(tmp_path / "chunks.fasta")

    strat = su.ChunkedStratification(2)

    spectra = su.Parser(
        source=vcf,
        fasta=fasta,
        n=4,
        seed=0,
        skip_non_polarized=False,
        subsample_mode="random",
        stratifications=[strat],
        filtrations=[su.SNPFiltration()],
        target_site_counter=su.TargetSiteCounter(n_samples=2000, n_target_sites=100000),
    ).parse()

    # the first pass split the variants evenly, so the boundary sits at the eleventh one
    assert strat._chunk_starts == [(0, 1), (0, 500)]

    monomorphic = spectra.data.iloc[0]
    span = {"chunk0": 500 - 1, "chunk1": 950 - 500}

    for t, expected in span.items():
        assert monomorphic[t] / monomorphic.sum() == pytest.approx(expected / sum(span.values()), abs=0.05)


def test_chunked_stratification_counts_sites_on_the_first_pass(tmp_path):
    """Without a second pass the chunks still hold equal numbers of sites, which is what the chunk sizes
    are computed for."""
    Settings.disable_pbar = True

    rows = [["1", str(pos), ".", "A", "T", ".", ".", ".", "GT", "0|1", "0|0"] for pos in range(1, 21)]
    vcf = _write_vcf(tmp_path / "counted.vcf", rows)

    spectra = su.Parser(
        source=vcf,
        n=4,
        skip_non_polarized=False,
        subsample_mode="random",
        stratifications=[su.ChunkedStratification(4)],
    ).parse()

    assert spectra.data.sum().to_dict() == {f"chunk{i}": 5.0 for i in range(4)}


def test_two_sfs_target_site_counter_does_not_reopen_the_source(tmp_path, monkeypatch):
    """The region length is read before the reader is closed, so the source is opened exactly once and no
    reader is left behind on the parser."""
    Settings.disable_pbar = True

    rows = [["1", str(pos), ".", "A", "T", ".", ".", ".", "GT", "0|1", "0|0"] for pos in range(1, 200, 10)]
    vcf = _write_vcf(tmp_path / "two_sfs.vcf", rows)

    opens = []
    original = VCFHandler._open_reader

    def counting_open(self, *args, **kwargs):
        opens.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VCFHandler, "_open_reader", counting_open)

    parser = su.Parser(
        source=vcf,
        n=4,
        two_sfs=True,
        d=50,
        skip_non_polarized=False,
        subsample_mode="random",
        filtrations=[su.SNPFiltration()],
        target_site_counter=su.TargetSiteCounter(n_samples=10, n_target_sites=10000),
    )

    parser.parse()

    assert len(opens) == 1
    assert "_reader" not in parser.__dict__
