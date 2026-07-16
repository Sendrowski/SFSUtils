"""
Validate the non-VCF input backends against the VCF backend. A tree sequence, the VCF written from it, and
the VCF-Zarr store converted from that VCF all encode the same genotypes, so parsing any of them must give the
same site-frequency spectrum. These tests need the optional ``tskit`` / ``zarr`` packages (and the committed
``.trees`` / ``.vcz`` fixtures) and are skipped otherwise.
"""
import importlib.util
import os

import numpy as np
import pytest

import sfsutils as su
from sfsutils.settings import Settings

VCF = "resources/msprime/two_epoch.vcf"
TREES = "resources/msprime/two_epoch.trees"
VCZ = "resources/msprime/two_epoch.vcz"
REF = "resources/msprime/two_epoch.ref.fasta.gz"

_has_tskit = importlib.util.find_spec("tskit") is not None
_has_zarr = importlib.util.find_spec("zarr") is not None

requires_trees = pytest.mark.skipif(
    not (_has_tskit and os.path.exists(TREES) and os.path.exists(VCF)),
    reason="tskit or the tree-sequence fixture is absent",
)
requires_zarr = pytest.mark.skipif(
    not (_has_zarr and os.path.exists(VCZ) and os.path.exists(VCF)),
    reason="zarr or the VCF-Zarr fixture is absent",
)

_KW = dict(n=20, skip_non_polarized=False, subsample_mode="random")


def _sfs(source):
    Settings.disable_pbar = True
    return np.array(su.Parser(vcf=source, **_KW).parse().all.to_list()).astype(int)


@requires_trees
def test_tree_sequence_reproduces_vcf():
    """Parsing the tree sequence gives the same SFS as parsing the VCF written from it."""
    np.testing.assert_array_equal(_sfs(TREES), _sfs(VCF))
    assert _sfs(TREES)[1:20].sum() > 0


@requires_trees
def test_tree_sequence_object_and_path_agree():
    """Passing an in-memory TreeSequence gives the same result as passing the .trees path."""
    import tskit
    ts = tskit.load(TREES)
    np.testing.assert_array_equal(_sfs(ts), _sfs(TREES))


@requires_zarr
def test_vcf_zarr_reproduces_vcf():
    """Parsing the VCF-Zarr store gives the same SFS as parsing the VCF it was converted from."""
    np.testing.assert_array_equal(_sfs(VCZ), _sfs(VCF))
    assert _sfs(VCZ)[1:20].sum() > 0


@requires_trees
def test_contig_stratification_on_tree_sequence():
    """ContigStratification must work on a tree sequence: its get_types() reads the reader's seqnames,
    which the tree/zarr readers now expose (previously only cyvcf2.VCF did, so this raised AttributeError)."""
    Settings.disable_pbar = True
    spectra = su.Parser(vcf=TREES, n=20, skip_non_polarized=False, subsample_mode="random",
                        stratifications=[su.ContigStratification()]).parse()
    # the tree sequence has a single synthetic contig "1"
    assert spectra.types == ["1"]
    assert spectra["1"].n_polymorphic > 0


@requires_trees
def test_reader_seqnames_exposed():
    """Both non-VCF readers expose seqnames (used by ContigStratification and the contig filtration)."""
    import tskit
    from sfsutils.io_handlers import TskitVariantReader
    reader = TskitVariantReader(tskit.load(TREES))
    assert reader.seqnames == ["1"]


@requires_trees
def test_tree_sequence_joint_matches_vcf():
    """The joint SFS from the tree sequence matches the joint SFS from the VCF (shared sample identity)."""
    Settings.disable_pbar = True
    pops = {"A": [f"tsk_{i}" for i in range(5)], "B": [f"tsk_{i}" for i in range(5, 10)]}
    kw = dict(pops=pops, n={"A": 10, "B": 10}, skip_non_polarized=False, subsample_mode="random")
    from_vcf = np.asarray(su.Parser(vcf=VCF, **kw).parse()["all"]).astype(int)
    from_trees = np.asarray(su.Parser(vcf=TREES, **kw).parse()["all"]).astype(int)
    np.testing.assert_array_equal(from_trees, from_vcf)


@pytest.mark.skipif(not os.path.exists(REF), reason="the reference FASTA fixture is absent")
@pytest.mark.parametrize("source", ["vcf", "trees", "vcz"])
def test_target_site_counter_input_agnostic(source):
    """The TargetSiteCounter extrapolates to the same total from any input: it depends only on the per-contig
    variant bounds (populated by the source-agnostic parse loop) and the FASTA, not on the input format. The
    tree/zarr fixtures share the VCF's synthetic contig '1', matching the reference FASTA."""
    paths = {"vcf": VCF, "trees": TREES, "vcz": VCZ}
    if source == "trees" and not _has_tskit:
        pytest.skip("tskit is absent")
    if source == "vcz" and not _has_zarr:
        pytest.skip("zarr is absent")

    Settings.disable_pbar = True
    spectra = su.Parser(
        vcf=paths[source], n=20, skip_non_polarized=False, subsample_mode="random", fasta=REF,
        target_site_counter=su.TargetSiteCounter(n_samples=50_000, n_target_sites=50_000),
    ).parse()

    assert spectra.n_sites.sum() == pytest.approx(50_000)
    assert spectra.n_polymorphic.sum() == 608
