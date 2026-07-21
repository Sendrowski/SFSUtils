"""
Validate the non-VCF input backends against the VCF backend. A tree sequence, the VCF written from it, and
the VCF-Zarr store converted from that VCF all encode the same genotypes, so parsing any of them must give the
same site-frequency spectrum. These tests need the optional ``tskit`` / ``zarr`` packages (and the committed
``.trees`` / ``.vcz`` fixtures) and are skipped otherwise.
"""
import os

import numpy as np
import pytest

import sfsutils as su
from sfsutils.settings import Settings

VCF = "resources/msprime/two_epoch.vcf"
TREES = "resources/msprime/two_epoch.trees"
VCZ = "resources/msprime/two_epoch.vcz"
REF = "resources/msprime/two_epoch.ref.fasta.gz"


requires_trees = pytest.mark.skipif(
    not (os.path.exists(TREES) and os.path.exists(VCF)),
    reason="the tree-sequence fixture is absent",
)
requires_zarr = pytest.mark.skipif(
    not (os.path.exists(VCZ) and os.path.exists(VCF)),
    reason="the VCF-Zarr fixture is absent",
)

_KW = dict(n=20, skip_non_polarized=False, subsample_mode="random")


def _sfs(source):
    Settings.disable_pbar = True
    return np.array(su.Parser(source=source, **_KW).parse().all.to_list()).astype(int)


@pytest.mark.skipif(not os.path.exists(VCF), reason="the VCF fixture is absent")
def test_source_and_vcf_alias_agree():
    """The preferred ``source=`` and the deprecated ``source=`` alias yield the same SFS."""
    Settings.disable_pbar = True
    from_source = np.array(su.Parser(source=VCF, **_KW).parse().all.to_list()).astype(int)
    from_vcf = np.array(su.Parser(vcf=VCF, **_KW).parse().all.to_list()).astype(int)
    np.testing.assert_array_equal(from_source, from_vcf)


def test_source_vcf_mutually_exclusive_and_required():
    """Providing both ``source`` and ``vcf``, or neither, is an error raised at construction."""
    with pytest.raises(ValueError):
        su.Parser(source=VCF, vcf=VCF, **_KW)

    with pytest.raises(ValueError):
        su.Parser(**_KW)  # neither source nor vcf


@pytest.mark.skipif(not os.path.exists(VCF), reason="the VCF fixture is absent")
def test_vcf_alias_emits_deprecation_warning():
    """The deprecated ``source=`` alias still constructs a working parser but emits a DeprecationWarning; the
    preferred ``source=`` does not warn."""
    import warnings

    Settings.disable_pbar = True

    with pytest.warns(DeprecationWarning, match="vcf"):
        su.Parser(vcf=VCF, **_KW)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        su.Parser(source=VCF, **_KW)  # must not warn


@pytest.mark.skipif(not os.path.exists(VCF), reason="the VCF fixture is absent")
def test_pathlike_source():
    """An ``os.PathLike`` (``pathlib.Path``) source parses like the equivalent string path."""
    import pathlib
    np.testing.assert_array_equal(_sfs(VCF), _sfs(pathlib.Path(VCF)))


@requires_trees
def test_parser_from_variant_reader():
    """A pre-built VariantReader can be passed directly as ``source`` and reproduces the SFS parsed from the
    .trees fixture."""
    import tskit
    from sfsutils.io_handlers import TskitVariantReader

    Settings.disable_pbar = True
    reader = TskitVariantReader(tskit.load(TREES))
    from_reader = np.array(su.Parser(source=reader, **_KW).parse().all.to_list()).astype(int)
    np.testing.assert_array_equal(from_reader, _sfs(TREES))


def test_bare_iterable_source_raises():
    """A bare (non-VariantReader) iterable of sites is not a supported source: it lacks ``samples`` /
    ``seqnames`` / ``count_sites`` and re-iterability, so the parser rejects it with a clear ``TypeError``
    instead of crashing later in ``_prepare_samples_mask``. Covers both a materialised list and a one-shot
    generator."""
    Settings.disable_pbar = True

    for bad in ([object(), object()], (x for x in [object(), object()])):
        with pytest.raises(TypeError, match="VariantReader"):
            su.Parser(source=bad, **_KW).parse()


@requires_trees
def test_variant_reader_source_counted_and_parsed():
    """A supported VariantReader source can be both counted (``n_sites``) and parsed without the count pass
    exhausting the source: counting opens a fresh iteration, so the subsequent parse still sees every site
    and produces a non-empty SFS."""
    import tskit
    from sfsutils.io_handlers import TskitVariantReader

    Settings.disable_pbar = True
    reader = TskitVariantReader(tskit.load(TREES))
    parser = su.Parser(source=reader, **_KW)

    n = parser.n_sites  # counting pass
    assert n > 0

    sfs = parser.parse()  # parse pass, must not see an exhausted source
    assert sfs.all.data.sum() > 0


@pytest.mark.skipif(not os.path.exists(VCF), reason="the VCF fixture is absent")
def test_chunked_stratification_warns_under_filtration(caplog):
    """With active filtrations, ChunkedStratification sizes its chunks from the raw record count while sites
    are chunked only after surviving filtration/projection, so the trailing chunks can come out empty. This is
    documented and a warning is logged at setup."""
    import logging

    Settings.disable_pbar = True

    with caplog.at_level(logging.WARNING, logger="sfsutils"):
        su.Parser(
            source=VCF,
            n=20,
            skip_non_polarized=False,
            stratifications=[su.ChunkedStratification(n_chunks=5)],
            filtrations=[su.SNPFiltration()],
        ).parse()

    assert any(
        "ChunkedStratification sizes its" in r.message and "filtration" in r.message
        for r in caplog.records
    )


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
    spectra = su.Parser(source=TREES, n=20, skip_non_polarized=False, subsample_mode="random",
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
    from_vcf = np.asarray(su.Parser(source=VCF, **kw).parse()["all"]).astype(int)
    from_trees = np.asarray(su.Parser(source=TREES, **kw).parse()["all"]).astype(int)
    np.testing.assert_array_equal(from_trees, from_vcf)


@pytest.mark.skipif(not os.path.exists(REF), reason="the reference FASTA fixture is absent")
@pytest.mark.parametrize("source", ["vcf", "trees", "vcz"])
def test_target_site_counter_input_agnostic(source):
    """The TargetSiteCounter extrapolates to the same total from any input: it depends only on the per-contig
    variant bounds (populated by the source-agnostic parse loop) and the FASTA, not on the input format. The
    tree/zarr fixtures share the VCF's synthetic contig '1', matching the reference FASTA."""
    paths = {"vcf": VCF, "trees": TREES, "vcz": VCZ}

    Settings.disable_pbar = True
    spectra = su.Parser(
        source=paths[source], n=20, skip_non_polarized=False, subsample_mode="random", fasta=REF,
        target_site_counter=su.TargetSiteCounter(n_samples=50_000, n_target_sites=50_000),
    ).parse()

    assert spectra.n_sites.sum() == pytest.approx(50_000)
    assert spectra.n_polymorphic.sum() == 608
