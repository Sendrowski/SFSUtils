"""
Validate the variant writers behind ``Filterer``/``Annotator``: the output format is chosen by the output
file's extension. A VCF-Zarr store can be written from any input; a ``.trees`` file only from a tree-sequence
input (a genealogy cannot be reconstructed from genotypes), and the written store/tree sequence must parse to
the same spectrum as the equivalent VCF output. These need the optional ``tskit`` / ``zarr`` packages and the
committed fixtures, and are skipped otherwise.
"""
import importlib.util
import os

import numpy as np
import pytest

import sfsutils as su
from sfsutils.settings import Settings
from sfsutils.io_handlers import Variant

VCF = "resources/msprime/two_epoch.vcf"
TREES = "resources/msprime/two_epoch.trees"
VCZ = "resources/msprime/two_epoch.vcz"

_has_tskit = importlib.util.find_spec("tskit") is not None
_has_zarr = importlib.util.find_spec("zarr") is not None

requires_zarr = pytest.mark.skipif(
    not (_has_zarr and os.path.exists(VCZ) and os.path.exists(VCF)),
    reason="zarr or the VCF-Zarr fixture is absent",
)
requires_trees = pytest.mark.skipif(
    not (_has_tskit and os.path.exists(TREES) and os.path.exists(VCF)),
    reason="tskit or the tree-sequence fixture is absent",
)

_KW = dict(n=20, skip_non_polarized=False, subsample_mode="random")


def _sfs(source):
    Settings.disable_pbar = True
    return np.array(su.Parser(source=source, **_KW).parse().all.to_list()).astype(int)


def _filter(source, output, filtrations):
    Settings.disable_pbar = True
    su.Filterer(source=source, output=output, filtrations=filtrations).filter()
    return output


# --- format detection ------------------------------------------------------------------------------

def test_output_format_detection():
    from sfsutils.io_handlers import _output_format
    assert _output_format("x.vcz") == "zarr"
    assert _output_format("x.zarr") == "zarr"
    assert _output_format("x.zarr/") == "zarr"
    assert _output_format("x.trees") == "tskit"
    assert _output_format("x.vcf") == "vcf"
    assert _output_format("x.vcf.gz") == "vcf"
    assert _output_format("x.bcf") == "vcf"


# --- VCF-Zarr output (from any input) --------------------------------------------------------------

@requires_zarr
def test_vcf_to_zarr_filter_roundtrips(tmp_path):
    """Filtering a VCF to a VCF-Zarr store gives the same spectrum as filtering it to a VCF."""
    out_vcz = _filter(VCF, str(tmp_path / "out.vcz"), [su.SNPFiltration()])
    out_vcf = _filter(VCF, str(tmp_path / "out.vcf"), [su.SNPFiltration()])
    np.testing.assert_array_equal(_sfs(out_vcz), _sfs(out_vcf))
    assert _sfs(out_vcz)[1:20].sum() > 0


@requires_zarr
def test_zarr_to_zarr_filter_roundtrips(tmp_path):
    """A VCF-Zarr store filtered to a VCF-Zarr store still matches the VCF-filtered spectrum."""
    out_vcz = _filter(VCZ, str(tmp_path / "out.vcz"), [su.SNPFiltration()])
    out_vcf = _filter(VCF, str(tmp_path / "out.vcf"), [su.SNPFiltration()])
    np.testing.assert_array_equal(_sfs(out_vcz), _sfs(out_vcf))


@requires_trees
@requires_zarr
def test_tree_sequence_to_zarr_roundtrips(tmp_path):
    """A tree sequence filtered to a VCF-Zarr store matches the VCF-filtered spectrum (any input -> Zarr)."""
    out_vcz = _filter(TREES, str(tmp_path / "out.vcz"), [su.SNPFiltration()])
    out_vcf = _filter(VCF, str(tmp_path / "out.vcf"), [su.SNPFiltration()])
    np.testing.assert_array_equal(_sfs(out_vcz), _sfs(out_vcf))


@requires_zarr
def test_zarr_writer_persists_info_ancestral(tmp_path):
    """INFO fields (e.g. an annotated ancestral allele) written to a store are read back under INFO."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader

    out = str(tmp_path / "info.vcz")
    writer = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    writer.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"],
                                        is_snp=True, info={"AA": "A"}))
    writer.write(Variant(ref="C", pos=20, chrom="1", gt_bases=["G|G"], alt=["G"],
                                        is_snp=True, info={"AA": "G"}))
    writer.close()

    variants = list(ZarrVariantReader(out, info_ancestral="AA"))
    assert [v.INFO["AA"] for v in variants] == ["A", "G"]
    assert [v.POS for v in variants] == [10, 20]


@requires_zarr
def test_zarr_writer_handles_large_positions(tmp_path):
    """Positions beyond the int32 range must round-trip exactly (large-genome contigs exceed 2^31 bp)."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader

    big = 3_000_000_000  # > 2^31 - 1
    out = str(tmp_path / "big.vcz")
    writer = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    writer.write(Variant(ref="A", pos=big, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True))
    writer.close()

    assert [v.POS for v in ZarrVariantReader(out)] == [big]


@requires_zarr
def test_zarr_writer_skips_reserved_info_names(tmp_path):
    """An INFO field whose name would collide with a reserved coordinate dataset must be skipped, leaving the
    real positions intact rather than overwriting them with strings."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader

    out = str(tmp_path / "reserved.vcz")
    writer = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    writer.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True,
                         info={"position": "junk", "AA": "A"}))
    writer.write(Variant(ref="C", pos=20, chrom="1", gt_bases=["G|G"], alt=["G"], is_snp=True,
                         info={"position": "junk", "AA": "G"}))
    writer.close()

    variants = list(ZarrVariantReader(out, info_ancestral="AA"))
    assert [v.POS for v in variants] == [10, 20]  # not clobbered by the 'position' INFO field
    assert [v.INFO["AA"] for v in variants] == ["A", "G"]


@requires_trees
def test_tree_sequence_to_zarr_preserves_phasing(tmp_path):
    """Tree-sequence haplotypes are phased; the tskit reader emits '|' so a written store records phased genotypes."""
    import tskit
    from sfsutils.io_handlers import TskitVariantReader

    variants = list(TskitVariantReader(tskit.load(TREES)))
    assert variants and all("|" in str(gt) for v in variants for gt in v.gt_bases)

    out = _filter(TREES, str(tmp_path / "phased.vcz"), [su.SNPFiltration()])
    import zarr
    root = zarr.open(out, mode="r")
    assert bool(root["call_genotype_phased"][:].all())


@pytest.mark.skipif(not os.path.exists(VCF), reason="the VCF fixture is absent")
def test_filterer_closes_writer_on_exception(tmp_path):
    """If a filtration raises mid-stream, the writer is still closed (finally), so the partial output is a complete,
    readable file rather than an unflushed, truncated one."""
    Settings.disable_pbar = True

    class _BoomAfterOne(su.Filtration):
        def __init__(self):
            super().__init__()
            self._seen = 0

        def filter_site(self, variant):
            self._seen += 1
            if self._seen > 1:
                raise RuntimeError("boom")
            return True

    out = str(tmp_path / "partial.vcf")
    with pytest.raises(RuntimeError, match="boom"):
        su.Filterer(source=VCF, output=out, filtrations=[_BoomAfterOne()]).filter()

    assert os.path.exists(out)
    from cyvcf2 import VCF as CyVCF
    assert len(list(CyVCF(out))) == 1  # the one site written before the error was flushed by close()


# --- tree-sequence output (from tree-sequence input only) ------------------------------------------

@requires_trees
def test_tree_sequence_subset_write_preserves_topology(tmp_path):
    """Filtering a tree sequence to .trees keeps only surviving sites via delete_sites; genealogy untouched."""
    import tskit

    out = _filter(TREES, str(tmp_path / "out.trees"), [su.SNPFiltration()])
    ts_in, ts_out = tskit.load(TREES), tskit.load(out)

    assert ts_out.num_sites <= ts_in.num_sites
    # the genealogy (edges/trees) is unchanged; only the site table is subset
    assert ts_out.num_edges == ts_in.num_edges
    assert ts_out.num_trees == ts_in.num_trees


@requires_trees
def test_tree_sequence_subset_matches_vcf_filter(tmp_path):
    """Parsing the site-subset .trees equals parsing the same filter applied to the VCF."""
    out_trees = _filter(TREES, str(tmp_path / "out.trees"), [su.SNPFiltration()])
    out_vcf = _filter(VCF, str(tmp_path / "out.vcf"), [su.SNPFiltration()])
    np.testing.assert_array_equal(_sfs(out_trees), _sfs(out_vcf))


@requires_trees
def test_tree_sequence_output_attaches_info_metadata(tmp_path):
    """INFO added while writing a .trees is stored as JSON site metadata on the kept sites (best-effort)."""
    import tskit
    from sfsutils.io_handlers import TskitVariantWriter

    ts = tskit.load(TREES)
    out = str(tmp_path / "meta.trees")

    writer = TskitVariantWriter(ts, out)
    # keep exactly the first two sites, tagging them with INFO
    kept = sorted(int(s.position) + 1 for s in ts.sites())[:2]
    for site in ts.sites():
        pos = int(site.position) + 1
        if pos in kept:
            writer.write(Variant(ref="A", pos=pos, chrom="1", info={"tag": pos}))
    writer.close()

    sub = tskit.load(out)
    assert sub.num_sites == 2
    assert all(site.metadata.get("tag") == int(site.position) + 1 for site in sub.sites())


@requires_trees
def test_tree_sequence_subset_handles_colliding_integer_positions(tmp_path):
    """
    Continuous-genome tree sequences carry non-integer positions that collide under the integer VCF ``POS``.
    The writer must identify kept sites by their exact tskit position, not the truncated ``POS``, or filtered-out
    sites are silently retained.
    """
    import tskit

    # two SNP sites at 5.2 and 5.8 (both -> POS 6), each with a distinct derived allele
    tables = tskit.TableCollection(sequence_length=10)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    root = tables.nodes.add_row(flags=0, time=1)
    tables.edges.add_row(left=0, right=10, parent=root, child=0)
    tables.edges.add_row(left=0, right=10, parent=root, child=1)
    s0 = tables.sites.add_row(position=5.2, ancestral_state="A")
    s1 = tables.sites.add_row(position=5.8, ancestral_state="A")
    tables.mutations.add_row(site=s0, node=0, derived_state="T")
    tables.mutations.add_row(site=s1, node=1, derived_state="G")
    tables.sort()
    ts = tables.tree_sequence()

    trees_in = str(tmp_path / "collide.trees")
    ts.dump(trees_in)

    # keep only the second site; both share POS 6, so a POS-keyed writer would retain both
    class KeepG(su.Filtration):
        def filter_site(self, variant):
            return "G" in list(variant.ALT)

    out = _filter(trees_in, str(tmp_path / "collide_out.trees"), [KeepG()])
    ts_out = tskit.load(out)

    assert ts_out.num_sites == 1
    assert ts_out.site(0).position == 5.8


# --- unsupported combinations ----------------------------------------------------------------------

def test_vcf_to_trees_raises(tmp_path):
    """Writing a .trees from a VCF is rejected: a genealogy cannot be reconstructed from genotypes."""
    Settings.disable_pbar = True
    with pytest.raises(ValueError, match="tree sequence"):
        su.Filterer(source=VCF, output=str(tmp_path / "bad.trees"),
                    filtrations=[su.SNPFiltration()]).filter()


@requires_zarr
def test_zarr_to_trees_raises(tmp_path):
    """Writing a .trees from a VCF-Zarr store is likewise rejected (non-tree-sequence input)."""
    Settings.disable_pbar = True
    with pytest.raises(ValueError, match="tree sequence"):
        su.Filterer(source=VCZ, output=str(tmp_path / "bad.trees"),
                    filtrations=[su.SNPFiltration()]).filter()
