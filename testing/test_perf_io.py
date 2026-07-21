"""
The streaming VCF-Zarr writer and the index-derived site count.

:class:`~sfsutils.io_handlers.ZarrVariantWriter` flushes each complete chunk to the store, so what it
holds is one chunk rather than the whole input. The chunk size is the only thing that changes with it, so
the tests here compare a store written chunk by chunk against one written in a single flush: the arrays
must agree element for element, including on the ragged ploidy and allele axes, which grow after the
first chunk has already reached disk.

:func:`~sfsutils.io_handlers.count_indexed_sites` reads the record count of an indexed VCF from its tabix
index instead of decompressing the file a second time, and must agree with the count a full pass gives.
"""
import ctypes
import glob
import os
import shutil
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

from sfsutils.io_handlers import Variant, ZarrVariantWriter, count_indexed_sites, count_sites
from sfsutils.settings import Settings

Settings.disable_pbar = True

VCF_FIXTURE = "resources/genome/sapiens/chr21_test.vcf.gz"
PLAIN_FIXTURE = "resources/msprime/two_epoch.vcf"


def _vcztools_bin():
    """Locate the vcztools console script (VCZTOOLS_BIN overrides), or None if it is not installed. The
    script sits next to this interpreter even when the env's bin is not on PATH."""
    override = os.environ.get("VCZTOOLS_BIN")
    if override:
        return override
    local = os.path.join(os.path.dirname(sys.executable), "vcztools")
    return local if os.path.exists(local) else shutil.which("vcztools")


class SmallChunkWriter(ZarrVariantWriter):
    """A writer whose chunks hold seven variants, so a short input still spans several of them."""

    _variant_chunk = 7


class SingleChunkWriter(ZarrVariantWriter):
    """A writer whose chunk exceeds any input used here, so everything is written in one flush at close."""

    _variant_chunk = 10 ** 6


def _variants(n):
    """A ragged run of variants: the ploidy and the allele count both grow well after the first chunks
    have been flushed, and an INFO field first appears at the very last variant."""
    for i in range(n):
        alt, gt = ["T"], ["A|T", "T/T"]
        if i == n - 3:
            alt, gt = ["T", "G"], ["A|G", "T/T"]
        if i == n - 2:
            alt, gt = ["T", "G", "C"], ["A|G|C", "T/T/T"]

        info = {"AA": "A", "DP": i}
        if i % 3 == 0:
            info["MAF"] = 0.25
        if i == n - 1:
            info["LATE"] = "x"

        yield Variant(ref="A", pos=i + 1, chrom="1" if i % 4 else "2", gt_bases=gt, alt=alt,
                      is_snp=True, info=info)


def _write(writer_class, path, n):
    """Write ``n`` ragged variants with the given writer class."""
    writer = writer_class(str(path), samples=["a", "b"], seqnames=["1"])
    for variant in _variants(n):
        writer.write(variant)
    writer.close()

    return str(path)


def _assert_stores_equal(a, b):
    """Every array of the two stores holds the same values, with the same shape, dtype and attributes."""
    import zarr

    a, b = zarr.open(a, mode="r"), zarr.open(b, mode="r")

    assert dict(a.attrs) == dict(b.attrs)
    assert sorted(a.array_keys()) == sorted(b.array_keys())

    for name in sorted(a.array_keys()):
        assert a[name].shape == b[name].shape, name
        assert a[name].dtype == b[name].dtype, name
        assert dict(a[name].attrs) == dict(b[name].attrs), name

        left, right = a[name][...], b[name][...]

        if left.dtype.kind == "f":
            # compare the bit patterns, so the missing-float sentinel has to match exactly
            left, right = left.view(np.uint64), right.view(np.uint64)

        assert np.array_equal(left, right), name


def test_streamed_store_matches_a_single_flush(tmp_path):
    """A store written chunk by chunk holds exactly what a store written in one flush holds, including on
    the ragged axes that grow after the first chunks are already on disk."""
    _assert_stores_equal(_write(SmallChunkWriter, tmp_path / "streamed.vcz", 30),
                         _write(SingleChunkWriter, tmp_path / "single.vcz", 30))


def test_streamed_store_matches_a_single_flush_across_the_real_chunk_size(tmp_path):
    """The same at the chunk size the writer uses, so the store spans three chunks of ten thousand."""
    _assert_stores_equal(_write(ZarrVariantWriter, tmp_path / "streamed.vcz", 20005),
                         _write(SingleChunkWriter, tmp_path / "single.vcz", 20005))


def test_store_smaller_than_a_chunk(tmp_path):
    """A store below the chunk size chunks the variants axis whole, and matches the single flush."""
    import zarr

    store = _write(ZarrVariantWriter, tmp_path / "small.vcz", 12)

    assert zarr.open(store, mode="r")["variant_position"].chunks == (12,)
    _assert_stores_equal(store, _write(SingleChunkWriter, tmp_path / "single.vcz", 12))


def test_completed_chunks_reach_the_store_before_close(tmp_path):
    """The writer does not hold the input: the completed chunks are in the store while the writer is still
    being fed."""
    import zarr

    store = str(tmp_path / "streamed.vcz")
    writer = SmallChunkWriter(store, samples=["a", "b"], seqnames=["1"])

    for variant in _variants(20):
        writer.write(variant)

    assert zarr.open(store, mode="r")["variant_position"].shape == (14,)

    writer.close()

    assert zarr.open(store, mode="r")["variant_position"].shape == (20,)


def test_widening_leaves_no_staging_array(tmp_path):
    """Growing a ragged axis rebuilds the array through a staging one, which the finished store must not
    carry."""
    import zarr

    root = zarr.open(_write(SmallChunkWriter, tmp_path / "wide.vcz", 30), mode="r")

    assert not [name for name in root.array_keys() if "staging" in name]
    assert root["call_genotype"].shape == (30, 2, 3)
    assert root["variant_allele"].shape == (30, 4)


def test_an_info_field_of_the_last_variant_covers_all_of_them(tmp_path):
    """A field first seen at the last variant is written for every variant, missing on the earlier ones."""
    import zarr

    root = zarr.open(_write(SmallChunkWriter, tmp_path / "late.vcz", 30), mode="r")
    late = root["variant_LATE"][...]

    assert len(late) == 30
    assert list(late[:-1]) == [""] * 29 and late[-1] == "x"


def test_memory_stays_bounded_by_a_chunk(tmp_path):
    """Writing four chunks allocates no more than a few chunks' worth of buffers: holding every variant
    instead would cost two orders of magnitude more."""
    samples = [f"s{i}" for i in range(20)]
    variant = Variant(ref="A", pos=1, chrom="1", gt_bases=["A|T"] * 20, alt=["T"], is_snp=True)
    writer = ZarrVariantWriter(str(tmp_path / "big.vcz"), samples=samples, seqnames=["1"])

    tracemalloc.start()
    for i in range(40000):
        variant.POS = i + 1
        writer.write(variant)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    writer.close()

    assert peak < 25e6, f"peak of {peak / 1e6:.1f} MB while writing 40000 variants"


@pytest.mark.skipif(_vcztools_bin() is None, reason="no vcztools binary reachable")
def test_vcztools_reads_a_streamed_ragged_store(tmp_path):
    """vcztools reads back a store whose ragged axes were widened mid-stream, so the rebuilt arrays keep
    the chunk grid the reference reader requires."""
    store = _write(ZarrVariantWriter, tmp_path / "ragged.vcz", 20005)
    result = subprocess.run([_vcztools_bin(), "view", store], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    body = [line for line in result.stdout.splitlines() if line and not line.startswith("#")]
    assert len(body) == 20005
    assert body[-2].split("\t")[4] == "T,G,C"


# --- the site count of an indexed VCF ---------------------------------------------------------------

def _index(path):
    """Build a tabix index next to a bgzipped VCF through the htslib that cyvcf2 links, there being no
    tabix binary in this environment. Returns whether the index was built."""
    import cyvcf2

    libraries = glob.glob(os.path.join(os.path.dirname(cyvcf2.__file__), "*.so"))
    if not libraries:
        return False

    library = ctypes.CDLL(libraries[0])
    library.bcf_index_build.argtypes = [ctypes.c_char_p, ctypes.c_int]
    library.bcf_index_build.restype = ctypes.c_int

    return library.bcf_index_build(str(path).encode(), 0) == 0 and os.path.exists(f"{path}.tbi")


@pytest.fixture
def indexed_vcf(tmp_path):
    """A copy of the bgzipped fixture with a tabix index next to it."""
    if not os.path.exists(VCF_FIXTURE):
        pytest.skip("the bgzipped VCF fixture is absent")

    path = tmp_path / os.path.basename(VCF_FIXTURE)
    shutil.copy(VCF_FIXTURE, path)

    if not _index(path):
        pytest.skip("no way to build a tabix index in this environment")

    return str(path)


def test_index_count_matches_a_full_pass(indexed_vcf):
    """The count read from the index is the count a pass over the records gives."""
    counted = sum(1 for line in __import__("gzip").open(indexed_vcf, "rt") if not line.startswith("#"))

    assert count_indexed_sites(indexed_vcf) == counted
    assert count_sites(indexed_vcf) == counted


def test_index_count_respects_max_sites(indexed_vcf):
    """``max_sites`` still caps the count, as the counting path does."""
    assert count_sites(indexed_vcf, max_sites=100) == 100


def test_count_falls_back_without_an_index(tmp_path):
    """Without an index next to it the count comes from a pass over the records, as before."""
    assert count_indexed_sites(PLAIN_FIXTURE) is None
    assert count_indexed_sites("https://example.org/some.vcf.gz") is None
    assert count_indexed_sites(str(tmp_path / "absent.vcf.gz")) is None
    assert count_sites(PLAIN_FIXTURE) == 608


def test_count_falls_back_on_an_unreadable_index(tmp_path, indexed_vcf):
    """A truncated index is not fatal: the count comes from a pass over the records instead."""
    with open(f"{indexed_vcf}.tbi", "wb") as f:
        f.write(b"not an index")

    assert count_indexed_sites(indexed_vcf) is None
    assert count_sites(indexed_vcf) == 1517
