"""
Chunking of the VCF-Zarr stores our writer produces: the spec demands one chunk size along
``variants`` across every variant and call array, and one along ``samples``, which zarr's
auto-chunking does not provide. ``vcztools`` is the reference reader that enforces it, so where the
binary is reachable we check a store it rejected before.
"""
import os
import shutil
import subprocess
import sys

import pytest

from sfsutils.io_handlers import Variant


def _vcztools_bin():
    """Locate the vcztools console script (VCZTOOLS_BIN overrides), or None if it is not installed. The
    script sits next to this interpreter even when the env's bin is not on PATH."""
    override = os.environ.get("VCZTOOLS_BIN")
    if override:
        return override
    local = os.path.join(os.path.dirname(sys.executable), "vcztools")
    return local if os.path.exists(local) else shutil.which("vcztools")


def _write_store(path, n_variants, n_samples):
    """Write a store of ``n_variants`` biallelic phased sites over ``n_samples`` samples."""
    from sfsutils.io_handlers import ZarrVariantWriter

    samples = [f"s{i}" for i in range(n_samples)]
    w = ZarrVariantWriter(path, samples=samples, seqnames=["1"], info_ancestral="AA")
    for i in range(n_variants):
        w.write(Variant(ref="A", pos=i + 1, chrom="1", gt_bases=["A|T"] * n_samples, alt=["T"],
                        is_snp=True, info={"AA": "A"}))
    w.close()
    return path


def test_variant_arrays_share_a_chunk_grid(tmp_path):
    """Enough variants and samples to reach the chunk sizes: every array carrying a ``variants`` axis
    chunks it identically, and every ``call_*`` array chunks ``samples`` identically."""
    import zarr

    root = zarr.open(_write_store(str(tmp_path / "big.vcz"), n_variants=25000, n_samples=1500), mode="r")

    variant_chunks, sample_chunks = set(), set()
    for name in root.array_keys():
        dimensions = list(root[name].attrs["_ARRAY_DIMENSIONS"])
        chunks = root[name].chunks
        for dim, chunk in zip(dimensions, chunks):
            if dim == "variants":
                variant_chunks.add(chunk)
            elif dim == "samples" and name.startswith("call_"):
                sample_chunks.add(chunk)

    assert variant_chunks == {10000}
    assert sample_chunks == {1000}


def test_chunks_do_not_exceed_the_array(tmp_path):
    """A store smaller than the chunk sizes chunks each axis whole, and still uniformly."""
    import zarr

    root = zarr.open(_write_store(str(tmp_path / "small.vcz"), n_variants=50, n_samples=4), mode="r")

    assert root["variant_position"].chunks == (50,)
    assert root["variant_contig"].chunks == (50,)
    assert root["variant_allele"].chunks == (50, 2)
    assert root["call_genotype"].chunks == (50, 4, 2)
    assert root["call_genotype_phased"].chunks == (50, 4)


@pytest.mark.skipif(_vcztools_bin() is None,
                    reason="no vcztools binary reachable (needs a zarr-3 env; set VCZTOOLS_BIN)")
def test_vcztools_reads_a_store_larger_than_one_chunk(tmp_path):
    """vcztools reconstructs the VCF from a store spanning several chunks along the variants axis, which
    it rejects unless the chunk grids line up."""
    store = _write_store(str(tmp_path / "multi.vcz"), n_variants=21000, n_samples=4)
    result = subprocess.run([_vcztools_bin(), "view", store], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    body = [ln for ln in result.stdout.splitlines() if ln and not ln.startswith("#")]
    assert len(body) == 21000
