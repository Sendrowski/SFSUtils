"""
VCF-Zarr interoperability, both directions:

  * we read a store written by bio2zarr, with INFO fields typed from the VCF header (a committed
    fixture, so no bio2zarr is needed at test time), and
  * a store written by our own writer is spec-compliant VCF-Zarr: it carries the vcf_zarr_version and
    an ``_ARRAY_DIMENSIONS`` on every array, and ``vcztools`` reads it back to the expected VCF.

``vcztools`` is a dev dependency (it shares the zarr>=3 requirement with bio2zarr), so the live check
runs in this environment; it is skipped only where the binary is absent (a bare install). The
structural compliance assertions run everywhere and cover the metadata vcztools needs.
"""
import os
import shutil
import subprocess
import sys

import pytest

from sfsutils.io_handlers import Variant


BIO2ZARR_FIXTURE = "resources/msprime/typed_info.vcz"


def _vcztools_bin():
    """Locate the vcztools console script (VCZTOOLS_BIN overrides), or None if it is not installed. The
    script sits next to this interpreter even when the env's bin is not on PATH."""
    override = os.environ.get("VCZTOOLS_BIN")
    if override:
        return override
    local = os.path.join(os.path.dirname(sys.executable), "vcztools")
    return local if os.path.exists(local) else shutil.which("vcztools")


# --- direction 1: we read bio2zarr output, with INFO typed from the VCF header ----------------------

@pytest.mark.skipif(not os.path.exists(BIO2ZARR_FIXTURE), reason="the bio2zarr fixture is absent")
def test_reads_bio2zarr_typed_info():
    """A store written by bio2zarr is read with the INFO types the VCF header declared: String -> str,
    Float -> float, Integer -> int."""
    from sfsutils.io_handlers import ZarrVariantReader

    variants = list(ZarrVariantReader(BIO2ZARR_FIXTURE))
    info = variants[0].INFO

    assert isinstance(info["AA"], str) and info["AA"] == "A"
    assert isinstance(info["AA_prob"], float) and info["AA_prob"] == pytest.approx(0.9, abs=1e-6)
    assert isinstance(info["DP"], int) and info["DP"] == 30


# --- direction 2: our writer produces spec-compliant VCF-Zarr --------------------------------------

def _write_store(path):
    from sfsutils.io_handlers import ZarrVariantWriter
    w = ZarrVariantWriter(path, samples=["s1", "s2"], seqnames=["1"], info_ancestral="AA")
    w.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T", "T|T"], alt=["T"], is_snp=True,
                    info={"AA": "A", "AA_prob": 0.9}))
    w.write(Variant(ref="C", pos=20, chrom="1", gt_bases=["C|C", "C|G"], alt=["G"], is_snp=True,
                    info={"AA": "C", "AA_prob": 0.8}))
    w.close()
    return path


def test_writer_output_is_spec_compliant_vcz(tmp_path):
    """The written store carries the vcf_zarr_version and an _ARRAY_DIMENSIONS on every array, the
    metadata a VCF-Zarr reader (vcztools/sgkit) requires."""
    import zarr

    root = zarr.open(_write_store(str(tmp_path / "out.vcz")), mode="r")

    assert root.attrs["vcf_zarr_version"] == "0.5"
    for name in root.array_keys():
        assert "_ARRAY_DIMENSIONS" in root[name].attrs, f"{name} is missing _ARRAY_DIMENSIONS"
    # a few of the axis names that matter for reconstructing the VCF
    assert list(root["variant_position"].attrs["_ARRAY_DIMENSIONS"]) == ["variants"]
    assert list(root["call_genotype"].attrs["_ARRAY_DIMENSIONS"]) == ["variants", "samples", "ploidy"]
    assert list(root["variant_allele"].attrs["_ARRAY_DIMENSIONS"]) == ["variants", "alleles"]


@pytest.mark.skipif(_vcztools_bin() is None,
                    reason="no vcztools binary reachable (needs a zarr-3 env; set VCZTOOLS_BIN)")
def test_vcztools_reads_writer_output(tmp_path):
    """vcztools (the reference VCF-Zarr reader, from a zarr-3 env) reconstructs the expected VCF from a
    store our writer produced."""
    store = _write_store(str(tmp_path / "out.vcz"))
    result = subprocess.run([_vcztools_bin(), "view", store], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    body = [ln for ln in result.stdout.splitlines() if ln and not ln.startswith("#")]
    assert len(body) == 2
    # POS / REF / ALT / genotypes / INFO survive the round-trip
    first = body[0].split("\t")
    assert first[1] == "10" and first[3] == "A" and first[4] == "T"
    assert first[9] == "0|1" and first[10] == "1|1"
    assert "AA=A" in first[7] and "AA_prob=0.9" in first[7]


VCF_FIXTURE = "resources/msprime/two_epoch.vcf"


@pytest.mark.skipif(_vcztools_bin() is None, reason="no vcztools binary reachable")
@pytest.mark.skipif(not os.path.exists(VCF_FIXTURE), reason="the VCF fixture is absent")
def test_vcztools_reads_filterer_vcf_to_vcz_output(tmp_path):
    """The real VCF-in -> vcz-out pipeline (Filterer, no template) produces a store vcztools reads back
    to a VCF, so a plain VCF input yields spec-compliant VCF-Zarr without any template."""
    import sfsutils as su
    from sfsutils.settings import Settings
    Settings.disable_pbar = True

    out = str(tmp_path / "filtered.vcz")
    su.Filterer(source=VCF_FIXTURE, output=out, filtrations=[su.SNPFiltration()]).filter()

    result = subprocess.run([_vcztools_bin(), "view", out], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    body = [ln for ln in result.stdout.splitlines() if ln and not ln.startswith("#")]
    assert len(body) == 608  # every SNP survived and round-tripped
    assert any("##contig" in ln for ln in result.stdout.splitlines())


@pytest.mark.skipif(_vcztools_bin() is None, reason="no vcztools binary reachable")
def test_vcztools_exports_missing_float_as_missing(tmp_path):
    """A float INFO field absent on some sites must export as a missing token ('.'), not the literal
    'nan': the writer uses the VCF-Zarr float-missing sentinel, which vcztools recognises."""
    from sfsutils.io_handlers import ZarrVariantWriter
    out = str(tmp_path / "f.vcz")
    w = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    w.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True, info={"AAP": 0.9}))
    w.write(Variant(ref="C", pos=20, chrom="1", gt_bases=["C|G"], alt=["G"], is_snp=True, info={}))
    w.close()

    result = subprocess.run([_vcztools_bin(), "view", out], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    body = [ln for ln in result.stdout.splitlines() if ln and not ln.startswith("#")]
    assert "nan" not in body[1].lower()  # site 2's INFO must not contain the literal 'nan'
