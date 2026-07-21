"""
Regression tests for the defects found by the release-readiness scan: the est-sfs ingroup-count bug,
the Zarr INFO read-back asymmetry, the DataFrame-index serialization, the divide-by-zero in
normalize, and ContigFiltration alias handling. Kept fast and unmarked so they run in the default suite.
"""
import importlib.util

import numpy as np
import pandas as pd
import pytest

import sfsutils as su
from sfsutils.settings import Settings
from sfsutils.io_handlers import Variant
from sfsutils.json_handlers import DataframeHandler

_has_zarr = importlib.util.find_spec("zarr") is not None
requires_zarr = pytest.mark.skipif(not _has_zarr, reason="zarr is absent")


def test_from_est_sfs_ingroup_count_uses_sum_not_max(tmp_path):
    """n_ingroups is the sum of the first row's A,C,G,T counts, not their max; a polymorphic first row
    (max < sum) must not shrink the sample size."""
    # first data row is polymorphic: 6 + 0 + 14 + 0 = 20 ingroups, but max is 14
    est = tmp_path / "polymorphic_first.txt"
    est.write_text("6,0,14,0\t0,0,1,0\n20,0,0,0\t0,0,1,0\n0,0,20,0\t0,0,1,0\n")

    anc = su.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
        file=str(est), model=su.JCSubstitutionModel(), n_runs=1, prior=None, parallelize=False)

    assert anc.n_ingroups == 20


@requires_zarr
def test_zarr_reader_surfaces_all_info_fields(tmp_path):
    """Every INFO field the writer persisted must be readable back, not just the ancestral tag, so an
    annotated store re-parsed by a stratification sees its field."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader

    out = str(tmp_path / "annotated.vcz")
    writer = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    writer.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True,
                         info={"AA": "A", "Degeneracy": "4"}))
    writer.write(Variant(ref="C", pos=20, chrom="1", gt_bases=["G|G"], alt=["G"], is_snp=True,
                         info={"AA": "G", "Degeneracy": "0"}))
    writer.close()

    variants = list(ZarrVariantReader(out, info_ancestral="AA"))
    assert [v.INFO.get("Degeneracy") for v in variants] == ["4", "0"]
    assert [v.INFO.get("AA") for v in variants] == ["A", "G"]


def test_dataframe_handler_preserves_integer_index():
    """A round-trip through DataframeHandler must keep an integer index integer (the default to_dict
    orient turns integer index labels into strings once JSON-encoded)."""
    df = pd.DataFrame({"x": [10, 20, 30]}, index=[0, 1, 2])
    handler = DataframeHandler(context=None)

    # emulate the JSON round-trip: integer dict keys become strings through json, lists do not
    import json
    flat = json.loads(json.dumps(handler.flatten(df, {})))
    restored = handler.restore(flat)

    assert restored.index.tolist() == [0, 1, 2]
    assert restored.index.dtype == df.index.dtype
    pd.testing.assert_frame_equal(restored, df)


def test_dataframe_handler_restores_legacy_payload():
    """A dataframe serialized in the legacy column->index mapping still restores (backward compatibility)."""
    df = pd.DataFrame({"x": [1, 2]})
    restored = DataframeHandler(context=None).restore({"data": df.to_dict()})
    assert restored["x"].tolist() == [1, 2]


def test_normalize_on_spectrum_without_polymorphic_sites():
    """Normalising a spectrum with no polymorphic sites leaves the interior zero rather than dividing by
    zero into NaNs."""
    normalized = su.Spectrum([100, 0, 0, 0, 0]).normalize()
    assert not np.isnan(normalized.to_list()).any()
    assert normalized.to_list()[1:-1] == [0, 0, 0]


def test_contig_filtration_matches_through_aliases():
    """ContigFiltration must match a site whose contig is an alias of a requested contig, not only an
    exact string match."""
    Settings.disable_pbar = True

    vcf = tmp_vcf = None
    import tempfile, os
    tmp = tempfile.mkdtemp()
    vcf = os.path.join(tmp, "v.vcf")
    with open(vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr1,length=100>\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n')
        fh.write("#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "s1"]) + "\n")
        fh.write("\t".join(["chr1", "10", ".", "A", "T", ".", ".", ".", "GT", "0/1"]) + "\n")

    # request contig '1'; the input names it 'chr1' -> only matches if the alias is honoured
    out = os.path.join(tmp, "out.vcf")
    su.Filterer(source=vcf, output=out, filtrations=[su.ContigFiltration(contigs=["1"])],
                aliases={"chr1": ["1"]}).filter()

    from cyvcf2 import VCF
    assert len(list(VCF(out))) == 1


# --- round 2: regression from the round-1 zarr fix, plus edge-case crashes/NaNs -------------------

_FIXTURE = "resources/msprime/two_epoch.vcz"


@requires_zarr
@pytest.mark.skipif(not __import__("os").path.exists(_FIXTURE), reason="the VCF-Zarr fixture is absent")
def test_zarr_reader_does_not_surface_reserved_metadata():
    """A plain vcf2zarr store must not have its reserved variant_* metadata (quality/filter/id/length)
    surfaced as bogus INFO, which would corrupt the typed layout on a round-trip."""
    from sfsutils.io_handlers import ZarrVariantReader
    variant = next(iter(ZarrVariantReader(_FIXTURE)))
    assert dict(variant.INFO) == {}


def test_scale_theta_on_all_monomorphic_spectrum():
    """Scaling theta on a spectrum with no polymorphic sites (theta == 0) must not divide by zero."""
    scaled = su.Spectrum([100, 0, 0, 0]).scale_theta(0.01)
    assert not np.isnan(scaled.to_list()).any()


def test_spectra_normalize_on_empty_spectra():
    """Normalising an entirely empty spectra must not produce NaN columns."""
    empty = su.Spectra.from_spectra({"a": su.Spectrum([0, 0, 0, 0]), "b": su.Spectrum([0, 0, 0, 0])})
    assert not np.isnan(empty.normalize().to_numpy()).any()


def test_target_site_counter_single_position_does_not_crash(tmp_path):
    """A TargetSiteCounter on input whose every contig spans a single position must skip monomorphic
    sampling rather than dividing by a zero range span and raising in rng.multinomial."""
    Settings.disable_pbar = True
    fasta = tmp_path / "g.fasta"
    fasta.write_text(">1\nACGTACGTAC\n")
    vcf = tmp_path / "one.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=1,length=10>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
        "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "s1", "s2"]) + "\n"
        + "\t".join(["1", "5", ".", "A", "T", ".", ".", ".", "GT", "0/1", "0/0"]) + "\n")

    # a single polymorphic site -> its contig spans a single position (range 0)
    spectra = su.Parser(source=str(vcf), n=4, skip_non_polarized=False, fasta=str(fasta),
                        target_site_counter=su.TargetSiteCounter(n_samples=100, n_target_sites=1000)).parse()
    assert spectra["all"].n_polymorphic == 1


def _prob_vcf(tmp_path):
    """A small VCF carrying AA + a Float AA_prob tag."""
    vcf = tmp_path / "prob.vcf"
    header = [
        "##fileformat=VCFv4.2", "##contig=<ID=1,length=100>",
        '##INFO=<ID=AA,Number=1,Type=String,Description="aa">',
        '##INFO=<ID=AA_prob,Number=1,Type=Float,Description="p">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">',
        "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "a", "b"]),
    ]
    # numeric GT allele indices (0=REF A, 1=ALT T), as htslib requires for a real VCF
    rows = ["\t".join(["1", str(p), ".", "A", "T", ".", ".", "AA=A;AA_prob=0.9", "GT", "0|1", "1|1"])
            for p in (10, 20, 30)]
    vcf.write_text("\n".join(header + rows) + "\n")
    return str(vcf)


def _prob_vcz(tmp_path):
    """The same sites as a VCF-Zarr store (INFO stored as strings)."""
    from sfsutils.io_handlers import ZarrVariantWriter
    out = str(tmp_path / "prob.vcz")
    w = ZarrVariantWriter(out, samples=["a", "b"], seqnames=["1"], info_ancestral="AA")
    for p in (10, 20, 30):
        w.write(Variant(ref="A", pos=p, chrom="1", gt_bases=["A|T", "T|T"], alt=["T"], is_snp=True,
                        info={"AA": "A", "AA_prob": 0.9}))
    w.close()
    return out


@requires_zarr
def test_probabilistic_polarization_agrees_across_vcf_and_zarr(tmp_path):
    """AA_prob is typed by cyvcf2 but a string from the Zarr backend; probabilistic polarization must
    cast it and give the same spectrum from either source (previously the Zarr path raised on '0.9'*array)."""
    Settings.disable_pbar = True

    def spectrum(source):
        return su.Parser(source=source, n=4, skip_non_polarized=True,
                         polarize_probabilistically=True).parse()["all"].to_list()

    from_vcf = spectrum(_prob_vcf(tmp_path))
    from_vcz = spectrum(_prob_vcz(tmp_path))
    # equal within float precision (cyvcf2 returns AA_prob as float32, the Zarr string casts to float64)
    np.testing.assert_allclose(from_vcf, from_vcz, atol=1e-6)
    assert sum(from_vcf) > 0  # sites were actually kept and polarized


def test_ancestral_prob_sentinels_treated_as_unpolarized():
    """Empty / '.' AA_prob values are treated as certain (probability 1), like a missing tag."""
    from sfsutils.io_handlers import Variant as V
    p = su.Parser(source=None, vcf="x", n=4, polarize_probabilistically=True) if False else None
    # exercise _get_ancestral_prob directly against the sentinels
    parser = su.Parser.__new__(su.Parser)
    parser.polarize_probabilistically = True
    parser.info_ancestral_prob = "AA_prob"
    parser.n_aa_prob = 0
    for sentinel in ("", ".", None):
        v = V(ref="A", pos=1, chrom="1", alt=["T"], is_snp=True,
              info={} if sentinel is None else {"AA_prob": sentinel})
        assert parser._get_ancestral_prob(v) == 1.0
    # a real string value is cast to float
    v = V(ref="A", pos=1, chrom="1", alt=["T"], is_snp=True, info={"AA_prob": "0.75"})
    assert parser._get_ancestral_prob(v) == 0.75


@requires_zarr
def test_zarr_info_round_trips_with_native_types(tmp_path):
    """INFO written through our own Zarr writer round-trips with native types (str/float/int), so a
    numeric field is a number on read, matching cyvcf2 rather than becoming a string."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader

    out = str(tmp_path / "typed.vcz")
    w = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    w.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True,
                    info={"AA": "A", "AA_prob": 0.9, "DP": 30}))
    w.close()

    info = next(iter(ZarrVariantReader(out))).INFO
    assert info["AA"] == "A" and isinstance(info["AA"], str)
    assert info["AA_prob"] == 0.9 and isinstance(info["AA_prob"], float)
    assert info["DP"] == 30 and isinstance(info["DP"], int)


@requires_zarr
def test_zarr_info_missing_value_is_absent(tmp_path):
    """A numeric INFO field present on some sites but not others reads back as absent (NaN omitted), the
    way cyvcf2 reports a missing INFO field, rather than as a NaN or empty string."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader

    out = str(tmp_path / "miss.vcz")
    w = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    w.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True, info={"AA_prob": 0.9}))
    w.write(Variant(ref="C", pos=20, chrom="1", gt_bases=["C|G"], alt=["G"], is_snp=True, info={}))
    w.close()

    variants = list(ZarrVariantReader(out))
    assert variants[0].INFO["AA_prob"] == 0.9
    assert "AA_prob" not in variants[1].INFO
