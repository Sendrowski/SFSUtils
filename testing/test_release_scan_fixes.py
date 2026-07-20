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
