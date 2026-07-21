"""
Unit tests for the duck-typed :class:`Variant` abstraction and the monomorphic-site classifier, covering
regressions found in review: an insertion from the tskit/Zarr backends must not be classified as a
monomorphic SNP, and the mono-allelic :class:`DummyVariant` must expose the full Variant interface.
"""
import numpy as np

from sfsutils.io_handlers import Variant, DummyVariant, is_monomorphic_snp


def test_dummy_variant_has_full_interface():
    v = DummyVariant("A", 10, "1")
    # delegates to Variant.__init__, so the whole duck-typed surface is present
    assert v.REF == "A" and v.POS == 10 and v.CHROM == "1"
    assert v.ALT == [] and v.INFO == {}
    assert v.is_snp is False and v.is_mnp is False
    assert isinstance(v.gt_bases, np.ndarray) and len(v.gt_bases) == 0


def test_monomorphic_classifier_true_for_mono_allelic():
    # a genuine mono-allelic site (no ALT) is monomorphic
    assert is_monomorphic_snp(DummyVariant("A", 1, "1")) is True
    assert is_monomorphic_snp(Variant(ref="C", pos=1, chrom="1", alt=[])) is True


def test_monomorphic_classifier_false_for_insertion():
    # a biallelic insertion (REF='A', ALT=['AT']) is NOT a monomorphic SNP; the tskit/Zarr readers do
    # not set is_indel, so the ALT guard is what keeps it out of the monomorphic (derived 0) bin
    insertion = Variant(ref="A", pos=1, chrom="1", alt=["AT"], is_snp=False)
    assert is_monomorphic_snp(insertion) is False


def test_monomorphic_classifier_false_for_snp():
    snp = Variant(ref="A", pos=1, chrom="1", alt=["T"], is_snp=True)
    assert is_monomorphic_snp(snp) is False


def test_dummy_variant_satisfies_site_contract():
    """DummyVariant must expose the full Site interface; with n_samples its gt_bases is a per-sample
    array aligned with the sample masks (a monomorphic reference site), built lazily."""
    import numpy as np
    from sfsutils.io_handlers import DummyVariant, Site

    v = DummyVariant(ref="A", pos=10, chrom="1", n_samples=3)
    assert isinstance(v, Site)
    assert list(v.gt_bases) == ["A/A", "A/A", "A/A"]
    assert list(v.gt_bases[np.array([True, False, True])]) == ["A/A", "A/A"]  # indexable by a mask
    assert v.ALT == [] and v.INFO == {} and v.is_snp is False
