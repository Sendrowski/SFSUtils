"""
Focused unit tests for otherwise-uncovered public behaviour: Spectrum/Spectra helpers and
properties, jsonpickle (de)serialization handlers, the parallelization helpers, and the simple
site filters. All synthetic, no VCF/FASTA fixtures required, so they run in the fast tier.
"""
import numpy as np
import jsonpickle
import pytest

import sfsutils as su
from sfsutils import Spectrum, Spectra
from sfsutils._parallelization import parallelize, check_bounds
from sfsutils.io_handlers import DummyVariant


# --- Spectrum ---------------------------------------------------------------------------------

def test_spectrum_to_numpy():
    data = [0, 10, 6, 3, 1, 0]
    np.testing.assert_array_equal(Spectrum(data).to_numpy(), np.array(data, dtype=float))


def test_spectrum_from_polymorphic_pads_with_zeros():
    sfs = Spectrum.from_polymorphic([5, 3, 1])
    assert sfs.to_list() == [0, 5, 3, 1, 0]


def test_spectrum_from_list():
    assert Spectrum.from_list([0, 1, 2, 0]).to_list() == [0, 1, 2, 0]


def test_spectrum_resample_accepts_generator_seed():
    sfs = Spectrum([0, 10, 6, 3, 1, 0])
    resampled = sfs.resample(np.random.default_rng(0))
    assert resampled.n == sfs.n


def test_spectrum_subsample_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown subsampling mode"):
        Spectrum([0, 10, 6, 3, 1, 0]).subsample(3, mode="bogus")


def test_get_neutral_with_explicit_rates():
    n = 10
    sfs = Spectrum.get_neutral(theta=0.01, n_sites=1000, n=n, r=[1.0] * (n - 1))
    assert sfs.n == n


def test_get_neutral_wrong_rate_length_raises():
    with pytest.raises(ValueError, match="length of r must be"):
        Spectrum.get_neutral(theta=0.01, n_sites=1000, n=10, r=[1.0] * 5)


def test_spectrum_watterson_estimators():
    # Theta = n_polymorphic / sum(1/i, i=1..n-1); theta = Theta / n_sites
    sfs = Spectrum([0, 10, 6, 3, 1, 0])
    assert sfs.Theta == pytest.approx(9.6)
    assert sfs.theta == pytest.approx(0.48)


# --- Spectra ----------------------------------------------------------------------------------

def _spectra():
    return Spectra.from_dict({"a": [0, 10, 6, 3, 1, 0], "b": [0, 5, 2, 1, 0, 0]})


def test_spectra_watterson_series():
    sp = _spectra()
    assert sp.Theta["a"] == pytest.approx(9.6)
    assert sp.theta["a"] == pytest.approx(0.48)


def test_spectra_iter_yields_types():
    assert list(iter(_spectra())) == ["a", "b"]


def test_spectra_get_empty_same_shape_zero_counts():
    empty = _spectra().get_empty()
    assert empty.to_dict() == {"a": [0, 0, 0, 0, 0, 0], "b": [0, 0, 0, 0, 0, 0]}


def test_spectra_combine():
    combined = _spectra().combine(Spectra.from_dict({"c": [0, 1, 1, 0, 0, 0]}))
    assert combined.types == ["a", "b", "c"]


def test_spectra_resample_all_types():
    assert _spectra().resample(0).types == ["a", "b"]


def test_spectra_setitem():
    sp = Spectra.from_dict({"a": [0, 1, 2, 0]})
    sp["z"] = Spectrum([0, 3, 1, 0])
    assert sp.types == ["a", "z"]


def test_spectra_print_runs(capsys):
    _spectra().print()
    assert "a" in capsys.readouterr().out


# --- jsonpickle handlers ----------------------------------------------------------------------

def test_jsonpickle_spectrum_roundtrip():
    sfs = Spectrum([0, 5, 3, 0])
    assert jsonpickle.decode(jsonpickle.encode(sfs)).to_list() == sfs.to_list()


def test_jsonpickle_spectra_roundtrip():
    sp = _spectra()
    assert jsonpickle.decode(jsonpickle.encode(sp)).to_dict() == sp.to_dict()


# --- parallelization helpers ------------------------------------------------------------------

def test_parallelize_sequential_without_array_wrap():
    assert parallelize(str, [1, 2, 3], parallelize=False, wrap_array=False) == ["1", "2", "3"]


def test_check_bounds_linear_scale():
    near_lower, near_upper = check_bounds({"x": (0.0, 10.0)}, {"x": 0.05}, scale="lin")
    assert near_lower["x"] == (0.0, 0.05, 10.0)


# --- simple filters ---------------------------------------------------------------------------

def _dummy(ref="A", alt=None):
    v = DummyVariant(ref, 1, "chr1")
    v.ALT = alt or []
    return v


def test_snv_filtration_keeps_snv_drops_non_snv():
    f = su.SNVFiltration()
    assert f.filter_site(_dummy(ref="A")) is np.True_ or f.filter_site(_dummy(ref="A"))
    assert not f.filter_site(_dummy(ref="AT"))
    assert not f.filter_site(_dummy(ref="A", alt=["<DEL>"]))


def test_polyallelic_filtration_drops_multiallelic():
    f = su.PolyAllelicFiltration()
    assert f.filter_site(_dummy(alt=["C"]))
    assert not f.filter_site(_dummy(alt=["C", "G"]))


def test_all_and_no_filtration():
    assert su.AllFiltration().filter_site(_dummy()) is False
    assert su.NoFiltration().filter_site(_dummy()) is True
