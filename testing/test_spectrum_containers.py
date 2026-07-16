"""
Unit tests for the higher-dimensional spectrum containers pulled from PhaseGen: the square two-dimensional
:class:`~sfsutils.spectrum.TwoSFS` (and its :class:`~sfsutils.spectrum.TwoLocusSFS` specialization), the
multi-population :class:`~sfsutils.spectrum.JointSFS`, and the :class:`~sfsutils.spectrum.JointSpectra` collection.
These need only numpy / matplotlib / jsonpickle and run in the light suite.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pytest

import sfsutils as su
from sfsutils.spectrum import AbstractSpectrum, AbstractSpectra


def test_abstract_bases_not_instantiable():
    for base in (AbstractSpectrum, AbstractSpectra):
        with pytest.raises(TypeError):
            base()


def test_hierarchy():
    assert issubclass(su.Spectrum, AbstractSpectrum)
    assert issubclass(su.TwoSFS, AbstractSpectrum)
    assert issubclass(su.TwoLocusSFS, su.TwoSFS)
    assert issubclass(su.JointSFS, AbstractSpectrum)
    assert issubclass(su.Spectra, AbstractSpectra)
    assert issubclass(su.JointSpectra, AbstractSpectra)


# --- TwoSFS -----------------------------------------------------------------------------------------

def test_sfs2_validation():
    with pytest.raises(ValueError):
        su.TwoSFS(np.arange(8))  # not 2-D
    with pytest.raises(ValueError):
        su.TwoSFS(np.arange(6).reshape(2, 3))  # not square


def test_sfs2_shape_and_totals():
    data = np.arange(16).reshape(4, 4).astype(float)
    s = su.TwoSFS(data)
    assert s.n == 4 and s.w == 2
    assert s.shape == (4, 4)
    assert s.n_sites == data.sum()
    assert np.array_equal(np.asarray(s), data)


def test_sfs2_fold_is_idempotent_and_conserves_mass():
    rng = np.random.default_rng(0)
    s = su.TwoSFS(rng.integers(0, 10, size=(5, 5)).astype(float))
    folded = s.fold()
    assert folded.is_folded()
    assert folded.n_sites == pytest.approx(s.n_sites)
    # folding an already-folded spectrum is a no-op
    np.testing.assert_allclose(folded.fold().data, folded.data)


def test_sfs2_arithmetic_and_masks():
    s = su.TwoSFS(np.ones((4, 4)))
    assert ((s + s).data == 2).all()
    assert ((s * 3).data == 3).all()
    assert ((s - su.TwoSFS(np.ones((4, 4)))).data == 0).all()
    assert np.isnan(s.symmetrize().mask_diagonal().data).any()
    assert np.isnan(s.fill_monomorphic().data[0]).all()
    assert s.get_max_abs() == 1


def test_sfs2_roundtrip_and_copy_type(tmp_path):
    s = su.TwoSFS(np.arange(9).reshape(3, 3).astype(float))
    f = tmp_path / "two_sfs.json"
    s.to_file(str(f))
    loaded = su.TwoSFS.from_file(str(f))
    assert type(loaded) is su.TwoSFS
    assert np.array_equal(loaded.data, s.data)
    assert type(s.copy()) is su.TwoSFS


def test_two_locus_sfs_roundtrip_type(tmp_path):
    t = su.TwoLocusSFS(np.arange(9).reshape(3, 3).astype(float))
    assert isinstance(t, su.TwoSFS)
    f = tmp_path / "two_locus.json"
    t.to_file(str(f))
    assert type(su.TwoLocusSFS.from_file(str(f))) is su.TwoLocusSFS


def test_sfs2_mask_upper_masks_the_upper_triangle():
    s = su.TwoSFS(np.arange(9).reshape(3, 3).astype(float))
    masked = s.mask_upper(fill_value=-1.0).data
    # strictly-upper entries are masked; the diagonal and lower triangle are retained
    assert masked[0, 1] == -1 and masked[0, 2] == -1 and masked[1, 2] == -1
    assert masked[1, 0] == s.data[1, 0] and masked[2, 0] == s.data[2, 0] and masked[1, 1] == s.data[1, 1]


def test_sfs2_interior_covariance_correlation_and_monomorphic_invariance():
    """covariance()/correlation() use only the segregating interior (both derived counts in 1..n-1), so they ignore
    the monomorphic first/last row and column: an independent (product-form) interior gives zero covariance and
    correlation, a diagonal (perfectly linked) interior gives correlation 1, and adding arbitrary mass to the
    monomorphic bins changes neither. A single segregating class has no interior variance, so correlation raises."""
    m = np.array([1.0, 3.0, 2.0])  # a marginal over the three segregating classes (sample size n = 4)

    # independent sites: P(i, j) = p_i p_j -> zero covariance and correlation
    indep = np.zeros((5, 5)); indep[1:4, 1:4] = np.outer(m, m)
    assert su.TwoSFS(indep).covariance() == pytest.approx(0.0, abs=1e-12)
    assert su.TwoSFS(indep).correlation() == pytest.approx(0.0, abs=1e-12)

    # perfectly linked (diagonal) interior: the two frequencies are identical -> correlation 1
    diag = np.zeros((5, 5)); diag[1:4, 1:4] = np.diag(m)
    assert su.TwoSFS(diag).correlation() == pytest.approx(1.0)

    # the monomorphic first/last row and column are ignored entirely
    for base in (indep, diag):
        mono = base.copy()
        mono[0, :] += 7.0; mono[:, 0] += 7.0; mono[-1, :] += 5.0; mono[:, -1] += 5.0
        assert su.TwoSFS(mono).covariance() == pytest.approx(su.TwoSFS(base).covariance())
        assert su.TwoSFS(mono).correlation() == pytest.approx(su.TwoSFS(base).correlation())

    # a spectrum with a single segregating class has no interior variance -> correlation is undefined
    with pytest.raises(ValueError, match="segregating"):
        su.TwoSFS(np.ones((3, 3))).correlation()


def test_sfs2_plot_smoke():
    s = su.TwoSFS(np.arange(16).reshape(4, 4).astype(float))
    assert s.plot(show=False) is not None
    s.plot_surface(show=False)
    plt.close('all')


# --- JointSFS -------------------------------------------------------------------------------------

def test_jointsfs_validation():
    with pytest.raises(ValueError):
        su.JointSFS(np.arange(6).reshape(2, 3), pop_names=["only_one"])


def test_jointsfs_basics():
    data = np.arange(12).reshape(3, 4)
    j = su.JointSFS(data, pop_names=["A", "B"])
    assert j.n_pops == 2 and j.shape == (3, 4)
    assert j.n_sites == float(data.sum())
    assert j.pop_names == ["A", "B"]
    assert np.array_equal(np.asarray(j), data)
    assert j[1, 2] == data[1, 2]
    # default population names
    assert su.JointSFS(data).pop_names == ["pop_0", "pop_1"]


def test_jointsfs_arithmetic_preserves_pop_names():
    j = su.JointSFS(np.ones((2, 3)), pop_names=["A", "B"])
    assert (j + j).pop_names == ["A", "B"]
    assert ((j * 2).data == 2).all()
    assert ((j - j).data == 0).all()
    assert type((j ** 2)) is su.JointSFS


def test_jointsfs_marginalize():
    data = np.arange(24).reshape(2, 3, 4)
    j = su.JointSFS(data, pop_names=["A", "B", "C"])

    # marginalizing onto one axis sums over the others
    assert np.array_equal(j.marginalize([0]).data, data.sum(axis=(1, 2)))
    assert j.marginalize([0]).pop_names == ["A"]

    # keep two axes, reordered
    m = j.marginalize([2, 0])
    assert m.shape == (4, 2)
    assert np.array_equal(m.data, data.sum(axis=1).T)
    assert m.pop_names == ["C", "A"]

    with pytest.raises(ValueError):
        j.marginalize([5])


def test_jointsfs_roundtrip_and_copy_type(tmp_path):
    j = su.JointSFS(np.arange(6).reshape(2, 3), pop_names=["A", "B"])
    f = tmp_path / "jsfs.json"
    j.to_file(str(f))
    loaded = su.JointSFS.from_file(str(f))
    assert type(loaded) is su.JointSFS
    assert np.array_equal(loaded.data, j.data) and loaded.pop_names == ["A", "B"]
    assert type(j.copy()) is su.JointSFS


def test_jointsfs_restores_without_pop_names():
    """A JointSFS serialized without pop_names (legacy or cross-version JSON) must restore and resolve default
    names rather than raising AttributeError, per the backward-compat class-attribute convention."""
    import json

    j = su.JointSFS(np.arange(6).reshape(2, 3), pop_names=["A", "B"])
    payload = json.loads(j.to_json())
    payload.pop("pop_names", None)  # simulate JSON that predates the attribute
    loaded = su.JointSFS.from_json(json.dumps(payload))

    assert loaded.pop_names is None  # the raw attribute is absent
    assert loaded._names() == ["pop_0", "pop_1"]  # but consumers resolve defaults
    assert loaded.plot(show=False) is not None  # and plotting no longer raises
    plt.close('all')


def test_two_sfs_restores_without_n_w():
    """A TwoSFS serialized without n/w restores via the class-level defaults instead of raising AttributeError."""
    import json

    s = su.TwoSFS(np.arange(9).reshape(3, 3))
    payload = json.loads(s.to_json())
    for k in ("n", "w"):
        payload.pop(k, None)
    loaded = su.TwoSFS.from_json(json.dumps(payload))

    assert np.array_equal(loaded.data, s.data)


def test_jointsfs_plot_smoke():
    j = su.JointSFS(np.arange(9).reshape(3, 3).astype(float), pop_names=["A", "B"])
    assert j.plot(show=False) is not None
    j.plot_surface(show=False)
    plt.close('all')
    # 3-D input marginalizes to 2-D before plotting
    j3 = su.JointSFS(np.arange(27).reshape(3, 3, 3).astype(float))
    assert j3.plot(pops=(0, 1), show=False) is not None
    j3.plot_surface(pops=(0, 1), show=False)
    plt.close('all')


def test_jointsfs_plot_requires_two_pops():
    j = su.JointSFS(np.arange(9).reshape(3, 3).astype(float))
    with pytest.raises(ValueError):
        j.plot(pops=(0,), show=False)
    with pytest.raises(ValueError):
        j.plot_surface(pops=(0, 1, 2), show=False)


# --- JointSpectra ---------------------------------------------------------------------------------

def test_jointspectra_collection():
    a = np.arange(6).reshape(2, 3)
    js = su.JointSpectra({"neutral": a, "selected": a * 2}, pop_names=["A", "B"])

    assert js.types == ["neutral", "selected"]
    assert js.n_pops == 2 and js.shape == (2, 3)
    assert js.pop_names == ["A", "B"]
    assert len(js) == 2
    assert set(iter(js)) == {"neutral", "selected"}
    assert np.array_equal(js["neutral"].data, a)
    assert np.array_equal(js.all.data, a * 3)
    assert set(js.to_dict()) == {"neutral", "selected"}


def test_jointspectra_marginalize():
    a = np.arange(24).reshape(2, 3, 4)
    js = su.JointSpectra({"neutral": a, "selected": a * 2})
    marg = js.marginalize([0, 1])
    assert marg.shape == (2, 3)
    assert np.array_equal(marg["selected"].data, (a * 2).sum(axis=2))


def test_jointspectra_roundtrip(tmp_path):
    a = np.arange(6).reshape(2, 3)
    js = su.JointSpectra({"neutral": a, "selected": a * 2}, pop_names=["A", "B"])
    f = tmp_path / "jspectra.json"
    js.to_file(str(f))
    loaded = su.JointSpectra.from_file(str(f))
    assert type(loaded) is su.JointSpectra
    assert loaded.types == ["neutral", "selected"]
    assert np.array_equal(loaded["selected"].data, a * 2)
    assert loaded.pop_names == ["A", "B"]


def test_jointspectra_empty_raises():
    empty = su.JointSpectra({})
    for accessor in ("pop_names", "n_pops", "shape"):
        with pytest.raises(ValueError):
            getattr(empty, accessor)
    with pytest.raises(ValueError):
        _ = empty.all


# --- TwoSpectra -----------------------------------------------------------------------------------

def test_twospectra_collection():
    a = np.arange(9).reshape(3, 3).astype(float)
    ts = su.TwoSpectra({"neutral": a, "selected": a * 2})

    assert issubclass(su.TwoSpectra, AbstractSpectra)
    assert ts.types == ["neutral", "selected"]
    assert ts.shape == (3, 3)
    assert len(ts) == 2
    assert set(iter(ts)) == {"neutral", "selected"}
    assert isinstance(ts["neutral"], su.TwoSFS)
    assert np.array_equal(ts["neutral"].data, a)
    # `all` pools the per-type (within-stratum) spectra
    assert np.array_equal(ts.all.data, a * 3)
    assert set(ts.to_dict()) == {"neutral", "selected"}


def test_twospectra_roundtrip(tmp_path):
    a = np.arange(9).reshape(3, 3).astype(float)
    ts = su.TwoSpectra({"neutral": a, "selected": a * 2})
    f = tmp_path / "twospectra.json"
    ts.to_file(str(f))
    loaded = su.TwoSpectra.from_file(str(f))
    assert type(loaded) is su.TwoSpectra
    assert loaded.types == ["neutral", "selected"]
    assert type(loaded["selected"]) is su.TwoSFS
    assert np.array_equal(loaded["selected"].data, a * 2)


def test_twospectra_empty_raises():
    empty = su.TwoSpectra({})
    for accessor in ("shape",):
        with pytest.raises(ValueError):
            getattr(empty, accessor)
    with pytest.raises(ValueError):
        _ = empty.all
