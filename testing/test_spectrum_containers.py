"""
Unit tests for the higher-dimensional spectrum containers pulled from PhaseGen: the square two-dimensional
:class:`~sfsutils.spectrum.SFS2` (and its :class:`~sfsutils.spectrum.TwoLocusSFS` specialization), the
multi-population :class:`~sfsutils.spectrum.JointSFS`, and the :class:`~sfsutils.spectrum.JointSpectra` collection.
These need only numpy / matplotlib / jsonpickle and run in the light suite.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pytest

import sfsutils as sf
from sfsutils.spectrum import AbstractSpectrum, AbstractSpectra


def test_abstract_bases_not_instantiable():
    for base in (AbstractSpectrum, AbstractSpectra):
        with pytest.raises(TypeError):
            base()


def test_hierarchy():
    assert issubclass(sf.Spectrum, AbstractSpectrum)
    assert issubclass(sf.SFS2, AbstractSpectrum)
    assert issubclass(sf.TwoLocusSFS, sf.SFS2)
    assert issubclass(sf.JointSFS, AbstractSpectrum)
    assert issubclass(sf.Spectra, AbstractSpectra)
    assert issubclass(sf.JointSpectra, AbstractSpectra)


# --- SFS2 -----------------------------------------------------------------------------------------

def test_sfs2_validation():
    with pytest.raises(ValueError):
        sf.SFS2(np.arange(8))  # not 2-D
    with pytest.raises(ValueError):
        sf.SFS2(np.arange(6).reshape(2, 3))  # not square


def test_sfs2_shape_and_totals():
    data = np.arange(16).reshape(4, 4).astype(float)
    s = sf.SFS2(data)
    assert s.n == 4 and s.w == 2
    assert s.shape == (4, 4)
    assert s.n_sites == data.sum()
    assert np.array_equal(np.asarray(s), data)


def test_sfs2_fold_is_idempotent_and_conserves_mass():
    rng = np.random.default_rng(0)
    s = sf.SFS2(rng.integers(0, 10, size=(5, 5)).astype(float))
    folded = s.fold()
    assert folded.is_folded()
    assert folded.n_sites == pytest.approx(s.n_sites)


def test_sfs2_arithmetic_and_masks():
    s = sf.SFS2(np.ones((4, 4)))
    assert ((s + s).data == 2).all()
    assert ((s * 3).data == 3).all()
    assert ((s - sf.SFS2(np.ones((4, 4)))).data == 0).all()
    assert np.isnan(s.symmetrize().mask_diagonal().data).any()
    assert np.isnan(s.fill_monomorphic().data[0]).all()
    assert s.get_max_abs() == 1


def test_sfs2_roundtrip_and_copy_type(tmp_path):
    s = sf.SFS2(np.arange(9).reshape(3, 3).astype(float))
    f = tmp_path / "sfs2.json"
    s.to_file(str(f))
    loaded = sf.SFS2.from_file(str(f))
    assert type(loaded) is sf.SFS2
    assert np.array_equal(loaded.data, s.data)
    assert type(s.copy()) is sf.SFS2


def test_two_locus_sfs_roundtrip_type(tmp_path):
    t = sf.TwoLocusSFS(np.arange(9).reshape(3, 3).astype(float))
    assert isinstance(t, sf.SFS2)
    f = tmp_path / "two_locus.json"
    t.to_file(str(f))
    assert type(sf.TwoLocusSFS.from_file(str(f))) is sf.TwoLocusSFS


def test_sfs2_plot_smoke():
    ax = sf.SFS2(np.arange(16).reshape(4, 4).astype(float)).plot(show=False)
    assert ax is not None
    plt.close('all')


# --- JointSFS -------------------------------------------------------------------------------------

def test_jointsfs_validation():
    with pytest.raises(ValueError):
        sf.JointSFS(np.arange(6).reshape(2, 3), pop_names=["only_one"])


def test_jointsfs_basics():
    data = np.arange(12).reshape(3, 4)
    j = sf.JointSFS(data, pop_names=["A", "B"])
    assert j.n_pops == 2 and j.shape == (3, 4)
    assert j.n_sites == float(data.sum())
    assert j.pop_names == ["A", "B"]
    assert np.array_equal(np.asarray(j), data)
    assert j[1, 2] == data[1, 2]
    # default population names
    assert sf.JointSFS(data).pop_names == ["pop_0", "pop_1"]


def test_jointsfs_arithmetic_preserves_pop_names():
    j = sf.JointSFS(np.ones((2, 3)), pop_names=["A", "B"])
    assert (j + j).pop_names == ["A", "B"]
    assert ((j * 2).data == 2).all()
    assert ((j - j).data == 0).all()
    assert type((j ** 2)) is sf.JointSFS


def test_jointsfs_marginalize():
    data = np.arange(24).reshape(2, 3, 4)
    j = sf.JointSFS(data, pop_names=["A", "B", "C"])

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
    j = sf.JointSFS(np.arange(6).reshape(2, 3), pop_names=["A", "B"])
    f = tmp_path / "jsfs.json"
    j.to_file(str(f))
    loaded = sf.JointSFS.from_file(str(f))
    assert type(loaded) is sf.JointSFS
    assert np.array_equal(loaded.data, j.data) and loaded.pop_names == ["A", "B"]
    assert type(j.copy()) is sf.JointSFS


def test_jointsfs_plot_smoke():
    j = sf.JointSFS(np.arange(9).reshape(3, 3).astype(float), pop_names=["A", "B"])
    assert j.plot(show=False) is not None
    plt.close('all')
    # 3-D input marginalizes to 2-D before plotting
    j3 = sf.JointSFS(np.arange(27).reshape(3, 3, 3).astype(float))
    assert j3.plot(pops=(0, 1), show=False) is not None
    plt.close('all')


# --- JointSpectra ---------------------------------------------------------------------------------

def test_jointspectra_collection():
    a = np.arange(6).reshape(2, 3)
    js = sf.JointSpectra({"neutral": a, "selected": a * 2}, pop_names=["A", "B"])

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
    js = sf.JointSpectra({"neutral": a, "selected": a * 2})
    marg = js.marginalize([0, 1])
    assert marg.shape == (2, 3)
    assert np.array_equal(marg["selected"].data, (a * 2).sum(axis=2))


def test_jointspectra_roundtrip(tmp_path):
    a = np.arange(6).reshape(2, 3)
    js = sf.JointSpectra({"neutral": a, "selected": a * 2}, pop_names=["A", "B"])
    f = tmp_path / "jspectra.json"
    js.to_file(str(f))
    loaded = sf.JointSpectra.from_file(str(f))
    assert type(loaded) is sf.JointSpectra
    assert loaded.types == ["neutral", "selected"]
    assert np.array_equal(loaded["selected"].data, a * 2)
    assert loaded.pop_names == ["A", "B"]


def test_jointspectra_empty_raises():
    empty = sf.JointSpectra({})
    for accessor in ("pop_names", "n_pops", "shape"):
        with pytest.raises(ValueError):
            getattr(empty, accessor)
    with pytest.raises(ValueError):
        _ = empty.all
