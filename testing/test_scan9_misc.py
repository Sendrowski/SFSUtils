"""
Regression tests for the ninth-scan defects in the spectrum containers and the CLI.

Covered here: the branch-length correlation of independent loci, the Poisson resampling of a SNP-only
spectrum, the round trip of a joint spectrum whose population names came from a numpy array, indexing an
empty or non-matching ``Spectra``, the CLI refusing an empty two-SFS parse, and ``--n`` rejecting
non-positive values.
"""

import os

import numpy as np
import pytest

import sfsutils as su
from sfsutils.cli import run

VCF_PATH = "resources/msprime/two_epoch.vcf"

requires_vcf = pytest.mark.skipif(not os.path.exists(VCF_PATH), reason="msprime fixtures absent")


# --- C16: correlation of independent loci ----------------------------------------------------------

def test_corr_of_independent_loci_is_zero():
    """A two-SFS that factorizes exactly describes independent loci, whose branch-length correlation is zero
    rather than the maximal +/-1 a covariance-scaled zero-variance floor produces."""
    f = np.array([1e6, 500.0, 200.0, 120.0, 90.0, 3000.0])
    two = su.TwoSFS(np.outer(f, f))

    # the covariance is at roundoff, as independence demands
    assert np.abs(np.asarray(two.cov())).max() < 1e-15

    np.testing.assert_array_equal(np.asarray(two.corr()), np.zeros((6, 6)))


def test_corr_of_perturbed_independent_loci_is_zero():
    """A perturbation at the roundoff scale of the covariance carries no signal, so no class may read as
    correlated. The perturbation is relative because the roundoff of ``P(i, j) - P(i) P(j)`` is."""
    f = np.array([1e6, 500.0, 200.0, 120.0, 90.0, 3000.0])
    rng = np.random.default_rng(0)

    data = np.outer(f, f) * (1 + 1e-14 * rng.standard_normal((6, 6)))

    np.testing.assert_allclose(np.asarray(su.TwoSFS(data).corr()), 0.0, atol=1e-9)


def test_corr_keeps_unit_diagonal_for_a_genuine_signal():
    """A spectrum with real branch-length variance keeps the unit diagonal the correlation promises."""
    rng = np.random.default_rng(1)

    data = np.zeros((6, 6))
    data[1:-1, 1:-1] = rng.random((4, 4)) + 0.5
    data[0, 0] = 1e4
    data = data + data.T

    r = np.asarray(su.TwoSFS(data).corr())

    np.testing.assert_allclose(np.diag(r)[1:-1], 1.0, atol=1e-9)
    assert np.all(np.abs(r) <= 1 + 1e-12)


# --- C17: resampling a SNP-only spectrum -----------------------------------------------------------

def test_resample_never_produces_negative_counts():
    """A SNP-only spectrum carries no monomorphic mass, so deriving bin 0 as a residual makes it negative;
    every bin has to be its own Poisson draw."""
    sfs = su.Spectrum([0.0, 100.0, 50.0, 20.0, 10.0])

    for seed in range(200):
        assert np.all(np.asarray(sfs.resample(seed)) >= 0)


def test_resample_draws_the_monomorphic_bin_independently():
    """Bin 0 is resampled like any other class, so it varies around its own mean instead of absorbing the
    noise of the remaining bins."""
    sfs = su.Spectrum([500.0, 100.0, 50.0, 20.0, 10.0])

    draws = np.array([np.asarray(sfs.resample(seed))[0] for seed in range(300)])

    assert draws.std() > 0
    assert abs(draws.mean() - 500) < 10


# --- C18: numpy-typed metadata in a payload --------------------------------------------------------

def test_joint_sfs_with_numpy_population_names_round_trips(tmp_path):
    """Population names taken from a numpy array must not make the user's own file unreadable."""
    file = str(tmp_path / 'joint.json')

    su.JointSFS(np.zeros((2, 2)), pop_names=np.array(['A', 'B'])).to_file(file)

    assert su.JointSFS.from_file(file).pop_names == ['A', 'B']


def test_joint_spectra_with_numpy_population_names_round_trips(tmp_path):
    """The same holds for the dict-backed collection, whose names travel through its spectra."""
    file = str(tmp_path / 'joint_spectra.json')

    su.JointSpectra({'all': np.zeros((2, 2))}, pop_names=np.unique(np.array(['A', 'B']))).to_file(file)

    assert su.JointSpectra.from_file(file).pop_names == ['A', 'B']


# --- C19: indexing a Spectra -----------------------------------------------------------------------

def test_indexing_an_empty_spectra_raises_key_error():
    """An empty Spectra has no string columns to match against, which must read as a missing type rather
    than a pandas accessor error."""
    with pytest.raises(KeyError):
        su.Spectra({})['all']


def test_indexing_a_non_matching_key_raises_key_error():
    """A mistyped type name is an error, not an empty selection."""
    spectra = su.Spectra({'neutral': [0, 1, 2, 0], 'selected': [0, 3, 4, 0]})

    with pytest.raises(KeyError):
        spectra['neutrol']

    # an array of keys keeps its selection semantics
    assert spectra[['neutrol']].types == []
    assert spectra['neutral'].to_list() == [0, 1, 2, 0]


# --- C20: empty two-SFS parse ----------------------------------------------------------------------

def test_is_empty_across_the_collections():
    """The emptiness of a collection is about parsed mass, not about the number of types: a two-SFS parse
    always carries an ``all`` type."""
    assert su.Spectra({}).is_empty
    assert su.Spectra({'all': [0, 0, 0, 0]}).is_empty
    assert not su.Spectra({'all': [0, 1, 0, 0]}).is_empty

    assert su.TwoSpectra({'all': np.zeros((5, 5))}).is_empty
    assert not su.TwoSpectra({'all': np.eye(5)}).is_empty

    assert su.JointSpectra({'all': np.zeros((3, 3))}).is_empty
    assert not su.JointSpectra({'all': np.eye(3)}).is_empty


@requires_vcf
def test_parse_two_sfs_fails_when_nothing_was_included(tmp_path):
    """The fixture is unpolarized, so every site is skipped; the two-SFS parse must not write an all-zero
    spectrum and report success."""
    out = str(tmp_path / 'two.json')

    assert run(['-q', 'parse', '--vcf', VCF_PATH, '--n', '8', '--two-sfs', '--output', out]) == 1
    assert not os.path.exists(out)


@requires_vcf
def test_parse_two_sfs_succeeds_when_sites_were_included(tmp_path):
    """The guard must not fire on a parse that did include sites."""
    out = str(tmp_path / 'two.json')

    assert run(['-q', 'parse', '--vcf', VCF_PATH, '--n', '8', '--two-sfs',
                '--no-skip-non-polarized', '--output', out]) == 0
    assert not su.TwoSpectra.from_file(out).is_empty


# --- C21: validation of --n ------------------------------------------------------------------------

@requires_vcf
@pytest.mark.parametrize('n', ['0', '-3'])
def test_parse_rejects_non_positive_n(tmp_path, n):
    """A sample size below one either writes a nonsense single-bin spectrum or raises a raw numpy error."""
    out = str(tmp_path / 'sfs.json')

    with pytest.raises(SystemExit):
        run(['-q', 'parse', '--vcf', VCF_PATH, '--n', n, '--output', out])

    assert not os.path.exists(out)


@requires_vcf
@pytest.mark.parametrize('option', ['--two-sfs-distance', '--n-ingroups'])
def test_parse_rejects_non_positive_counts(tmp_path, option):
    """The remaining count-valued options are validated alongside ``--n``."""
    with pytest.raises(SystemExit):
        run(['-q', 'parse', '--vcf', VCF_PATH, '--n', '8', option, '0',
             '--output', str(tmp_path / 'sfs.json')])
