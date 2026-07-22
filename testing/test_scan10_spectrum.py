"""
Regression tests for the tenth-scan defects in :mod:`sfsutils.spectrum`.

Covered here: the branch-length correlation of a low-diversity spectrum, folding a single-bin spectrum, and
merging groups of types whose names have unequal depth.
"""

import numpy as np
import pytest

import sfsutils as su


def _two_sfs(theta: float, n: int = 8, rho: float = 0.4, eps: float = 0.05, sites: float = 3e9):
    """
    An all-sites-like two-SFS with an exactly known branch-length covariance.

    The joint class distribution is ``P = outer(m, m) + K`` with ``K`` symmetric and of zero row sum, so the
    marginal is exactly the site-frequency spectrum ``m`` and the branch-length covariance is exactly ``K``. Over
    the polymorphic interior ``K`` is built to have unit diagonal and off-diagonal correlation ``rho``.

    :param theta: Per-site polymorphism probability, setting the scale of the polymorphic classes
    :param n: Sample size
    :param rho: Built-in interior branch-length correlation
    :param eps: Relative amplitude of the branch-length fluctuations
    :param sites: Number of sites the probabilities are scaled to
    :return: The two-SFS, its exact covariance and its exact interior correlation
    """
    m = np.zeros(n + 1)
    m[1:n] = theta / np.arange(1, n)
    m[n] = theta / 2
    m[0] = 1 - m[1:].sum()

    corr = np.full((n - 1, n - 1), rho)
    np.fill_diagonal(corr, 1.0)

    cov = np.zeros((n + 1, n + 1))
    cov[1:n, 1:n] = eps ** 2 * np.outer(m[1:n], m[1:n]) * corr

    # push the residual into the monomorphic bin so every row sums to zero and the marginal stays exactly m
    residual = cov.sum(axis=1)
    cov[0, :] -= residual
    cov[:, 0] -= residual
    cov[0, 0] += residual.sum()

    joint = np.outer(m, m) + cov
    assert joint.min() > 0

    return su.TwoSFS(joint * sites), cov, corr


# --- D16: correlation floor scaled to the probability of each class ---------------------------------

@pytest.mark.parametrize('theta', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10])
def test_corr_recovers_a_genuine_correlation_at_low_diversity(theta):
    """The covariance of a low-diversity spectrum lives on the scale of the class probabilities, not on the
    scale of the monomorphic bin, so a floor set by the latter censors a genuine signal however much data it
    rests on."""
    two, cov, corr = _two_sfs(theta)

    np.testing.assert_allclose(np.asarray(two.cov())[1:-1, 1:-1], cov[1:-1, 1:-1], rtol=1e-10, atol=0)
    np.testing.assert_allclose(np.asarray(two.corr())[1:-1, 1:-1], corr, atol=1e-9)


def test_corr_is_invariant_to_the_scale_of_the_input():
    """The correlation is a ratio, so a spectrum supplied as probabilities has to give the same answer as the
    same spectrum supplied as counts."""
    counts, _, corr = _two_sfs(1e-6)
    probs, _, _ = _two_sfs(1e-6, sites=1.0)

    np.testing.assert_allclose(np.asarray(probs.corr()), np.asarray(counts.corr()), atol=1e-12)
    np.testing.assert_allclose(np.asarray(probs.corr())[1:-1, 1:-1], corr, atol=1e-9)


def test_corr_matches_a_shared_genealogy_reference():
    """An exact Poisson-on-shared-genealogy reference: two sites mutate independently on one random genealogy,
    so their class correlation follows from the branch-length fluctuations alone."""
    rng = np.random.default_rng(7)
    n, reps = 5, 20000

    lengths = rng.gamma(shape=np.array([3.0, 2.0, 1.5, 1.0]), size=(reps, n - 1)) * rng.gamma(2.0, size=(reps, 1))

    for theta in [1e-3, 1e-5, 1e-7]:
        q = np.zeros((reps, n + 1))
        q[:, 1:n] = theta * lengths / lengths.sum(axis=1).mean()
        q[:, 0] = 1 - q[:, 1:].sum(axis=1)

        joint = (q[:, :, None] * q[:, None, :]).mean(axis=0)
        cov = joint - np.outer(q.mean(axis=0), q.mean(axis=0))
        sd = np.sqrt(np.diag(cov)[1:n])
        expected = cov[1:n, 1:n] / np.outer(sd, sd)

        # the reference carries a real signal, so the test would be vacuous if it did not
        assert np.abs(expected - np.eye(n - 1)).max() > 0.3

        np.testing.assert_allclose(np.asarray(su.TwoSFS(joint * 1e9).corr())[1:n, 1:n], expected, atol=1e-9)


# --- D17: folding a single-bin spectrum -------------------------------------------------------------

def test_fold_leaves_a_single_bin_spectrum_alone():
    """A one-bin spectrum has no upper half to fold into the lower one, so folding is the identity."""
    sfs = su.Spectrum([5.0])

    np.testing.assert_array_equal(sfs.fold().data, [5.0])
    assert sfs.fold().n_sites == 5.0
    assert sfs.is_folded()


@pytest.mark.parametrize('n', range(9))
def test_fold_matches_a_brute_force_reference(n):
    """Folding maps bin i onto min(i, n - i)."""
    data = np.arange(1.0, n + 2)

    expected = np.zeros(n + 1)
    for i, count in enumerate(data):
        expected[min(i, n - i)] += count

    np.testing.assert_allclose(su.Spectrum(data).fold().data, expected)


def test_subsampling_a_folded_spectrum_to_one_bin_keeps_its_sites():
    """The n = 0 subsample of a folded spectrum takes the folding branch, which used to blank it."""
    sfs = su.Spectrum([10.0, 1.0, 2.0, 0.0])

    np.testing.assert_array_equal(sfs.fold().subsample(0).data, sfs.subsample(0).data)
    assert sfs.fold().subsample(0).n_sites == 13.0


# --- D18: merging groups of unequally deep type names -----------------------------------------------

def test_merge_groups_rejects_a_level_some_type_does_not_have():
    """A type with fewer levels has no name at the requested level, and used to be dropped together with all
    of its sites."""
    spectra = su.Spectra({'a.b': [0.0, 1, 2, 3, 4, 5, 6], 'c': [0.0, 1, 2, 3, 4, 5, 6]})

    with pytest.raises(ValueError, match="'c'"):
        spectra.merge_groups(1)

    with pytest.raises(ValueError, match="'c'"):
        spectra.merge_groups([0, 1])


def test_merge_groups_keeps_every_site_at_a_shared_level():
    """Level 0 exists for every type, so merging over it conserves the sites."""
    spectra = su.Spectra({'a.b': [0.0, 1, 2, 3, 4, 5, 6], 'c': [0.0, 1, 2, 3, 4, 5, 6]})
    merged = spectra.merge_groups(0)

    assert merged.types == ['a', 'c']
    assert merged.n_sites.sum() == spectra.n_sites.sum() == 42.0


def test_merge_groups_still_merges_equally_deep_names():
    """Well-formed names are unaffected, including negative levels."""
    spectra = su.Spectra({'a.b.c': [0.0, 1, 2], 'd.e.f': [0.0, 1, 2]})

    assert spectra.merge_groups([1, 2]).types == ['b.c', 'e.f']
    assert spectra.merge_groups(-1).types == ['c', 'f']
    assert spectra.merge_groups(-1).n_sites.sum() == spectra.n_sites.sum()


def test_corr_reports_how_much_of_the_interior_it_zeroed(caplog):
    """Zeroing is indistinguishable from a genuine zero correlation in the returned matrix, so the count is
    the only way a caller learns their spectrum sits at the resolution limit."""
    import logging

    f = np.array([1e6, 500, 200, 120, 90, 3000])

    with caplog.at_level(logging.INFO, logger='sfsutils'):
        su.TwoSFS(np.outer(f, f)).corr()

    assert any('Zeroed 16 of 16 interior correlations' in record.message for record in caplog.records)


def test_corr_stays_quiet_when_it_zeroes_nothing(caplog):
    import logging

    rng = np.random.default_rng(0)
    f = np.array([1e6, 500, 200, 120, 90, 3000])
    base = np.outer(f, f) * (1 + 0.3 * rng.random((6, 6)))

    with caplog.at_level(logging.INFO, logger='sfsutils'):
        su.TwoSFS((base + base.T) / 2).corr()

    assert not any('interior correlations' in record.message for record in caplog.records)
