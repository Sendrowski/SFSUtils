"""
Equivalence tests for the fast paths of the parser: the cached hypergeometric down-projection and the running
window sums of the two-SFS. Both are algebraic rewrites, so every test here compares against a naive parser that
evaluates the definitions directly (an uncached ``hypergeom.pmf`` call per site, one outer product per pair).
"""
import os
from collections import deque

import numpy as np
import pytest
from scipy.stats import hypergeom

import sfsutils as su
from sfsutils.settings import Settings

TWO_SFS_VCF = "resources/msprime/two_sfs.vcf"
ALL_SITES_VCF = "resources/msprime/two_sfs_kingman.all.vcf.gz"
TWO_EPOCH_VCF = "resources/msprime/two_epoch.vcf"

requires_fixtures = pytest.mark.skipif(
    not (os.path.exists(TWO_SFS_VCF) and os.path.exists(ALL_SITES_VCF) and os.path.exists(TWO_EPOCH_VCF)),
    reason="msprime fixtures not available"
)


class NaiveParser(su.Parser):
    """
    A parser evaluating the definitions directly: the hypergeometric down-projection is recomputed at every site,
    and the two-SFS pairs the current site with each buffered site individually.
    """

    def _projection(self, n_samples: int, n_der: int, n: int) -> np.ndarray:
        """
        The uncached hypergeometric down-projection.

        :param n_samples: The number of called genotypes at the site.
        :param n_der: The number of derived alleles among them.
        :param n: The number of genotypes drawn.
        :return: The mass over derived-allele counts.
        """
        return hypergeom.pmf(k=range(n + 1), M=n_samples, n=n_der, N=n)

    def _parse_site_two_sfs(self, variant) -> bool:
        """
        Add a site to the two-SFS with one outer product per within-window pair.

        :param variant: The variant.
        :return: Whether the site was included.
        """
        m = self._project(variant)

        if m is None:
            return False

        try:
            t = '.'.join([s.get_type(variant) for s in self.stratifications]) or 'all'
        except su.io_handlers.NoTypeException:
            return False

        _ = self._two_sfs_matrices[t]
        self._two_sfs_marginal[t] += m

        if not hasattr(self, '_naive_buffer') or variant.CHROM != self._two_sfs_contig:
            self._naive_buffer = deque()
            self._two_sfs_contig = variant.CHROM

        max_distance = self.two_sfs_offset + self.d

        while self._naive_buffer and variant.POS - self._naive_buffer[0][0] > max_distance:
            self._naive_buffer.popleft()

        for pos, m_prev, t_prev in self._naive_buffer:
            distance = variant.POS - pos

            if self.two_sfs_offset < distance <= max_distance and t_prev == t:
                self._two_sfs_matrices[t] += np.multiply.outer(m_prev, m)

        self._naive_buffer.append((variant.POS, m, t))

        return True


def _two_sfs(cls, **kwargs) -> dict:
    """
    Parse a two-SFS and return the per-type matrices.

    :param cls: The parser class.
    :param kwargs: The parser arguments.
    :return: Dictionary of matrices keyed by type.
    """
    Settings.disable_pbar = True
    spectra = cls(two_sfs=True, skip_non_polarized=False, **kwargs).parse()

    return {t: np.asarray(spectra[t].data) for t in spectra.types}


def _sfs(cls, **kwargs) -> np.ndarray:
    """
    Parse a one-dimensional SFS.

    :param cls: The parser class.
    :param kwargs: The parser arguments.
    :return: The spectra as an array.
    """
    Settings.disable_pbar = True
    spectra = cls(skip_non_polarized=False, **kwargs).parse()

    return np.asarray(spectra.data)


@requires_fixtures
@pytest.mark.parametrize("n", [5, 10])
def test_cached_projection_matches_uncached_sfs(n):
    """The cached down-projection reproduces the spectrum obtained from an uncached ``hypergeom.pmf`` per site."""
    kwargs = dict(source=TWO_SFS_VCF, n=n)

    np.testing.assert_allclose(_sfs(su.Parser, **kwargs), _sfs(NaiveParser, **kwargs), rtol=0, atol=0)


@requires_fixtures
def test_cached_projection_matches_uncached_joint_sfs():
    """The cached down-projection reproduces the joint SFS, where it is called once per population."""
    kwargs = dict(source="resources/msprime/two_epoch_joint.vcf", n={"A": 8, "B": 6},
                  pops={"A": [f"tsk_{i}" for i in range(4)], "B": [f"tsk_{i}" for i in range(4, 7)]},
                  skip_non_polarized=False)

    Settings.disable_pbar = True
    fast, naive = su.Parser(**kwargs).parse(), NaiveParser(**kwargs).parse()

    for t in fast.types:
        np.testing.assert_allclose(np.asarray(fast[t].data), np.asarray(naive[t].data), rtol=0, atol=0)


@requires_fixtures
def test_projection_cache_hands_out_read_only_vectors():
    """The cached vectors are shared across sites, so they must be read-only to keep a caller from corrupting the
    cache, and the cache must hold at most one entry per distinct ``(n_samples, n_der, n)``."""
    Settings.disable_pbar = True
    parser = su.Parser(source=TWO_SFS_VCF, n=10, skip_non_polarized=False)
    parser.parse()

    assert len(parser._projection_cache) > 0

    for key, vec in parser._projection_cache.items():
        assert not vec.flags.writeable
        assert len(vec) == key[2] + 1
        np.testing.assert_allclose(vec, hypergeom.pmf(k=range(key[2] + 1), M=key[0], n=key[1], N=key[2]),
                                   rtol=0, atol=0)


@requires_fixtures
def test_projection_cache_does_not_leak_across_parses():
    """A repeated parse starts from an empty cache, so a cached vector can never outlive the sample size it was
    computed for."""
    Settings.disable_pbar = True
    parser = su.Parser(source=TWO_SFS_VCF, n=10, skip_non_polarized=False)

    first = np.asarray(parser.parse().data)
    populated = dict(parser._projection_cache)
    second = np.asarray(parser.parse().data)

    assert len(populated) > 0
    np.testing.assert_allclose(first, second, rtol=0, atol=0)

    parser._reset()
    assert parser._projection_cache == {}


@requires_fixtures
@pytest.mark.parametrize("d,offset", [(1, 0), (100, 0), (1000, 0), (500, 250), (100, 1000)])
def test_two_sfs_window_sums_match_pairwise(d, offset):
    """The running window sums reproduce the pairwise accumulation for windows with and without an offset."""
    kwargs = dict(source=TWO_SFS_VCF, n=10, d=d, two_sfs_offset=offset)

    fast, naive = _two_sfs(su.Parser, **kwargs), _two_sfs(NaiveParser, **kwargs)

    assert set(fast) == set(naive)

    for t in fast:
        np.testing.assert_allclose(fast[t], naive[t], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(fast[t].sum(), naive[t].sum(), rtol=1e-9)


@requires_fixtures
@pytest.mark.parametrize("d,offset", [(100, 0), (300, 100)])
def test_two_sfs_window_sums_match_pairwise_stratified(d, offset):
    """Only within-stratum pairs are counted, so the running sums must be kept per type."""
    kwargs = dict(source=TWO_SFS_VCF, n=10, d=d, two_sfs_offset=offset)

    fast = _two_sfs(su.Parser, stratifications=[su.RandomStratification(3, seed=0)], **kwargs)
    naive = _two_sfs(NaiveParser, stratifications=[su.RandomStratification(3, seed=0)], **kwargs)

    assert len(fast) == 3
    assert set(fast) == set(naive)

    for t in fast:
        np.testing.assert_allclose(fast[t], naive[t], rtol=1e-9, atol=1e-9)


@requires_fixtures
@pytest.mark.parametrize("d,offset", [(50, 0), (30, 20)])
def test_two_sfs_window_sums_match_pairwise_all_sites(d, offset):
    """On all-sites input the window holds a site per base pair, which is where the running sums matter most, and
    where the repeated subtraction has the most opportunity to drift."""
    kwargs = dict(source=ALL_SITES_VCF, n=10, d=d, two_sfs_offset=offset, max_sites=20000)

    fast, naive = _two_sfs(su.Parser, **kwargs), _two_sfs(NaiveParser, **kwargs)

    np.testing.assert_allclose(fast['all'], naive['all'], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(fast['all'].sum(), naive['all'].sum(), rtol=1e-9)


@requires_fixtures
def test_two_sfs_window_resets_at_contig_boundary():
    """Pairs never cross contigs: the all-sites fixture holds many short contigs, so a window wider than a contig
    must still reproduce the pairwise result rather than carry a running sum across the boundary."""
    kwargs = dict(source=ALL_SITES_VCF, n=10, d=1500, two_sfs_offset=0, max_sites=5000)

    fast, naive = _two_sfs(su.Parser, **kwargs), _two_sfs(NaiveParser, **kwargs)

    np.testing.assert_allclose(fast['all'], naive['all'], rtol=1e-9, atol=1e-9)


@requires_fixtures
def test_two_sfs_state_is_reset_between_parses():
    """A second parse starts from an empty window, so repeated parses give the same matrix."""
    Settings.disable_pbar = True
    parser = su.Parser(source=TWO_SFS_VCF, n=10, two_sfs=True, d=200, skip_non_polarized=False)

    first = np.asarray(parser.parse()['all'].data)
    second = np.asarray(parser.parse()['all'].data)

    np.testing.assert_allclose(first, second, rtol=0, atol=0)

    parser._reset()
    assert len(parser._two_sfs_pending) == 0
    assert len(parser._two_sfs_active) == 0
    assert parser._two_sfs_window_sums == {}
    assert parser._two_sfs_window_counts == {}
    assert parser._two_sfs_since_resync == 0


@requires_fixtures
def test_random_subsample_mode_is_unaffected():
    """The random subsampling mode draws from the hypergeometric distribution instead of evaluating its mass
    function, so it bypasses the cache and must be bit-for-bit what it was."""
    kwargs = dict(source=TWO_EPOCH_VCF, n=10, subsample_mode='random', seed=7)

    np.testing.assert_allclose(_sfs(su.Parser, **kwargs), _sfs(NaiveParser, **kwargs), rtol=0, atol=0)


@requires_fixtures
def test_two_sfs_resync_is_exercised_and_exact():
    """The periodic rebuild of the running sums runs mid-parse on a long enough input and leaves the result equal
    to the pairwise accumulation."""
    from sfsutils.parser import _TWO_SFS_RESYNC_INTERVAL

    n_sites = 4 * _TWO_SFS_RESYNC_INTERVAL
    kwargs = dict(source=ALL_SITES_VCF, n=10, d=40, two_sfs_offset=0, max_sites=n_sites)

    fast, naive = _two_sfs(su.Parser, **kwargs), _two_sfs(NaiveParser, **kwargs)

    np.testing.assert_allclose(fast['all'], naive['all'], rtol=1e-9, atol=1e-9)
