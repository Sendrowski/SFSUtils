"""
Validate the two-site SFS parser against an exact analytic ground truth from PhaseGen.

At recombination rate zero the two loci share one genealogy, so PhaseGen's expected two-locus SFS
``Coalescent(n, loci=2, recombination_rate=0).sfs2.mean`` is the within-tree cross-moment
``E[L_i · L_j]`` of the branch lengths. We reproduce this empirically: simulate many independent
non-recombining trees, drop infinite-sites mutations on each, and pair all sites within a tree with
sfsutils' two-SFS (fully linked). Pooled over trees and normalised over the polymorphic block, the
parser's pair counts must converge to PhaseGen's exact matrix. The check spans the Kingman coalescent
and Beta (multiple-merger) coalescents at two intensities, so it exercises the down-projection, the
within-tree pairing and the monomorphic handling against an independent analytic engine.

Requires the optional ``phasegen`` and ``msprime`` packages; skipped otherwise.
"""
import importlib.util
import logging

import numpy as np
import pytest

import sfsutils as su
from sfsutils.settings import Settings

# the pooling loop parses hundreds of trees; keep its per-parse INFO logging out of the test output
logging.getLogger('sfsutils').setLevel(logging.WARNING)

_has_phasegen = importlib.util.find_spec("phasegen") is not None
_has_msprime = importlib.util.find_spec("msprime") is not None

pytestmark = pytest.mark.skipif(not (_has_phasegen and _has_msprime), reason="phasegen or msprime is absent")

N = 4          # haploid sample size
L = 1e4        # sequence length per tree (one genealogy, recombination_rate=0)
THETA = 20.0   # expected mutation intensity per tree
REPS = 500
SEED = 42


def _models(name):
    """Return the matched (phasegen, msprime) coalescent models for a model name."""
    import msprime
    import phasegen as pg

    if name == "kingman":
        return pg.StandardCoalescent(), msprime.StandardCoalescent()

    # a Beta(2 - alpha) multiple-merger coalescent; smaller alpha -> more skewed offspring (stronger multiple mergers)
    alpha = float(name.split("-")[1])
    return pg.BetaCoalescent(alpha=alpha), msprime.BetaCoalescent(alpha=alpha)


def _empirical_two_sfs(ms_model):
    """Pool the fully-linked (within-tree) two-SFS over many independent non-recombining trees."""
    import msprime

    Settings.disable_pbar = True
    emp = np.zeros((N + 1, N + 1))
    rate = THETA / (4 * L)

    trees = msprime.sim_ancestry(samples=N, ploidy=1, sequence_length=L, recombination_rate=0,
                                 model=ms_model, num_replicates=REPS, random_seed=SEED)

    for i, ts in enumerate(trees):
        mts = msprime.sim_mutations(ts, rate=rate, discrete_genome=False, random_seed=SEED + 1 + i)
        if mts.num_sites < 2:
            continue
        # pair every pair of sites within the tree (a window spanning the whole sequence)
        two = su.Parser(vcf=mts, n=N, two_sfs=True, two_sfs_distance=int(L) + 1,
                        skip_non_polarized=False, subsample_mode="random").parse()
        emp += two.data

    return emp


@pytest.mark.parametrize("name", ["kingman", "beta-1.5", "beta-1.8"])
def test_two_sfs_converges_to_phasegen_expectation(name):
    import phasegen as pg

    pg_model, ms_model = _models(name)

    # exact analytic ground truth: the within-tree cross-moment E[L_i L_j] at recombination rate 0
    expected = np.asarray(pg.Coalescent(n=N, loci=2, recombination_rate=0, model=pg_model).sfs2.mean.data)
    empirical = _empirical_two_sfs(ms_model)

    # both are symmetric by construction
    np.testing.assert_allclose(empirical, empirical.T)
    np.testing.assert_allclose(expected, expected.T, atol=1e-9)

    # compare on the polymorphic block (PhaseGen leaves the monomorphic 0 and n bins at zero), each
    # normalised to sum 1; the parser's counts are proportional to E[L_i L_j] in the low-mutation limit
    block = (slice(1, N), slice(1, N))
    e = expected[block] / expected[block].sum()
    o = empirical[block] / empirical[block].sum()

    assert np.corrcoef(e.ravel(), o.ravel())[0, 1] > 0.99
    assert np.abs(e - o).max() < 0.04

    # the interior class-resolved covariance/correlation matrices recovered from the parser match the analytic
    # ones, and, crucially, are invariant to whether monomorphic sites are included (they populate only bins 0/n)
    exact, parsed = su.TwoSFS(expected), su.TwoSFS(empirical)
    np.testing.assert_allclose(parsed.covariance(), exact.covariance(), atol=0.02)
    np.testing.assert_allclose(parsed.correlation(), exact.correlation(), atol=0.06)

    with_mono = empirical.copy()
    with_mono[0, :] += 1e6; with_mono[:, 0] += 1e6      # monomorphic-involving pairs (all-ancestral)
    with_mono[-1, :] += 1e6; with_mono[:, -1] += 1e6    # and the all-derived bin
    np.testing.assert_array_equal(su.TwoSFS(with_mono).covariance(), parsed.covariance())   # interior untouched
    np.testing.assert_array_equal(su.TwoSFS(with_mono).correlation(), parsed.correlation())
