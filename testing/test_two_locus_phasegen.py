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
    np.testing.assert_allclose(parsed.covariance().data, exact.covariance().data, atol=0.02)
    np.testing.assert_allclose(parsed.correlation().data, exact.correlation().data, atol=0.06)

    with_mono = empirical.copy()
    with_mono[0, :] += 1e6; with_mono[:, 0] += 1e6      # monomorphic-involving pairs (all-ancestral)
    with_mono[-1, :] += 1e6; with_mono[:, -1] += 1e6    # and the all-derived bin
    np.testing.assert_array_equal(su.TwoSFS(with_mono).covariance().data, parsed.covariance().data)  # untouched
    np.testing.assert_array_equal(su.TwoSFS(with_mono).correlation().data, parsed.correlation().data)


def _empirical_recombining(offset, width, ne, r, mu, seq_len, reps, seed):
    """Pool the two-SFS over pairs at separation in ``(offset, offset + width]`` from recombining replicates, each
    spanning many distinct trees. Returns the pooled matrix and the total number of trees seen."""
    import msprime

    Settings.disable_pbar = True
    emp = np.zeros((N + 1, N + 1))
    total_trees = 0

    trees = msprime.sim_ancestry(samples=N, ploidy=1, population_size=ne, sequence_length=seq_len,
                                 recombination_rate=r, num_replicates=reps, random_seed=seed)
    for i, ts in enumerate(trees):
        total_trees += ts.num_trees
        mts = msprime.sim_mutations(ts, rate=mu, discrete_genome=False, random_seed=seed + 1 + i)
        if mts.num_sites < 2:
            continue
        emp += su.Parser(vcf=mts, n=N, two_sfs=True, two_sfs_offset=offset, two_sfs_distance=width,
                         skip_non_polarized=False, subsample_mode="random").parse().data

    return emp, total_trees


@pytest.mark.slow
@pytest.mark.parametrize("offset,width", [(300, 400), (800, 400), (1800, 400)])
def test_two_sfs_matches_phasegen_with_recombination(offset, width):
    """A two-SFS validation that is NOT based on a single genealogy, swept over three recombination rates. Simulate
    a recombining sequence (each replicate spans many distinct trees), pool the parser's two-SFS over pairs of sites
    at a fixed genetic distance, and check it matches PhaseGen's exact two-locus SFS at the corresponding
    recombination rate ``rho = Ne * r * d``. PhaseGen scales recombination so a linked lineage recombines at rate
    ``rho`` per coalescent time unit; with the unit population size used here (as for the rho = 0 test) that is
    ``Ne * r * d`` for two sites at distance ``d``. The three windows give rho ~ 0.5, 1, 2, and at each the pooled
    two-SFS must be far closer to its own rho than to the fully linked rho = 0 limit."""
    import phasegen as pg

    ne, r, mu, seq_len, reps, seed = 1.0, 1e-3, 1e-3, 20_000, 2_500, 11
    rho = ne * r * (offset + width / 2)  # the two-locus recombination rate for pairs in this window

    emp, total_trees = _empirical_recombining(offset, width, ne, r, mu, seq_len, reps, seed)

    assert total_trees > 3 * reps  # the sequence genuinely recombines: many distinct genealogies, not one tree
    parsed = su.TwoSFS(emp)
    at_rho = su.TwoSFS(np.asarray(pg.Coalescent(n=N, loci=2, recombination_rate=rho).sfs2.mean.data))
    at_zero = su.TwoSFS(np.asarray(pg.Coalescent(n=N, loci=2, recombination_rate=0.0).sfs2.mean.data))

    # the pooled two-SFS matches PhaseGen at the recombination rate implied by the genetic distance
    np.testing.assert_allclose(parsed.covariance().data, at_rho.covariance().data, atol=0.012)
    np.testing.assert_allclose(parsed.correlation().data, at_rho.correlation().data, atol=0.05)

    # and this is a real recombination signal, not the single-tree limit: the parsed correlation is much closer to
    # its own rho than to the fully linked rho = 0 one (which recombination has decorrelated away)
    to_rho = np.abs(parsed.correlation().data - at_rho.correlation().data).max()
    to_zero = np.abs(parsed.correlation().data - at_zero.correlation().data).max()
    assert to_rho < 0.5 * to_zero
