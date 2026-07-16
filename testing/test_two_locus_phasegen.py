"""
Validate the empirical two-site SFS branch-length covariance/correlation against exact analytic ground truth from
PhaseGen, across a range of multiple-merger (Beta coalescent) intensities.

At recombination rate zero both loci share one genealogy, so PhaseGen's two-locus SFS
``Coalescent(n, loci=2, recombination_rate=0).sfs2.mean`` is the within-tree cross-moment ``E[L_i L_j]`` of the
branch lengths, and the class-resolved branch-length covariance is ``Cov(L_i, L_j) = sfs2.mean - outer(sfs.mean,
sfs.mean)``. We reproduce this empirically from all-sites data: simulate many independent non-recombining trees,
drop infinite-sites mutations, and for each tree add ``outer(c, c) - diag(c)`` over its per-class site counts
(monomorphic sites included). That is exactly the fully linked all-sites two-SFS the parser produces from an
all-sites input (see ``test_two_sfs_includes_monomorphic_sites``). :meth:`TwoSFS.cov` / :meth:`TwoSFS.corr`, which
normalise over the full spectrum, must then match PhaseGen's branch-length covariance/correlation, and must track
the multiple-merger intensity: the low-frequency correlation of two linked sites rises with merger strength,
crossing from negative (Kingman) to positive (strong multiple mergers).

Requires the optional ``phasegen`` and ``msprime`` packages; skipped otherwise.
"""
import importlib.util
import logging

import numpy as np
import pytest

import sfsutils as su

# the pooling loop parses thousands of trees; keep its per-parse INFO logging out of the test output
logging.getLogger('sfsutils').setLevel(logging.WARNING)

_has_phasegen = importlib.util.find_spec("phasegen") is not None
_has_msprime = importlib.util.find_spec("msprime") is not None

pytestmark = pytest.mark.skipif(not (_has_phasegen and _has_msprime), reason="phasegen or msprime is absent")

N = 6          # haploid sample size (PhaseGen's two-locus state space grows quickly, so keep n small)
L = 400        # discrete sites per tree
MU = 5e-3      # per-site mutation rate
REPS = 8000
SEED = 11


def _models(name):
    """Return the matched (msprime, phasegen) coalescent models for a model name."""
    import msprime
    import phasegen as pg

    if name == "kingman":
        return msprime.StandardCoalescent(), pg.StandardCoalescent()

    # a Beta(2 - alpha) multiple-merger coalescent; smaller alpha -> stronger multiple mergers
    alpha = float(name.split("-")[1])
    return msprime.BetaCoalescent(alpha=alpha), pg.BetaCoalescent(alpha=alpha)


def _cov2corr(cov):
    """Correlation from a covariance matrix, with zero-variance classes returned as zero."""
    v = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(np.outer(v, v) > 0, cov / np.outer(v, v), 0.0)


def _phasegen_branch_length(pg_model):
    """PhaseGen's exact interior branch-length covariance ``Cov(L_i, L_j)`` and its correlation, at recombination 0."""
    import phasegen as pg

    m2 = np.asarray(pg.Coalescent(n=N, loci=2, recombination_rate=0.0, model=pg_model).sfs2.mean.data)
    m1 = np.asarray(pg.Coalescent(n=N, model=pg_model).sfs.mean.data)
    cov = (m2 - np.outer(m1, m1))[1:-1, 1:-1]

    return cov, _cov2corr(cov)


def _empirical_all_sites_two_sfs(ms_model):
    """Pool the fully linked all-sites two-SFS over independent non-recombining trees. Each tree contributes
    ``outer(c, c) - diag(c)`` over its per-class site counts ``c`` with the monomorphic-ancestral count anchoring
    the total, which is the exact all-sites two-SFS the parser produces at full linkage."""
    import msprime

    S = np.zeros((N + 1, N + 1))
    trees = msprime.sim_ancestry(samples=N, ploidy=1, sequence_length=L, recombination_rate=0,
                                 model=ms_model, num_replicates=REPS, random_seed=SEED)

    for i, ts in enumerate(trees):
        mts = msprime.sim_mutations(ts, rate=MU, discrete_genome=True, random_seed=SEED + 1 + i)
        c = mts.allele_frequency_spectrum(polarised=True, span_normalise=False).astype(float)
        c[0] = L - c[1:].sum()  # the monomorphic-ancestral sites anchor the total to the number of target sites
        S += np.outer(c, c) - np.diag(c)

    return su.TwoSFS(S)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["kingman", "beta-1.7", "beta-1.5"])
def test_two_sfs_cov_corr_match_phasegen_branch_length(name):
    """The empirical all-sites two-SFS branch-length covariance/correlation match PhaseGen's exact ``Cov(L_i, L_j)``
    across multiple-merger intensities: the correlation matrices agree in structure (near-perfect) and on the
    informative low-frequency block, and the covariance agrees up to the (non-recoverable) mutational scale. The
    low-frequency block moves with intensity, from strongly negative under Kingman to positive under Beta."""
    ms_model, pg_model = _models(name)

    pcov, pcorr = _phasegen_branch_length(pg_model)
    emp = _empirical_all_sites_two_sfs(ms_model)
    ecov = emp.cov().data[1:-1, 1:-1]
    ecorr = emp.corr().data[1:-1, 1:-1]

    # the correlation matrices are equivalent: near-perfect structural agreement...
    assert np.corrcoef(ecorr.ravel(), pcorr.ravel())[0, 1] > 0.99

    # ...and a close low-frequency block, where the multiple-merger signal lives and the counts are reliable
    block = (slice(0, 3), slice(0, 3))
    assert np.abs(ecorr[block] - pcorr[block]).max() < 0.06

    # the covariance matches PhaseGen up to a single positive scale (the mutational constant is not recoverable)
    scale = float(np.sum(ecov * pcov) / np.sum(ecov * ecov))
    assert np.abs(scale * ecov - pcov).max() / np.abs(pcov).max() < 0.15


def _phasegen_branch_length_at(rho):
    """PhaseGen's exact interior Kingman branch-length covariance and correlation at recombination rate ``rho``."""
    import phasegen as pg

    m2 = np.asarray(pg.Coalescent(n=N, loci=2, recombination_rate=rho).sfs2.mean.data)
    m1 = np.asarray(pg.Coalescent(n=N).sfs.mean.data)
    cov = (m2 - np.outer(m1, m1))[1:-1, 1:-1]

    return cov, _cov2corr(cov)


def _empirical_recombining_all_sites(offset, width, ne, r, mu, seq_len, reps, seed):
    """The all-sites windowed two-SFS of a recombining sequence, pooled over replicates. Each discrete position
    carries its derived-allele count (zero for monomorphic sites), and positions are paired over separations in
    ``(offset, offset + width]`` — exactly the all-sites windowed two-SFS the parser builds, done directly here for
    speed. The sequence genuinely recombines, so pairs at this separation span many distinct genealogies."""
    import msprime

    S = np.zeros((N + 1, N + 1))
    trees = msprime.sim_ancestry(samples=N, ploidy=1, population_size=ne, sequence_length=seq_len,
                                 recombination_rate=r, num_replicates=reps, random_seed=seed)

    for i, ts in enumerate(trees):
        mts = msprime.sim_mutations(ts, rate=mu, discrete_genome=True, random_seed=seed + 1 + i)
        der = np.zeros(seq_len, dtype=np.int64)
        pos = mts.tables.sites.position.astype(int)
        dc = mts.genotype_matrix().sum(axis=1)
        keep = (dc <= N) & (pos < seq_len)  # drop any recurrent/multi-allelic overflow and out-of-range positions
        der[pos[keep]] = dc[keep]
        for delta in range(offset + 1, offset + width + 1):
            idx = der[:-delta] * (N + 1) + der[delta:]
            S += np.bincount(idx, minlength=(N + 1) ** 2).reshape(N + 1, N + 1)

    return su.TwoSFS(S + S.T)


@pytest.mark.slow
def test_two_sfs_cov_corr_match_phasegen_with_recombination():
    """A branch-length validation that is NOT based on a single genealogy. An all-sites recombining sequence is
    pooled over pairs of sites at a fixed genetic distance ``d``; its two-SFS covariance/correlation must match
    PhaseGen at the corresponding recombination rate ``rho = Ne * r * d``, and be far closer to that rho than to the
    fully linked ``rho = 0`` limit (recombination has decorrelated the two loci)."""
    ne, r, mu, seq_len, reps, seed = 1.0, 1e-3, 3e-3, 40_000, 400, 7
    offset, width = 800, 400
    rho = ne * r * (offset + width / 2)  # = 1.0 for pairs in this window

    emp = _empirical_recombining_all_sites(offset, width, ne, r, mu, seq_len, reps, seed)
    ecov = emp.cov().data[1:-1, 1:-1]
    ecorr = emp.corr().data[1:-1, 1:-1]

    rho_cov, rho_corr = _phasegen_branch_length_at(rho)
    _, zero_corr = _phasegen_branch_length_at(0.0)

    # matches PhaseGen at the recombination rate implied by the genetic distance
    assert np.corrcoef(ecorr.ravel(), rho_corr.ravel())[0, 1] > 0.99
    assert np.abs(ecorr - rho_corr).max() < 0.06
    scale = float(np.sum(ecov * rho_cov) / np.sum(ecov * ecov))
    assert np.abs(scale * ecov - rho_cov).max() / np.abs(rho_cov).max() < 0.12

    # a real recombination signal, not the single-tree limit: much closer to its own rho than to rho = 0
    to_rho = np.abs(ecorr - rho_corr).max()
    to_zero = np.abs(ecorr - zero_corr).max()
    assert to_rho < 0.4 * to_zero
