"""
Validate the empirical two-site SFS branch-length covariance/correlation against exact analytic ground truth from
PhaseGen, for RECOMBINING sequences across a range of multiple-merger (Beta coalescent) intensities.

Every test simulates an all-sites sequence that genuinely recombines, so pairs of sites at a fixed genomic
separation ``d`` span many distinct genealogies rather than a single tree. The empirical windowed two-SFS
covariance/correlation must then match PhaseGen's two-locus branch-length covariance ``Cov(L_i, L_j) = sfs2.mean -
outer(sfs.mean, sfs.mean)`` at the corresponding recombination rate ``rho = Ne * r * d``. The Kingman and the Beta
coalescent share the same ``rho = Ne * r * d`` calibration (verified numerically: the best-fitting PhaseGen rho
equals the naive Kingman value to within the grid for both alpha = 1.5 and 1.7). The multiple-merger signal
survives recombination: the low-frequency correlation of two linked sites is negative under Kingman and rises to
positive under strong multiple mergers.

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

N = 6            # haploid sample size (PhaseGen's two-locus state space grows quickly, so keep n small)

# a recombining all-sites sequence: pairs at separation ~d span many genealogies at rho = Ne * r * d
NE = 1.0
R = 1e-3         # per-base recombination rate
MU = 3e-3        # per-base mutation rate
SEQ_LEN = 40_000
REPS = 500
SEED = 7
OFFSET, WIDTH = 800, 400
RHO = NE * R * (OFFSET + WIDTH / 2)  # = 1.0 for pairs in this window


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


def _phasegen_branch_length_at(pg_model, rho):
    """PhaseGen's exact interior branch-length covariance ``Cov(L_i, L_j)`` and its correlation at recombination
    rate ``rho`` under the given model."""
    import phasegen as pg

    m2 = np.asarray(pg.Coalescent(n=N, loci=2, recombination_rate=rho, model=pg_model).sfs2.mean.data)
    m1 = np.asarray(pg.Coalescent(n=N, model=pg_model).sfs.mean.data)
    cov = (m2 - np.outer(m1, m1))[1:-1, 1:-1]

    return cov, _cov2corr(cov)


def _empirical_recombining_all_sites(ms_model, offset, width, ne, r, mu, seq_len, reps, seed):
    """The all-sites windowed two-SFS of a recombining sequence, pooled over replicates. Each discrete position
    carries its derived-allele count (zero for monomorphic sites), and positions are paired over separations in
    ``(offset, offset + width]`` — exactly the all-sites windowed two-SFS the parser builds, done directly here for
    speed. The sequence genuinely recombines, so pairs at this separation span many distinct genealogies."""
    import msprime

    S = np.zeros((N + 1, N + 1))
    trees = msprime.sim_ancestry(samples=N, ploidy=1, population_size=ne, sequence_length=seq_len,
                                 recombination_rate=r, model=ms_model, num_replicates=reps, random_seed=seed)

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
@pytest.mark.parametrize("name", ["kingman", "beta-1.7", "beta-1.5"])
def test_two_sfs_cov_corr_match_phasegen_with_recombination(name):
    """The empirical recombining all-sites two-SFS covariance/correlation match PhaseGen's exact branch-length
    ``Cov(L_i, L_j)`` at ``rho = Ne * r * d``, across multiple-merger intensities and NOT from a single genealogy:
    the correlation matrices agree (near-perfect structure and a close low-frequency block), the covariance agrees
    up to the non-recoverable mutational scale, and the empirical correlation is much closer to its own rho than to
    the fully linked ``rho = 0`` limit (recombination has genuinely decorrelated the two loci). The low-frequency
    block moves with intensity, from negative under Kingman to positive under strong multiple mergers."""
    ms_model, pg_model = _models(name)

    emp = _empirical_recombining_all_sites(ms_model, OFFSET, WIDTH, NE, R, MU, SEQ_LEN, REPS, SEED)
    ecov = emp.cov().data[1:-1, 1:-1]
    ecorr = emp.corr().data[1:-1, 1:-1]

    rho_cov, rho_corr = _phasegen_branch_length_at(pg_model, RHO)
    _, zero_corr = _phasegen_branch_length_at(pg_model, 0.0)

    # matches PhaseGen at the recombination rate implied by the genetic distance
    assert np.corrcoef(ecorr.ravel(), rho_corr.ravel())[0, 1] > 0.99
    assert np.abs(ecorr - rho_corr).max() < 0.08

    # the covariance matches PhaseGen up to a single positive scale (the mutational constant is not recoverable)
    scale = float(np.sum(ecov * rho_cov) / np.sum(ecov * ecov))
    assert np.abs(scale * ecov - rho_cov).max() / np.abs(rho_cov).max() < 0.15

    # a real recombination signal, not the single-tree limit: closer to its own rho than to rho = 0
    to_rho = np.abs(ecorr - rho_corr).max()
    to_zero = np.abs(ecorr - zero_corr).max()
    assert to_rho < 0.7 * to_zero


@pytest.mark.slow
@pytest.mark.parametrize("name", ["kingman", "beta-1.5"])
def test_two_sfs_fpmi_recovers_phasegen(name):
    """The empirical fpmi() recovers PhaseGen's ground-truth fPMI (the same ratio-PMI on the polymorphic two-locus
    branch-length moments) for Kingman and multiple mergers, and is exactly invariant to the monomorphic sites: a
    SNP-only spectrum gives the same fPMI as the all-sites spectrum."""
    import phasegen as pg
    ms_model, pg_model = _models(name)

    # PhaseGen ground truth: fPMI of the polymorphic two-locus branch-length cross-moment (same fpmi definition)
    gt = su.TwoSFS(np.asarray(pg.Coalescent(n=N, loci=2, recombination_rate=RHO, model=pg_model).sfs2.mean.data)
                   ).fpmi().data[1:-1, 1:-1]

    emp = _empirical_recombining_all_sites(ms_model, OFFSET, WIDTH, NE, R, MU, SEQ_LEN, REPS, SEED)
    efpmi = emp.fpmi().data[1:-1, 1:-1]

    # recovers the ground truth: near-perfect structure and a close low-frequency block
    assert np.corrcoef(efpmi.ravel(), gt.ravel())[0, 1] > 0.99
    assert np.abs(efpmi[:3, :3] - gt[:3, :3]).max() < 0.06

    # exactly invariant to the monomorphic sites: dropping them (SNP-only) leaves fpmi unchanged
    snp_only = emp.data.copy(); snp_only[[0, -1], :] = 0.0; snp_only[:, [0, -1]] = 0.0
    np.testing.assert_array_equal(su.TwoSFS(snp_only).fpmi().data, emp.fpmi().data)
