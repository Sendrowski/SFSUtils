"""
Ground-truth tests for the TargetSiteCounter extrapolation of monomorphic sites into the joint SFS and the two-SFS.

For the joint SFS the monomorphic corner is fixed exactly by the target-site count. For the two-SFS the counter
extrapolates the monomorphic-involving pairs from the target-site count (only approximate for the branch-length
covariance/correlation, which prefer a real all-sites input; the ratio-based ``fpmi`` needs no monomorphic sites).
The two-SFS branch-length covariance/correlation are validated against PhaseGen in ``test_two_locus_phasegen.py``.
"""
import os

import numpy as np
import pytest

import sfsutils as su
from sfsutils.settings import Settings


def test_two_sfs_target_site_counter_extrapolates_monomorphic_pairs():
    """A TargetSiteCounter's two-SFS extrapolation adds the monomorphic-involving pairs from the target-site count:
    the monomorphic (row/column 0) bins are populated and scale with the target-site count, while the polymorphic
    interior is left unchanged."""
    poly = np.zeros((5, 5)); poly[1:-1, 1:-1] = np.array([[3.0, 2, 1], [2, 4, 2], [1, 2, 3]])
    marginal = np.array([0.0, 6.0, 8.0, 6.0, 0.0])  # polymorphic SFS (no monomorphic sites)

    def extrapolated(n_target):
        return su.TargetSiteCounter(n_target_sites=n_target)._extrapolate_two_sfs(
            poly.copy(), marginal, region_length=1000.0, distance=100)

    a = extrapolated(10_000)
    b = extrapolated(20_000)
    assert a[0, 0] > 0 and a[0, 1:-1].sum() > 0                  # monomorphic-involving pairs added
    assert b[0, 0] > a[0, 0]                                     # scale with the target-site count
    np.testing.assert_allclose(a[1:-1, 1:-1], poly[1:-1, 1:-1])  # polymorphic interior unchanged


@pytest.mark.skipif(not (os.path.exists("resources/msprime/two_epoch.ref.fasta.gz")
                         and os.path.exists("resources/msprime/two_epoch.vcf")),
                    reason="the reference FASTA / VCF fixture is absent")
def test_two_sfs_target_site_counter_populates_monomorphic_bins():
    """A TargetSiteCounter is supported with the two-SFS: parsing a SNP-only VCF with one populates the monomorphic
    (row/column 0) bins of the two-SFS, which are empty without it."""
    Settings.disable_pbar = True
    kw = dict(vcf="resources/msprime/two_epoch.vcf", n=20, two_sfs=True, d=1000,
              skip_non_polarized=False, subsample_mode="random", fasta="resources/msprime/two_epoch.ref.fasta.gz")

    without = su.Parser(**kw).parse().data
    with_tsc = su.Parser(**kw, target_site_counter=su.TargetSiteCounter(
        n_samples=50_000, n_target_sites=500_000)).parse().data

    assert without[0].sum() == 0                          # the msprime VCF has no monomorphic sites
    assert with_tsc[0].sum() > 0 and with_tsc[0, 0] > 0   # the counter extrapolated the monomorphic pairs


# --- joint SFS: the monomorphic corner is fixed exactly by the target-site count -------------------

REF = "resources/msprime/two_epoch.ref.fasta.gz"
VCF = "resources/msprime/two_epoch.vcf"


@pytest.mark.skipif(not (os.path.exists(REF) and os.path.exists(VCF)),
                    reason="the reference FASTA / VCF fixture is absent")
def test_joint_target_site_counter_sets_corner_exactly():
    Settings.disable_pbar = True
    pops = {"A": [f"tsk_{i}" for i in range(5)], "B": [f"tsk_{i}" for i in range(5, 10)]}
    kw = dict(vcf=VCF, pops=pops, n={"A": 10, "B": 10}, skip_non_polarized=False, subsample_mode="random")

    without = su.Parser(**kw).parse()["all"].data
    n_poly = without.sum()

    n_target = 50_000
    with_tsc = su.Parser(**kw, fasta=REF,
                         target_site_counter=su.TargetSiteCounter(n_samples=50_000,
                                                                  n_target_sites=n_target)).parse()["all"].data

    # every monomorphic site maps to the all-ancestral origin, so the corner is fixed exactly
    assert with_tsc.sum() == pytest.approx(n_target)
    assert with_tsc[0, 0] == pytest.approx(n_target - n_poly)

    # the polymorphic cells (everything but the origin) are unchanged
    without_no_corner = without.copy(); without_no_corner[0, 0] = 0
    with_no_corner = with_tsc.copy(); with_no_corner[0, 0] = 0
    np.testing.assert_allclose(with_no_corner, without_no_corner)


@pytest.mark.skipif(not (os.path.exists(REF) and os.path.exists(VCF)),
                    reason="the reference FASTA / VCF fixture is absent")
def test_joint_target_site_counter_degenerate_leaves_corner_unchanged():
    """When the target-site count does not exceed the polymorphic count (a misconfiguration), the joint SFS must be
    left as the observed polymorphic spectrum, not contaminated by the monomorphic sites sampled during counting."""
    Settings.disable_pbar = True
    pops = {"A": [f"tsk_{i}" for i in range(5)], "B": [f"tsk_{i}" for i in range(5, 10)]}
    kw = dict(vcf=VCF, pops=pops, n={"A": 10, "B": 10}, skip_non_polarized=False, subsample_mode="random")

    without = su.Parser(**kw).parse()["all"].data
    n_poly = int(without.sum())

    # n_target_sites below the polymorphic count triggers the degenerate branch; a large n_samples would inflate the
    # origin if the sampled monomorphic mass leaked through
    degenerate = su.Parser(**kw, fasta=REF,
                           target_site_counter=su.TargetSiteCounter(n_samples=100_000,
                                                                    n_target_sites=n_poly - 1)).parse()["all"].data

    np.testing.assert_allclose(degenerate, without)
