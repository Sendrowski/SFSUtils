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


def test_two_sfs_extrapolation_matches_all_sites_ground_truth(tmp_path):
    """Ground truth: parse an all-sites VCF directly (which counts the monomorphic-involving pairs for real) and
    compare against the SNP-only projection of the SAME data parsed with a TargetSiteCounter. The extrapolated
    monomorphic row/column and (0, 0) corner must reproduce the real ones.

    The previous version of this test hand-built the polymorphic block and asserted an algebraic convention, which
    made it self-consistent with the very factor-of-2 it was meant to pin down."""
    L, n_hap, d = 4000, 6, 50
    rng = np.random.default_rng(0)

    # a site at every position, so the site density is exactly 1/bp and the extrapolation has no sampling noise
    derived = np.where(rng.random(L) < 0.03, rng.integers(1, n_hap, size=L), 0)

    header = ('##fileformat=VCFv4.2\n##contig=<ID=1,length=%d>\n'
              '##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral">\n'
              '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
              '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'
              % L + '\t'.join(f's{i}' for i in range(n_hap // 2)) + '\n')

    all_sites, snps = tmp_path / "all.vcf", tmp_path / "snp.vcf"
    with open(all_sites, 'w') as fa, open(snps, 'w') as fs:
        fa.write(header)
        fs.write(header)
        for pos, k in enumerate(derived, start=1):
            hap = np.array([1] * int(k) + [0] * (n_hap - int(k)))
            rng.shuffle(hap)
            row = (f'1\t{pos}\t.\tA\t{"T" if k else "."}\t.\tPASS\tAA=A\tGT\t'
                   + '\t'.join(f'{a}|{b}' for a, b in hap.reshape(-1, 2)) + '\n')
            fa.write(row)
            if k:
                fs.write(row)

    Settings.disable_pbar = True
    kw = dict(n=n_hap, two_sfs=True, d=d, skip_non_polarized=False, subsample_mode="random")
    truth = np.asarray(su.Parser(source=str(all_sites), **kw).parse()["all"].data)
    extrapolated = np.asarray(su.Parser(source=str(snps), **kw,
                                        target_site_counter=su.TargetSiteCounter(n_target_sites=L)).parse()["all"].data)

    # the polymorphic-polymorphic block is observed directly and must match exactly, which also confirms the two
    # parses are on the same scale, so any difference in the monomorphic entries is the extrapolation's own
    np.testing.assert_allclose(extrapolated[1:-1, 1:-1], truth[1:-1, 1:-1])

    # the extrapolated monomorphic entries reproduce the real ones (the sites near the contig edges have fewer
    # partners than the uniform-density assumption predicts, hence the tolerance)
    assert extrapolated[0, 0] == pytest.approx(truth[0, 0], rel=0.02)
    assert extrapolated[0, 1:-1].sum() == pytest.approx(truth[0, 1:-1].sum(), rel=0.02)
    assert extrapolated[1:-1, 0].sum() == pytest.approx(truth[1:-1, 0].sum(), rel=0.02)

@pytest.mark.skipif(not (os.path.exists("resources/msprime/two_epoch.ref.fasta.gz")
                         and os.path.exists("resources/msprime/two_epoch.vcf")),
                    reason="the reference FASTA / VCF fixture is absent")
def test_two_sfs_target_site_counter_populates_monomorphic_bins():
    """A TargetSiteCounter is supported with the two-SFS: parsing a SNP-only VCF with one populates the monomorphic
    (row/column 0) bins of the two-SFS, which are empty without it."""
    Settings.disable_pbar = True
    kw = dict(source="resources/msprime/two_epoch.vcf", n=20, two_sfs=True, d=1000,
              skip_non_polarized=False, subsample_mode="random", fasta="resources/msprime/two_epoch.ref.fasta.gz")

    without = su.Parser(**kw).parse()["all"].data
    with_tsc = su.Parser(**kw, target_site_counter=su.TargetSiteCounter(
        n_samples=50_000, n_target_sites=500_000)).parse()["all"].data

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
    kw = dict(source=VCF, pops=pops, n={"A": 10, "B": 10}, skip_non_polarized=False, subsample_mode="random")

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
    kw = dict(source=VCF, pops=pops, n={"A": 10, "B": 10}, skip_non_polarized=False, subsample_mode="random")

    without = su.Parser(**kw).parse()["all"].data
    n_poly = int(without.sum())

    # n_target_sites below the polymorphic count triggers the degenerate branch; a large n_samples would inflate the
    # origin if the sampled monomorphic mass leaked through
    degenerate = su.Parser(**kw, fasta=REF,
                           target_site_counter=su.TargetSiteCounter(n_samples=100_000,
                                                                    n_target_sites=n_poly - 1)).parse()["all"].data

    np.testing.assert_allclose(degenerate, without)
