"""
Ground-truth tests for the TargetSiteCounter extrapolation of monomorphic sites into the two-site and joint SFS.

For the two-SFS the reference is a full site list (polymorphic sites plus monomorphic sites laid out at a known
density) whose windowed two-SFS is computed directly; stripping the monomorphic sites and running the parser with
a TargetSiteCounter must recover it. The polymorphic-polymorphic block is reproduced exactly (the extrapolation
only touches the monomorphic row and column), and the monomorphic-involving pairs match to within the edge-effect
tolerance of the uniform-density approximation. For the joint SFS the monomorphic corner is fixed exactly by the
target-site count.
"""
import os
import textwrap

import numpy as np
import pytest

import sfsutils as su
from sfsutils.settings import Settings


def _windowed_two_sfs(derived, positions, n, distance, offset=0):
    """Forward-pair every pair of sites within (offset, offset + distance] and symmetrize (independent reference)."""
    ref = np.zeros((n + 1, n + 1))
    lo, hi = offset, offset + distance
    for a in range(len(positions)):
        b = a + 1
        while b < len(positions) and positions[b] - positions[a] <= hi:
            if positions[b] - positions[a] > lo:
                ref[derived[a], derived[b]] += 1
            b += 1
    return (ref + ref.T) / 2


def _write_vcf(path, positions, derived):
    """Write a one-diploid-sample VCF: derived count 1 -> 0/1, 2 -> 1/1 (monomorphic sites are omitted)."""
    header = textwrap.dedent("""\
        ##fileformat=VCFv4.2
        ##contig=<ID=1>
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
        """)
    gt = {1: "0/1", 2: "1/1"}
    rows = "".join(f"1\t{p}\t.\tA\tT\t.\t.\t.\tGT\t{gt[d]}\n" for p, d in zip(positions, derived))
    path.write_text(header + rows)


# --- unit check of the extrapolation kernel --------------------------------------------------------

def test_extrapolate_two_sfs_kernel():
    """The kernel adds marginal[j]*rho_m*d to (0,j)/(j,0) and n_m*rho_m*d to (0,0), leaving the interior intact."""
    tsc = su.TargetSiteCounter(n_target_sites=1000)

    interior = np.array([[0, 0, 0], [0, 5, 3], [0, 3, 2]], dtype=float)  # a symmetric polymorphic 2-SFS (n=2)
    marginal = np.array([0.0, 8.0, 4.0])  # 8 singletons, 4 fixed -> n_poly = 12
    L, d = 1000.0, 100

    out = tsc._extrapolate_two_sfs(interior, marginal, region_length=L, distance=d)

    n_m = 1000 - 12
    rho_m = n_m / L
    np.testing.assert_allclose(out[1:, 1:], interior[1:, 1:])          # interior untouched
    np.testing.assert_allclose(out[0, 1], 8.0 * rho_m * d)             # mono-singleton pairs
    np.testing.assert_allclose(out[0, 2], 4.0 * rho_m * d)             # mono-fixed pairs
    np.testing.assert_allclose(out[0, 1], out[1, 0])                   # symmetric
    np.testing.assert_allclose(out[0, 0], n_m * rho_m * d)             # mono-mono pairs


# --- two-SFS against a full-sequence ground truth --------------------------------------------------

def test_two_sfs_target_site_counter_matches_ground_truth(tmp_path):
    Settings.disable_pbar = True
    rng = np.random.default_rng(0)

    L, d = 50_000, 1_000

    # polymorphic sites: random positions (spanning nearly all of L) with derived count 1 or 2
    poly_pos = np.sort(rng.choice(np.arange(3, L - 2), size=60, replace=False))
    poly_pos[0], poly_pos[-1] = 3, L - 2  # anchor the span so it covers the region
    poly_der = rng.integers(1, 3, size=poly_pos.size)

    # monomorphic sites laid out at uniform density over the same span, avoiding the polymorphic positions
    mono_pos = np.linspace(3, L - 2, 2_000).astype(int)
    mono_pos = np.array(sorted(set(mono_pos) - set(poly_pos.tolist())))

    # full site list -> ground-truth windowed two-SFS (n = 2)
    positions = np.concatenate([poly_pos, mono_pos])
    derived = np.concatenate([poly_der, np.zeros(mono_pos.size, dtype=int)])
    order = np.argsort(positions)
    positions, derived = positions[order], derived[order]
    truth = _windowed_two_sfs(derived, positions, n=2, distance=d)

    # parse only the polymorphic sites, extrapolating the monomorphic pairs from the target-site count
    vcf = tmp_path / "poly.vcf"
    _write_vcf(vcf, poly_pos, poly_der)
    n_target = poly_pos.size + mono_pos.size
    sfs2 = su.Parser(vcf=str(vcf), n=2, two_sfs=True, two_sfs_distance=d, skip_non_polarized=False,
                     subsample_mode="random",
                     target_site_counter=su.TargetSiteCounter(n_target_sites=n_target)).parse()

    # the polymorphic-polymorphic block is reproduced exactly (same positions, same window)
    np.testing.assert_array_equal(sfs2.data[1:, 1:], truth[1:, 1:])

    # the monomorphic-involving pairs match to within the edge-effect tolerance of the uniform-density model
    assert sfs2.data[0, 0] == pytest.approx(truth[0, 0], rel=0.06)
    assert sfs2.data[0, 1:].sum() == pytest.approx(truth[0, 1:].sum(), rel=0.06)
    np.testing.assert_allclose(sfs2.data, sfs2.data.T)  # symmetric


def test_two_sfs_target_site_counter_leaves_polymorphic_block_unchanged(tmp_path):
    """The extrapolation must only add the monomorphic row/column, never alter the observed polymorphic pairs."""
    Settings.disable_pbar = True
    rng = np.random.default_rng(1)

    pos = np.sort(rng.choice(np.arange(3, 20_000), size=40, replace=False))
    der = rng.integers(1, 3, size=pos.size)
    vcf = tmp_path / "poly.vcf"
    _write_vcf(vcf, pos, der)

    kw = dict(vcf=str(vcf), n=2, two_sfs=True, two_sfs_distance=1_000, skip_non_polarized=False,
              subsample_mode="random")
    without = su.Parser(**kw).parse()
    with_tsc = su.Parser(**kw, target_site_counter=su.TargetSiteCounter(n_target_sites=100_000)).parse()

    np.testing.assert_array_equal(without.data[1:, 1:], with_tsc.data[1:, 1:])
    assert with_tsc.data[0, 0] > 0  # but the monomorphic corner is now populated


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
