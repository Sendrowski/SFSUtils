"""
Ground-truth tests for the TargetSiteCounter extrapolation of monomorphic sites into the joint SFS, and for its
refusal to combine with the two-SFS.

For the joint SFS the monomorphic corner is fixed exactly by the target-site count. The two-SFS is different: its
covariance/correlation require the real monomorphic sites (an all-sites input), so a TargetSiteCounter is not
supported together with ``two_sfs=True`` and must raise. The two-SFS branch-length covariance/correlation are
validated against PhaseGen in ``test_two_locus_phasegen.py``.
"""
import os
import textwrap

import numpy as np
import pytest

import sfsutils as su
from sfsutils.settings import Settings


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


def test_target_site_counter_not_supported_with_two_sfs(tmp_path):
    """The two-SFS covariance/correlation require the real monomorphic sites, so a target-site extrapolation is
    refused: combining a :class:`~sfsutils.parser.TargetSiteCounter` with ``two_sfs=True`` raises."""
    Settings.disable_pbar = True
    vcf = tmp_path / "poly.vcf"
    _write_vcf(vcf, positions=[10, 30, 60, 90], derived=[1, 2, 1, 1])

    with pytest.raises(NotImplementedError, match="two_sfs"):
        su.Parser(vcf=str(vcf), n=2, two_sfs=True, two_sfs_distance=100, skip_non_polarized=False,
                  subsample_mode="random",
                  target_site_counter=su.TargetSiteCounter(n_target_sites=10_000)).parse()


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
