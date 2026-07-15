"""
Unit tests for the two-SFS distance window and contig handling, on a tiny hand-constructed VCF so the expected
pair counts can be verified by hand. This exercises the sliding-window offset boundary and the per-contig reset,
which the msprime fixtures (single contig, offset 0) do not cover.
"""
import textwrap

import numpy as np
import pytest

import sfsutils as sf
from sfsutils.settings import Settings

# one diploid sample (n = 2 haplotypes), all heterozygous -> every site has derived count 1 (index 1).
# three sites on contig "1" at 10/20/30 and one on contig "2" at 15.
VCF = textwrap.dedent("""\
    ##fileformat=VCFv4.2
    ##contig=<ID=1>
    ##contig=<ID=2>
    #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
    1\t10\t.\tA\tT\t.\t.\t.\tGT\t0/1
    1\t20\t.\tA\tT\t.\t.\t.\tGT\t0/1
    1\t30\t.\tA\tT\t.\t.\t.\tGT\t0/1
    2\t15\t.\tA\tT\t.\t.\t.\tGT\t0/1
    """)


@pytest.fixture()
def vcf_path(tmp_path):
    p = tmp_path / "tiny.vcf"
    p.write_text(VCF)
    return str(p)


def _two_sfs(vcf_path, distance, offset=0):
    Settings.disable_pbar = True
    return sf.Parser(vcf=vcf_path, n=2, two_sfs=True, two_sfs_distance=distance, two_sfs_offset=offset,
                     skip_non_polarized=False, subsample_mode="random").parse()


def test_pairs_within_window_only_and_never_cross_contig(vcf_path):
    # window (0, 15]: on contig 1 the pairs (10,20) and (20,30) qualify, (10,30) at 20 bp does not;
    # the contig-2 site at 15 is never paired with contig-1 sites -> exactly 2 pairs, all at derived (1, 1).
    sfs2 = _two_sfs(vcf_path, distance=15)
    assert sfs2.data.shape == (3, 3)
    assert sfs2.data[1, 1] == 2
    assert sfs2.data.sum() == 2


def test_offset_lower_bound_excludes_close_pairs(vcf_path):
    # window (15, 30]: only (10,30) at 20 bp qualifies; the two 10-bp pairs are now below the offset -> 1 pair.
    sfs2 = _two_sfs(vcf_path, distance=15, offset=15)
    assert sfs2.data[1, 1] == 1
    assert sfs2.data.sum() == 1


def test_offset_shifts_window_upper_bound(vcf_path):
    # window (5, 20]: all three contig-1 pairs (10 bp, 10 bp, 20 bp) qualify -> 3 pairs.
    sfs2 = _two_sfs(vcf_path, distance=15, offset=5)
    assert sfs2.data[1, 1] == 3
    assert sfs2.data.sum() == 3
