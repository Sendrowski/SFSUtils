"""
Unit tests for the two-SFS distance window and contig handling, on a tiny hand-constructed VCF so the expected
pair counts can be verified by hand. This exercises the sliding-window offset boundary and the per-contig reset,
which the msprime fixtures (single contig, offset 0) do not cover.
"""
import logging
import textwrap

import numpy as np
import pytest

import sfsutils as su
from sfsutils.settings import Settings


class _ListHandler(logging.Handler):
    """Collect emitted log messages; the ``sfsutils`` logger does not propagate to root, so caplog cannot see it."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _capture_sfsutils_logs(fn):
    handler = _ListHandler()
    log = logging.getLogger("sfsutils")
    log.addHandler(handler)
    try:
        fn()
    finally:
        log.removeHandler(handler)
    return handler.messages

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
    return su.Parser(vcf=vcf_path, n=2, two_sfs=True, d=distance, two_sfs_offset=offset,
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


# --- monomorphic-inclusive ground truth -----------------------------------------------------------
# one diploid sample (n = 2); an all-sites contig with monomorphic (0/0 -> 0 derived), heterozygous
# (0/1 -> 1) and homozygous-alternate (1/1 -> 2) sites. The two-SFS must place the monomorphic sites at
# derived-count 0, anchoring the (0, .) row/column, which the segregating-sites-only msprime fixture
# cannot test.
ALL_SITES_VCF = textwrap.dedent("""\
    ##fileformat=VCFv4.2
    ##contig=<ID=1>
    #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
    1\t10\t.\tA\t.\t.\t.\t.\tGT\t0/0
    1\t20\t.\tA\tT\t.\t.\t.\tGT\t0/1
    1\t30\t.\tA\tT\t.\t.\t.\tGT\t1/1
    1\t40\t.\tA\t.\t.\t.\t.\tGT\t0/0
    """)


def _naive_two_sfs(derived, positions, n, distance, offset=0):
    """Independent reference: forward-pair every pair of (all) sites within the window, then symmetrize."""
    ref = np.zeros((n + 1, n + 1))
    max_distance = offset + distance
    for a in range(len(positions)):
        for b in range(a + 1, len(positions)):
            sep = positions[b] - positions[a]
            if offset < sep <= max_distance:
                ref[derived[a], derived[b]] += 1
    return (ref + ref.T) / 2


def test_two_sfs_includes_monomorphic_sites(tmp_path):
    p = tmp_path / "all_sites.vcf"
    p.write_text(ALL_SITES_VCF)
    Settings.disable_pbar = True

    sfs2 = su.Parser(vcf=str(p), n=2, two_sfs=True, d=100,
                     skip_non_polarized=False, subsample_mode="random").parse()

    # derived counts: 0 (monomorphic), 1 (het), 2 (hom-alt), 0 (monomorphic)
    expected = _naive_two_sfs(derived=[0, 1, 2, 0], positions=[10, 20, 30, 40], n=2, distance=100)

    np.testing.assert_allclose(sfs2.data, expected)
    # the monomorphic sites must contribute: the (0, .) row carries mass
    assert sfs2.data[0].sum() > 0
    assert sfs2.data[0, 0] == 1  # the single (pos 10, pos 40) monomorphic-monomorphic pair


# --- monomorphic-missing warning ------------------------------------------------------------------
# the covariance/correlation need the monomorphic sites to anchor the marginal; if the input carries
# (almost) none, parsing must warn (mirroring the AAA monomorphic-sites warning).
_MONO_WARNING = "monomorphic sites is unusually low"


def _all_sites_vcf(n_mono, n_poly, spacing=10):
    """Build an all-sites VCF string with ``n_mono`` monomorphic (0/0) then ``n_poly`` heterozygous (0/1) sites."""
    header = ("##fileformat=VCFv4.2\n##contig=<ID=1>\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n")
    rows, pos = [], spacing
    for _ in range(n_mono):
        rows.append(f"1\t{pos}\t.\tA\t.\t.\t.\t.\tGT\t0/0")
        pos += spacing
    for _ in range(n_poly):
        rows.append(f"1\t{pos}\t.\tA\tT\t.\t.\t.\tGT\t0/1")
        pos += spacing
    return header + "\n".join(rows) + "\n"


def test_two_sfs_warns_when_monomorphic_sites_missing(vcf_path):
    # the tiny VCF is all heterozygous (no monomorphic sites at all) -> warn.
    Settings.disable_pbar = True
    messages = _capture_sfsutils_logs(lambda: _two_sfs(vcf_path, distance=15))
    assert any(_MONO_WARNING in m for m in messages)


def test_two_sfs_no_warning_with_all_sites_input(tmp_path):
    # 40 monomorphic + 1 polymorphic site (~98% monomorphic) -> no warning.
    p = tmp_path / "all_sites.vcf"
    p.write_text(_all_sites_vcf(n_mono=40, n_poly=1))
    Settings.disable_pbar = True
    messages = _capture_sfsutils_logs(lambda: su.Parser(
        vcf=str(p), n=2, two_sfs=True, d=100,
        skip_non_polarized=False, subsample_mode="random").parse())
    assert not any(_MONO_WARNING in m for m in messages)


# --- within-stratum pairing -----------------------------------------------------------------------

class _ParityStratification(su.Stratification):
    """Minimal stratification assigning each site a type from the parity of its position (for testing)."""

    def get_type(self, variant):
        return "even" if variant.POS % 2 == 0 else "odd"

    def get_types(self):
        return ["even", "odd"]


# four heterozygous sites on one contig at positions chosen so parity splits them: 10, 12 (even) and
# 15, 17 (odd). Within a wide window all six forward pairs qualify, but only same-parity pairs are counted.
PARITY_VCF = textwrap.dedent("""\
    ##fileformat=VCFv4.2
    ##contig=<ID=1>
    #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
    1\t10\t.\tA\tT\t.\t.\t.\tGT\t0/1
    1\t12\t.\tA\tT\t.\t.\t.\tGT\t0/1
    1\t15\t.\tA\tT\t.\t.\t.\tGT\t0/1
    1\t17\t.\tA\tT\t.\t.\t.\tGT\t0/1
    """)


def test_stratified_two_sfs_counts_only_within_stratum_pairs(tmp_path):
    p = tmp_path / "parity.vcf"
    p.write_text(PARITY_VCF)
    Settings.disable_pbar = True

    result = su.Parser(vcf=str(p), n=2, two_sfs=True, d=100,
                       skip_non_polarized=False, subsample_mode="random",
                       stratifications=[_ParityStratification()]).parse()

    assert isinstance(result, su.TwoSpectra)
    assert sorted(result.types) == ["even", "odd"]

    # every site is heterozygous (derived count 1); one within-stratum pair per parity: (10,12) and (15,17)
    for t in ("even", "odd"):
        assert result[t].data[1, 1] == 1
        assert result[t].data.sum() == 1

    # cross-parity pairs (four of the six) are not counted, so the pooled total is 2, not 6
    assert result.all.data.sum() == 2


# one even-parity pair (10, 12) but only a single odd-parity site (15): the odd stratum forms no pair.
PARITY_UNPAIRED_VCF = textwrap.dedent("""\
    ##fileformat=VCFv4.2
    ##contig=<ID=1>
    #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
    1\t10\t.\tA\tT\t.\t.\t.\tGT\t0/1
    1\t12\t.\tA\tT\t.\t.\t.\tGT\t0/1
    1\t15\t.\tA\tT\t.\t.\t.\tGT\t0/1
    """)


def test_stratified_two_sfs_keeps_strata_without_pairs(tmp_path):
    p = tmp_path / "parity_unpaired.vcf"
    p.write_text(PARITY_UNPAIRED_VCF)
    Settings.disable_pbar = True

    result = su.Parser(vcf=str(p), n=2, two_sfs=True, d=100,
                       skip_non_polarized=False, subsample_mode="random",
                       stratifications=[_ParityStratification()]).parse()

    # the odd stratum has a site but forms no within-window pair; it must still be present (as zeros),
    # not silently dropped
    assert sorted(result.types) == ["even", "odd"]
    assert result["even"].data[1, 1] == 1
    assert result["odd"].data.sum() == 0

