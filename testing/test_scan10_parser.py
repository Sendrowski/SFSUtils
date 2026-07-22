"""
Regression tests for the defects found by the tenth release-readiness scan: the ``--output`` guard letting a hard
link and a case-insensitive spelling of the input through, ``--pops`` dropping a repeated population, the region
length being estimated from the observed variant span, the two-SFS monomorphic extrapolation ignoring the
per-contig geometry, and the SFS sample size going unvalidated.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2026-07-22"

import argparse
import os

import numpy as np
import pytest

import sfsutils as su
from sfsutils.cli import _check_output_distinct_from_input, _parse_pops, _sample_size, main
from sfsutils.settings import Settings

VCF = "resources/msprime/two_epoch.vcf"


def args_for(source, output):
    """
    Build the parsed arguments the output guard reads.

    :param source: The input source path.
    :param output: The output path.
    :return: The namespace.
    """
    return argparse.Namespace(vcf=source, zarr=None, trees=None, output=output)


class TestOutputGuard:
    """
    An output resolving to the input destroys the input, so it must be rejected however it is spelled (C1).
    """

    def test_hard_link_to_the_input_is_rejected(self, tmp_path):
        """A hard link shares the inode but not the path, which a comparison of the resolved paths misses."""
        source, link = tmp_path / "h1.vcf", tmp_path / "h2.vcf"
        source.write_text("##fileformat=VCFv4.2\n")
        os.link(source, link)

        with pytest.raises(SystemExit, match="input source"):
            _check_output_distinct_from_input(args_for(str(source), str(link)))

        assert source.read_text() == "##fileformat=VCFv4.2\n"

    def test_case_insensitive_spelling_of_the_input_is_rejected(self, tmp_path):
        """On a case-insensitive filesystem the upper-cased path is the very same store."""
        store = tmp_path / "z.vcz"
        store.mkdir()
        upper = tmp_path / "Z.VCZ"

        if not upper.exists():
            pytest.skip("the filesystem is case-sensitive")

        with pytest.raises(SystemExit, match="input source"):
            _check_output_distinct_from_input(args_for(str(store), str(upper)))

    def test_output_inside_the_input_store_is_rejected(self, tmp_path):
        """A zarr output is opened for writing and empties the directory holding the input."""
        store = tmp_path / "in.vcz"
        store.mkdir()

        with pytest.raises(SystemExit, match="input source"):
            _check_output_distinct_from_input(args_for(str(store), str(store / "out.vcz")))

    def test_output_directory_holding_the_input_is_rejected(self, tmp_path):
        """The other direction: an output directory is wiped whole, taking the input below it with it."""
        directory = tmp_path / "data"
        (directory / "nested").mkdir(parents=True)
        source = directory / "nested" / "in.vcz"
        source.mkdir()

        with pytest.raises(SystemExit, match="containing the input source"):
            _check_output_distinct_from_input(args_for(str(source), str(directory)))

    @pytest.mark.parametrize("output", ["out.vcf", "sub/out.vcf", "in.vcf.gz"])
    def test_distinct_outputs_are_allowed(self, tmp_path, output):
        """A distinct output beside the input, or in a directory that does not exist yet, passes."""
        source = tmp_path / "in.vcf"
        source.write_text("##fileformat=VCFv4.2\n")

        _check_output_distinct_from_input(args_for(str(source), str(tmp_path / output)))

    def test_a_missing_or_remote_input_is_not_checked(self, tmp_path):
        """Nothing to compare against, and a remote source is never written to."""
        _check_output_distinct_from_input(args_for(str(tmp_path / "absent.vcf"), str(tmp_path / "absent.vcf")))
        _check_output_distinct_from_input(args_for("https://example.org/in.vcf", str(tmp_path / "out.vcf")))


class TestPopulationSpec:
    """
    A repeated population name used to overwrite the earlier group, yielding a joint SFS of the wrong dimension
    that looks entirely valid downstream (C2).
    """

    def test_repeated_population_is_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="more than once"):
            _parse_pops("A=tsk_0,tsk_1;A=tsk_2,tsk_3")

    def test_empty_sample_list_is_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="no samples"):
            _parse_pops("A=tsk_0;B=")

    def test_empty_name_is_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="name is empty"):
            _parse_pops("=tsk_0")

    def test_missing_separator_is_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid population spec"):
            _parse_pops("A")

    def test_valid_spec_is_parsed(self):
        assert _parse_pops("A=tsk_0, tsk_1 ;B=tsk_2;") == {"A": ["tsk_0", "tsk_1"], "B": ["tsk_2"]}


class TestSampleSize:
    """
    A sample size below two makes bin 1 the divergence bin (``n == 1``), so every segregating site is booked as a
    fixed difference, or collapses the ancestral and the divergence bin onto one index (``n == 0``) (C15).
    """

    @pytest.mark.parametrize("n", [1, 0, -1, -3])
    def test_sample_size_below_two_is_rejected(self, n):
        with pytest.raises(ValueError, match="at least 2"):
            su.Parser(source=VCF, n=n)

    def test_missing_sample_size_is_rejected(self):
        with pytest.raises(ValueError, match="'n' must be given"):
            su.Parser(source=VCF)

    @pytest.mark.parametrize("n", [1, {"A": 4, "B": 1}, [4, 1]])
    def test_sample_size_below_two_is_rejected_per_population(self, n):
        with pytest.raises(ValueError, match="at least 2 for every population"):
            su.Parser(source=VCF, pops={"A": ["tsk_0", "tsk_1"], "B": ["tsk_2", "tsk_3"]}, n=n)

    def test_cli_rejects_a_sample_size_below_two(self):
        with pytest.raises(argparse.ArgumentTypeError, match="at least 2"):
            _sample_size("1")

        assert _sample_size("2") == 2

    def test_cli_parse_rejects_a_sample_size_below_two(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main(["parse", "--vcf", VCF, "--output", str(tmp_path / "sfs.csv"), "--n", "1"])

        assert "at least 2" in capsys.readouterr().err


def write_sites(path, n_contigs, span, derived, n_hap, only_polymorphic=False):
    """
    Write a VCF holding one site per position of every contig.

    :param path: The path to write to.
    :param n_contigs: The number of contigs.
    :param span: The declared length of each contig, and the number of sites on it.
    :param derived: The derived counts, of shape ``(n_contigs, span)``.
    :param n_hap: The number of haplotypes.
    :param only_polymorphic: Whether to write the segregating sites alone.
    :return: The path as a string.
    """
    rng = np.random.default_rng(1)

    header = ('##fileformat=VCFv4.2\n'
              + ''.join(f'##contig=<ID=c{c},length={span}>\n' for c in range(n_contigs))
              + '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
              '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
              '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'
              + '\t'.join(f's{i}' for i in range(n_hap // 2)) + '\n')

    with open(path, 'w') as out:
        out.write(header)

        for contig in range(n_contigs):
            for pos, k in enumerate(derived[contig], start=1):
                if only_polymorphic and not k:
                    continue

                haplotypes = np.array([1] * int(k) + [0] * (n_hap - int(k)))
                rng.shuffle(haplotypes)

                out.write(f'c{contig}\t{pos}\t.\tA\t{"T" if k else "."}\t.\tPASS\tAA=A\tGT\t'
                          + '\t'.join(f'{a}|{b}' for a, b in haplotypes.reshape(-1, 2)) + '\n')

    return str(path)


class TestRegionLength:
    """
    The region length used to be the summed span of the observed variants, which for the SNP-only input a
    TargetSiteCounter exists for falls far short of the contigs and inflates the site density (C13).
    """

    def test_declared_contig_lengths_are_preferred_over_the_observed_span(self, tmp_path):
        """The variants cover the middle of each contig only, so their span underestimates the region."""
        n_contigs, span, n_hap = 5, 2000, 4

        derived = np.zeros((n_contigs, span), dtype=int)
        derived[:, span // 4:3 * span // 4:100] = 2

        vcf = write_sites(tmp_path / "snp.vcf", n_contigs, span, derived, n_hap, only_polymorphic=True)

        Settings.disable_pbar = True
        parser = su.Parser(source=vcf, n=n_hap, two_sfs=True, d=100, skip_non_polarized=False,
                           target_site_counter=su.TargetSiteCounter(n_target_sites=n_contigs * span))
        parser.parse()

        # the reader is closed after the parse, so the spans are captured while it is open
        assert parser._region_length() == pytest.approx(n_contigs * span)

    def test_the_observed_span_is_the_last_resort(self, tmp_path, caplog):
        """A source declaring no length falls back to the observed span, which is warned about."""
        parser = su.Parser(source=VCF, n=4)
        parser._contig_bounds.update({'c0': (100, 1100), 'c1': (10, 510)})
        parser._declared_contig_lengths = lambda: {}

        assert parser._region_length() == pytest.approx(1500)
        assert "no length" in caplog.text


class TestTwoSFSExtrapolation:
    """
    Pairs form within a contig only, so the number of partners a site has saturates once the distance approaches
    the contig span; the extrapolation used to apply a flat ``2 * rho * d`` to every site (C14).
    """

    @pytest.mark.parametrize("span, distance, expected", [
        (1000, 100, 0.95),  # the window fits, losing only the sites past the contig ends
        (100, 100, 0.5),  # the two expressions meet where the window is exactly the span
        (100, 1000, 0.05),  # saturated: the partners are the contig's own sites, not the window's
    ])
    def test_window_factor(self, span, distance, expected):
        assert su.TargetSiteCounter._window_factor([span], distance) == pytest.approx(expected)

    def test_window_factor_of_mixed_contigs_is_weighted_by_span(self):
        spans, distance = [100.0, 1000.0], 100
        mixed = su.TargetSiteCounter._window_factor(spans, distance)

        assert mixed == pytest.approx((100 * 0.5 + 1000 * 0.95) / 1100)

    def test_window_factor_degenerates_to_one(self):
        assert su.TargetSiteCounter._window_factor([], 100) == 1.0
        assert su.TargetSiteCounter._window_factor([1000.0], 0) == 1.0

    @pytest.mark.parametrize("n_contigs, span, distance", [(20, 200, 500), (5, 1000, 200)])
    def test_extrapolation_matches_all_sites_ground_truth_across_contigs(self, tmp_path, n_contigs, span, distance):
        """Ground truth: the same data parsed as an all-sites input, which counts the monomorphic-involving pairs
        for real. Before the fix the short contigs were overestimated by an order of magnitude."""
        n_hap = 4
        rng = np.random.default_rng(0)
        derived = np.where(rng.random((n_contigs, span)) < 0.05, rng.integers(1, n_hap, size=(n_contigs, span)), 0)

        all_sites = write_sites(tmp_path / "all.vcf", n_contigs, span, derived, n_hap)
        snps = write_sites(tmp_path / "snp.vcf", n_contigs, span, derived, n_hap, only_polymorphic=True)

        Settings.disable_pbar = True
        kw = dict(n=n_hap, two_sfs=True, d=distance, skip_non_polarized=False, subsample_mode="random")

        truth = np.asarray(su.Parser(source=all_sites, **kw).parse()["all"].data)
        extrapolated = np.asarray(su.Parser(
            source=snps, **kw,
            target_site_counter=su.TargetSiteCounter(n_target_sites=n_contigs * span)
        ).parse()["all"].data)

        # the polymorphic block is observed directly, so any difference elsewhere is the extrapolation's own
        np.testing.assert_allclose(extrapolated[1:-1, 1:-1], truth[1:-1, 1:-1])

        assert extrapolated.sum() == pytest.approx(truth.sum(), rel=0.03)
        assert extrapolated[0, 0] == pytest.approx(truth[0, 0], rel=0.03)
        assert extrapolated[0, 1:-1].sum() == pytest.approx(truth[0, 1:-1].sum(), rel=0.03)
