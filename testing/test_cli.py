"""
Tests for the command-line interface. Parser-level tests assert on the parsed ``argparse.Namespace`` and on the
private option-building helpers; end-to-end tests invoke the entry function ``run([...argv])`` against the
committed msprime fixtures and check the exit code and the written artifact.
"""
import os

import numpy as np
import pytest

import sfsutils as su
from sfsutils.settings import Settings
from sfsutils.cli import (build_parser, run, _split_csv, _parse_pops, _lookup,
                          _build_filtrations, _build_stratifications, _build_annotations)
import pandas as pd
from sfsutils.io_handlers import get_called_alleles
from sfsutils.spectrum import Spectrum, Spectra
from sfsutils.cli import run
import shutil
from cyvcf2 import VCF as VCFReader
from testing.test_degeneracy import _write_inputs

VCF = "resources/msprime/two_epoch.vcf"
JOINT_VCF = "resources/msprime/two_epoch_joint.vcf"
TWO_SFS_VCF = "resources/msprime/two_sfs.vcf"

requires_fixtures = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (VCF, JOINT_VCF, TWO_SFS_VCF)),
    reason="msprime fixtures absent",
)


# --- parser-level ---------------------------------------------------------------------------------

def test_defaults():
    ns = build_parser().parse_args(["parse", "--vcf", "x.vcf", "--n", "10", "--output", "o.csv"])
    assert ns.command == "parse"
    assert ns.n == 10
    assert ns.filter == ["poly-allelic"]
    assert ns.stratify == [] and ns.annotate == []
    assert ns.skip_non_polarized is True
    assert ns.subsample_mode == "probabilistic"
    assert ns.two_sfs is False


def test_csv_and_flag_parsing():
    ns = build_parser().parse_args(
        ["parse", "--vcf", "x", "--n", "8", "--output", "o", "--stratify", "degeneracy,synonymy",
         "--filter", "snp,coding-sequence", "--no-skip-non-polarized", "--two-sfs"]
    )
    assert ns.stratify == ["degeneracy", "synonymy"]
    assert ns.filter == ["snp", "coding-sequence"]
    assert ns.skip_non_polarized is False
    assert ns.two_sfs is True


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert "sfsutils" in capsys.readouterr().out


def test_subcommand_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_verbose_quiet_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-v", "-q", "parse", "--vcf", "x", "--n", "2", "--output", "o"])


# --- helper units ---------------------------------------------------------------------------------

def test_split_csv():
    assert _split_csv("a, b ,,c") == ["a", "b", "c"]


def test_parse_pops():
    assert _parse_pops("A=s1,s2;B=s3") == {"A": ["s1", "s2"], "B": ["s3"]}


def test_parse_pops_invalid():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_pops("no-equals-sign")


def test_lookup_unknown_raises():
    with pytest.raises(SystemExit, match="Unknown filtration"):
        _lookup({"snp": object}, "nope", "filtration")


def test_build_filtrations():
    filtrations = _build_filtrations(["snp", "poly-allelic", "no", "all"], None)
    assert [type(f).__name__ for f in filtrations] == \
        ["SNPFiltration", "PolyAllelicFiltration", "NoFiltration", "AllFiltration"]


def test_build_stratifications():
    strat = _build_stratifications(["degeneracy", "synonymy"])
    assert [type(s).__name__ for s in strat] == ["DegeneracyStratification", "SynonymyStratification"]


def test_build_filtrations_contig_requires_contigs():
    with pytest.raises(SystemExit, match="requires --contigs"):
        _build_filtrations(["contig"], None)
    assert type(_build_filtrations(["contig"], ["1"])[0]).__name__ == "ContigFiltration"


def test_two_sfs_offset_flag_parsed():
    ns = build_parser().parse_args(
        ["parse", "--vcf", "x", "--n", "2", "--output", "o", "--two-sfs", "--two-sfs-offset", "50"])
    assert ns.two_sfs_offset == 50


def test_build_annotations_degeneracy():
    ann = _build_annotations(["degeneracy"], None, 11)
    assert [type(a).__name__ for a in ann] == ["DegeneracyAnnotation"]


def test_build_annotations_mle_requires_outgroups():
    with pytest.raises(SystemExit, match="requires --outgroups"):
        _build_annotations(["maximum-likelihood-ancestral"], None, 11)


# --- end-to-end -----------------------------------------------------------------------------------

GROUND_TRUTH_SFS = "resources/msprime/two_epoch.sfs.txt"


@requires_fixtures
def test_run_parse_one_dimensional(tmp_path):
    Settings.disable_pbar = True
    out = tmp_path / "sfs.csv"
    code = run(["-q", "parse", "--vcf", VCF, "--n", "20", "--no-skip-non-polarized",
                "--subsample-mode", "random", "--output", str(out)])
    assert code == 0 and out.exists()
    # at full sample size the CLI must reproduce the tskit ground-truth SFS bin for bin
    expected = np.loadtxt(GROUND_TRUTH_SFS, dtype=int)
    parsed = np.array(su.Spectra.from_file(str(out)).all.to_list()).astype(int)
    np.testing.assert_array_equal(parsed[1:20], expected[1:20])
    assert parsed[1:20].sum() > 0


@requires_fixtures
def test_run_parse_joint(tmp_path):
    Settings.disable_pbar = True
    out = tmp_path / "jsfs.json"
    code = run(["-q", "parse", "--vcf", JOINT_VCF, "--n", "6",
                "--pops", "A=tsk_0,tsk_1,tsk_2,tsk_3;B=tsk_4,tsk_5,tsk_6",
                "--no-skip-non-polarized", "--subsample-mode", "random", "--output", str(out)])
    assert code == 0 and out.exists()
    loaded = su.JointSpectra.from_file(str(out))
    assert loaded.types == ["all"] and loaded.n_pops == 2
    # the CLI must produce the same joint spectrum as the equivalent Parser call (shared default seed)
    direct = su.Parser(source=JOINT_VCF, n=6, pops={"A": ["tsk_0", "tsk_1", "tsk_2", "tsk_3"],
                                                 "B": ["tsk_4", "tsk_5", "tsk_6"]},
                       skip_non_polarized=False, subsample_mode="random").parse()
    np.testing.assert_array_equal(np.asarray(loaded["all"]), np.asarray(direct["all"]))


@requires_fixtures
def test_run_parse_two_sfs(tmp_path):
    Settings.disable_pbar = True
    out = tmp_path / "two_sfs.json"
    code = run(["-q", "parse", "--vcf", TWO_SFS_VCF, "--n", "20", "--two-sfs",
                "--two-sfs-distance", "1000", "--no-skip-non-polarized",
                "--subsample-mode", "random", "--output", str(out)])
    assert code == 0 and out.exists()
    # the two-SFS parse mode writes a single-entry TwoSpectra collection (keyed 'all')
    sfs2 = su.TwoSpectra.from_file(str(out))["all"]
    assert sfs2.data.shape == (21, 21)
    np.testing.assert_allclose(sfs2.data, sfs2.data.T)
    # the CLI must produce the same two-SFS as the equivalent Parser call
    direct = su.Parser(source=TWO_SFS_VCF, n=20, two_sfs=True, d=1000,
                       skip_non_polarized=False, subsample_mode="random").parse()["all"]
    np.testing.assert_array_equal(sfs2.data, direct.data)
    assert sfs2.data.sum() > 0


@requires_fixtures
def test_run_filter(tmp_path):
    Settings.disable_pbar = True
    out = tmp_path / "filtered.vcf"
    code = run(["-q", "filter", "--vcf", VCF, "--filter", "snp,poly-allelic", "--output", str(out)])
    assert code == 0 and out.exists() and out.stat().st_size > 0


class TestCLIWiring:
    """
    ``--contigs`` reaches the contig stratification, and a malformed ``--pops`` exits cleanly.
    """

    def test_contigs_reach_stratification(self):
        from sfsutils.cli import _build_stratifications

        assert _build_stratifications(['contig'], ['chr1'])[0].contigs == ['chr1']

    def test_malformed_pops_exits(self):
        from sfsutils.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(['parse', '--source', 'x.vcf', '--n', '10', '--out', 'o.csv',
                                       '--pops', 'nonsense'])


VCF_PATH = "resources/msprime/two_epoch.vcf"


requires_vcf = pytest.mark.skipif(not os.path.exists(VCF_PATH), reason="msprime fixtures absent")


@requires_vcf
@pytest.mark.parametrize('n', ['0', '-3'])
def test_parse_rejects_non_positive_n(tmp_path, n):
    """A sample size below one either writes a nonsense single-bin spectrum or raises a raw numpy error."""
    out = str(tmp_path / 'sfs.json')

    with pytest.raises(SystemExit):
        run(['-q', 'parse', '--vcf', VCF_PATH, '--n', n, '--output', out])

    assert not os.path.exists(out)


@requires_vcf
@pytest.mark.parametrize('option', ['--two-sfs-distance', '--n-ingroups'])
def test_parse_rejects_non_positive_counts(tmp_path, option):
    """The remaining count-valued options are validated alongside ``--n``."""
    with pytest.raises(SystemExit):
        run(['-q', 'parse', '--vcf', VCF_PATH, '--n', '8', option, '0',
             '--output', str(tmp_path / 'sfs.json')])




ZARR_PATH = "resources/msprime/two_epoch.vcz"




requires_zarr = pytest.mark.skipif(not os.path.exists(ZARR_PATH), reason="msprime fixtures absent")


def _n_records(file: str) -> int:
    """
    Count the variant records of a VCF.

    :param file: The VCF path.
    :return: The number of records.
    """
    return sum(1 for _ in VCFReader(file))


@requires_vcf
@pytest.mark.parametrize("argv_tail", [
    ["filter", "--filter", "no"],
    ["parse", "--n", "10"],
    ["annotate", "--annotation", "degeneracy"],
])
def test_output_equal_to_input_is_refused(tmp_path, argv_tail):
    """Every subcommand refuses to write its output over the input, leaving the input untouched."""
    target = tmp_path / "self.vcf"
    shutil.copy(VCF_PATH, target)
    before = _n_records(str(target))

    with pytest.raises(SystemExit) as exc:
        run([argv_tail[0], "--vcf", str(target), *argv_tail[1:], "--output", str(target)])

    assert "resolves to the input source" in str(exc.value)
    assert _n_records(str(target)) == before


@requires_vcf
def test_output_equal_to_input_is_refused_through_a_non_canonical_path(tmp_path):
    """The comparison resolves both paths, so a detour through '.' and '..' is caught as well."""
    target = tmp_path / "self.vcf"
    shutil.copy(VCF_PATH, target)
    (tmp_path / "sub").mkdir()

    with pytest.raises(SystemExit):
        run(["filter", "--vcf", str(target), "--filter", "no",
             "--output", str(tmp_path / "sub" / ".." / "self.vcf")])

    assert _n_records(str(target)) == 608


@requires_zarr
def test_zarr_output_equal_to_zarr_input_is_refused(tmp_path):
    """A zarr store is a directory, and overwriting it in place is the same hazard as for a VCF."""
    target = tmp_path / "store.vcz"
    shutil.copytree(ZARR_PATH, target)
    before = sorted(p.name for p in target.iterdir())

    with pytest.raises(SystemExit):
        run(["filter", "--zarr", str(target), "--filter", "no", "--output", str(target)])

    assert sorted(p.name for p in target.iterdir()) == before


@requires_vcf
def test_distinct_output_is_still_accepted(tmp_path):
    """The guard only rejects a genuine collision."""
    out = tmp_path / "filtered.vcf"

    assert run(["filter", "--vcf", VCF_PATH, "--filter", "no", "--output", str(out)]) == 0
    assert _n_records(str(out)) == 608


def test_synonymy_annotation_is_reachable_from_the_cli(tmp_path):
    """--annotate synonymy produces the tag --stratify synonymy consumes."""
    vcf, fasta, gff = _write_inputs(tmp_path)
    out = tmp_path / "sfs.csv"

    code = run(["parse", "--vcf", vcf, "--n", "2", "--fasta", fasta, "--gff", gff,
                "--annotate", "synonymy", "--stratify", "synonymy", "--filter", "snp",
                "--output", str(out)])

    assert code == 0

    spectra = su.Spectra.from_file(str(out))
    assert set(spectra.types) == {"neutral", "selected"}
    assert spectra.n_sites.sum() > 0


@pytest.mark.parametrize("command,extra", [
    ("parse", ["--n", "10"]),
    ("filter", ["--filter", "no"]),
    ("annotate", ["--annotation", "degeneracy"]),
])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_max_sites_below_one_is_a_usage_error(command, extra, value):
    """A limit the library stop conditions can never reach is rejected as a usage error."""
    with pytest.raises(SystemExit) as exc:
        run([command, "--vcf", "x.vcf", *extra, "--max-sites", value, "--output", "out"])

    assert exc.value.code == 2


def test_max_sites_of_one_is_accepted():
    """The smallest limit the stop conditions can reach stays valid."""
    from sfsutils.cli import build_parser

    ns = build_parser().parse_args(["filter", "--vcf", "x.vcf", "--filter", "no",
                                    "--max-sites", "1", "--output", "o.vcf"])

    assert ns.max_sites == 1
