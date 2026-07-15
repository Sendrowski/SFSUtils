"""
Tests for the command-line interface. Parser-level tests assert on the parsed ``argparse.Namespace`` and on the
private option-building helpers; end-to-end tests invoke the entry function ``run([...argv])`` against the
committed msprime fixtures and check the exit code and the written artifact.
"""
import os

import numpy as np
import pytest

import sfsutils as sf
from sfsutils.settings import Settings
from sfsutils.cli import (build_parser, run, _split_csv, _parse_pops, _lookup,
                          _build_filtrations, _build_stratifications, _build_annotations)

VCF = "resources/msprime/two_epoch.vcf"
JOINT_VCF = "resources/msprime/two_epoch_joint.vcf"
TWO_SFS_VCF = "resources/msprime/two_sfs.vcf"

requires_fixtures = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (VCF, JOINT_VCF, TWO_SFS_VCF)),
    reason="msprime fixtures absent",
)


# --- parser-level ---------------------------------------------------------------------------------

def test_defaults():
    ns = build_parser().parse_args(["parse", "--vcf", "x.vcf", "--n", "10", "--out", "o.csv"])
    assert ns.command == "parse"
    assert ns.n == 10
    assert ns.filter == ["poly-allelic"]
    assert ns.stratify == [] and ns.annotate == []
    assert ns.skip_non_polarized is True
    assert ns.subsample_mode == "probabilistic"
    assert ns.two_sfs is False


def test_csv_and_flag_parsing():
    ns = build_parser().parse_args(
        ["parse", "--vcf", "x", "--n", "8", "--out", "o", "--stratify", "degeneracy,synonymy",
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
        build_parser().parse_args(["-v", "-q", "parse", "--vcf", "x", "--n", "2", "--out", "o"])


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


def test_build_annotations_degeneracy():
    ann = _build_annotations(["degeneracy"], None, 11)
    assert [type(a).__name__ for a in ann] == ["DegeneracyAnnotation"]


def test_build_annotations_mle_requires_outgroups():
    with pytest.raises(SystemExit, match="requires --outgroups"):
        _build_annotations(["maximum-likelihood-ancestral"], None, 11)


# --- end-to-end -----------------------------------------------------------------------------------

@requires_fixtures
def test_run_parse_one_dimensional(tmp_path):
    Settings.disable_pbar = True
    out = tmp_path / "sfs.csv"
    code = run(["-q", "parse", "--vcf", VCF, "--n", "20", "--no-skip-non-polarized",
                "--subsample-mode", "random", "--out", str(out)])
    assert code == 0 and out.exists()
    assert sf.Spectra.from_file(str(out)).all.n_polymorphic > 0


@requires_fixtures
def test_run_parse_joint(tmp_path):
    Settings.disable_pbar = True
    out = tmp_path / "jsfs.json"
    code = run(["-q", "parse", "--vcf", JOINT_VCF, "--n", "6",
                "--pops", "A=tsk_0,tsk_1,tsk_2,tsk_3;B=tsk_4,tsk_5,tsk_6",
                "--no-skip-non-polarized", "--subsample-mode", "random", "--out", str(out)])
    assert code == 0 and out.exists()
    loaded = sf.JointSpectra.from_file(str(out))
    assert loaded.types == ["all"] and loaded.n_pops == 2


@requires_fixtures
def test_run_parse_two_sfs(tmp_path):
    Settings.disable_pbar = True
    out = tmp_path / "sfs2.json"
    code = run(["-q", "parse", "--vcf", TWO_SFS_VCF, "--n", "20", "--two-sfs",
                "--two-sfs-distance", "1000", "--no-skip-non-polarized",
                "--subsample-mode", "random", "--out", str(out)])
    assert code == 0 and out.exists()
    sfs2 = sf.SFS2.from_file(str(out))
    assert sfs2.data.shape == (21, 21)
    np.testing.assert_allclose(sfs2.data, sfs2.data.T)


@requires_fixtures
def test_run_filter(tmp_path):
    Settings.disable_pbar = True
    out = tmp_path / "filtered.vcf"
    code = run(["-q", "filter", "--vcf", VCF, "--filter", "snp,poly-allelic", "--out", str(out)])
    assert code == 0 and out.exists() and out.stat().st_size > 0
