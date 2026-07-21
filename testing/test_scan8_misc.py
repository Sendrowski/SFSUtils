"""
Regression tests for the eighth-scan defects in the CLI, the spectrum containers and the annotations.

Covered here: the CLI refusing an ``--output`` that resolves to the input source, the ``synonymy``
annotation being reachable from ``--annotate``, ``--max-sites`` rejecting values below one, the dict-backed
spectra serializing aliased spectra, the JSON decoders refusing payloads that execute code, the annotations
resetting their per-pass counters on rewind, and the annotator releasing the reader when setup fails.
"""

import os
import shutil

import numpy as np
import pytest
from cyvcf2 import VCF

import sfsutils as su
from sfsutils.cli import run

from testing.test_degeneracy import _write_inputs

VCF_PATH = "resources/msprime/two_epoch.vcf"
ZARR_PATH = "resources/msprime/two_epoch.vcz"

requires_vcf = pytest.mark.skipif(not os.path.exists(VCF_PATH), reason="msprime fixtures absent")
requires_zarr = pytest.mark.skipif(not os.path.exists(ZARR_PATH), reason="msprime fixtures absent")


def _n_records(file: str) -> int:
    """
    Count the variant records of a VCF.

    :param file: The VCF path.
    :return: The number of records.
    """
    return sum(1 for _ in VCF(file))


# --- C2: --output equal to the input ---------------------------------------------------------------

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


# --- C12: --stratify synonymy ----------------------------------------------------------------------

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


def test_synonymy_annotation_factory():
    """The factory builds the annotation the stratification needs."""
    from sfsutils.cli import _build_annotations

    built = _build_annotations(["synonymy"], None, 11)

    assert len(built) == 1
    assert isinstance(built[0], su.SynonymyAnnotation)


# --- C14: --max-sites below one --------------------------------------------------------------------

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


# --- C15: aliased spectra and open-then-render -----------------------------------------------------

@pytest.mark.parametrize("cls,spectrum", [
    (su.TwoSpectra, su.TwoSFS(np.arange(25, dtype=float).reshape(5, 5))),
    (su.JointSpectra, su.JointSFS(np.arange(9, dtype=float).reshape(3, 3))),
])
def test_aliased_spectra_serialize(cls, spectrum):
    """One spectrum object stored under two keys is converted once, not once per key."""
    restored = cls.from_json(cls({'a': spectrum, 'b': spectrum}).to_json())

    assert set(restored.data) == {'a', 'b'}
    assert np.allclose(restored.data['a'].data, spectrum.data)
    assert np.allclose(restored.data['b'].data, spectrum.data)


def test_to_file_leaves_an_existing_file_intact_when_rendering_fails(tmp_path, monkeypatch):
    """The JSON is rendered before the target is opened, so a failure cannot truncate a good file."""
    target = tmp_path / "spectra.json"
    good = su.TwoSpectra({'a': su.TwoSFS(np.zeros((3, 3)))})
    good.to_file(str(target))
    content = target.read_text()

    monkeypatch.setattr(su.TwoSpectra, 'to_json', lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        good.to_file(str(target))

    assert target.read_text() == content


def test_spectrum_to_file_leaves_an_existing_file_intact_when_rendering_fails(tmp_path, monkeypatch):
    """The same ordering holds for the single-spectrum containers."""
    target = tmp_path / "sfs.json"
    sfs = su.TwoSFS(np.zeros((3, 3)))
    sfs.to_file(str(target))
    content = target.read_text()

    monkeypatch.setattr(su.TwoSFS, 'to_json', lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        sfs.to_file(str(target))

    assert target.read_text() == content


# --- C16: decoding a hostile payload ---------------------------------------------------------------

#: A payload that runs a shell command while it is being decoded.
_PAYLOAD = '{{"py/reduce": [{{"py/function": "os.system"}}, {{"py/tuple": ["touch {marker}"]}}]}}'

DECODERS = [su.Spectrum, su.TwoSFS, su.TwoLocusSFS, su.JointSFS, su.JointSpectra, su.TwoSpectra,
            su.MaximumLikelihoodAncestralAnnotation]


@pytest.mark.parametrize("cls", DECODERS)
def test_hostile_payload_does_not_execute(cls, tmp_path):
    """from_json refuses the payload instead of running the command it names."""
    marker = tmp_path / "executed"

    with pytest.raises(ValueError):
        cls.from_json(_PAYLOAD.format(marker=marker))

    assert not marker.exists()


@pytest.mark.parametrize("cls", [c for c in DECODERS if c is not su.Spectrum])
def test_hostile_payload_does_not_execute_through_from_file(cls, tmp_path):
    """The file-based entry point goes through the same screening. Spectrum.from_file reads CSV through
    pandas and never reaches a decoder, so it is covered by its from_json counterpart alone."""
    marker = tmp_path / "executed"
    payload = tmp_path / "payload.json"
    payload.write_text(_PAYLOAD.format(marker=marker))

    with pytest.raises(ValueError):
        cls.from_file(str(payload))

    assert not marker.exists()


def test_payload_of_the_wrong_class_is_refused():
    """A payload decoding to another spectrum type does not pass as this one."""
    with pytest.raises(ValueError):
        su.JointSFS.from_json(su.TwoSFS(np.zeros((3, 3))).to_json())


@pytest.mark.parametrize("obj", [
    su.Spectrum([0, 1, 2, 3, 0]),
    su.TwoSFS(np.arange(25, dtype=float).reshape(5, 5)),
    su.TwoLocusSFS(np.arange(25, dtype=float).reshape(5, 5)),
    su.JointSFS(np.arange(9, dtype=float).reshape(3, 3)),
])
def test_spectrum_round_trip_still_works(obj, tmp_path):
    """Screening the payload does not break the legitimate round-trip."""
    target = tmp_path / "spectrum.json"
    obj.to_file(str(target))
    restored = type(obj).from_file(str(target))

    assert isinstance(restored, type(obj))
    assert np.allclose(restored.data, obj.data)


@pytest.mark.parametrize("cls,spectrum", [
    (su.TwoSpectra, su.TwoSFS(np.arange(25, dtype=float).reshape(5, 5))),
    (su.JointSpectra, su.JointSFS(np.arange(9, dtype=float).reshape(3, 3))),
])
def test_spectra_container_round_trip_still_works(cls, spectrum, tmp_path):
    """The dict-backed containers round-trip through file as well."""
    target = tmp_path / "spectra.json"
    obj = cls({'all': spectrum})
    obj.to_file(str(target))
    restored = cls.from_file(str(target))

    assert isinstance(restored, cls)
    assert np.allclose(restored.data['all'].data, spectrum.data)


@pytest.mark.skipif(not os.path.exists("resources/EST-SFS/test-data.txt"), reason="EST-SFS fixture absent")
def test_ancestral_annotation_round_trip_still_works(tmp_path):
    """The annotation payload carries numpy and scipy helpers, which stay decodable."""
    anc = su.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
        file="resources/EST-SFS/test-data.txt",
        model=su.JCSubstitutionModel(),
        n_runs=1,
        prior=su.KingmanPolarizationPrior(),
        parallelize=False
    )
    anc.infer()

    target = tmp_path / "anc.json"
    anc.to_file(str(target))
    restored = su.MaximumLikelihoodAncestralAnnotation.from_file(str(target))

    assert isinstance(restored, su.MaximumLikelihoodAncestralAnnotation)
    assert restored.params_mle == anc.params_mle


# --- C13: per-pass counters ------------------------------------------------------------------------

def test_degeneracy_counters_reset_on_rewind():
    """The counters describe a single pass, so a rewind clears them alongside n_annotated."""
    ann = su.DegeneracyAnnotation()
    ann.n_annotated = 7
    ann.n_skipped = 13
    ann.mismatches = ['a']
    ann.errors = ['b']

    ann._rewind()

    assert (ann.n_annotated, ann.n_skipped, ann.mismatches, ann.errors) == (0, 0, [], [])


def test_synonymy_counters_reset_on_rewind():
    """The VEP and SnpEff comparison counters follow the same per-pass lifetime."""
    ann = su.SynonymyAnnotation()
    ann.n_skipped = 3
    ann.vep_mismatches = ['a']
    ann.snpeff_mismatches = ['b']
    ann.n_vep_comparisons = 4
    ann.n_snpeff_comparisons = 5

    ann._rewind()

    assert ann.n_skipped == 0
    assert ann.vep_mismatches == [] and ann.snpeff_mismatches == []
    assert ann.n_vep_comparisons == 0 and ann.n_snpeff_comparisons == 0


def test_counters_describe_a_single_parse(tmp_path):
    """Parsing twice with the same annotation reports the same counts, not the accumulated ones."""
    vcf, fasta, gff = _write_inputs(tmp_path)

    ann = su.DegeneracyAnnotation()
    parser = su.Parser(source=vcf, n=2, fasta=fasta, gff=gff,
                       annotations=[ann], stratifications=[su.DegeneracyStratification()])

    parser.parse()
    first = (ann.n_annotated, ann.n_skipped, len(ann.mismatches), len(ann.errors))

    parser.parse()
    second = (ann.n_annotated, ann.n_skipped, len(ann.mismatches), len(ann.errors))

    assert first == second


def test_counters_come_from_a_single_pass_with_a_target_site_counter(tmp_path):
    """The target site counter rewinds mid-parse, so all counters must describe the same pass rather than
    mixing an annotated count from one with a skipped count accumulated over both."""
    vcf, fasta, gff = _write_inputs(tmp_path)

    ann = su.DegeneracyAnnotation()
    parser = su.Parser(
        source=vcf, n=2, fasta=fasta, gff=gff,
        annotations=[ann],
        stratifications=[su.DegeneracyStratification()],
        target_site_counter=su.TargetSiteCounter(n_samples=4, n_target_sites=100)
    )

    parser.parse()
    first = (ann.n_annotated, ann.n_skipped, len(ann.mismatches), len(ann.errors))

    parser.parse()

    # a single pass sees at most the six sites of the input plus the four sampled monomorphic ones
    assert ann.n_annotated + ann.n_skipped <= 10
    assert (ann.n_annotated, ann.n_skipped, len(ann.mismatches), len(ann.errors)) == first


# --- C18: annotator setup failure ------------------------------------------------------------------

@requires_vcf
def test_annotator_without_annotations(tmp_path):
    """The default annotation list is empty rather than absent."""
    out = tmp_path / "out.vcf"
    ann = su.Annotator(source=VCF_PATH, output=str(out))

    assert ann.annotations == []

    ann.annotate()

    assert _n_records(str(out)) == 608


@requires_vcf
def test_annotator_releases_the_reader_when_setup_fails(tmp_path):
    """A writer that cannot be opened must not leave the reader behind."""

    class RecordingReader:
        """A stand-in reader that records whether it was closed."""

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    ann = su.Annotator(source=VCF_PATH, output=str(tmp_path / "out.trees"), annotations=[])

    # pre-empt the cached property, so the reader the teardown has to release is observable
    reader = RecordingReader()
    ann.__dict__['_reader'] = reader

    with pytest.raises(ValueError):
        ann.annotate()

    assert reader.closed
