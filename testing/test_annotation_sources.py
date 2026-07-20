"""
Cross-source equivalence for the Annotator and MaximumLikelihoodAncestralAnnotation: the same small
dataset, presented as a VCF, a VCF-Zarr store and a tree sequence, must yield the same annotation
regardless of input source, and the same values whichever output format it is written to. The sources
are only checked against each other (and, for degeneracy, against the hand-worked genetic code), so no
external tool or ground truth is involved. These need the optional ``tskit`` / ``zarr`` packages and
are skipped otherwise; they are deliberately small so they run in the default (fast) suite.
"""
import importlib.util
import os

import numpy as np
import pytest

import sfsutils as su
from sfsutils.settings import Settings
from sfsutils.io_handlers import Variant

_has_tskit = importlib.util.find_spec("tskit") is not None
_has_zarr = importlib.util.find_spec("zarr") is not None
requires_tskit = pytest.mark.skipif(not _has_tskit, reason="tskit is absent")
requires_zarr = pytest.mark.skipif(not _has_zarr, reason="zarr is absent")

# committed msprime fixtures: the same data as a VCF, a tree sequence and a VCF-Zarr store
VCF_FIX = "resources/msprime/two_epoch.vcf"
TREES_FIX = "resources/msprime/two_epoch.trees"
VCZ_FIX = "resources/msprime/two_epoch.vcz"
requires_fixtures = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (VCF_FIX, TREES_FIX, VCZ_FIX)),
    reason="the msprime fixtures are absent",
)

# an 18 bp forward-strand CDS (phase 0) with SNPs at hand-worked codon positions, reused from
# test_degeneracy: (POS, REF, ALT, expected n-fold degeneracy)
CODING_SEQ = "ATGGTTTTTCGTTATTAA"
SITES = [(4, "G", "A", 0), (5, "T", "C", 0), (6, "T", "C", 4),
         (9, "T", "C", 2), (12, "T", "C", 4), (15, "T", "C", 2)]
SAMPLES = ["S0", "S1"]
EXPECTED_DEGENERACY = {pos: deg for pos, _, _, deg in SITES}


# --- building the same synthetic coding contig as each input source --------------------------------

def _fasta_gff(tmp_path):
    fasta = tmp_path / "genome.fasta"
    fasta.write_text(f">1\n{CODING_SEQ}\n")
    gff = tmp_path / "genome.gff"
    gff.write_text("\t".join(["1", "synthetic", "CDS", "1", str(len(CODING_SEQ)), ".", "+", "0", "."]) + "\n")
    return str(fasta), str(gff)


def _write_vcf(tmp_path):
    header = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=1,length={len(CODING_SEQ)}>",
        '##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + SAMPLES),
    ]
    rows = ["\t".join(["1", str(pos), ".", ref, alt, ".", ".", f"AA={ref}", "GT", "0/1", "0/0"])
            for pos, ref, alt, _ in SITES]
    path = tmp_path / "variants.vcf"
    path.write_text("\n".join(header + rows) + "\n")
    return str(path)


def _write_vcz(tmp_path):
    from sfsutils.io_handlers import ZarrVariantWriter

    path = str(tmp_path / "variants.vcz")
    writer = ZarrVariantWriter(path, samples=SAMPLES, seqnames=["1"], info_ancestral="AA")
    for pos, ref, alt, _ in SITES:
        writer.write(Variant(ref=ref, pos=pos, chrom="1", alt=[alt],
                             gt_bases=[f"{ref}/{alt}", f"{ref}/{ref}"], is_snp=True, info={"AA": ref}))
    writer.close()
    return path


def _write_trees(tmp_path):
    import tskit

    tables = tskit.TableCollection(sequence_length=len(CODING_SEQ))
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    root = tables.nodes.add_row(flags=0, time=1)
    tables.edges.add_row(left=0, right=len(CODING_SEQ), parent=root, child=0)
    tables.edges.add_row(left=0, right=len(CODING_SEQ), parent=root, child=1)
    # a tskit site at position p-1 is read back as VCF POS = int(p-1) + 1 = p
    for pos, ref, alt, _ in SITES:
        sid = tables.sites.add_row(position=pos - 1, ancestral_state=ref)
        tables.mutations.add_row(site=sid, node=0, derived_state=alt)
    tables.sort()
    path = str(tmp_path / "variants.trees")
    tables.tree_sequence().dump(path)
    return path


# --- reading the annotated degeneracy back out of each output format --------------------------------

def _degeneracy_from_vcf(path):
    from cyvcf2 import VCF
    return {v.POS: int(v.INFO.get("Degeneracy")) for v in VCF(path)}


def _degeneracy_from_vcz(path):
    import zarr
    root = zarr.open(path, mode="r")
    return {int(p): int(d) for p, d in zip(root["variant_position"][:], root["variant_Degeneracy"][:])}


def _annotate(source, output, fasta, gff):
    Settings.disable_pbar = True
    su.Annotator(source=source, output=output, fasta=fasta, gff=gff,
                 annotations=[su.DegeneracyAnnotation()]).annotate()
    return output


# --- Annotator: the input source must not change the annotation -------------------------------------

@requires_zarr
@requires_tskit
def test_degeneracy_input_sources_agree(tmp_path):
    """Annotating the identical coding contig read as a VCF, a VCF-Zarr store and a tree sequence gives
    the same per-site degeneracy (and it matches the hand-worked genetic code)."""
    fasta, gff = _fasta_gff(tmp_path)
    sources = {"vcf": _write_vcf(tmp_path), "vcz": _write_vcz(tmp_path), "trees": _write_trees(tmp_path)}

    # VCF output is only written from a VCF input, so normalise every source to a VCF-Zarr output
    results = {name: _degeneracy_from_vcz(_annotate(src, str(tmp_path / f"out_{name}.vcz"), fasta, gff))
               for name, src in sources.items()}

    assert results["vcf"] == EXPECTED_DEGENERACY
    assert results["vcf"] == results["vcz"] == results["trees"]


# --- Annotator: the output format must not change the annotation ------------------------------------

@requires_zarr
def test_degeneracy_vcf_and_zarr_output_agree(tmp_path):
    """Annotating a VCF to a VCF output and to a VCF-Zarr output records the same degeneracy values."""
    fasta, gff = _fasta_gff(tmp_path)
    vcf = _write_vcf(tmp_path)

    from_vcf = _degeneracy_from_vcf(_annotate(vcf, str(tmp_path / "out.vcf"), fasta, gff))
    from_vcz = _degeneracy_from_vcz(_annotate(vcf, str(tmp_path / "out.vcz"), fasta, gff))

    assert from_vcf == EXPECTED_DEGENERACY == from_vcz


@requires_tskit
def test_degeneracy_trees_output(tmp_path):
    """Annotating a tree sequence to a ``.trees`` output stores the degeneracy as site metadata."""
    import tskit

    fasta, gff = _fasta_gff(tmp_path)
    trees = _write_trees(tmp_path)
    out = _annotate(trees, str(tmp_path / "out.trees"), fasta, gff)

    ts = tskit.load(out)
    stored = {int(s.position) + 1: s.metadata.get("Degeneracy") for s in ts.sites()}
    assert stored == EXPECTED_DEGENERACY


# --- MaximumLikelihoodAncestralAnnotation: the input source must not change the inference -----------

@requires_zarr
@requires_tskit
@requires_fixtures
def test_ml_aaa_input_sources_agree():
    """Running maximum-likelihood ancestral annotation over the same data read as a VCF, a VCF-Zarr store
    and a tree sequence yields the same polarised spectrum."""
    Settings.disable_pbar = True

    def spectrum(source):
        anc = su.MaximumLikelihoodAncestralAnnotation(
            outgroups=["tsk_0"], n_ingroups=8, model=su.JCSubstitutionModel(),
            n_runs=1, parallelize=False, seed=0)
        parsed = su.Parser(source=source, n=8, skip_non_polarized=False,
                           subsample_mode="random", annotations=[anc]).parse()
        return np.array(parsed.all.to_list())

    reference = spectrum(VCF_FIX)
    np.testing.assert_array_almost_equal(reference, spectrum(VCZ_FIX))
    np.testing.assert_array_almost_equal(reference, spectrum(TREES_FIX))
    assert reference.sum() > 0
