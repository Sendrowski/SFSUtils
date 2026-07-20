"""
Fast, synthetic correctness check for :class:`sfsutils.annotation.DegeneracyAnnotation`.

We build a tiny coding contig with a known reading frame (single forward-strand CDS, phase 0),
place SNPs at hand-worked codon positions, and assert that the per-site n-fold degeneracy tag
matches what the genetic code dictates. This exercises the classification directly on committed-free
synthetic data, complementing the slow Betula/CLI degeneracy tests.
"""

import pytest
from cyvcf2 import VCF

import sfsutils as su

# An 18 bp coding sequence, one forward-strand CDS (phase 0), codons:
#   ATG(Met) GTT(Val) TTT(Phe) CGT(Arg) TAT(Tyr) TAA(stop)
CODING_SEQ = "ATGGTTTTTCGTTATTAA"

# (POS, REF, ALT, expected Degeneracy). Degeneracy depends only on the reference codon and the
# position within it; the ALT is irrelevant to the classification. Hand-worked from the genetic code
# with the implementation's {0:0, 1:2, 2:2, 3:4} synonymous-count -> fold mapping (so no distinct
# 3-fold class exists: e.g. a 2-synonymous third position is reported as 2-fold).
#   pos 4  G  GTT 1st  -> ATT/CTT/TTT (Ile/Leu/Phe), 0 synonymous            -> 0-fold
#   pos 5  T  GTT 2nd  -> GAT/GCT/GGT (Asp/Ala/Gly), 0 synonymous            -> 0-fold
#   pos 6  T  GTT 3rd  -> GTA/GTC/GTG (all Val),      3 synonymous           -> 4-fold
#   pos 9  T  TTT 3rd  -> TTA/TTC/TTG (Leu/Phe/Leu),  1 synonymous           -> 2-fold
#   pos 12 T  CGT 3rd  -> CGA/CGC/CGG (all Arg),      3 synonymous           -> 4-fold
#   pos 15 T  TAT 3rd  -> TAA/TAC/TAG (stop/Tyr/stop),1 synonymous           -> 2-fold
SITES = [
    (4, "G", "A", 0),
    (5, "T", "C", 0),
    (6, "T", "C", 4),
    (9, "T", "C", 2),
    (12, "T", "C", 4),
    (15, "T", "C", 2),
]


def _write_inputs(tmp_path):
    """Write the synthetic FASTA, GFF, and VCF and return their paths."""
    fasta = tmp_path / "genome.fasta"
    fasta.write_text(f">1\n{CODING_SEQ}\n")

    # single CDS spanning the whole contig, + strand, phase 0
    gff = tmp_path / "genome.gff"
    gff.write_text("\t".join(["1", "synthetic", "CDS", "1", str(len(CODING_SEQ)), ".", "+", "0", "."]) + "\n")

    samples = ["S0", "S1"]
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=1,length={len(CODING_SEQ)}>",
        '##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples),
    ]
    for pos, ref, alt, _ in SITES:
        lines.append("\t".join(["1", str(pos), ".", ref, alt, ".", ".", f"AA={ref}", "GT", "0/1", "0/0"]))

    vcf = tmp_path / "variants.vcf"
    vcf.write_text("\n".join(lines) + "\n")

    return str(vcf), str(fasta), str(gff)


def test_degeneracy_annotation_synthetic_codons(tmp_path):
    """Per-site degeneracy tags match the genetic code for hand-worked codon positions."""
    vcf, fasta, gff = _write_inputs(tmp_path)
    out = tmp_path / "degeneracy.vcf"

    ann = su.Annotator(
        source=vcf,
        output=str(out),
        fasta=fasta,
        gff=gff,
        annotations=[su.DegeneracyAnnotation()],
    )
    ann.annotate()

    observed = {v.POS: v.INFO.get("Degeneracy") for v in VCF(str(out))}
    expected = {pos: deg for pos, _, _, deg in SITES}

    assert observed == expected, f"degeneracy mismatch: observed {observed}, expected {expected}"

    # the classification must produce genuinely different classes, not a constant
    assert set(expected.values()) == {0, 2, 4}


def test_degeneracy_classification_from_codon_table():
    """The codon-degeneracy table itself classifies representative positions correctly."""
    # _get_degeneracy(codon, position_within_codon) -> 0 / 2 / 4
    assert su.DegeneracyAnnotation._get_degeneracy("GTT", 0) == 0  # Val 1st position
    assert su.DegeneracyAnnotation._get_degeneracy("GTT", 1) == 0  # Val 2nd position
    assert su.DegeneracyAnnotation._get_degeneracy("GTT", 2) == 4  # Val 3rd position, fourfold
    assert su.DegeneracyAnnotation._get_degeneracy("TTT", 2) == 2  # Phe 3rd position, twofold
    assert su.DegeneracyAnnotation._get_degeneracy("CGT", 2) == 4  # Arg 3rd position, fourfold
    assert su.DegeneracyAnnotation._get_degeneracy("TAT", 2) == 2  # Tyr 3rd position, twofold
