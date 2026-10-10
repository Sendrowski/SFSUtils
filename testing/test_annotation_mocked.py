"""
Drive the per-variant annotation pipeline (``DegeneracyAnnotation.annotate_site`` and its codon
parsing) directly with synthetic, integer-free inputs -- a hand-built reference contig, a CDS
record and ``DummyVariant`` sites -- instead of a real VCF/FASTA/GFF. This reaches the genome-
dependent annotation code that the end-to-end tests (inference/slow tier) need the full betula
genome for, while staying instantaneous.

Reference contig (1-based), single CDS on the + strand, phase 0:

    pos:  1 2 3 | 4 5 6 | 7 8 9 | 10 11 12 ...
    base: A T G | G T T | G T A |  G  T  C ...
    aa:    Met  |  Val  |  Val  |    Val
"""
import pandas as pd
import pytest

from sfsutils.annotation import DegeneracyAnnotation, SynonymyAnnotation
from sfsutils.io_handlers import DummyVariant

# Met ATG, then Val GTT/GTA/GTC..., Arg CGG, Pro CCC, Lys AAA, stop TAA
CONTIG = 'ATGGTTGTAGTCGTGGTACGGCCCAAATAA'


class _Handler:
    @staticmethod
    def get_aliases(chrom):
        return {chrom}


def _cds(start, end, strand):
    return pd.Series({'seqid': 'chr1', 'start': start, 'end': end, 'strand': strand, 'phase': 0})


def _make(cls, strand='+', phase=0, start=1, end=len(CONTIG), prev=None, contig=CONTIG):
    # prev is the (start, end) of the preceding CDS of the same transcript
    ann = cls()
    ann._handler = _Handler()
    ann._cd = pd.Series({'seqid': 'chr1', 'start': start, 'end': end, 'strand': strand, 'phase': phase})
    ann._cd_prev = None if prev is None else _cds(*prev, strand)
    ann._cd_next = None
    ann._contig = contig
    ann._fetch = lambda v: None  # state is pre-injected, skip the file-based CDS fetch
    return ann


def _make_ann(**kwargs):
    return _make(DegeneracyAnnotation, **kwargs)


def _annotate(ann, ref, pos):
    v = DummyVariant(ref=ref, pos=pos, chrom='chr1')
    ann.annotate_site(v)
    return v


def test_degeneracy_fourfold_site():
    # pos 6 is the 3rd position of the Val codon GTT -> 4-fold degenerate
    ann = _make_ann()
    v = _annotate(ann, ref='T', pos=6)
    assert v.INFO['Degeneracy'] == 4
    assert v.INFO['Degeneracy_Info'] == '2,+,GTT'
    assert ann.n_annotated == 1


def test_degeneracy_zerofold_site():
    # pos 4 is the 1st position of the Val codon GTT -> 0-fold degenerate
    ann = _make_ann()
    v = _annotate(ann, ref='G', pos=4)
    assert v.INFO['Degeneracy'] == 0


def test_degeneracy_minus_strand_site():
    # on the minus strand the codon is read complemented from the CDS end; at pos 28 the codon is
    # 'TTA' (Leu) and the variant sits at its 3rd position -> 2-fold degenerate
    ann = _make_ann(strand='-')
    v = _annotate(ann, ref='T', pos=28)
    assert v.INFO['Degeneracy'] == 2
    assert v.INFO['Degeneracy_Info'] == '2,-,TTA'


def test_degeneracy_reference_mismatch_recorded():
    # ref allele 'A' does not match the contig base 'T' at pos 6 -> recorded as a mismatch, not annotated
    ann = _make_ann()
    v = _annotate(ann, ref='A', pos=6)
    assert v in ann.mismatches
    assert v.INFO['Degeneracy'] == '.'
    assert ann.n_annotated == 0


def test_degeneracy_skips_indel():
    # multi-base REF (indel) is skipped
    ann = _make_ann()
    _annotate(ann, ref='AT', pos=6)
    assert ann.n_skipped == 1


def test_degeneracy_skips_on_missing_cds():
    # a LookupError from the CDS fetch (no overlapping CDS) skips the site
    ann = _make_ann()
    ann._fetch = lambda v: (_ for _ in ()).throw(LookupError('no cds'))
    v = _annotate(ann, ref='T', pos=6)
    assert ann.n_skipped == 1
    assert v.INFO['Degeneracy'] == '.'


@pytest.mark.parametrize('pos, degeneracy', [(19, 2), (20, 0), (21, 4)])
def test_degeneracy_of_each_codon_position(pos, degeneracy):
    # Arg CGG at pos 19-21: AGG is also Arg (2-fold), no change at the 2nd position keeps Arg (0-fold),
    # every change at the 3rd does (4-fold)
    v = _annotate(_make_ann(), ref=CONTIG[pos - 1], pos=pos)
    assert v.INFO['Degeneracy'] == degeneracy
    assert v.INFO['Degeneracy_Info'] == f'{pos - 19},+,CGG'


def test_degeneracy_twofold_third_position():
    # Lys AAA at pos 25-27: only AAG keeps Lys
    v = _annotate(_make_ann(), ref='A', pos=27)
    assert v.INFO['Degeneracy'] == 2
    assert v.INFO['Degeneracy_Info'] == '2,+,AAA'


def test_degeneracy_phase_shifts_reading_frame():
    # phase 1 starts the first codon at pos 2, so pos 4 is the 3rd position of Trp TGG -> 0-fold
    v = _annotate(_make_ann(phase=1), ref='G', pos=4)
    assert v.INFO['Degeneracy'] == 0
    assert v.INFO['Degeneracy_Info'] == '2,+,TGG'


def test_degeneracy_codon_spliced_from_previous_cds():
    # the CDS at 11-30 has phase 2, so its first two bases T, C complete a codon whose first base is
    # the last base T (pos 2) of the previous CDS: Phe TTC, 2-fold at its 3rd position
    v = _annotate(_make_ann(start=11, phase=2, prev=(1, 2)), ref='C', pos=12)
    assert v.INFO['Degeneracy'] == 2
    assert v.INFO['Degeneracy_Info'] == '2,+,TTC'


def test_degeneracy_minus_strand_codon_spliced_from_previous_cds():
    # the minus-strand CDS at 11-30 has phase 1, so the codon starting at its lowest base (pos 11)
    # continues with pos 5 and 4 of the previous CDS: bases T, T, G complemented give Asn AAC
    v = _annotate(_make_ann(strand='-', start=11, phase=1, prev=(1, 5)), ref='T', pos=11)
    assert v.INFO['Degeneracy'] == 0
    assert v.INFO['Degeneracy_Info'] == '0,-,AAC'


@pytest.mark.parametrize('kwargs, pos', [
    (dict(start=11, phase=2), 12),  # codon starts before the CDS
    (dict(strand='-', start=11, phase=1), 11),  # minus-strand codon ends before the CDS
    (dict(strand='-', end=20, phase=1), 20),  # minus-strand codon starts after the CDS
    (dict(end=20), 20),  # codon ends after the CDS
])
def test_degeneracy_codon_overlapping_cds_boundary_without_neighbour_is_an_error(kwargs, pos):
    ann = _make_ann(**kwargs)
    v = _annotate(ann, ref=CONTIG[pos - 1], pos=pos)
    assert ann.errors == [v]
    assert v.INFO['Degeneracy'] == '.'
    assert ann.n_annotated == 0


def test_degeneracy_codon_with_unknown_base_not_counted():
    # an N in the codon leaves the degeneracy undetermined
    ann = _make_ann(contig='ATGGNTGTA')
    v = _annotate(ann, ref='T', pos=6)
    assert v.INFO['Degeneracy'] == '.'
    assert v.INFO['Degeneracy_Info'] == '2,+,GNT'
    assert ann.n_annotated == 0


def test_degeneracy_site_outside_cds_not_annotated():
    ann = _make_ann(start=4)
    v = _annotate(ann, ref='T', pos=2)
    assert v.INFO['Degeneracy'] == '.'
    assert 'Degeneracy_Info' not in v.INFO
    assert ann.n_annotated == 0


# --------------------------------------------------------------------------- synonymy annotation

def _make_syn(**kwargs):
    return _make(SynonymyAnnotation, **kwargs)


def _snp(ref, alt, pos, **info):
    v = DummyVariant(ref=ref, pos=pos, chrom='chr1')
    v.is_snp = True
    v.ALT = [alt]
    v.INFO.update(info)
    return v


def test_synonymy_synonymous_change():
    # GTT -> GTC keeps the amino acid Valine -> synonymous (Synonymy == 1)
    ann = _make_syn()
    v = _snp('T', 'C', 6)
    ann.annotate_site(v)
    assert v.INFO['Synonymy'] == 1
    assert v.INFO['Synonymy_Info'] == 'GTT/GTC'


def test_synonymy_nonsynonymous_change():
    # GTT -> ATT changes Valine to Isoleucine -> non-synonymous (Synonymy == 0)
    ann = _make_syn()
    v = _snp('G', 'A', 4)
    ann.annotate_site(v)
    assert v.INFO['Synonymy'] == 0
    assert v.INFO['Synonymy_Info'] == 'GTT/ATT'


def test_synonymy_skips_non_snp():
    # a non-SNP (mono-allelic) site is not annotated for synonymy
    ann = _make_syn()
    v = DummyVariant(ref='T', pos=6, chrom='chr1')  # DummyVariant.is_snp is False
    ann.annotate_site(v)
    assert v.INFO['Synonymy'] == '.'


@pytest.mark.parametrize('ref, alt, pos, synonymy, info', [
    ('G', 'A', 4, 1, 'AAC/AAT'),  # Asn -> Asn
    ('T', 'C', 6, 0, 'AAC/GAC'),  # Asn -> Asp
])
def test_synonymy_minus_strand(ref, alt, pos, synonymy, info):
    # on the minus strand pos 4-6 (GTT) are read as Asn AAC, pos 6 being its 1st position
    v = _snp(ref, alt, pos)
    _make_syn(strand='-').annotate_site(v)
    assert v.INFO['Synonymy'] == synonymy
    assert v.INFO['Synonymy_Info'] == info


def test_synonymy_stop_gained():
    # Lys AAA -> stop TAA
    v = _snp('A', 'T', 25)
    _make_syn().annotate_site(v)
    assert v.INFO['Synonymy'] == 0
    assert v.INFO['Synonymy_Info'] == 'AAA/TAA,stop_gained'


def test_synonymy_start_gained():
    # Val GTG -> Met ATG
    v = _snp('G', 'A', 13)
    _make_syn().annotate_site(v)
    assert v.INFO['Synonymy'] == 0
    assert v.INFO['Synonymy_Info'] == 'GTG/ATG,start_gained'


def test_synonymy_reference_mismatch_recorded():
    ann = _make_syn()
    v = _snp('A', 'C', 6)
    ann.annotate_site(v)
    assert ann.mismatches == [v]
    assert v.INFO['Synonymy'] == '.'
    assert ann.n_annotated == 0


def test_synonymy_codon_overlapping_cds_boundary_without_neighbour_is_an_error():
    ann = _make_syn(end=20)
    v = _snp('G', 'A', 20)
    ann.annotate_site(v)
    assert ann.errors == [v]
    assert v.INFO['Synonymy'] == '.'


@pytest.mark.parametrize('strand, csq', [
    ('+', 'synonymous_variant|gtT/gtC'),
    ('-', 'missense_variant|Aac/Gac'),  # VEP reports the codons on the transcript strand
])
def test_synonymy_agrees_with_vep_codons(strand, csq):
    ann = _make_syn(strand=strand)
    v = _snp('T', 'C', 6, CSQ=csq)
    ann.annotate_site(v)
    assert ann.n_vep_comparisons == 1
    assert ann.vep_mismatches == []
    assert v.INFO['Synonymy'] == (1 if strand == '+' else 0)


def test_synonymy_vep_codon_mismatch_recorded():
    # VEP reports Gtt/Att where the site gives GTT/GTC, so the site is recorded and left unannotated
    ann = _make_syn()
    v = _snp('T', 'C', 6, CSQ='missense_variant|Gtt/Att')
    ann.annotate_site(v)
    assert ann.n_vep_comparisons == 1
    assert ann.vep_mismatches == [v]
    assert v.INFO['Synonymy'] == '.'
    assert ann.n_annotated == 0


def test_synonymy_vep_consequence_without_codons_not_compared():
    ann = _make_syn()
    v = _snp('T', 'C', 6, CSQ='intron_variant|')
    ann.annotate_site(v)
    assert ann.n_vep_comparisons == 0
    assert v.INFO['Synonymy'] == 1


@pytest.mark.parametrize('ref, alt, pos, ann_tag, synonymy', [
    ('T', 'C', 6, 'C|synonymous_variant|LOW', 1),
    ('G', 'A', 4, 'A|missense_variant|MODERATE', 0),
    ('T', 'C', 6, 'C|intron_variant|MODIFIER', 1),  # neither synonymous nor missense, nothing to contradict
])
def test_synonymy_agrees_with_snpeff(ref, alt, pos, ann_tag, synonymy):
    ann = _make_syn()
    v = _snp(ref, alt, pos, ANN=ann_tag)
    ann.annotate_site(v)
    assert ann.n_snpeff_comparisons == 1
    assert ann.snpeff_mismatches == []
    assert v.INFO['Synonymy'] == synonymy


@pytest.mark.parametrize('ref, alt, pos, ann_tag', [
    ('T', 'C', 6, 'C|missense_variant|MODERATE'),  # synonymous GTT/GTC
    ('G', 'A', 4, 'A|synonymous_variant|LOW'),  # non-synonymous GTT/ATT
])
def test_synonymy_snpeff_mismatch_recorded(ref, alt, pos, ann_tag):
    ann = _make_syn()
    v = _snp(ref, alt, pos, ANN=ann_tag)
    ann.annotate_site(v)
    assert ann.n_snpeff_comparisons == 1
    assert ann.snpeff_mismatches == [v]
    assert v.INFO['Synonymy'] == '.'
    assert ann.n_annotated == 0
