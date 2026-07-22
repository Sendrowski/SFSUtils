"""
Regression tests for the input/output defects found by the tenth release-readiness scan: a GFF3 whose
appended sequences make up a whole block of the chunked read, the two VCF-Zarr sentinels conflated into
one when a multi-valued INFO field is read back, the optional-backend install hints naming another
project, the allele counts of a site recomputed for every consumer of it, the allele re-indexing running
on every site of a padded store, and the INFO backfill of a late field allocating the whole input.
Alongside them the per-contig lengths the readers now surface.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2026-07-22"

import inspect

import numpy as np
import pytest

import sfsutils.io_handlers as io_handlers
from sfsutils.io_handlers import (GFFHandler, SiteAlleles, TskitVariantReader, Variant, VCFHandler,
                                  ZarrVariantReader, ZarrVariantWriter)

HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length=1000>\n"
    "##contig=<ID=2,length=2500>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
)


def _write_vcf(path, rows):
    """
    Write a minimal VCF holding the given data rows.

    :param path: The path to write to.
    :param rows: The data rows, each a tab-separated string.
    :return: The path as a string.
    """
    with open(path, 'w') as f:
        f.write(HEADER + ''.join(row + '\n' for row in rows))

    return str(path)


def _snp(pos, info=None, ref='A', alt=('T',), gt_bases=('A|T',)):
    """
    A single-nucleotide variant carrying the given INFO.

    :param pos: The position.
    :param info: The INFO fields.
    :param ref: The reference allele.
    :param alt: The alternate alleles.
    :param gt_bases: The genotypes.
    :return: The variant.
    """
    return Variant(ref=ref, pos=pos, chrom='1', gt_bases=list(gt_bases), alt=list(alt), is_snp=True,
                   info=dict(info or {}))


def _store(path, variants, samples=('s1',), seqnames=('1',)):
    """
    Write the given variants to a VCF-Zarr store.

    :param path: The path to write to.
    :param variants: The variants.
    :param samples: The sample names.
    :param seqnames: The contig names.
    :return: The path as a string.
    """
    writer = ZarrVariantWriter(str(path), samples=list(samples), seqnames=list(seqnames))

    for variant in variants:
        writer.write(variant)

    writer.close()

    return str(path)


def _with_array(store, name, data, dtype, dimensions):
    """
    Add an array to a store already written.

    :param store: The path to the store.
    :param name: The array name.
    :param data: The data.
    :param dtype: The dtype.
    :param dimensions: The names of its axes.
    :return: The path to the store.
    """
    import zarr

    root = zarr.open(store, mode='r+')
    array = root.create_array(name, shape=np.shape(data), dtype=dtype)
    array[:] = np.asarray(data, dtype=dtype)
    array.attrs['_ARRAY_DIMENSIONS'] = list(dimensions)

    return store


# --- 3. the appended sequences of a GFF3 -------------------------------------------------------------

def _gff_with_fasta(path, n_cds, n_sequence):
    """
    A GFF3 carrying the given number of coding sequences followed by an appended FASTA section.

    :param path: The path to write to.
    :param n_cds: The number of coding sequences.
    :param n_sequence: The number of sequence lines.
    :return: The path as a string.
    """
    with open(path, 'w') as f:
        f.write('##gff-version 3\n')

        for i in range(n_cds):
            start = 10 + i * 100
            f.write(f'c1\tx\tCDS\t{start}\t{start + 59}\t.\t+\t0\tID=cds{i};Parent=t{i}\n')

        f.write('##FASTA\n>c1\n')
        f.write(''.join('ACGT' * 15 + '\n' for _ in range(n_sequence)))

    return str(path)


def test_gff_with_an_appended_fasta_longer_than_a_block_loads(tmp_path, monkeypatch):
    """The sequences a GFF3 appends after ``##FASTA`` carry one field each, so past a certain length they
    make up a whole block of the chunked read. The read must end at the pragma rather than resolve the
    nine columns of the format against a block of sequence."""
    monkeypatch.setattr(GFFHandler, '_gff_block', 10)

    cds = GFFHandler(_gff_with_fasta(tmp_path / 'fasta.gff3', n_cds=15, n_sequence=40))._load_cds()

    assert len(cds) == 15
    assert list(cds['seqid'].unique()) == ['c1']

    # a sequence line must not read back as a contig of its own
    assert list(cds['seqid'].dtype.categories) == ['c1']


def test_gff_with_an_appended_fasta_counts_target_sites(tmp_path, monkeypatch):
    """The consequence for the caller: TargetSiteCounter and DegeneracyAnnotation reach the annotation of
    such a file at all."""
    monkeypatch.setattr(GFFHandler, '_gff_block', 10)

    handler = GFFHandler(_gff_with_fasta(tmp_path / 'fasta.gff3', n_cds=15, n_sequence=40))

    assert handler._count_target_sites() == {'c1': 15 * 60}


def test_a_gff_of_nothing_but_sequences_carries_no_coding_sequence(tmp_path):
    """The pragma on the first line leaves no annotation at all rather than an aborted read."""
    path = str(tmp_path / 'only.gff3')
    with open(path, 'w') as f:
        f.write('##FASTA\n>c1\nACGT\n')

    assert len(GFFHandler(path)._load_cds()) == 0


@pytest.mark.parametrize('reads', [1, 3, 7, 4096])
def test_the_annotation_records_are_handed_out_whole(tmp_path, reads):
    """The pragma must be found however the reads fall across it, and the last record of a file not
    ending in a newline is a record all the same."""
    path = str(tmp_path / 'lines.gff3')
    text = '##gff-version 3\nc1\tx\tCDS\t1\t9\t.\t+\t0\tID=a\nc1\tx\tCDS\t20\t29\t.\t+\t0\tID=b'
    with open(path, 'w') as f:
        f.write(text + '\n##FASTA\n>c1\nACGTACGT\n')

    with io_handlers._GFFAnnotationLines(path) as lines:
        read = ''.join(iter(lambda: lines.read(reads), ''))

    assert read == text + '\n'


def test_gff_read_in_blocks_matches_a_single_block(tmp_path, monkeypatch):
    """The coding sequences do not depend on where the blocks fall."""
    path = _gff_with_fasta(tmp_path / 'fasta.gff3', n_cds=25, n_sequence=40)

    monkeypatch.setattr(GFFHandler, '_gff_block', 3)
    blocked = GFFHandler(path)._load_cds()

    monkeypatch.setattr(GFFHandler, '_gff_block', 10 ** 6)
    single = GFFHandler(path)._load_cds()

    assert blocked.reset_index(drop=True).equals(single.reset_index(drop=True))


# --- 4. the fill and the missing sentinel of a multi-valued INFO field --------------------------------

def test_missing_integer_element_keeps_its_position(tmp_path):
    """``AC=.,2`` is stored as the missing sentinel followed by the count of the second allele. Dropping
    the missing element would book the count of allele 2 against allele 1."""
    store = _with_array(_store(tmp_path / 'ac.vcz', [_snp(pos=10), _snp(pos=20)]),
                        'variant_AC', [[3, -2], [-1, 2]], 'int32', ['variants', 'values'])

    assert [variant.INFO['AC'] for variant in ZarrVariantReader(store)] == [3, (None, 2)]


def test_missing_float_element_keeps_its_position(tmp_path):
    """A float array marks fill and missing by two NaN payloads, which a plain ``isnan`` cannot tell
    apart."""
    # the payload the spec gives each marker, alongside 0.5 and 0.25
    data = np.array([[0x3F000000, 0x7F800002], [0x7F800001, 0x3E800000]], dtype=np.uint32).view(np.float32)

    store = _with_array(_store(tmp_path / 'af.vcz', [_snp(pos=10), _snp(pos=20)]),
                        'variant_AF', data, 'float32', ['variants', 'values'])

    assert [variant.INFO['AF'] for variant in ZarrVariantReader(store)] == [0.5, (None, 0.25)]


def test_missing_string_element_reads_back_as_the_vcf_spells_it(tmp_path):
    """A string field is handed out as the comma-separated string cyvcf2 gives, a missing element
    included."""
    store = _with_array(_store(tmp_path / 'tag.vcz', [_snp(pos=10), _snp(pos=20)]),
                        'variant_TAG', [['x', '', ''], ['x', '.', 'y']], '<U8', ['variants', 'values'])

    assert [variant.INFO['TAG'] for variant in ZarrVariantReader(store)] == ['x', 'x,.,y']


def test_missing_element_between_two_values_survives(tmp_path):
    """``DP4=1,.,3,4`` keeps its four positions, so the counts stay with the allele they belong to."""
    store = _with_array(_store(tmp_path / 'dp4.vcz', [_snp(pos=10)]),
                        'variant_DP4', [[1, -1, 3, 4]], 'int32', ['variants', 'values'])

    assert [variant.INFO['DP4'] for variant in ZarrVariantReader(store)] == [(1, None, 3, 4)]


def test_a_row_of_nothing_but_markers_carries_no_field(tmp_path):
    """A site carrying no value at all still reports no field rather than a tuple of ``None``."""
    store = _with_array(_store(tmp_path / 'none.vcz', [_snp(pos=10)]),
                        'variant_AC', [[-1, -2]], 'int32', ['variants', 'values'])

    assert [variant.INFO for variant in ZarrVariantReader(store)] == [{}]


def test_multivalued_info_matches_cyvcf2(tmp_path):
    """The reference: the same records read through cyvcf2, which keeps the two apart."""
    cyvcf2 = pytest.importorskip('cyvcf2')
    bio2zarr = pytest.importorskip('bio2zarr.vcf')

    header = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=1000>\n"
        '##INFO=<ID=AC,Number=A,Type=Integer,Description="ac">\n'
        '##INFO=<ID=AF,Number=A,Type=Float,Description="af">\n'
        '##INFO=<ID=DP4,Number=4,Type=Integer,Description="dp4">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
    )

    vcf = str(tmp_path / 'multi.vcf')
    with open(vcf, 'w') as f:
        f.write(header)
        f.write("1\t10\t.\tA\tT\t.\tPASS\tAC=3;AF=0.5;DP4=1,2,3,4\tGT\t0/1\n")
        f.write("1\t20\t.\tC\tG,T\t.\tPASS\tAC=.,2;AF=.,0.25;DP4=1,.,3,4\tGT\t0/1\n")

    store = str(tmp_path / 'multi.vcz')
    bio2zarr.convert([vcf], store)

    expected = [{key: value for key, value in variant.INFO} for variant in cyvcf2.VCF(vcf)]
    read = [variant.INFO for variant in ZarrVariantReader(store)]

    for want, got in zip(expected, read):
        assert {key: got[key] for key in want} == want


def test_a_one_dimensional_field_still_reads_both_markers_as_absent(tmp_path):
    """A field of one value per site carries no padding, so both markers mean the site has no value."""
    store = _with_array(_store(tmp_path / 'dp.vcz', [_snp(pos=10), _snp(pos=20), _snp(pos=30)]),
                        'variant_DP', [3, -1, -2], 'int32', ['variants'])

    assert [variant.INFO for variant in ZarrVariantReader(store)] == [{'DP': 3}, {}, {}]


# --- 5. the optional-backend install hints -----------------------------------------------------------

def test_install_hints_name_the_distribution():
    """``sfsutils`` on PyPI is an unrelated project, so an install hint naming it installs the wrong
    package and shadows the import name."""
    source = inspect.getsource(io_handlers)

    assert 'pip install sfsutils[' not in source

    for extra in ('vcf', 'zarr', 'arg'):
        assert f'pip install \\"sfsutils-popgen[{extra}]\\"' in source


# --- 19. the allele counts of a site -----------------------------------------------------------------

def _counting_view(monkeypatch, indices, alleles):
    """
    A view of a site alongside the number of times it walks its index array.

    :param monkeypatch: The monkeypatch fixture.
    :param indices: The per-haplotype allele indices.
    :param alleles: The allele strings.
    :return: The view and a one-element list holding the count.
    """
    walks = [0]
    count = SiteAlleles._count

    def counted(self, mask):
        walks[0] += 1

        return count(self, mask)

    monkeypatch.setattr(SiteAlleles, '_count', counted)

    return SiteAlleles(np.asarray(indices), alleles), walks


def test_the_counts_of_a_site_are_computed_once_per_mask(monkeypatch):
    """Every filtration and the parser itself ask the same view for the counts under the same mask, which
    is where the bulk of an unstratified parse was spent."""
    view, walks = _counting_view(monkeypatch, [[0, 1], [1, 1], [0, 0]], ['A', 'T'])

    assert view.n_called(None) == 6
    assert view.counts(None) == {'A': 3, 'T': 3}
    assert view.distinct(None) == {'A', 'T'}
    assert walks[0] == 1


def test_the_counts_of_a_site_follow_the_mask(monkeypatch):
    """A second mask is counted on its own rather than served the counts of the first."""
    mask = np.array([True, False, False])
    view, walks = _counting_view(monkeypatch, [[0, 1], [1, 1], [0, 0]], ['A', 'T'])

    assert view.counts(None) == {'A': 3, 'T': 3}
    assert view.counts(mask) == {'A': 1, 'T': 1}
    assert view.counts(mask) == {'A': 1, 'T': 1}
    assert view.counts(None) == {'A': 3, 'T': 3}
    assert walks[0] == 3


def test_the_counts_of_two_sites_are_kept_apart(monkeypatch):
    """The counts are held on the view rather than on the class, so one site does not answer for another."""
    first, _ = _counting_view(monkeypatch, [[0, 0], [0, 0]], ['A', 'T'])
    second, _ = _counting_view(monkeypatch, [[1, 1], [1, 1]], ['A', 'T'])

    assert first.counts(None) == {'A': 4}
    assert second.counts(None) == {'T': 4}
    assert first.counts(None) == {'A': 4}


# --- 20. the allele re-indexing of a padded store ----------------------------------------------------

def test_trailing_padding_leaves_the_calls_untouched():
    """A store pads every site to the widest allele count it holds anywhere, and that padding is
    referenced by no call, so dropping it is the whole of the work."""
    rows = np.array([[0, 1], [1, 1]])

    alleles, remapped = ZarrVariantReader._alleles(['A', 'C', ''], rows)

    assert alleles == ['A', 'C']
    assert remapped is rows


def test_an_interior_empty_allele_is_still_re_indexed():
    """A tree sequence writes a mutation to the empty allele, which sits among the alleles rather than
    past them and does shift the numbering."""
    rows = np.array([[0, 1], [2, -1]])

    alleles, remapped = ZarrVariantReader._alleles(['A', '', 'C'], rows)

    assert alleles == ['A', 'C']
    assert remapped.tolist() == [[0, -1], [1, -1]]


def test_a_padded_store_parses_as_the_same_data_unpadded(tmp_path):
    """The whole point: a store carrying one multi-allelic site reads back the same as one without it."""
    variants = [_snp(pos=10 * (i + 1), ref='A', alt=['C'], gt_bases=['A|C']) for i in range(6)]
    padded = _store(tmp_path / 'padded.vcz', variants + [_snp(pos=100, ref='A', alt=['C', 'G'],
                                                              gt_bases=['A|C'])])
    plain = _store(tmp_path / 'plain.vcz', variants)

    import zarr

    assert zarr.open(padded, mode='r')['variant_allele'].shape[1] == 3
    assert zarr.open(plain, mode='r')['variant_allele'].shape[1] == 2

    read = [(variant.REF, variant.ALT) for variant in ZarrVariantReader(padded)]

    assert read[:6] == [(variant.REF, variant.ALT) for variant in ZarrVariantReader(plain)]
    assert read[6] == ('A', ['C', 'G'])


# --- 21. the INFO backfill of a field first appearing late -------------------------------------------

def test_a_late_info_field_is_backfilled_a_chunk_at_a_time(tmp_path, monkeypatch):
    """The class holds one chunk of the input rather than the whole of it, which a backfill encoding one
    array over every variant already written breaks."""
    monkeypatch.setattr(ZarrVariantWriter, '_variant_chunk', 8)

    widest = [0]
    encode = ZarrVariantWriter._encode

    def encoded(self, values, kind):
        widest[0] = max(widest[0], len(values))

        return encode(self, values, kind)

    monkeypatch.setattr(ZarrVariantWriter, '_encode', encoded)

    store = _store(tmp_path / 'late.vcz',
                   [_snp(pos=i + 1, info={'LATE': 'x'} if i >= 40 else {}) for i in range(48)])

    assert widest[0] == 8

    read = [variant.INFO.get('LATE') for variant in ZarrVariantReader(store)]

    assert read == [None] * 40 + ['x'] * 8


def test_a_late_info_field_is_written_out_in_full(tmp_path, monkeypatch):
    """The values themselves are unaffected by how the backfill is cut into blocks."""
    monkeypatch.setattr(ZarrVariantWriter, '_variant_chunk', 5)

    store = _store(tmp_path / 'late.vcz',
                   [_snp(pos=i + 1, info={'LATE': i} if i >= 12 else {}) for i in range(17)])

    assert [variant.INFO.get('LATE') for variant in ZarrVariantReader(store)] == \
           [None] * 12 + [12, 13, 14, 15, 16]


# --- 13. the per-contig lengths the readers surface --------------------------------------------------

def test_vcf_contig_lengths_come_from_the_header(tmp_path):
    """The observed variants span a fraction of a sparsely covered contig, where the header declares the
    whole of it."""
    pytest.importorskip('cyvcf2')

    handler = VCFHandler(_write_vcf(tmp_path / 'lengths.vcf',
                                    ["1\t10\t.\tA\tT\t.\tPASS\t.\tGT\t0|1"]))

    assert handler.contig_lengths == {'1': 1000, '2': 2500}


def test_a_vcf_declaring_no_lengths_has_none(tmp_path):
    """A header without a length is no length at all rather than a length of zero."""
    pytest.importorskip('cyvcf2')

    path = str(tmp_path / 'nolengths.vcf')
    with open(path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n##contig=<ID=1>\n"
                '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
                "1\t10\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\n")

    assert VCFHandler(path).contig_lengths is None


def test_zarr_contig_lengths_come_from_the_store(tmp_path):
    """The VCF-Zarr layout carries the lengths in a ``contig_length`` array of its own."""
    store = _store(tmp_path / 'lengths.vcz', [_snp(pos=250)])

    assert ZarrVariantReader(store).contig_lengths == {'1': 250}
    assert VCFHandler(store).contig_lengths == {'1': 250}


def test_tskit_contig_length_is_the_genome_length():
    """A tree sequence knows the region its sites are distributed over exactly."""
    tskit = pytest.importorskip('tskit')

    reader = TskitVariantReader(tskit.load('resources/msprime/two_epoch.trees'))

    assert reader.contig_lengths == {'1': int(reader.sequence_length)}
    assert VCFHandler('resources/msprime/two_epoch.trees').contig_lengths == reader.contig_lengths
