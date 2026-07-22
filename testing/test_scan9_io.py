"""
The VCF-Zarr backends against the store layouts the converters actually write: a multi-valued INFO
field spread over a second axis, an integer field carrying the missing sentinel, an allele array with
an empty allele in the middle, and a site with more alleles than a signed byte holds. Alongside them,
the two streaming paths that must not depend on how the input is cut into chunks: the writer's INFO
arrays and the GFF reader.
"""
import numpy as np
import pytest

from sfsutils.io_handlers import (GFFHandler, SiteAlleles, Variant, ZarrVariantReader,
                                  ZarrVariantWriter, get_called_bases)


def _store(path, variants, samples=('s1',), seqnames=('1',)):
    """Write the given variants to a store and return its path."""
    writer = ZarrVariantWriter(str(path), samples=list(samples), seqnames=list(seqnames))

    for variant in variants:
        writer.write(variant)

    writer.close()

    return str(path)


def _snp(pos, info=None, ref='A', alt=('T',), gt_bases=('A|T',)):
    """A single-nucleotide variant carrying the given INFO."""
    return Variant(ref=ref, pos=pos, chrom='1', gt_bases=list(gt_bases), alt=list(alt), is_snp=True,
                   info=dict(info or {}))


# --- multi-valued INFO fields (C1) ------------------------------------------------------------------

def _multivalued(path, rows, key='CSQ', dtype='<U32'):
    """A store whose variant_<key> is the given 2-D array, as vcf2zarr writes a Number != 1 field."""
    import zarr

    store = _store(path, [_snp(pos=10 * (i + 1), info={'AA': 'A'}) for i in range(len(rows))])
    root = zarr.open(store, mode='r+')

    array = root.create_array(f'variant_{key}', shape=np.shape(rows), dtype=dtype)
    array[:] = np.asarray(rows, dtype=dtype)
    array.attrs['_ARRAY_DIMENSIONS'] = ['variants', 'values']

    return store


def test_multivalued_string_info_is_joined_as_the_vcf_spells_it(tmp_path):
    """A VEP CSQ of several transcripts is stored one transcript per column, padded to the widest site,
    and reads back as the comma-separated string cyvcf2 hands out."""
    store = _multivalued(tmp_path / 'csq.vcz',
                         [['T|synonymous_variant|g1', 'T|intron_variant|g2'],
                          ['G|missense_variant|g1', '']])

    info = [variant.INFO['CSQ'] for variant in ZarrVariantReader(store)]

    assert info == ['T|synonymous_variant|g1,T|intron_variant|g2', 'G|missense_variant|g1']


def test_multivalued_numeric_info_drops_the_padding(tmp_path):
    """A numeric field of Number != 1 surfaces as the tuple cyvcf2 hands out, or as the scalar where the
    site carries a single value; the sentinels padding the shorter sites are not values."""
    store = _multivalued(tmp_path / 'ac.vcz', [[3, 5], [7, -2]], key='AC', dtype='int32')

    assert [variant.INFO['AC'] for variant in ZarrVariantReader(store)] == [(3, 5), 7]


def test_multivalued_info_stratifies_a_spectrum(tmp_path):
    """The headline consequence: a store carrying a multi-transcript CSQ stratifies into a neutral and a
    selected spectrum rather than raising at every site."""
    from sfsutils import Parser
    from sfsutils.parser import VEPStratification
    from sfsutils.settings import Settings

    Settings.disable_pbar = True

    store = _multivalued(tmp_path / 'strat.vcz',
                         [['T|synonymous_variant|g1', 'T|intron_variant|g2'],
                          ['G|missense_variant|g1', '']])

    spectra = Parser(source=store, n=2, stratifications=[VEPStratification()],
                     skip_non_polarized=False).parse()

    assert np.asarray(spectra['neutral'].data).sum() == 1
    assert np.asarray(spectra['selected'].data).sum() == 1


# --- the integer missing sentinel (C2) --------------------------------------------------------------

def test_integer_sentinel_reads_as_an_absent_field(tmp_path):
    """An integer INFO field a site does not carry is stored as the -1 sentinel by every converter, and
    must read back as no field at all, as cyvcf2 reports it."""
    import zarr

    store = _store(tmp_path / 'dp.vcz', [_snp(pos=10), _snp(pos=20), _snp(pos=30)])
    root = zarr.open(store, mode='r+')

    array = root.create_array('variant_DP', shape=(3,), dtype='int8')
    array[:] = [3, -1, -2]
    array.attrs['_ARRAY_DIMENSIONS'] = ['variants']

    assert [variant.INFO for variant in ZarrVariantReader(store)] == [{'DP': 3}, {}, {}]


def test_a_negative_integer_value_survives_the_store(tmp_path):
    """A field whose value is genuinely -1 is written through the numeric encoding that has a missing
    marker of its own, so it does not read back as an absent field."""
    store = _store(tmp_path / 'neg.vcz', [_snp(pos=10, info={'Score': -1}), _snp(pos=20, info={'Score': 4})])

    assert [variant.INFO['Score'] for variant in ZarrVariantReader(store)] == [-1, 4]


def test_a_partly_missing_integer_field_is_absent_where_it_is_missing(tmp_path):
    """A field only some sites carry round-trips as those sites' values alone, so a probability read off
    it is never the sentinel."""
    store = _store(tmp_path / 'part.vcz', [_snp(pos=10, info={'AA_prob': 1}), _snp(pos=20),
                                           _snp(pos=30, info={'AA_prob': 0})])

    assert [variant.INFO.get('AA_prob') for variant in ZarrVariantReader(store)] == [1, None, 0]


# --- the allele index range (C3) --------------------------------------------------------------------

def test_a_site_of_many_alleles_keeps_every_haplotype(tmp_path):
    """A site with as many alleles as a signed byte holds is stored with byte-wide calls, whose highest
    index must not wrap round when the view shifts the sentinels out of the way."""
    alleles = ['A', 'C'] + [f'AC{i}' for i in range(2, 126)] + ['T']
    calls = np.array([[0, 1], [0, 126], [126, 126]], dtype=np.int64)

    store = _store(tmp_path / 'many.vcz',
                   [Variant(ref=alleles[0], pos=10, chrom='1', alt=alleles[1:], is_snp=False,
                            allele_indices=calls)],
                   samples=('s1', 's2', 's3'))

    variant = next(iter(ZarrVariantReader(store)))
    site = SiteAlleles.from_site(variant)

    assert np.asarray(variant.allele_indices).dtype == np.int8
    assert site.counts() == {'A': 2, 'C': 1, 'T': 3}
    assert site.n_called() == 6
    assert dict(zip(*np.unique(get_called_bases(variant.gt_bases), return_counts=True))) == \
           {'A': 2, 'C': 1, 'T': 3}


# --- empty alleles (C4) -----------------------------------------------------------------------------

def test_an_empty_allele_is_dropped_and_the_calls_re_indexed(tmp_path):
    """A tree sequence mutating to the empty allele reaches a store as an allele array with a hole in
    it. The hole is not an alternate allele, and the calls past it belong to the alleles that remain."""
    import zarr

    store = _store(tmp_path / 'hole.vcz',
                   [Variant(ref='A', pos=10, chrom='1', alt=['x', 'T'], is_snp=False,
                            allele_indices=np.array([[2, 0]], dtype=np.int64))])

    root = zarr.open(store, mode='r+')
    root['variant_allele'][:] = np.array([['A', '', 'T']], dtype=object)

    variant = next(iter(ZarrVariantReader(store)))

    assert variant.REF == 'A'
    assert variant.ALT == ['T']
    assert variant.is_snp
    assert list(np.asarray(variant.allele_indices).ravel()) == [1, 0]
    assert SiteAlleles.from_site(variant).counts() == {'A': 1, 'T': 1}


def test_the_padding_of_a_narrower_site_is_still_dropped(tmp_path):
    """The padding a site with fewer alleles than the widest carries is not an allele either, and the
    calls of the sites around it keep pointing at the alleles they did."""
    store = _store(tmp_path / 'pad.vcz',
                   [_snp(pos=10, alt=('T', 'G'), gt_bases=('T|G',)), _snp(pos=20)])

    variants = list(ZarrVariantReader(store))

    assert [variant.ALT for variant in variants] == [['T', 'G'], ['T']]
    assert [SiteAlleles.from_site(variant).counts() for variant in variants] == [{'T': 1, 'G': 1},
                                                                                {'A': 1, 'T': 1}]


# --- the read batch follows the store (P2) ----------------------------------------------------------

def test_the_read_batch_follows_the_stores_own_chunking(tmp_path):
    """A batch that straddles the stored chunks fetches each of them once per batch overlapping it, so
    the default batch is the store's own chunk length."""
    import zarr

    store = _store(tmp_path / 'grid.vcz', [_snp(pos=10 * (i + 1)) for i in range(25)])
    chunk = zarr.open(store, mode='r')['call_genotype'].chunks[0]

    assert ZarrVariantReader(store)._chunk_size == chunk
    assert ZarrVariantReader(store, chunk_size=7)._chunk_size == 7


def test_the_read_batch_does_not_change_what_is_read(tmp_path):
    """Reading in batches of any length yields the same variants."""
    store = _store(tmp_path / 'batches.vcz', [_snp(pos=10 * (i + 1), info={'DP': i}) for i in range(25)])

    def read(chunk_size):
        return [(v.POS, v.REF, v.ALT, v.INFO, np.asarray(v.allele_indices).tolist())
                for v in ZarrVariantReader(store, chunk_size=chunk_size)]

    assert read(None) == read(1) == read(7) == read(1000)


# --- the streamed INFO arrays (P3) ------------------------------------------------------------------

class SmallChunkWriter(ZarrVariantWriter):
    """A writer whose chunks hold four variants, so a handful of them still stream."""

    _variant_chunk = 4


def _contents(store):
    """Every array of a store as a list, so two stores can be compared whole. A float array is compared
    through its bits, so that the missing sentinel is one value among others rather than an unequal
    NaN."""
    import zarr

    root = zarr.open(store, mode='r')

    def values(array):
        data = np.asarray(array[...])

        return data.view(np.uint64).tolist() if data.dtype.kind == 'f' else data.tolist()

    return {name: values(root[name]) for name in sorted(root.array_keys())}


@pytest.mark.parametrize('info', [
    # a field every site carries, in each of the encodings
    [{'AA': 'A'}, {'AA': 'C'}, {'AA': 'G'}, {'AA': 'T'}, {'AA': 'A'}, {'AA': 'C'}],
    [{'DP': 3}, {'DP': 4}, {'DP': 5}, {'DP': 6}, {'DP': 7}, {'DP': 8}],
    [{'P': 0.5}, {'P': 0.25}, {'P': 0.125}, {'P': 1.0}, {'P': 0.0}, {'P': 0.75}],
    [{'F': True}, {'F': True}, {'F': True}, {'F': True}, {'F': True}, {'F': True}],
    # a field appearing only after the first chunk has been written
    [{}, {}, {}, {}, {'DP': 3}, {'DP': 4}],
    [{}, {}, {}, {}, {'AA': 'A'}, {'AA': 'C'}],
    [{}, {}, {}, {}, {'F': True}, {}],
    # a field whose values widen the encoding it was first written in
    [{'DP': 3}, {'DP': 4}, {'DP': 5}, {'DP': 6}, {'DP': 0.5}, {'DP': 7}],
    [{'DP': 3}, {'DP': 4}, {'DP': 5}, {'DP': 6}, {'DP': 'high'}, {'DP': 7}],
    [{'P': 0.5}, {'P': 0.25}, {'P': 0.125}, {'P': 1.0}, {'P': 'high'}, {'P': 0.75}],
    [{'F': True}, {'F': True}, {'F': True}, {'F': True}, {'F': 3}, {'F': True}],
    # a field that stops being carried, so an integer array can no longer hold it
    [{'DP': 3}, {'DP': 4}, {'DP': 5}, {'DP': 6}, {}, {'DP': 7}],
    [{'DP': 3}, {'DP': 4}, {'DP': 5}, {'DP': 6}, {'DP': '.'}, {'DP': 7}],
    # several fields at once, each on its own schedule
    [{'AA': 'A', 'DP': 1}, {'DP': 2}, {'AA': 'C'}, {'AA': 'G', 'DP': 4}, {}, {'AA': 'T', 'DP': 6}],
])
def test_a_streamed_store_holds_what_a_single_chunk_holds(tmp_path, info):
    """The INFO arrays are written a chunk at a time, so the store must not depend on where the chunk
    boundaries fall, whichever encoding the values ask for and whenever a field first appears."""
    variants = [_snp(pos=10 * (i + 1), info=values) for i, values in enumerate(info)]

    # the one store is written in chunks of four, the other in a single chunk holding all six variants
    writer = SmallChunkWriter(str(tmp_path / 'streamed.vcz'), samples=['s1'], seqnames=['1'])
    for variant in variants:
        writer.write(variant)
    writer.close()

    streamed, whole = str(tmp_path / 'streamed.vcz'), _store(tmp_path / 'whole.vcz', variants)

    assert _contents(streamed) == _contents(whole)
    assert [v.INFO for v in ZarrVariantReader(streamed)] == [v.INFO for v in ZarrVariantReader(whole)]


def test_the_writer_does_not_hold_the_info_of_every_variant(tmp_path):
    """The INFO values are buffered by the chunk, not by the input, so the writer holds no more of them
    once a chunk has been flushed."""
    writer = SmallChunkWriter(str(tmp_path / 'bounded.vcz'), samples=['s1'], seqnames=['1'])

    for i in range(20):
        writer.write(_snp(pos=10 * (i + 1), info={'DP': i}))

    assert sum(len(values) for values in writer._info.values()) <= writer._variant_chunk

    writer.close()

    assert [variant.INFO['DP'] for variant in ZarrVariantReader(str(tmp_path / 'bounded.vcz'))] == \
           list(range(20))


# --- the site view is built once per site (P5) ------------------------------------------------------

def test_the_view_of_a_site_is_built_once(tmp_path):
    """Every filtration and the parser ask for the view of the site they are handed, which is the same
    site, so it is built once and handed out again."""
    store = _store(tmp_path / 'view.vcz', [_snp(pos=10), _snp(pos=20)])

    first, second = list(ZarrVariantReader(store))

    assert SiteAlleles.from_site(first) is SiteAlleles.from_site(first)
    assert SiteAlleles.from_site(second) is not SiteAlleles.from_site(first)


def test_the_view_of_a_released_site_is_not_handed_to_its_successor(tmp_path):
    """The sites are transient, so a view kept against a site's identity alone would be handed to a
    later site allocated at the same address."""
    store = _store(tmp_path / 'transient.vcz', [_snp(pos=10, gt_bases=('A|A',)), _snp(pos=20)])

    views = []
    for variant in ZarrVariantReader(store):
        views.append((variant.POS, SiteAlleles.from_site(variant).counts()))

    assert views == [(10, {'A': 2}), (20, {'A': 1, 'T': 1})]


# --- the GFF is read in blocks (P1) -----------------------------------------------------------------

def _gff(path):
    """A small annotation over three contigs, one of which carries no coding sequence at all, with the
    coding sequences of two transcripts of one gene sharing coordinates."""
    rows = [
        ('ctgB', 'gene', 1, 900, '.', '.', 'ID=g1'),
        ('ctgB', 'CDS', 10, 60, '+', '0', 'Parent=t1'),
        ('ctgB', 'CDS', 10, 60, '+', '0', 'Parent=t2'),
        ('ctgB', 'CDS', 100, 160, '+', '0', 'Parent=t1'),
        ('ctgA', 'CDS', 30, 90, '-', '0', 'Parent=t3'),
        ('ctgA', 'exon', 30, 90, '-', '.', 'Parent=t3'),
        ('ctgA', 'CDS', 5, 20, '-', '2', 'Parent=t3'),
        ('ctgC', 'exon', 1, 50, '+', '.', 'Parent=t4'),
        ('ctgA', 'CDS', 300, 360, '-', '1', 'Parent=t5'),
    ]

    with open(path, 'w') as f:
        f.write('##gff-version 3\n')
        for seqid, kind, start, end, strand, phase, attributes in rows:
            f.write(f'{seqid}\tsrc\t{kind}\t{start}\t{end}\t.\t{strand}\t{phase}\t{attributes}\n')

    return str(path)


@pytest.mark.parametrize('block', [1, 2, 3, 5, 100])
def test_the_coding_sequences_do_not_depend_on_the_block_size(tmp_path, monkeypatch, block):
    """The GFF is read a block of lines at a time to bound the memory its attributes occupy, which must
    leave the coding sequences, their order and their contig categories as a single pass gives them."""
    gff = _gff(tmp_path / 'small.gff')

    monkeypatch.setattr(GFFHandler, '_gff_block', 10 ** 6)
    whole = GFFHandler(gff)._load_cds()

    monkeypatch.setattr(GFFHandler, '_gff_block', block)
    blocked = GFFHandler(gff)._load_cds()

    assert list(blocked.seqid.cat.categories) == list(whole.seqid.cat.categories)
    assert blocked.reset_index(drop=True).astype(object).equals(whole.reset_index(drop=True).astype(object))


def test_a_contig_without_coding_sequences_keeps_its_category(tmp_path):
    """A per-contig count reports every contig of the annotation, including those no coding sequence
    falls on, so the categories are those of the file rather than those of the coding sequences."""
    cds = GFFHandler(_gff(tmp_path / 'small.gff'))._load_cds()

    assert list(cds.seqid.cat.categories) == ['ctgA', 'ctgB', 'ctgC']
    assert set(cds.seqid.unique()) == {'ctgA', 'ctgB'}


def test_the_coding_sequences_of_a_whole_annotation_survive_the_blocks(monkeypatch):
    """The same on a real annotation, whose blocks each see their own contigs and transcripts."""
    gff = 'resources/genome/betula/genome.gff.gz'

    monkeypatch.setattr(GFFHandler, '_gff_block', 10 ** 7)
    whole = GFFHandler(gff)._load_cds()

    monkeypatch.setattr(GFFHandler, '_gff_block', 20000)
    blocked = GFFHandler(gff)._load_cds()

    assert list(blocked.seqid.cat.categories) == list(whole.seqid.cat.categories)
    assert blocked.reset_index(drop=True).astype(object).equals(whole.reset_index(drop=True).astype(object))
