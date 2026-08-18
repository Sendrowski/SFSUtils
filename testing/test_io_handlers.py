import os
import tempfile
import uuid
from unittest.mock import patch, MagicMock

from sfsutils.io_handlers import FileHandler, download_if_url
from testing import TestCase
import shutil
import subprocess
import sys
import pytest
from sfsutils.io_handlers import Variant
import numpy as np
from cyvcf2 import VCF
from sfsutils.io_handlers import (FASTAHandler, SiteAlleles, TskitVariantReader, Variant,
                                  ZarrVariantReader, ZarrVariantWriter)
from sfsutils.io_handlers import (GFFHandler, SiteAlleles, Variant, ZarrVariantReader,
                                  ZarrVariantWriter, get_called_bases)
import inspect
import sfsutils.io_handlers as io_handlers
from sfsutils.io_handlers import (GFFHandler, SiteAlleles, TskitVariantReader, Variant, VCFHandler,
                                  ZarrVariantReader, ZarrVariantWriter)
import random
from sfsutils.io_handlers import Variant, VariantWriter, ZarrVariantReader, ZarrVariantWriter
import pandas as pd
from sfsutils.io_handlers import get_called_alleles
from sfsutils.spectrum import Spectrum, Spectra
import sfsutils as su
from sfsutils.settings import Settings
from sfsutils.io_handlers import Variant, DummyVariant
from sfsutils.json_handlers import DataframeHandler


def _fake_response(content: bytes) -> MagicMock:
    """
    Build a stand-in for the streaming ``requests`` response used by ``download_file`` so the
    download path can be exercised without any network access.
    """
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.headers = {'content-length': str(len(content))}
    resp.iter_content.return_value = iter([content])
    return resp


class IOHandlerURLTestCase(TestCase):
    """
    Fast tests for the URL-handling layer in :mod:`sfsutils.io_handlers` (URL detection,
    filename/hash derivation, downloading and caching). Network access is mocked, so these run in
    milliseconds and never touch a remote server -- unlike the heavy ``slow``-tier tests that pull
    whole chromosomes from Ensembl/Sanger/GitHub.
    """

    def setUp(self):
        # a unique URL per test keeps the cache path (a hash of the URL) fresh across runs
        self.url = f"https://example.com/data/{uuid.uuid4().hex}/file.vcf.gz"
        self.payload = b"##fileformat=VCFv4.2\nfoo\nbar\n"
        self.cached_path = (tempfile.gettempdir() + '/' +
                            FileHandler.hash(self.url) + '.' + FileHandler.get_filename(self.url))

    def tearDown(self):
        if os.path.exists(self.cached_path):
            os.remove(self.cached_path)

    def test_is_url_recognizes_remote_schemes(self):
        """URLs with a scheme and host are recognised as remote."""
        self.assertTrue(FileHandler.is_url("https://example.com/x.vcf.gz"))
        self.assertTrue(FileHandler.is_url("http://example.com/x.fa"))
        self.assertTrue(FileHandler.is_url("ftp://ftp.ensembl.org/pub/x.gff3.gz"))

    def test_is_url_rejects_local_paths(self):
        """Local paths (relative, absolute, bare filename) are not treated as URLs."""
        self.assertFalse(FileHandler.is_url("resources/genome/betula/genome.gff.gz"))
        self.assertFalse(FileHandler.is_url("/abs/path/file.vcf"))
        self.assertFalse(FileHandler.is_url("file.vcf"))

    def test_get_filename_from_url(self):
        """The cached filename is the basename of the URL path, ignoring any query string."""
        self.assertEqual("chr21.fa.gz",
                         FileHandler.get_filename("http://ftp.ensembl.org/pub/chr21.fa.gz"))
        self.assertEqual("data.vcf",
                         FileHandler.get_filename("https://host/dir/data.vcf?token=abc"))

    def test_hash_is_deterministic_and_distinct(self):
        """The URL hash is stable, collision-free for distinct inputs, and truncated to 12 chars."""
        self.assertEqual(FileHandler.hash("a"), FileHandler.hash("a"))
        self.assertNotEqual(FileHandler.hash("a"), FileHandler.hash("b"))
        self.assertEqual(12, len(FileHandler.hash("anything")))

    def test_download_file_writes_payload(self):
        """A URL download lands at the hashed cache path with the streamed content intact."""
        with patch('sfsutils.io_handlers.requests.get',
                   return_value=_fake_response(self.payload)) as mock_get:
            path = FileHandler.download_file(self.url)

        self.assertEqual(self.cached_path, path)
        self.assertTrue(os.path.exists(path))
        with open(path, 'rb') as f:
            self.assertEqual(self.payload, f.read())
        mock_get.assert_called_once()

    def test_download_file_uses_cache_on_second_call(self):
        """With caching on, a repeated download is served from disk without hitting the network."""
        with patch('sfsutils.io_handlers.requests.get',
                   return_value=_fake_response(self.payload)) as mock_get:
            FileHandler.download_file(self.url, cache=True)
            FileHandler.download_file(self.url, cache=True)

            mock_get.assert_called_once()

    def test_download_file_no_cache_redownloads(self):
        """With caching off, every call re-fetches even when the file is already on disk."""
        with patch('sfsutils.io_handlers.requests.get',
                   side_effect=lambda *a, **k: _fake_response(self.payload)) as mock_get:
            FileHandler.download_file(self.url, cache=False)
            FileHandler.download_file(self.url, cache=False)

            self.assertEqual(2, mock_get.call_count)

    def test_download_if_url_passes_through_local_path(self):
        """A local path is returned unchanged and never triggers a download."""
        local = "resources/genome/betula/genome.gff.gz"

        with patch('sfsutils.io_handlers.requests.get') as mock_get:
            result = download_if_url(local)

        self.assertEqual(local, result)
        mock_get.assert_not_called()

    def test_download_if_url_downloads_remote(self):
        """A URL is downloaded and the local cache path is returned."""
        with patch('sfsutils.io_handlers.requests.get',
                   return_value=_fake_response(self.payload)) as mock_get:
            result = download_if_url(self.url)

        self.assertEqual(self.cached_path, result)
        mock_get.assert_called_once()


def _vcztools_bin():
    """Locate the vcztools console script (VCZTOOLS_BIN overrides), or None if it is not installed. The
    script sits next to this interpreter even when the env's bin is not on PATH."""
    override = os.environ.get("VCZTOOLS_BIN")
    if override:
        return override
    local = os.path.join(os.path.dirname(sys.executable), "vcztools")
    return local if os.path.exists(local) else shutil.which("vcztools")


def _write_store(path, n_variants, n_samples):
    """Write a store of ``n_variants`` biallelic phased sites over ``n_samples`` samples."""
    from sfsutils.io_handlers import ZarrVariantWriter

    samples = [f"s{i}" for i in range(n_samples)]
    w = ZarrVariantWriter(path, samples=samples, seqnames=["1"], info_ancestral="AA")
    for i in range(n_variants):
        w.write(Variant(ref="A", pos=i + 1, chrom="1", gt_bases=["A|T"] * n_samples, alt=["T"],
                        is_snp=True, info={"AA": "A"}))
    w.close()
    return path


def test_variant_arrays_share_a_chunk_grid(tmp_path):
    """Enough variants and samples to reach the chunk sizes: every array carrying a ``variants`` axis
    chunks it identically, and every ``call_*`` array chunks ``samples`` identically."""
    import zarr

    root = zarr.open(_write_store(str(tmp_path / "big.vcz"), n_variants=25000, n_samples=1500), mode="r")

    variant_chunks, sample_chunks = set(), set()
    for name in root.array_keys():
        dimensions = list(root[name].attrs["_ARRAY_DIMENSIONS"])
        chunks = root[name].chunks
        for dim, chunk in zip(dimensions, chunks):
            if dim == "variants":
                variant_chunks.add(chunk)
            elif dim == "samples" and name.startswith("call_"):
                sample_chunks.add(chunk)

    assert variant_chunks == {10000}
    assert sample_chunks == {1000}


def test_chunks_do_not_exceed_the_array(tmp_path):
    """A store smaller than the chunk sizes chunks each axis whole, and still uniformly."""
    import zarr

    root = zarr.open(_write_store(str(tmp_path / "small.vcz"), n_variants=50, n_samples=4), mode="r")

    assert root["variant_position"].chunks == (50,)
    assert root["variant_contig"].chunks == (50,)
    assert root["variant_allele"].chunks == (50, 2)
    assert root["call_genotype"].chunks == (50, 4, 2)
    assert root["call_genotype_phased"].chunks == (50, 4)


@pytest.mark.skipif(_vcztools_bin() is None,
                    reason="no vcztools binary reachable (needs a zarr-3 env; set VCZTOOLS_BIN)")
def test_vcztools_reads_a_store_larger_than_one_chunk(tmp_path):
    """vcztools reconstructs the VCF from a store spanning several chunks along the variants axis, which
    it rejects unless the chunk grids line up."""
    store = _write_store(str(tmp_path / "multi.vcz"), n_variants=21000, n_samples=4)
    result = subprocess.run([_vcztools_bin(), "view", store], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    body = [ln for ln in result.stdout.splitlines() if ln and not ln.startswith("#")]
    assert len(body) == 21000


HEADER = ("##fileformat=VCFv4.2\n"
          "##contig=<ID=1,length=1000>\n"
          "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")


def _vcf(path, records, samples):
    """Write a minimal VCF holding the given records over the given samples, and return its path."""
    columns = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)

    with open(path, 'w') as f:
        f.write(HEADER + columns + "\n")
        for record in records:
            f.write(record + "\n")

    return str(path)


def _roundtrip(path, store):
    """Write every record of a VCF to a store and return the reader over it."""
    reader = VCF(path)
    writer = ZarrVariantWriter(str(store), samples=reader.samples, seqnames=reader.seqnames)

    for variant in reader:
        writer.write(variant)

    writer.close()

    return ZarrVariantReader(str(store))


def _vcztools():
    """The vcztools console script, which sits next to this interpreter, or None where it is absent."""
    local = os.path.join(os.path.dirname(sys.executable), 'vcztools')

    return local if os.path.exists(local) else shutil.which('vcztools')


def test_haploid_call_of_a_third_allele_survives_the_store(tmp_path):
    """A haploid call of an allele beyond the second is one cyvcf2 cannot render as a genotype string,
    so the store must take it from the numeric calls: all six haplotypes stay called."""
    path = _vcf(tmp_path / 'multi.vcf', ["1\t20\t.\tA\tT,G\t.\t.\t.\tGT\t0/1\t2\t./.\t1|2\t0/."],
                [f's{i}' for i in range(5)])

    source = SiteAlleles.from_site(next(iter(VCF(path))))
    target = SiteAlleles.from_site(next(iter(_roundtrip(path, tmp_path / 'multi.vcz'))))

    assert np.asarray(source.indices).tolist() == np.asarray(target.indices).tolist()
    assert source.counts() == target.counts() == {'A': 2, 'T': 2, 'G': 2}


def test_polyploid_records_are_written(tmp_path):
    """cyvcf2 refuses to assemble genotype strings above ploidy two, which the numeric calls do not need."""
    path = _vcf(tmp_path / 'triploid.vcf', ["1\t12\t.\tA\tT\t.\t.\t.\tGT\t0/0/0\t1/1\t0/1\t1/1"],
                [f's{i}' for i in range(4)])

    variant = next(iter(_roundtrip(path, tmp_path / 'triploid.vcz')))

    assert np.asarray(variant.allele_indices).tolist() == [[0, 0, 0], [1, 1, -2], [0, 1, -2], [1, 1, -2]]
    assert SiteAlleles.from_site(variant).counts() == {'A': 4, 'T': 5}


def test_a_shorter_call_is_padded_with_the_fill_sentinel(tmp_path):
    """The spec separates the fill of a haplotype a call does not reach from a missing call, so a
    haploid call exports as haploid rather than as a diploid one half of which is missing."""
    path = _vcf(tmp_path / 'mixed.vcf', ["1\t21\t.\tC\tG\t.\t.\t.\tGT\t0/1\t0\t0/0\t1|1\t./."],
                [f's{i}' for i in range(5)])

    store = str(tmp_path / 'mixed.vcz')
    _roundtrip(path, store)

    import zarr

    genotype = zarr.open(store, mode='r')['call_genotype'][0]

    assert genotype.tolist() == [[0, 1], [0, -2], [0, 0], [1, 1], [-1, -1]]

    binary = _vcztools()

    if binary is None:
        pytest.skip('vcztools is not installed')

    exported = subprocess.run([binary, 'view', store], capture_output=True, text=True, check=True)
    calls = [line.split('\t')[9:] for line in exported.stdout.splitlines() if not line.startswith('#')]

    assert calls == [['0/1', '0', '0/0', '1|1', './.']]


def test_phase_follows_the_call_rather_than_its_rendering(tmp_path):
    """The phase of each sample is the flag the record carries, which a missing call also sets."""
    path = _vcf(tmp_path / 'phase.vcf', ["1\t30\t.\tA\tT\t.\t.\t.\tGT\t0|1\t0/1\t.|.\t./."],
                [f's{i}' for i in range(4)])

    import zarr

    store = str(tmp_path / 'phase.vcz')
    _roundtrip(path, store)

    expected = np.asarray(next(iter(VCF(path))).genotype.array())[:, -1].astype(bool)

    assert zarr.open(store, mode='r')['call_genotype_phased'][0].tolist() == expected.tolist()


def test_an_empty_reference_allele_keeps_its_position(tmp_path):
    """A tree sequence may carry an empty ancestral state, which occupies allele zero: dropping it would
    shift every genotype code onto the following allele."""
    tskit = pytest.importorskip('tskit')

    tables = tskit.TableCollection(sequence_length=10)
    for _ in range(4):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)

    root = tables.nodes.add_row(time=1)
    for child in range(4):
        tables.edges.add_row(left=0, right=10, parent=root, child=child)

    individuals = [tables.individuals.add_row(), tables.individuals.add_row()]
    nodes = tables.nodes.copy()
    tables.nodes.clear()
    for i, node in enumerate(nodes):
        tables.nodes.append(node.replace(individual=individuals[i // 2] if i < 4 else -1))

    site = tables.sites.add_row(position=1, ancestral_state='')
    tables.mutations.add_row(site=site, node=0, derived_state='A')
    tables.sort()

    ts = tables.tree_sequence()
    reader = TskitVariantReader(ts)

    writer = ZarrVariantWriter(str(tmp_path / 'empty.vcz'), samples=reader.samples, seqnames=reader.seqnames)
    for variant in TskitVariantReader(ts):
        writer.write(variant)
    writer.close()

    source = next(iter(reader))
    target = next(iter(ZarrVariantReader(str(tmp_path / 'empty.vcz'))))

    assert (target.REF, target.ALT) == (source.REF, source.ALT) == ('', ['A'])
    assert np.asarray(target.allele_indices).tolist() == [[1, 0], [0, 0]]
    assert SiteAlleles.from_site(target).counts() == {'A': 1}


def test_trailing_allele_padding_is_still_dropped(tmp_path):
    """A site with fewer alleles than the widest of the store is padded on the right, and that padding
    is not part of its ALT."""
    path = _vcf(tmp_path / 'ragged.vcf',
                ["1\t10\t.\tA\tT,G\t.\t.\t.\tGT\t0/1", "1\t11\t.\tC\tG\t.\t.\t.\tGT\t0/1"], ['s0'])

    variants = list(_roundtrip(path, tmp_path / 'ragged.vcz'))

    assert [(v.REF, v.ALT) for v in variants] == [('A', ['T', 'G']), ('C', ['G'])]


@pytest.mark.parametrize('ref,alt', [('N', 'A'), ('A', 'T'), ('R', 'A'), ('A', 'T,*'), ('A', '*'),
                                     ('A', '<NON_REF>'), ('AT', 'A'), ('A', 'N')])
def test_site_types_agree_with_cyvcf2(tmp_path, ref, alt):
    """The site type a store reports is the one cyvcf2 reports for the same record: an ambiguity code in
    the reference is an SNP, a spanning deletion or a symbolic allele is not."""
    path = _vcf(tmp_path / f'type_{ref}_{alt}.vcf'.replace('*', 'star').replace('<', '').replace('>', ''),
                [f"1\t10\t.\t{ref}\t{alt}\t.\t.\t.\tGT\t0/1"], ['s0'])

    expected = next(iter(VCF(path))).is_snp
    store = _roundtrip(path, tmp_path / f'type_{ref}_{len(alt)}.vcz')

    assert next(iter(store)).is_snp == expected


def test_a_sites_only_store_streams(tmp_path):
    """A VCF without samples converts to a store without any call arrays, which streams as the sites it
    holds rather than raising."""
    pytest.importorskip('bio2zarr')

    path = str(tmp_path / 'sites.vcf')
    with open(path, 'w') as f:
        f.write(HEADER + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        f.write("1\t10\t.\tA\tT\t.\t.\t.\n1\t11\t.\tC\tG\t.\t.\t.\n")

    store = str(tmp_path / 'sites.vcz')
    subprocess.run([sys.executable, '-m', 'bio2zarr', 'vcf2zarr', 'convert', path, store],
                   capture_output=True, check=True)

    variants = list(ZarrVariantReader(store))

    assert [(v.CHROM, v.POS, v.REF, v.ALT) for v in variants] == [('1', 10, 'A', ['T']), ('1', 11, 'C', ['G'])]
    assert [np.asarray(v.allele_indices).shape[0] for v in variants] == [0, 0]


@pytest.mark.parametrize('indices,alleles,expected', [
    ([[0, 1], [1, 1]], ['A', 'T'], {'A': 1, 'T': 3}),
    ([[0, -2], [-1, -1]], ['A', 'T'], {'A': 1}),
    ([[0, 1], [2, 2]], ['A', 'N', 'T'], {'A': 1, 'T': 2}),
    ([[0, 5], [-1, 1]], ['A', 'T'], {'A': 1, 'T': 1}),
    ([[]], ['A', 'T'], {}),
])
def test_counting_covers_the_sentinels(indices, alleles, expected):
    """Both negative sentinels and an index beyond the alleles are uncalled, and an allele that is not a
    run of bases carries no count."""
    calls = np.asarray(indices, dtype=int).reshape(len(indices), -1) if indices else np.zeros((0, 0), dtype=int)
    site = SiteAlleles(calls, alleles)

    assert site.counts() == expected
    assert site.n_called() == sum(expected.values())
    assert site.distinct() == set(expected)


def test_counting_matches_the_genotype_strings():
    """Over a real VCF and several masks the counts agree with those of the assembled genotype strings."""
    from sfsutils.io_handlers import get_distinct_called_alleles

    reader = VCF('resources/genome/betula/all.polarized.subset.10000.vcf.gz')
    rng = np.random.default_rng(0)
    masks = [None, np.ones(len(reader.samples), bool), rng.random(len(reader.samples)) < 0.5]

    for i, variant in enumerate(reader):
        if i >= 200:
            break

        site = SiteAlleles.from_site(variant)

        for mask in masks:
            genotypes = variant.gt_bases if mask is None else variant.gt_bases[mask]
            counts = {}
            for genotype in genotypes:
                for allele in str(genotype).replace('|', '/').split('/'):
                    if allele in ('A', 'C', 'G', 'T'):
                        counts[allele] = counts.get(allele, 0) + 1

            assert site.counts(mask) == counts
            assert site.n_called(mask) == sum(counts.values())
            assert site.distinct(mask) == get_distinct_called_alleles(genotypes)


def test_contigs_are_found_in_any_order(tmp_path):
    """A contig is looked up by name, so visiting the FASTA backwards yields the same records as
    visiting it forwards, and an absent contig is reported without a pass over the file."""
    path = str(tmp_path / 'ref.fasta')
    contigs = {f'c{i}': ''.join('ACGT'[(i + j) % 4] for j in range(40)) for i in range(5)}

    with open(path, 'w') as f:
        for name, sequence in contigs.items():
            f.write(f'>{name} some description\n{sequence}\n')

    handler = FASTAHandler(path)

    assert handler.get_contig_names() == list(contigs)

    for name in reversed(list(contigs)):
        assert str(handler.get_contig([name]).seq) == contigs[name]

    # the same handler serves a second, forward pass without being rewound
    for name in contigs:
        assert str(handler.get_contig(['other', name]).seq) == contigs[name]

    with pytest.raises(LookupError):
        handler.get_contig(['absent'])

    with pytest.raises(LookupError):
        handler.get_contig(['absent'])


def test_the_tskit_positions_are_those_of_the_sites():
    """The site positions come from the table column, and a continuous genome keeps its exact
    (non-integer) position alongside the rounded VCF one."""
    msprime = pytest.importorskip('msprime')

    ts = msprime.sim_mutations(msprime.sim_ancestry(5, sequence_length=1e4, random_seed=1),
                               rate=1e-4, random_seed=2)

    variants = list(TskitVariantReader(ts))

    assert [v._tskit_position for v in variants] == list(ts.sites_position)
    assert [v.POS for v in variants] == [int(p) + 1 for p in ts.sites_position]


def test_the_writer_agrees_with_the_genotype_strings(tmp_path):
    """Over a diploid VCF the calls the store holds are those the genotype strings spell out."""
    reader = VCF('resources/genome/betula/all.polarized.subset.10000.vcf.gz')
    store = str(tmp_path / 'betula.vcz')
    writer = ZarrVariantWriter(store, samples=reader.samples, seqnames=reader.seqnames)

    expected = []
    for i, variant in enumerate(reader):
        if i >= 200:
            break

        writer.write(variant)
        index = {a: j for j, a in enumerate([variant.REF] + list(variant.ALT))}
        expected.append([[index.get(a, -1) for a in str(g).replace('|', '/').split('/')]
                         for g in variant.gt_bases])

    writer.close()

    import zarr

    assert zarr.open(store, mode='r')['call_genotype'][:].tolist() == expected


def test_a_variant_without_calls_is_written(tmp_path):
    """A site carrying neither allele indices nor genotypes still occupies a row of the store."""
    writer = ZarrVariantWriter(str(tmp_path / 'bare.vcz'), samples=[], seqnames=['1'])
    writer.write(Variant(ref='A', pos=1, chrom='1', alt=['T'], is_snp=True))
    writer.close()

    variants = list(ZarrVariantReader(str(tmp_path / 'bare.vcz')))

    assert [(v.POS, v.REF, v.ALT) for v in variants] == [(1, 'A', ['T'])]


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

    # the cache holds the site itself, which keeps it alive: were it keyed on the identity alone, the
    # address of a released site would be handed out again and match a later one
    site = _snp(pos=30)
    SiteAlleles.from_site(site)
    cached, _ = SiteAlleles._cached

    assert cached is site


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


HEADER_TWO_CONTIGS = (
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
        f.write(HEADER_TWO_CONTIGS + ''.join(row + '\n' for row in rows))

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

    # a row of missing markers alone carries no fill to strip, so the field is dropped by the row being
    # absent throughout rather than by the trailing padding being removed
    missing = _with_array(_store(tmp_path / 'missing.vcz', [_snp(pos=10)]),
                          'variant_AC', [[-1, -1]], 'int32', ['variants', 'values'])

    assert [variant.INFO for variant in ZarrVariantReader(missing)] == [{}]


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


def test_install_hints_name_the_distribution():
    """``sfsutils`` on PyPI is an unrelated project, so an install hint naming it installs the wrong
    package and shadows the import name."""
    source = inspect.getsource(io_handlers)

    assert 'pip install sfsutils[' not in source

    for extra in ('vcf', 'zarr', 'arg'):
        assert f'pip install \\"sfsutils-popgen[{extra}]\\"' in source


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


SAMPLES = ["s1"]


def _variant(pos, info):
    """A biallelic phased SNP at ``pos`` carrying ``info``."""
    return Variant(ref="A", pos=pos, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True, info=dict(info))


def _roundtrip_infos(path, infos, chunk=None):
    """Write one variant per INFO dict and read the INFO values back."""
    writer = ZarrVariantWriter(str(path), samples=SAMPLES, seqnames=["1"], info_ancestral="AA")

    if chunk is not None:
        writer._variant_chunk = chunk

    for i, info in enumerate(infos):
        writer.write(_variant(i + 1, info))

    writer.close()

    return [dict(variant.INFO) for variant in ZarrVariantReader(str(path))]


def test_field_absent_on_every_variant_of_its_first_chunk(tmp_path):
    """A field whose only values are the VCF missing marker, as DegeneracyAnnotation writes at a
    non-coding site, is written without a usable value and reads back as absent."""
    read = _roundtrip_infos(tmp_path / "empty.vcz", [{"Degeneracy": "."}, {"Degeneracy": "."}])

    assert [entry.get("Degeneracy") for entry in read] == [None, None]


def test_field_absent_on_every_variant_of_a_later_chunk(tmp_path):
    """An integer field no variant of a later chunk carries widens to the encoding that can mark an
    absent value, keeping the integers already written."""
    infos = [{"DP": 11 + i} for i in range(4)] + [{}] * 4

    read = _roundtrip_infos(tmp_path / "late.vcz", infos, chunk=4)

    assert [entry.get("DP") for entry in read] == [11, 12, 13, 14, None, None, None, None]


def test_flag_promoted_to_string_keeps_an_unset_flag_absent(tmp_path):
    """A Flag widened to a string by a later chunk carrying a string: the variants that never carried
    the flag stay absent instead of reading back as the string 'False'."""
    infos = [{"F": True}, {}, {"F": True}, {"F": "x"}, {"F": "x"}, {"F": "x"}]

    read = _roundtrip_infos(tmp_path / "flag.vcz", infos, chunk=3)

    assert [entry.get("F") for entry in read] == ["True", None, "True", "x", "x", "x"]


def test_integer_encoder_marks_an_absent_value(tmp_path):
    """The integer branch of the encoder marks an absent value with the sentinel the reader maps back
    to absent rather than failing on it."""
    writer = ZarrVariantWriter(str(tmp_path / "int.vcz"), samples=SAMPLES, seqnames=["1"])

    encoded = writer._encode([1, writer._missing, 3], "int")

    assert list(encoded) == [1, -1, 3]


def test_declared_contig_length_survives_the_round_trip(tmp_path):
    """A store written from a source declaring a contig length records that length, not the last
    position written, so the region a spectrum extrapolates over is not understated."""
    reader = _FakeReader(seqnames=["chr1", "chr2"], lengths={"chr1": 1_000_000})

    writer = VariantWriter.open(str(tmp_path / "declared.vcz"), reader)
    writer.write(_variant_on("chr1", 49800))
    writer.write(_variant_on("chr2", 700))
    writer.close()

    assert ZarrVariantReader(str(tmp_path / "declared.vcz")).contig_lengths == {"chr1": 1_000_000,
                                                                               "chr2": 700}


def test_undeclared_contig_length_falls_back_to_the_observed_span(tmp_path):
    """A source declaring no lengths leaves the store carrying the last position written, which the
    reader documents as a lower bound."""
    writer = ZarrVariantWriter(str(tmp_path / "observed.vcz"), samples=SAMPLES, seqnames=["1"])
    writer.write(_variant(500, {}))
    writer.close()

    assert ZarrVariantReader(str(tmp_path / "observed.vcz")).contig_lengths == {"1": 500}


def test_declared_length_reaches_the_store_through_a_vcf_source(tmp_path):
    """The ##contig header of a VCF input reaches the output store."""
    pytest.importorskip("cyvcf2")

    from sfsutils.filtration import Filterer, SNPFiltration

    vcf = tmp_path / "in.vcf"
    with open(vcf, "w") as handle:
        handle.write("##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000000>\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n")
        for i in range(50):
            handle.write(f"chr1\t{100 + i * 100}\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\n")

    out = str(tmp_path / "sub.vcz")
    Filterer(source=str(vcf), output=out, filtrations=[SNPFiltration()]).filter()

    assert ZarrVariantReader(out).contig_lengths == {"chr1": 1_000_000}


def _variant_on(chrom, pos):
    """A biallelic phased SNP at ``pos`` on ``chrom``."""
    return Variant(ref="A", pos=pos, chrom=chrom, gt_bases=["A|T"], alt=["T"], is_snp=True, info={})


class _FakeReader:
    """A stand-in cyvcf2 VCF exposing only what the writer takes from an input."""

    def __init__(self, seqnames, lengths):
        self.samples = SAMPLES
        self.seqnames = list(seqnames)
        self.seqlens = [lengths.get(name, 0) for name in seqnames]


def _matches(written, read):
    """Whether a value read back is the value written, allowing for the wider encoding a later chunk
    may have promoted the field to."""
    if isinstance(written, bool):
        return read is True or read == "True"

    if isinstance(written, str):
        return read == written

    if isinstance(read, str):
        try:
            return float(read) == float(written)
        except ValueError:
            return False

    return read is not None and float(read) == float(written)


FUZZ_SEEDS = 50


def _fuzz_case(tmp_path, seed):
    """
    Round-trip one randomly drawn sequence of INFO fields.

    :param tmp_path: Directory to write the store into.
    :param seed: Seed of the draw.
    :return: The written INFO mappings and the ones read back.
    """
    rng = random.Random(seed)

    def value():
        choice = rng.randrange(5)

        if choice == 0:
            return True
        if choice == 1:
            return rng.randrange(-1000, 1000)
        if choice == 2:
            return rng.uniform(-10, 10)
        if choice == 3:
            return rng.choice(["A", "T", "coding", "x"])

        return rng.choice([None, "."])

    n = rng.randrange(1, 20)
    infos = [{key: value() for key in rng.sample(["F", "DP", "AA"], rng.randrange(0, 4))}
             for _ in range(n)]

    return infos, _roundtrip_infos(tmp_path / f"fuzz{seed}.vcz", infos, chunk=rng.randrange(1, 6))


def test_info_encoding_round_trip_fuzz(tmp_path):
    """Across chunk sizes and sequences of types, a value written reads back as itself and a variant
    carrying no value reads back as carrying none. A boolean False is excluded: an unset VCF Flag is
    absent from INFO, so it is not a value a source can carry."""
    for seed in range(FUZZ_SEEDS):
        infos, read = _fuzz_case(tmp_path, seed)

        assert len(read) == len(infos), f"seed {seed}"

        for written, got in zip(infos, read):
            for key in ("F", "DP", "AA"):
                expected = written.get(key)

                if expected is None or expected == ".":
                    assert got.get(key) is None, f"seed {seed}: {key} {written} -> {got}"
                else:
                    assert _matches(expected, got.get(key)), f"seed {seed}: {key} {written} -> {got}"


def test_info_dtypes_cover_every_encoding_the_writer_reaches():
    """Every encoding the writer settles on is one the store has a dtype for."""
    assert set(ZarrVariantWriter._info_dtypes) == {"bool", "int", "float", "str"}
    assert np.dtype(ZarrVariantWriter._info_dtypes["int"]) == np.int64


class TestUnzipMemoised:
    """
    Decompressing the same file twice must reuse the temporary copy.
    """

    def test_same_path_returned(self, tmp_path):
        import gzip

        from sfsutils.io_handlers import FileHandler

        src = tmp_path / 'ref.fasta.gz'
        with gzip.open(src, 'wt') as f:
            f.write('>1\nACGT\n')

        first = FileHandler.unzip_if_zipped(str(src))

        assert first == FileHandler.unzip_if_zipped(str(src))


def test_zarr_reader_surfaces_all_info_fields(tmp_path):
    """Every INFO field the writer persisted must be readable back, not just the ancestral tag, so an
    annotated store re-parsed by a stratification sees its field."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader

    out = str(tmp_path / "annotated.vcz")
    writer = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    writer.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True,
                         info={"AA": "A", "Degeneracy": "4"}))
    writer.write(Variant(ref="C", pos=20, chrom="1", gt_bases=["G|G"], alt=["G"], is_snp=True,
                         info={"AA": "G", "Degeneracy": "0"}))
    writer.close()

    variants = list(ZarrVariantReader(out, info_ancestral="AA"))
    assert [v.INFO.get("Degeneracy") for v in variants] == ["4", "0"]
    assert [v.INFO.get("AA") for v in variants] == ["A", "G"]


_FIXTURE = "resources/msprime/two_epoch.vcz"


@pytest.mark.skipif(not __import__("os").path.exists(_FIXTURE), reason="the VCF-Zarr fixture is absent")
def test_zarr_reader_does_not_surface_reserved_metadata():
    """A plain vcf2zarr store must not have its reserved variant_* metadata (quality/filter/id/length)
    surfaced as bogus INFO, which would corrupt the typed layout on a round-trip."""
    from sfsutils.io_handlers import ZarrVariantReader
    variant = next(iter(ZarrVariantReader(_FIXTURE)))
    assert dict(variant.INFO) == {}


def test_zarr_info_round_trips_with_native_types(tmp_path):
    """INFO written through our own Zarr writer round-trips with native types (str/float/int), so a
    numeric field is a number on read, matching cyvcf2 rather than becoming a string."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader

    out = str(tmp_path / "typed.vcz")
    w = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    w.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True,
                    info={"AA": "A", "AA_prob": 0.9, "DP": 30}))
    w.close()

    info = next(iter(ZarrVariantReader(out))).INFO
    assert info["AA"] == "A" and isinstance(info["AA"], str)
    assert info["AA_prob"] == 0.9 and isinstance(info["AA_prob"], float)
    assert info["DP"] == 30 and isinstance(info["DP"], int)


def test_zarr_info_missing_value_is_absent(tmp_path):
    """A numeric INFO field present on some sites but not others reads back as absent (NaN omitted), the
    way cyvcf2 reports a missing INFO field, rather than as a NaN or empty string."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader

    out = str(tmp_path / "miss.vcz")
    w = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    w.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True, info={"AA_prob": 0.9}))
    w.write(Variant(ref="C", pos=20, chrom="1", gt_bases=["C|G"], alt=["G"], is_snp=True, info={}))
    w.close()

    variants = list(ZarrVariantReader(out))
    assert variants[0].INFO["AA_prob"] == 0.9
    assert "AA_prob" not in variants[1].INFO


def test_zarr_degeneracy_dot_sentinel_round_trips_numeric(tmp_path):
    """A Degeneracy field mixing ints (coding) with the VCF '.' marker (non-coding) must round-trip as a
    number, not a string: '.' is a missing sentinel, so the field stays numeric and the '4' == 4 test in
    DegeneracyStratification still works (the round-1..3 regression stored it as a string, emptying the
    stratified SFS silently)."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader

    out = str(tmp_path / "deg.vcz")
    w = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    w.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True, info={"Degeneracy": 4}))
    w.write(Variant(ref="C", pos=20, chrom="1", gt_bases=["C|G"], alt=["G"], is_snp=True, info={"Degeneracy": "."}))
    w.close()

    variants = list(ZarrVariantReader(out))
    assert variants[0].INFO["Degeneracy"] == 4          # numeric, so `== 4` holds (float 4.0 == 4)
    assert not isinstance(variants[0].INFO["Degeneracy"], str)
    assert "Degeneracy" not in variants[1].INFO          # '.' is absent, not the string "."


def test_zarr_reader_surfaces_multivalued_info(tmp_path):
    """A multi-valued INFO field (Number != 1, stored as a 2-D variant_<key> array by vcf2zarr) is
    surfaced in the form cyvcf2 hands out, a tuple of the values a numeric field carries at the site."""
    import zarr
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader

    out = str(tmp_path / "mv.vcz")
    w = ZarrVariantWriter(out, samples=["s1"], seqnames=["1"], info_ancestral="AA")
    w.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True, info={"AA": "A"}))
    w.close()
    root = zarr.open(out, mode="r+")
    ac = root.create_array("variant_AC", shape=(1, 2), dtype="float64")
    ac[:] = [[3.0, 5.0]]
    ac.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alt_alleles"]

    variant = next(iter(ZarrVariantReader(out)))  # must not raise
    assert variant.INFO["AC"] == (3.0, 5.0)
    assert variant.INFO["AA"] == "A"


def test_zarr_integer_info_minus_one_two_round_trip(tmp_path):
    """Legitimate integer INFO values of -1/-2 (e.g. SVLEN) must round-trip; the round-3 reader briefly
    treated them as the VCF-Zarr missing/fill sentinels and dropped them."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader
    out = str(tmp_path / "sv.vcz")
    w = ZarrVariantWriter(out, samples=["s"], seqnames=["1"], info_ancestral="AA")
    for pos, sv in [(10, 5), (20, -1), (30, -2), (40, 7)]:
        w.write(Variant(ref="A", pos=pos, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True, info={"SVLEN": sv}))
    w.close()
    assert [v.INFO.get("SVLEN") for v in ZarrVariantReader(out)] == [5, -1, -2, 7]


def test_zarr_flag_info_absent_is_omitted(tmp_path):
    """A bool/Flag INFO field is surfaced only where set (as cyvcf2 does); an absent flag must not read
    back as present-False."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader
    out = str(tmp_path / "flag.vcz")
    w = ZarrVariantWriter(out, samples=["s"], seqnames=["1"], info_ancestral="AA")
    w.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True, info={"DB": True}))
    w.write(Variant(ref="C", pos=20, chrom="1", gt_bases=["C|G"], alt=["G"], is_snp=True, info={}))
    w.close()
    variants = list(ZarrVariantReader(out))
    assert variants[0].INFO["DB"] is True
    assert "DB" not in variants[1].INFO


def test_zarr_out_of_int64_info_does_not_crash(tmp_path):
    """An integer INFO value beyond int64 must not crash the write; it is kept exactly as a string
    rather than truncated or lost to float precision."""
    from sfsutils.io_handlers import ZarrVariantWriter, ZarrVariantReader
    out = str(tmp_path / "big.vcz")
    w = ZarrVariantWriter(out, samples=["s"], seqnames=["1"], info_ancestral="AA")
    w.write(Variant(ref="A", pos=10, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True, info={"BIG": 10 ** 19}))
    w.close()
    assert next(iter(ZarrVariantReader(out))).INFO["BIG"] == "10000000000000000000"
