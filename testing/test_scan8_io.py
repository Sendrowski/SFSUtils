"""
The numeric path through the IO backends: the VCF-Zarr writer takes the calls a site already holds
rather than re-deriving them from the genotype strings, the reader keeps the allele positions of the
store it reads, the site types agree with cyvcf2 across the backends, and the FASTA is indexed by
contig instead of scanned.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from cyvcf2 import VCF

from sfsutils.io_handlers import (FASTAHandler, SiteAlleles, TskitVariantReader, Variant,
                                  ZarrVariantReader, ZarrVariantWriter)

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
