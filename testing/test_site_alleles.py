"""
The numeric genotype view must reach exactly the same verdicts and counts as decoding the genotype strings,
on every site of the real fixtures and on the pathological alleles the strings handle by hand.
"""

import os
from collections import Counter

import numpy as np
import pytest

from sfsutils.io_handlers import (
    SiteAlleles,
    get_called_bases,
    get_distinct_called_alleles,
    get_distinct_called_bases,
)

VCF = "resources/msprime/two_epoch.vcf"
VCZ = "resources/msprime/two_epoch.vcz"
TREES = "resources/msprime/two_epoch.trees"
SAPIENS = "resources/genome/sapiens/chr21_test.vcf.gz"
BETULA_VCF = "resources/genome/betula/all.polarized.subset.10000.vcf.gz"
BETULA_OUT = "resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz"


def _sources():
    """
    The available fixtures, one per backend.

    :return: The pairs of source path and identifier.
    """
    return [(path, name) for path, name in [
        (VCF, 'vcf'),
        (VCZ, 'zarr'),
        (TREES, 'tskit'),
        (SAPIENS, 'sapiens'),
        (BETULA_VCF, 'betula-vcf'),
        (BETULA_OUT, 'betula-outgroups'),
    ] if os.path.exists(path)]


def _variants(source: str):
    """
    Stream the sites of a source through the backend its extension selects.

    :param source: The path to the source.
    :return: An iterator over the sites.
    """
    if source.endswith('.vcz'):
        from sfsutils.io_handlers import ZarrVariantReader

        return iter(ZarrVariantReader(source))

    if source.endswith('.trees'):
        import tskit

        from sfsutils.io_handlers import TskitVariantReader

        return iter(TskitVariantReader(tskit.load(source)))

    from cyvcf2 import VCF as CyVCF

    return iter(CyVCF(source))


def _n_samples(source: str) -> int:
    """
    The number of samples of a source.

    :param source: The path to the source.
    :return: The number of samples.
    """
    first = next(_variants(source))

    return len(first.gt_bases)


def _masks(n: int):
    """
    The sample masks the comparisons are run under.

    :param n: The number of samples.
    :return: The named masks.
    """
    return {
        'all': np.ones(n, dtype=bool),
        'half': np.array([i % 2 == 0 for i in range(n)]),
        'one': np.array([i == 0 for i in range(n)]),
        'none': np.zeros(n, dtype=bool),
    }


class TestSiteAllelesAgreesWithTheStrings:
    """
    The view's counts, totals and distinct alleles must match the string helpers site by site.
    """

    @pytest.mark.parametrize('source,name', _sources())
    def test_counts_and_distinct_alleles(self, source, name):
        masks = _masks(_n_samples(source))

        compared = 0
        for variant in _variants(source):
            site = SiteAlleles.from_site(variant)

            assert site is not None, f"no numeric calls at {variant.CHROM}:{variant.POS} of {name}"

            for mask in masks.values():
                genotypes = variant.gt_bases[mask]

                assert site.distinct(mask) == get_distinct_called_alleles(genotypes)

                if site.single_character:
                    called = get_called_bases(genotypes)

                    assert site.n_called(mask) == len(called)
                    assert site.counts(mask) == dict(Counter(called))
                    assert set(site.distinct(mask)) == get_distinct_called_bases(genotypes)

            compared += 1

        assert compared > 100

    @pytest.mark.parametrize('source,name', _sources())
    def test_shape_and_dtype(self, source, name):
        variant = next(_variants(source))
        site = SiteAlleles.from_site(variant)

        assert site.indices.ndim == 2
        assert site.indices.shape[0] == len(variant.gt_bases)
        assert np.issubdtype(site.indices.dtype, np.integer)
        assert site.indices.min() >= -1
        assert site.alleles == [variant.REF] + list(variant.ALT)


class TestAdversarialAlleles:
    """
    The alleles the string helpers treat specially: the uncalled ones (``N``, ``*``, ``<NON_REF>``, ``.``),
    the multi-character ones, the partially missing calls and haploid data.
    """

    HEADER = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=10000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
    )

    ROWS = [
        "1\t10\t.\tA\tC\t.\t.\t.\tGT\t0|0\t0/1\t1|1",  # ordinary biallelic SNP
        "1\t11\t.\tN\tC\t.\t.\t.\tGT\t0|0\t0/1\t1|1",  # N as the reference
        "1\t12\t.\tA\tN\t.\t.\t.\tGT\t0|0\t0/1\t1|1",  # N as an alternate
        "1\t13\t.\tA\t*\t.\t.\t.\tGT\t0|0\t0/1\t1|1",  # spanning deletion
        "1\t14\t.\tA\t<NON_REF>\t.\t.\t.\tGT\t0|0\t0/1\t1|1",  # symbolic allele
        "1\t15\t.\tAT\tGC\t.\t.\t.\tGT\t0|0\t0/1\t1|1",  # MNP
        "1\t16\t.\tA\tAT\t.\t.\t.\tGT\t0|0\t0/1\t1|1",  # insertion
        "1\t17\t.\tAT\tA\t.\t.\t.\tGT\t0|0\t0/1\t1|1",  # deletion
        "1\t18\t.\tA\tC,G\t.\t.\t.\tGT\t0|0\t.|1\t2|2",  # tri-allelic with a half-missing call
        "1\t19\t.\tA\tC,G\t.\t.\t.\tGT\t0|0\t0/1\t0|0",  # third allele only in the excluded sample
        "1\t20\t.\tA\tC,G\t.\t.\t.\tGT\t0|0\t0/0\t0|0",  # alternates carried by no call at all
        "1\t21\t.\tA\tC\t.\t.\t.\tGT\t./.\t.|1\t1|1",  # fully and partially missing calls
        "1\t22\t.\tA\tC\t.\t.\t.\tGT\t0\t1\t.",  # haploid
        "1\t23\t.\tA\tN,*\t.\t.\t.\tGT\t0|1\t1/2\t2|2",  # only uncalled alternates
    ]

    @pytest.fixture(scope='class')
    @classmethod
    def vcf(cls, tmp_path_factory):
        path = tmp_path_factory.mktemp('adversarial') / 'adversarial.vcf'
        path.write_text(cls.HEADER + "\n".join(cls.ROWS) + "\n")

        return str(path)

    def test_agrees_with_the_strings(self, vcf):
        from cyvcf2 import VCF as CyVCF

        masks = _masks(3)

        compared = 0
        for variant in CyVCF(vcf):
            site = SiteAlleles.from_site(variant)

            for mask in masks.values():
                genotypes = variant.gt_bases[mask]

                assert site.distinct(mask) == get_distinct_called_alleles(genotypes), variant.POS

                if site.single_character:
                    called = get_called_bases(genotypes)

                    assert site.n_called(mask) == len(called), variant.POS
                    assert site.counts(mask) == dict(Counter(called)), variant.POS

            compared += 1

        assert compared == len(self.ROWS)

    def test_multi_character_alleles_are_flagged(self, vcf):
        from cyvcf2 import VCF as CyVCF

        flagged = {variant.POS: SiteAlleles.from_site(variant).single_character for variant in CyVCF(vcf)}

        # the MNP, the insertion, the deletion and the symbolic allele must all be kept off the
        # character-counting shortcut, since their genotype strings carry more than one character per call
        assert flagged[15] is False and flagged[16] is False and flagged[17] is False
        assert flagged[14] is False
        assert flagged[10] is True and flagged[13] is True

    def test_haploid_calls(self, vcf):
        from cyvcf2 import VCF as CyVCF

        variant = [v for v in CyVCF(vcf) if v.POS == 22][0]
        site = SiteAlleles.from_site(variant)

        assert site.indices.shape == (3, 1)
        assert site.indices.ravel().tolist() == [0, 1, -1]
        assert site.counts() == {'A': 1, 'C': 1}


class TestFiltrationVerdictsUnchanged:
    """
    The filtrations must reach the verdict the string implementations they replace reached, on every site.
    """

    @staticmethod
    def _snp_reference(variant, mask):
        if not variant.is_snp:
            return False

        return len(get_distinct_called_bases(variant.gt_bases[mask])) > 1

    @staticmethod
    def _polyallelic_reference(variant, mask):
        if len(variant.ALT) < 2:
            return True

        return len(get_distinct_called_alleles(variant.gt_bases[mask])) < 3

    @pytest.mark.parametrize('source,name', _sources())
    def test_verdicts(self, source, name):
        from sfsutils import PolyAllelicFiltration, SNPFiltration

        masks = _masks(_n_samples(source))

        compared = 0
        for variant in _variants(source):
            for mask in masks.values():
                snp = SNPFiltration()
                snp._samples_mask = mask

                poly = PolyAllelicFiltration()
                poly._samples_mask = mask

                assert snp.filter_site(variant) == self._snp_reference(variant, mask), variant.POS
                assert poly.filter_site(variant) == self._polyallelic_reference(variant, mask), variant.POS

            compared += 1

        assert compared > 100


@pytest.fixture
def strings_only(monkeypatch):
    """
    Withhold the numeric calls from every consumer, so that the parse runs entirely off the genotype strings.
    """
    monkeypatch.setattr(SiteAlleles, 'from_site', classmethod(lambda cls, variant: None))


class TestSpectraUnchanged:
    """
    A parse reading the numeric calls must produce bit-identical spectra to the same parse reading the
    genotype strings, on every backend.
    """

    @staticmethod
    def _sample_names(source):
        from sfsutils.io_handlers import ZarrVariantReader

        if source.endswith('.vcz'):
            return ZarrVariantReader(source).samples

        if source.endswith('.trees'):
            import tskit

            from sfsutils.io_handlers import TskitVariantReader

            return TskitVariantReader(tskit.load(source)).samples

        from cyvcf2 import VCF as CyVCF

        return list(CyVCF(source).samples)

    @staticmethod
    def _parse(source, **kwargs):
        import sfsutils as su
        from sfsutils.settings import Settings

        Settings.disable_pbar = True

        return su.Parser(source=source, skip_non_polarized=False, **kwargs).parse()

    @pytest.mark.parametrize('source,name', _sources())
    @pytest.mark.parametrize('subset', [False, True])
    def test_one_dimensional(self, source, name, subset, request):
        samples = self._sample_names(source)
        n = 4 if subset else 6
        kwargs = dict(n=n, include_samples=samples[::2] if subset else None)

        observed = self._parse(source, **kwargs).all.to_list()

        request.getfixturevalue('strings_only')
        expected = self._parse(source, **kwargs).all.to_list()

        assert observed == expected

    @pytest.mark.parametrize('source,name', _sources())
    def test_joint(self, source, name, request):
        samples = self._sample_names(source)

        if len(samples) < 4:
            pytest.skip("too few samples for a joint spectrum")

        pops = {'a': samples[:len(samples) // 2], 'b': samples[len(samples) // 2:]}
        kwargs = dict(n=4, pops=pops)

        observed = np.asarray(self._parse(source, **kwargs).all.data)

        request.getfixturevalue('strings_only')
        expected = np.asarray(self._parse(source, **kwargs).all.data)

        np.testing.assert_array_equal(observed, expected)

    @pytest.mark.parametrize('source,name', _sources())
    def test_two_sfs(self, source, name, request):
        kwargs = dict(n=4, two_sfs=True, d=1000)

        observed = np.asarray(self._parse(source, **kwargs).all.data)

        request.getfixturevalue('strings_only')
        expected = np.asarray(self._parse(source, **kwargs).all.data)

        np.testing.assert_array_equal(observed, expected)
