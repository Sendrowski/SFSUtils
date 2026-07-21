"""
Equivalence tests for the fast paths in :class:`~sfsutils.filtration.PolyAllelicFiltration`.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2026-07-20"

import os
from typing import List

import numpy as np

from sfsutils.filtration import PolyAllelicFiltration
from sfsutils.io_handlers import get_distinct_called_alleles

from testing import TestCase


def reference_filter_site(variant, samples_mask: np.ndarray | None) -> bool:
    """
    Decide the poly-allelic verdict by decoding the called bases of every included sample.

    :param variant: The variant to filter.
    :param samples_mask: The samples mask, or ``None`` for all samples.
    :return: ``True`` if the variant is not poly-allelic, ``False`` otherwise.
    """
    genotypes = variant.gt_bases if samples_mask is None else variant.gt_bases[samples_mask]

    return len(get_distinct_called_alleles(genotypes)) < 3


def make_filtration(samples_mask: np.ndarray | None) -> PolyAllelicFiltration:
    """
    Build a filtration with a given samples mask already in place.

    :param samples_mask: The samples mask, or ``None`` for all samples.
    :return: The filtration.
    """
    f = PolyAllelicFiltration()
    f._samples_mask = samples_mask

    return f


def masks(n: int) -> List[np.ndarray | None]:
    """
    A selection of samples masks covering the no-mask, all, strided and singleton cases.

    :param n: The number of samples.
    :return: The masks.
    """
    every_other = np.zeros(n, dtype=bool)
    every_other[::2] = True

    single = np.zeros(n, dtype=bool)
    single[0] = True

    last = np.zeros(n, dtype=bool)
    last[-1] = True

    return [None, np.ones(n, dtype=bool), every_other, single, last, np.zeros(n, dtype=bool)]


#: Hand-built sites exercising the multi-ALT cases the numeric genotype codes cannot settle on their own
multi_allelic_vcf = """##fileformat=VCFv4.2
##contig=<ID=1,length=10000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\ts2\ts3
1\t1\t.\tA\tC,G\t.\t.\t.\tGT\t0|0\t1|1\t2|2\t0|0
1\t2\t.\tA\tC,G\t.\t.\t.\tGT\t0|0\t1|1\t0|0\t2|2
1\t3\t.\tA\tC,G\t.\t.\t.\tGT\t0|0\t0|0\t0|0\t0|0
1\t4\t.\tA\tC,G\t.\t.\t.\tGT\t0|1\t0|2\t0|0\t.|.
1\t5\t.\tA\tC,G\t.\t.\t.\tGT\t.|1\t.|.\t0|0\t1|2
1\t6\t.\tA\tC,G\t.\t.\t.\tGT\t1|2\t1|1\t1|1\t1|1
1\t7\t.\tAT\tC,G\t.\t.\t.\tGT\t0|0\t1|1\t0|1\t.|2
1\t8\t.\tA\tC,N\t.\t.\t.\tGT\t0|0\t2|2\t1|1\t0|0
1\t9\t.\tN\tC,G\t.\t.\t.\tGT\t0|0\t1|1\t0|0\t2|2
1\t10\t.\tA\tAT\t.\t.\t.\tGT\t0|0\t1|1\t0|1\t0|0
1\t11\t.\tA\tAT,ATT\t.\t.\t.\tGT\t0|0\t1|1\t0|1\t2|2
1\t12\t.\tA\tC,G\t.\t.\t.\tGT\t.|0\t0|.\t0|0\t0|0
1\t13\t.\tA\tC,G\t.\t.\t.\tGT\t.|.\t.|.\t.|.\t.|.
1\t14\t.\tA\tC,G,T\t.\t.\t.\tGT\t1|1\t1|1\t2|2\t3|3
1\t15\t.\tA\tC,G\t.\t.\t.\tGT\t1|1\t1|1\t1|1\t2|2
1\t16\t.\tA\t<NON_REF>,C\t.\t.\t.\tGT\t0|0\t1|1\t2|2\t0|0
"""


def write_multi_allelic() -> str:
    """
    Write the hand-built multi-allelic sites to a VCF file.

    :return: Path to the VCF file.
    """
    path = os.path.join('scratch', 'poly_allelic_multi.vcf')

    with open(path, 'w') as fh:
        fh.write(multi_allelic_vcf)

    return path


class PolyAllelicFiltrationEquivalenceTestCase(TestCase):
    """
    Assert that the fast poly-allelic filter reaches the verdict of a full base-decoding implementation.
    """

    @staticmethod
    def _check(vcf: str):
        """
        Compare the verdicts of both implementations over every site and every mask.

        :param vcf: Path to the VCF file.
        """
        from cyvcf2 import VCF

        n = len(VCF(vcf).samples)

        for mask in masks(n):
            f = make_filtration(mask)

            n_sites = 0

            for variant in VCF(vcf):
                expected = reference_filter_site(variant, mask)
                n_sites += 1

                assert f.filter_site(variant) == expected, \
                    f'Disagreement at {variant.CHROM}:{variant.POS} for mask {mask}.'

            assert n_sites > 0

    def test_agrees_with_base_decoding_on_msprime_fixture(self):
        """
        The verdicts agree on every site of a real fixture.
        """
        self._check('resources/msprime/two_epoch.vcf')

    def test_agrees_with_base_decoding_on_multi_allelic_sites(self):
        """
        The verdicts agree on hand-built multi-allelic sites.
        """
        self._check(write_multi_allelic())

    def test_multi_allelic_verdicts(self):
        """
        Spot-check the verdicts that motivate the masked branch.
        """
        from cyvcf2 import VCF

        variants = {v.POS: v for v in VCF(write_multi_allelic())}

        all_samples = np.ones(4, dtype=bool)

        # two samples homozygous for different alternate alleles make the site poly-allelic
        assert not make_filtration(all_samples).filter_site(variants[1])

        # the third allele sits in an excluded sample only
        first_two = np.array([True, True, False, False])
        assert make_filtration(first_two).filter_site(variants[2])
        assert not make_filtration(all_samples).filter_site(variants[2])

        # alternate alleles present in the record but absent from all calls
        assert make_filtration(all_samples).filter_site(variants[3])

        # no call at all
        assert make_filtration(all_samples).filter_site(variants[13])

        # a bi-allelic MNP is kept whatever the mask
        assert make_filtration(all_samples).filter_site(variants[10])
        assert make_filtration(np.array([False, True, True, False])).filter_site(variants[10])

    def test_no_mask_covers_every_sample(self):
        """
        Without a samples mask the verdict is the one a mask naming every sample reaches.
        """
        from cyvcf2 import VCF

        path = write_multi_allelic()

        f = make_filtration(None)
        g = make_filtration(np.ones(len(VCF(path).samples), dtype=bool))

        for variant in VCF(path):
            assert f.filter_site(variant) == g.filter_site(variant)
