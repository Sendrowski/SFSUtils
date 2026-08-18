from functools import cached_property
from typing import List
from unittest.mock import Mock

import numpy as np
import pytest

import sfsutils as su
from sfsutils.io_handlers import count_sites
from testing import TestCase, requires
import pandas as pd
from sfsutils.annotation import DegeneracyAnnotation
from sfsutils.filtration import CodingSequenceFiltration
from sfsutils.io_handlers import MultiHandler, DummyVariant, SiteAlleles
from sfsutils.settings import Settings
import os
import subprocess
import sys
from sfsutils.filtration import CpGFiltration
from sfsutils.io_handlers import MultiHandler, SiteAlleles, ZarrVariantReader
import logging
from sfsutils.annotation import _CDSIndex
from sfsutils.filtration import CodingSequenceFiltration, CpGFiltration, PolyAllelicFiltration, SNPFiltration
from sfsutils.io_handlers import get_called_alleles
from sfsutils.spectrum import Spectrum, Spectra
from sfsutils.io_handlers import Variant, DummyVariant
from sfsutils.json_handlers import DataframeHandler


def _string_site(**attributes):
    """
    A site carrying its genotypes as strings alone, so the filtrations decide it from ``gt_bases``.

    :param attributes: The site's attributes.
    :return: The site.
    """
    attributes['gt_bases'] = np.array(attributes['gt_bases'], dtype=object)

    return type('_Site', (), attributes)()


class FiltrationTestCase(TestCase):
    """
    Test the filterer and filtration classes.
    """

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    @staticmethod
    def test_filter_snp_filtration():
        """
        Test the SNP filtration.
        """
        f = su.Filterer(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_filter_snp_filtration.vcf',
            filtrations=[su.SNPFiltration()],
        )

        f.filter()

        # assert no sites were filtered
        assert f.n_filtered == 2

        # assert number of sites is the same
        assert count_sites(f.vcf) == count_sites(f.output) + f.n_filtered

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_snp_filtration_use_sample_mask_from_parser_if_specified(self):
        """
        Make sure the SNP filtration uses the sample mask from the parser.
        """
        f = su.SNPFiltration(
            use_parser=True,
        )

        p = su.Parser(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            filtrations=[f],
            n=10,
            include_samples=['ASP01', 'ASP02', 'ASP03'],
        )

        p._setup()

        self.assertEqual(3, sum(p._samples_mask))

        np.testing.assert_array_equal(f._samples_mask, p._samples_mask)

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_snp_filtration_dont_use_sample_mask_from_parser_if_specified(self):
        """
        Make sure the SNP filtration doesn't use the sample mask from the parser.
        """
        f = su.SNPFiltration(
            use_parser=False,
        )

        p = su.Parser(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            filtrations=[f],
            n=10,
            include_samples=['ASP01', 'ASP02', 'ASP03'],
        )

        p._setup()

        self.assertEqual(3, sum(p._samples_mask))

        self.assertEqual(None, f._samples_mask)

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_snp_filtration_include(self):
        """
        Make sure the SNP filtration uses the sample mask from the parser.
        """
        f = su.SNPFiltration(
            include_samples=['ASP01', 'ASP02', 'ASP03']
        )

        filterer = su.Filterer(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_snp_filtration_include.vcf',
            filtrations=[f],
        )

        filterer._setup()

        self.assertEqual(3, sum(f._samples_mask))
        self.assertEqual(377, len(f._samples_mask))

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    @staticmethod
    def test_filter_no_poly_allelic_filtration():
        """
        Test the no poly-allelic filtration.
        """
        f = su.Filterer(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_filter_no_poly_allelic_filtration.vcf',
            filtrations=[su.PolyAllelicFiltration()],
        )

        f.filter()

        # assert no sites were filtered
        assert f.n_filtered == 0

        # assert number of sites is the same
        assert count_sites(f.vcf) == count_sites(f.output)

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    @staticmethod
    def test_annotator_load_vcf_from_url():
        """
        Test the annotator loading a VCF from a URL.
        """
        f = su.Filterer(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_filterer_load_vcf_from_url.vcf',
            filtrations=[su.PolyAllelicFiltration()]
        )

        f.filter()

        # assert number of sites is the same
        assert f.n_sites == 10000
        assert f.n_filtered == 0

    @requires('resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz')
    @staticmethod
    def test_deviant_outgroup_filtration():
        """
        Test the annotator loading a VCF from a URL.
        """
        f = su.Filterer(
            source="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_deviant_outgroup_filtration.vcf',
            filtrations=[su.DeviantOutgroupFiltration(outgroups=["ERR2103730", "ERR2103731"])]
        )

        f.filter()

        # assert number of sites is the same
        assert f.n_sites == 10000
        assert f.n_filtered == 510

    @staticmethod
    def test_deviant_outgroup_filtration_single_site():
        """
        Test the annotator loading a VCF from a URL.
        """
        from cyvcf2 import Variant

        mock_variant = Mock(spec=Variant)

        # test case 1: variant is not an SNP
        mock_variant.is_snp = False
        filter_obj = su.DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_masks()
        assert filter_obj.filter_site(mock_variant)  # expect True as the variant is not SNP

        # test case 2: variant is an SNP, strict mode is enabled and no outgroup sample is present
        mock_variant.is_snp = True
        filter_obj = su.DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'], strict_mode=True)
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', './.'])
        assert not filter_obj.filter_site(mock_variant)  # expect False as no outgroup sample is present

        # test case 3: variant is an SNP, strict mode is disabled and no outgroup sample is present
        filter_obj = su.DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'], strict_mode=False)
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', './.'])
        assert filter_obj.filter_site(mock_variant)  # # expect True as strict mode off and no outgroup sample present

        # test case 4: variant is an SNP and outgroup base is different from ingroup base
        filter_obj = su.DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'T/T'])
        assert not filter_obj.filter_site(mock_variant)  # expect False as outgroup base is different from ingroup base

        # test case 5: variant is an SNP and outgroup base is same as ingroup base
        filter_obj = su.DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/A'])
        assert filter_obj.filter_site(mock_variant)  # expect True as outgroup base is same as ingroup base

        # test case 6: multiple ingroups and outgroups with matching major bases
        mock_variant = Mock(spec=Variant)
        mock_variant.is_snp = True
        filter_obj = su.DeviantOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'],
                                                  ingroups=['ingroup1', 'ingroup2'])
        filter_obj.samples = np.array(['ingroup1', 'ingroup2', 'outgroup1', 'outgroup2'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/T', 'A/A', 'A/G'])
        assert filter_obj.filter_site(mock_variant)  # expect True as major base 'A' is common in ingroup and outgroup

        # test case 7: multiple ingroups and outgroups with differing major bases
        mock_variant = Mock(spec=Variant)
        mock_variant.is_snp = True
        filter_obj = su.DeviantOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'],
                                                  ingroups=['ingroup1', 'ingroup2'])
        filter_obj.samples = np.array(['ingroup1', 'ingroup2', 'outgroup1', 'outgroup2'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/T', 'T/T', 'T/G'])
        assert not filter_obj.filter_site(mock_variant)  # expect False as major base 'A' in ingroup and 'T' in outgroup

        # test case 8: make sure we retain mono-allelic sites if retain_monomorphic is True
        mock_variant = Mock(spec=Variant)
        mock_variant.is_snp = False
        filter_obj = su.DeviantOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'],
                                                  ingroups=['ingroup1', 'ingroup2'], retain_monomorphic=True)
        filter_obj.samples = np.array(['ingroup1', 'ingroup2', 'outgroup1', 'outgroup2'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/A', 'T/T', 'T/T'])
        assert filter_obj.filter_site(mock_variant)  # expect True as major base 'A' in ingroup and 'T' in outgroup

        # test case 9: make sure we don't retain mono-allelic sites if retain_monomorphic is False
        mock_variant = Mock(spec=Variant)
        mock_variant.is_snp = False
        filter_obj = su.DeviantOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'],
                                                  ingroups=['ingroup1', 'ingroup2'], retain_monomorphic=False)
        filter_obj.samples = np.array(['ingroup1', 'ingroup2', 'outgroup1', 'outgroup2'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/A', 'T/T', 'T/T'])
        assert not filter_obj.filter_site(mock_variant)  # expect False as major base 'A' in ingroup and 'T' in outgroup

    @staticmethod
    def test_existing_outgroup_filtration_single_site_1_missing():
        """
        Test the existing outgroup filtration.
        """
        from cyvcf2 import Variant

        # test case 1: variants has one fully defined outgroup sample
        mock_variant = Mock(spec=Variant)
        filter_obj = su.ExistingOutgroupFiltration(outgroups=['outgroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['A/A', 'T/T'])
        assert filter_obj.filter_site(mock_variant)

        # test case 2: variants has one missing outgroup sample
        mock_variant = Mock(spec=Variant)
        filter_obj = su.ExistingOutgroupFiltration(outgroups=['outgroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['A/A', './.'])
        assert not filter_obj.filter_site(mock_variant)

        # test case 3: variants has one fully defined outgroup sample and one missing outgroup sample
        mock_variant = Mock(spec=Variant)
        filter_obj = su.ExistingOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'])
        filter_obj.samples = np.array(['outgroup1', 'ingroup1', 'outgroup2'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['./.', 'A/A', 'T/T'])
        assert not filter_obj.filter_site(mock_variant)

        # test case 4: variants has one outgroup sample with one missing allele
        mock_variant = Mock(spec=Variant)
        filter_obj = su.ExistingOutgroupFiltration(outgroups=['outgroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['A/A', 'T/.'])
        assert filter_obj.filter_site(mock_variant)

        # test case 5: variants has three outgroup samples with one missing allele each
        mock_variant = Mock(spec=Variant)
        filter_obj = su.ExistingOutgroupFiltration(outgroups=['outgroup1', 'outgroup2', 'outgroup3'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1', 'outgroup2', 'outgroup3'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['A/A', 'T/.', 'T/.', 'T/.'])
        assert filter_obj.filter_site(mock_variant)

    @staticmethod
    def test_existing_outgroup_filtration_with_varied_n_missing():
        """
        Test the existing outgroup filtration with shuffled masks, extra unused samples, and ingroups.
        """
        from cyvcf2 import Variant

        # test case 6: n=2, one missing outgroup, extra unused sample -> should pass
        mock_variant = Mock(spec=Variant)
        filter_obj = su.ExistingOutgroupFiltration(outgroups=['outgroup2', 'outgroup3'], n_missing=2)
        filter_obj.samples = np.array(['unused1', 'ingroup1', 'outgroup2', 'outgroup3', 'unused2'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['A/A', 'C/C', './.', 'T/T', 'G/G'])
        assert filter_obj.filter_site(mock_variant)

        # test case 7: n=2, exactly two outgroups missing, shuffled sample order -> should fail
        mock_variant = Mock(spec=Variant)
        filter_obj = su.ExistingOutgroupFiltration(outgroups=['outgroup3', 'outgroup1'], n_missing=2)
        filter_obj.samples = np.array(['ingroup1', 'outgroup3', 'unused1', 'outgroup1', 'unused2'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['A/A', './.', 'G/G', './.', 'T/T'])
        assert not filter_obj.filter_site(mock_variant)

        # test case 8: n=3, two missing outgroups, extra ingroup sample -> should pass
        mock_variant = Mock(spec=Variant)
        filter_obj = su.ExistingOutgroupFiltration(outgroups=['outgroup1', 'outgroup2', 'outgroup3'], n_missing=3)
        filter_obj.samples = np.array(['ingroup1', 'outgroup1', 'ingroup2', 'outgroup2', 'outgroup3'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['A/A', './.', 'C/C', './.', 'T/T'])
        assert filter_obj.filter_site(mock_variant)

        # test case 9: n=3, exactly three outgroups missing -> should fail
        mock_variant = Mock(spec=Variant)
        filter_obj = su.ExistingOutgroupFiltration(outgroups=['outgroup1', 'outgroup2', 'outgroup3'], n_missing=3)
        filter_obj.samples = np.array(['outgroup1', 'ingroup1', 'outgroup2', 'outgroup3', 'unused1'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['./.', 'A/A', './.', './.', 'G/G'])
        assert not filter_obj.filter_site(mock_variant)

        # test case 10: n=3, mixed missing and defined outgroups, with unused samples -> should pass
        mock_variant = Mock(spec=Variant)
        filter_obj = su.ExistingOutgroupFiltration(outgroups=['outgroup1', 'outgroup2', 'outgroup3'], n_missing=3)
        filter_obj.samples = np.array(['unused1', 'outgroup1', 'ingroup1', 'outgroup2', 'unused2', 'outgroup3'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['G/G', './.', 'A/A', 'T/T', 'C/C', './.'])
        assert filter_obj.filter_site(mock_variant)

    @staticmethod
    def test_snp_filtration():
        """
        Test the SNP filtration.
        """
        f = su.SNPFiltration()

        assert not f.filter_site(variant=_string_site(is_snp=False, REF='A', ALT=['T'], gt_bases=['A/A', 'A/T']))
        assert f.filter_site(variant=_string_site(is_snp=True, REF='A', ALT=['T'], gt_bases=['A/A', 'A/T']))

        # an alternate allele the ``ALT`` field declares but no sample carries leaves the site monomorphic
        assert not f.filter_site(variant=_string_site(is_snp=True, REF='A', ALT=['T'], gt_bases=['A/A', 'A/A']))

    @staticmethod
    def test_snv_filtration():
        """
        Test the SNV filtration.
        """
        f = su.SNVFiltration()

        assert not f.filter_site(variant=Mock(REF='AG', ALT=['A', 'G']))
        assert f.filter_site(variant=Mock(REF='A', ALT=['G']))
        assert f.filter_site(variant=Mock(REF='A', ALT=['G', 'C']))
        assert not f.filter_site(variant=Mock(REF='A', ALT=['GA']))
        assert f.filter_site(variant=Mock(REF='A', ALT=['G', 'C', 'T']))

    @staticmethod
    def test_no_poly_allelic_filtration():
        """
        Test the no poly-allelic filtration.
        """
        f = su.PolyAllelicFiltration()

        assert not f.filter_site(variant=_string_site(REF='A', ALT=['T', 'G'], gt_bases=['A/T', 'G/G']))
        assert not f.filter_site(variant=_string_site(REF='A', ALT=['T', 'G', 'C'], gt_bases=['A/T', 'G/C']))
        assert f.filter_site(variant=_string_site(REF='A', ALT=['T'], gt_bases=['A/T', 'A/A']))
        assert f.filter_site(variant=_string_site(REF='A', ALT=[], gt_bases=['A/A', 'A/A']))

        # a further alternate allele the ``ALT`` field declares but no sample carries is not counted
        assert f.filter_site(variant=_string_site(REF='A', ALT=['T', 'G'], gt_bases=['A/T', 'A/A']))

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_coding_sequence_filtration_raises_error_if_no_fasta_given(self):
        """
        Test the coding sequence filtration.
        """
        with self.assertRaises(ValueError) as error:
            f = su.Filterer(
                source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
                output='scratch/test_coding_sequence_filtration.vcf',
                filtrations=[su.CodingSequenceFiltration()],
            )

            f.filter()

            print(error)

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz', 'resources/genome/betula/genome.gff.gz')
    @staticmethod
    def test_coding_sequence_filtration():
        """
        Test the coding sequence filtration.
        """
        f = su.Filterer(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_coding_sequence_filtration.vcf',
            gff="resources/genome/betula/genome.gff.gz",
            filtrations=[su.CodingSequenceFiltration()],
        )

        f.filter()

        # assert no sites were filtered
        assert f.n_filtered == 6434

        # assert number of sites is the same
        assert count_sites(f.vcf) - f.n_filtered == count_sites(f.output)

    @staticmethod
    def test_cpg_filtration_is_cpg():
        """
        Test CpG-context detection, including dinucleotide boundaries, on mocked reference bases.
        """
        is_cpg = su.CpGFiltration._is_cpg

        # CpG context (positions are 1-based): C followed by G, or G preceded by C
        assert is_cpg("CG", 1, 'C') is True
        assert is_cpg("CG", 2, 'G') is True
        assert is_cpg("ACGT", 2, 'C') is True
        assert is_cpg("ACGT", 3, 'G') is True

        # not in CpG context
        assert is_cpg("CA", 1, 'C') is False  # C not followed by G
        assert is_cpg("AG", 2, 'G') is False  # G not preceded by C

        # boundaries: no neighbouring base on the relevant side
        assert is_cpg("ATC", 3, 'C') is False  # C at the last position (no next base)
        assert is_cpg("GAT", 1, 'G') is False  # G at the first position (no previous base)

    def test_cpg_filtration_filter_site_mocked(self):
        """
        Test that ``filter_site`` drops CpG sites using the contig from a mocked handler.
        """
        from types import SimpleNamespace

        # contig (1-based): 1:G 2:A 3:C 4:G 5:T 6:C 7:G 8:C 9:A 10:G
        handler = Mock()
        handler.get_aliases.return_value = ['chr1']
        handler.get_contig.return_value = "GACGTCGCAG"

        f = su.CpGFiltration()
        f._handler = handler

        def keep(pos, ref):
            return f.filter_site(SimpleNamespace(CHROM='chr1', POS=pos, REF=ref))

        # CpG sites are dropped (filter_site returns False)
        assert keep(3, 'C') is False  # C followed by G
        assert keep(4, 'G') is False  # G preceded by C
        assert keep(6, 'C') is False
        assert keep(7, 'G') is False

        # kept (True)
        assert keep(1, 'G') is True   # G at the first position (no previous base)
        assert keep(8, 'C') is True   # C followed by A
        assert keep(10, 'G') is True  # G preceded by A
        assert keep(2, 'A') is True   # reference base is not C/G

        # n_filtered counts only the dropped (CpG) sites
        assert f.n_filtered == 4

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_cpg_filtration_raises_error_if_no_fasta_given(self):
        """
        Test that the CpG filtration requires a FASTA.
        """
        with self.assertRaises(ValueError):
            su.Filterer(
                source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
                output='scratch/test_cpg_filtration_no_fasta.vcf',
                filtrations=[su.CpGFiltration()],
            ).filter()

    @pytest.mark.slow
    @pytest.mark.very_slow
    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz', 'resources/genome/betula/genome.fasta')
    @pytest.mark.inference
    @staticmethod
    def test_cpg_filtration():
        """
        Test the CpG filtration end-to-end, obtaining the FASTA from the filterer.
        """
        f = su.Filterer(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_cpg_filtration.vcf',
            fasta="resources/genome/betula/genome.fasta",
            filtrations=[su.CpGFiltration()],
        )

        f.filter()

        assert f.n_filtered == 1123

        # assert number of sites is consistent
        assert count_sites(f.vcf) - f.n_filtered == count_sites(f.output)

    @requires('resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz')
    @staticmethod
    def test_existing_outgroup_filtration():
        """
        Test the existing outgroup filtration.
        """
        f = su.Filterer(
            source="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_existing_outgroup_filtration.vcf',
            filtrations=[su.ExistingOutgroupFiltration(outgroups=["ERR2103730", "ERR2103731"])]
        )

        f.filter()

        # assert number of sites is the same
        assert f.n_sites == 10000
        assert f.n_filtered == 3638

    @staticmethod
    def test_biased_gc_conversion_filtration():
        """
        Test the SNV filtration.
        """
        f = su.BiasedGCConversionFiltration()

        class VariantMock:
            """
            Mock the variant class.
            """

            def __init__(self, REF: str, ALT: List[str]):
                self.REF = REF
                self.ALT = ALT

            @cached_property
            def is_snp(self):
                """

                :return:
                """
                return len(self.ALT) == 0 or self.REF != self.ALT[0]

        assert f.filter_site(variant=VariantMock(REF='A', ALT=['A']))
        assert f.filter_site(variant=VariantMock(REF='T', ALT=['T']))
        assert f.filter_site(variant=VariantMock(REF='T', ALT=[]))

        assert not f.filter_site(variant=VariantMock(REF='A', ALT=['G']))
        assert not f.filter_site(variant=VariantMock(REF='G', ALT=['A']))
        assert not f.filter_site(variant=VariantMock(REF='C', ALT=['T']))
        assert not f.filter_site(variant=VariantMock(REF='T', ALT=['G']))

        assert f.filter_site(variant=VariantMock(REF='C', ALT=['G']))
        assert f.filter_site(variant=VariantMock(REF='G', ALT=['C']))
        assert f.filter_site(variant=VariantMock(REF='A', ALT=['T']))
        assert f.filter_site(variant=VariantMock(REF='T', ALT=['A']))

    def test_contig_filtration(self):
        """
        Test the contig filtration.
        """
        f = su.ContigFiltration(contigs=['chr1'])

        self.assertTrue(f.filter_site(variant=Mock(CHROM='chr1')))
        self.assertFalse(f.filter_site(variant=Mock(CHROM='chr2')))


HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length=100000>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)


def _write_vcf(path, rows, samples):
    """
    Write a minimal VCF holding the given data rows.

    :param path: The path to write to.
    :param rows: The data rows, each a sequence of the nine fixed columns followed by the genotypes.
    :param samples: The sample names.
    :return: The path as a string.
    """
    columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples]

    path.write_text(HEADER + "#" + "\t".join(columns) + "\n" + "".join("\t".join(r) + "\n" for r in rows))

    return str(path)


def _sites(source):
    """
    Read every site of the given source.

    :param source: The variant source.
    :return: The handler and its sites.
    """
    handler = MultiHandler(source=source)

    return handler, list(handler._reader)


def _setup(filtration, samples):
    """
    Set the filtration up against the given sample names.

    :param filtration: The filtration.
    :param samples: The sample names.
    :return: The filtration.
    """
    handler = type("_Handler", (), {"_reader": type("_Reader", (), {"samples": list(samples)})()})()

    filtration._setup(handler)

    return filtration


class TestHaploidLaterAlleleOutgroups:
    """
    A haploid call of the third or a later allele is rendered as a missing genotype wherever the site's
    maximum ploidy is two, so the outgroup filtrations must decide from the numeric calls.
    """

    samples = ("a", "b", "c", "d")

    def test_existing_keeps_haploid_later_allele_call(self, tmp_path):
        """The outgroup carries the second alternate allele as a haploid call, so it is not missing."""
        vcf = _write_vcf(tmp_path / "haploid.vcf", [
            ["1", "1", ".", "T", "C,A", ".", ".", ".", "GT", "2", "0/0", "0/0", "0/1"],
        ], self.samples)

        handler, sites = _sites(vcf)

        # the genotype strings render the outgroup's call as './.'
        assert sites[0].gt_bases[0] == "./."

        f = _setup(su.ExistingOutgroupFiltration(["a"], n_missing=1), self.samples)

        assert f.filter_site(sites[0])

        handler._reader.close()

    def test_deviant_reads_haploid_later_allele_call(self, tmp_path):
        """The outgroup's major base is the second alternate allele, which the ingroup shares."""
        vcf = _write_vcf(tmp_path / "deviant.vcf", [
            ["1", "1", ".", "T", "C,A", ".", ".", ".", "GT", "2", "2/2", "2/2", "2/2"],
            ["1", "2", ".", "T", "C,A", ".", ".", ".", "GT", "2", "0/0", "0/0", "0/0"],
        ], self.samples)

        handler, sites = _sites(vcf)

        f = _setup(su.DeviantOutgroupFiltration(["a"], ingroups=["b", "c", "d"]), self.samples)

        # the ingroup is fixed for the same allele the outgroup carries, then for the reference
        assert [f.filter_site(v) for v in sites] == [True, False]

        handler._reader.close()

    def test_verdicts_agree_across_encodings(self, tmp_path):
        """A mixed-ploidy input reaches the same verdicts read as a VCF and as a VCF-Zarr store."""
        pytest.importorskip("bio2zarr")
        pytest.importorskip("zarr")

        from bio2zarr import vcf as bio2zarr_vcf

        Settings.disable_pbar = True

        samples = ["a", "b", "c", "d", "e", "o1", "o2"]
        rng = np.random.default_rng(42)
        rows = []

        for i in range(200):
            alt = ["C", "A", "G"][:rng.integers(1, 4)]
            n = len(alt) + 1
            genotypes = []

            for _ in samples:
                if rng.random() < 0.12:
                    genotypes.append("./." if rng.random() < 0.5 else ".")
                elif rng.random() < 0.35:
                    genotypes.append(str(rng.integers(0, n)))
                else:
                    genotypes.append(f"{rng.integers(0, n)}/{rng.integers(0, n)}")

            rows.append(["1", str(i * 10 + 1), ".", "T", ",".join(alt), ".", ".", ".", "GT", *genotypes])

        vcf = _write_vcf(tmp_path / "mixed.vcf", rows, samples)
        vcz = str(tmp_path / "mixed.vcz")

        bio2zarr_vcf.convert([vcf], vcz)

        def verdicts(source, make):
            handler, sites = _sites(source)
            f = make()
            f._setup(handler)
            out = [f.filter_site(v) for v in sites]
            handler._reader.close()

            return out

        for make in [
            lambda: su.ExistingOutgroupFiltration(["o1", "o2"], n_missing=1),
            lambda: su.DeviantOutgroupFiltration(["o1", "o2"], ingroups=["a", "b", "c", "d", "e"]),
        ]:
            assert verdicts(vcf, make) == verdicts(vcz, make)

    def test_deviant_retains_monomorphic_and_dummy_sites(self, tmp_path):
        """The monomorphic shortcut and the dummy target site keep their verdicts."""
        vcf = _write_vcf(tmp_path / "mono.vcf", [
            ["1", "1", ".", "T", ".", ".", ".", ".", "GT", "./.", "0/0", "0/0", "0/0"],
        ], self.samples)

        handler, sites = _sites(vcf)

        f = _setup(su.DeviantOutgroupFiltration(["a"], ingroups=["b", "c", "d"]), self.samples)

        assert f.filter_site(sites[0])

        # without the shortcut the missing outgroup fails the strict-mode test
        g = _setup(su.DeviantOutgroupFiltration(["a"], ingroups=["b", "c", "d"], retain_monomorphic=False),
                   self.samples)

        assert not g.filter_site(sites[0])

        dummy = DummyVariant(ref="A", pos=1, chrom="1", n_samples=4)

        assert g.filter_site(dummy)
        assert _setup(su.ExistingOutgroupFiltration(["a"]), self.samples).filter_site(dummy)

        handler._reader.close()

    def test_deviant_counts_multi_character_alleles_as_one(self, tmp_path):
        """An MNP is majority-counted as one allele per haplotype rather than one per base."""
        vcf = _write_vcf(tmp_path / "mnp.vcf", [
            ["1", "1", ".", "AT", "GC", ".", ".", ".", "GT", "1/1", "1/1", "0/0", "0/0"],
        ], self.samples)

        handler, sites = _sites(vcf)

        f = _setup(su.DeviantOutgroupFiltration(["a"], ingroups=["b", "c", "d"], retain_monomorphic=False),
                   self.samples)

        # the majority is the allele a haplotype carries, not one of the bases spelling it out
        view = SiteAlleles.from_site(sites[0])
        assert f._get_major_allele(view, f.outgroup_mask) == "GC"
        assert f._get_major_allele(view, f.ingroup_mask) == "AT"

        assert not f.filter_site(sites[0])

        handler._reader.close()


class TestPolyAllelicAgreement:
    """
    A parser and a filterer must reach the same poly-allelic verdict, which is decided by the alleles the
    included samples actually carry rather than by the ``ALT`` field.
    """

    samples = ("s1", "s2")

    def test_parser_and_filterer_agree(self, tmp_path):
        """A site declaring three alleles of which the samples carry three is dropped on both paths."""
        Settings.disable_pbar = True

        rows = [["1", str(10 + i), ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/1", "2/2"] for i in range(20)]
        vcf = _write_vcf(tmp_path / "polyallelic.vcf", rows, self.samples)

        direct = su.Parser(source=vcf, n=4, filtrations=[su.PolyAllelicFiltration()]).parse()

        assert direct.data.sum().sum() == 0

        out = str(tmp_path / "filtered.vcf.gz")

        su.Filterer(source=vcf, output=out, filtrations=[su.PolyAllelicFiltration()]).filter()

        assert su.Parser(source=out, n=4).parse().data.sum().sum() == 0

    def test_all_true_mask_and_no_mask_agree(self, tmp_path):
        """Naming every sample and naming none of them reach the same verdict."""
        vcf = _write_vcf(tmp_path / "mask.vcf", [
            ["1", "10", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/0", "1/1"],
            ["1", "11", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/1", "2/2"],
        ], self.samples)

        handler, sites = _sites(vcf)

        f = su.PolyAllelicFiltration(include_samples=list(self.samples))
        f._setup(handler)

        assert f._samples_mask.tolist() == [True, True]

        g = su.PolyAllelicFiltration()
        g._setup(handler)

        assert g._samples_mask is None

        # the third allele is only declared at the first site and carried at the second
        assert [f.filter_site(v) for v in sites] == [True, False]
        assert [g.filter_site(v) for v in sites] == [True, False]

        # a genuine restriction is kept
        h = su.PolyAllelicFiltration(include_samples=["s1"])
        h._setup(handler)

        assert h._samples_mask is not None and h._samples_mask.tolist() == [True, False]

        handler._reader.close()


class TestCodingSequenceIndex:
    """
    The coding sequence lookup must not scan the whole frame per advancing site, and must reach the
    verdicts the scan reached.
    """

    def _scan(self, filtration, v):
        """
        Reach the verdict by scanning the whole coding sequence frame.

        :param filtration: The filtration holding the handler and the cursor.
        :param v: The variant.
        :return: The verdict.
        """
        aliases = filtration._handler.get_aliases(v.CHROM)
        cds = filtration._handler._cds

        if filtration.cd is None or filtration.cd.seqid not in aliases or v.POS > filtration.cd.end:
            filtration.cd = pd.Series({
                'seqid': v.CHROM,
                'start': DegeneracyAnnotation._pos_mock,
                'end': DegeneracyAnnotation._pos_mock
            })

            found = cds[cds['seqid'].isin(aliases) & (cds['end'] >= v.POS)]

            if not found.empty:
                filtration.cd = found.iloc[0]

        return filtration.cd.seqid in aliases and filtration.cd.start <= v.POS <= filtration.cd.end

    def test_matches_the_scan_on_a_real_gff(self):
        """Every site of a real input reaches the verdict the scan reaches."""
        Settings.disable_pbar = True

        handler = MultiHandler(
            source='resources/genome/betula/all.polarized.subset.10000.vcf.gz',
            gff='resources/genome/betula/genome.gff.gz'
        )

        indexed = CodingSequenceFiltration()
        indexed._setup(handler)

        scanned = CodingSequenceFiltration()
        scanned._setup(handler)

        n = kept = 0
        for v in handler._reader:
            verdict = indexed.filter_site(v)
            assert verdict == self._scan(scanned, v)
            n += 1
            kept += bool(verdict)

        assert n > 0 and kept > 0

        handler._reader.close()

    def test_lookup_does_not_scan_the_frame(self):
        """The cost of an advance is independent of the number of coding sequences on other contigs."""
        Settings.disable_pbar = True

        handler = MultiHandler(
            source='resources/genome/betula/all.polarized.subset.10000.vcf.gz',
            gff='resources/genome/betula/genome.gff.gz'
        )

        f = CodingSequenceFiltration()
        f._setup(handler)

        cds = handler._cds
        seqid = cds.seqid.value_counts().index[0]
        positions = cds[cds.seqid == seqid].start.to_numpy()[:200]

        # one site per coding sequence, so the cursor advances on every one of them
        for pos in positions:
            f.filter_site(DummyVariant(ref='A', pos=int(pos), chrom=seqid))

        index = f._get_index(seqid, handler.get_aliases(seqid))

        # the index covers only the contig it was built for
        assert len(index.cds) < len(cds)
        assert set(index.cds.seqid) <= set(handler.get_aliases(seqid))

        handler._reader.close()


HEADER_TWO_CONTIGS = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length=100000>\n"
    "##contig=<ID=2,length=100000>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)


MIXED_ROWS = [
    ["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0", "0/1", "1/1", "./."],
    ["1", "11", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/0", "0/1", "0/0", "0/0"],
    ["1", "12", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/1", "2/2", "0/0", "./."],
    ["1", "13", ".", "AT", "GC", ".", ".", "AA=AT", "GT", "1/1", "1/1", "0/0", "0/0"],
    ["1", "14", ".", "AT", "GC,AA", ".", ".", "AA=AT", "GT", "2", "0/0", "0/0", "1/1"],
    ["1", "15", ".", "A", "AT,ATT", ".", ".", "AA=A", "GT", "2", "0/0", "1/1", "1/1"],
    ["1", "16", ".", "ACG", "A", ".", ".", "AA=ACG", "GT", "0/1", "0/0", "0/0", "1/1"],
    ["1", "17", ".", "A", "C,G,T", ".", ".", "AA=A", "GT", "1/1", "2/2", "3/3", "0/0"],
    ["1", "18", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0", "0/0", "0/0", "0/0"],
    ["1", "19", ".", "A", "<NON_REF>,C", ".", ".", "AA=A", "GT", "0/0", "1/1", "2/2", "./."],
    ["1", "20", ".", "A", "T,G", ".", ".", "AA=A", "GT", "./.", "./.", "./.", "./."],
    ["1", "21", ".", "N", "C,G", ".", ".", "AA=N", "GT", "0/0", "1/1", "0/0", "2/2"],
    ["1", "22", ".", "C", "G", ".", ".", "AA=C", "GT", "1", "0/1", "0/0", "1/1"],
]


MIXED_SAMPLES = ["s1", "s2", "s3", "s4"]


TRIO = [
    ("resources/msprime/two_epoch.vcf", "vcf"),
    ("resources/msprime/two_epoch.vcz", "zarr"),
    ("resources/msprime/two_epoch.trees", "tskit"),
]


def write_vcf(path, rows, samples, header=HEADER_TWO_CONTIGS):
    """
    Write a minimal VCF holding the given data rows.

    :param path: The path to write to.
    :param rows: The data rows, each a sequence of the nine fixed columns followed by the genotypes.
    :param samples: The sample names.
    :param header: The header to write.
    :return: The path as a string.
    """
    columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples]

    path.write_text(header + "#" + "\t".join(columns) + "\n" + "".join("\t".join(r) + "\n" for r in rows))

    return str(path)


def to_vcz(vcf, store):
    """
    Convert a VCF to a VCF-Zarr store.

    :param vcf: The path to the VCF.
    :param store: The path to write the store to.
    :return: The path to the store.
    """
    pytest.importorskip("bio2zarr")

    subprocess.run([sys.executable, "-m", "bio2zarr", "vcf2zarr", "convert", vcf, store],
                   capture_output=True, check=True)

    return store


def read(source):
    """
    Stream the sites of a source through the backend its extension selects.

    :param source: The path to the source.
    :return: The list of sites.
    """
    if source.endswith(".vcz") or source.endswith(".zarr"):
        # an explicit chunk size, so the reader's own default cannot vary the comparison
        return list(ZarrVariantReader(source, chunk_size=1000))

    if source.endswith(".trees"):
        import tskit

        from sfsutils.io_handlers import TskitVariantReader

        return list(TskitVariantReader(tskit.load(source)))

    from cyvcf2 import VCF

    return list(VCF(source))


def samples_of(source):
    """
    The sample names of a source.

    :param source: The path to the source.
    :return: The sample names.
    """
    if source.endswith(".vcz") or source.endswith(".zarr"):
        return list(ZarrVariantReader(source, chunk_size=1000).samples)

    if source.endswith(".trees"):
        import tskit

        from sfsutils.io_handlers import TskitVariantReader

        return list(TskitVariantReader(tskit.load(source)).samples)

    from cyvcf2 import VCF

    return list(VCF(source).samples)


def setup(filtration, samples):
    """
    Set the filtration up against the given sample names.

    :param filtration: The filtration.
    :param samples: The sample names.
    :return: The filtration.
    """
    handler = type("_Handler", (), {"_reader": type("_Reader", (), {"samples": list(samples)})()})()

    filtration._setup(handler)

    return filtration


def verdicts(source, build):
    """
    The verdicts a freshly built filtration reaches on every site of a source.

    :param source: The path to the source.
    :param build: A callable taking the sample names and returning the filtration.
    :return: The verdicts, one per site.
    """
    samples = samples_of(source)
    filtration = setup(build(), samples)

    return [filtration.filter_site(v) for v in read(source)]


class TestAllTrueMaskIsHonoured:
    """
    A mask selecting every sample must decide a site exactly as no mask does, so that a sample belonging
    to no requested population cannot flip a verdict (C11).
    """

    pops = {"p1": ["s1", "s2"], "p2": ["s3", "s4"]}

    def _total(self, vcf):
        """
        The total mass of the joint spectra of a parse dropping the poly-allelic sites.

        :param vcf: The path to the VCF.
        :return: The total mass.
        """
        Settings.disable_pbar = True

        spectra = su.Parser(source=vcf, n=4, pops=self.pops, filtrations=[su.PolyAllelicFiltration()]).parse()

        return float(sum(np.asarray(s.data).sum() for s in spectra.to_dict().values()))

    def test_unparsed_sample_does_not_change_the_verdict(self, tmp_path):
        """The populations carry two alleles, so the sites are kept whether or not a fifth sample exists."""
        genotypes = ["0/0", "0/1", "0/0", "0/0"]

        four = write_vcf(
            tmp_path / "four.vcf",
            [["1", str(10 + i), ".", "A", "T,G", ".", ".", "AA=A", "GT", *genotypes] for i in range(20)],
            MIXED_SAMPLES
        )

        five = write_vcf(
            tmp_path / "five.vcf",
            [["1", str(10 + i), ".", "A", "T,G", ".", ".", "AA=A", "GT", *genotypes, "0/0"] for i in range(20)],
            MIXED_SAMPLES + ["x1"]
        )

        assert self._total(four) == self._total(five) == 20

    def test_naming_every_sample_agrees_with_naming_none(self, tmp_path):
        """The two ways of asking for every sample reach the same verdict site by site."""
        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)

        for build in (su.PolyAllelicFiltration, su.SNPFiltration):
            named = verdicts(vcf, lambda: build(include_samples=list(MIXED_SAMPLES)))
            unnamed = verdicts(vcf, build)

            assert named == unnamed
            assert len(named) == len(MIXED_ROWS)

    def test_declared_but_uncalled_allele_is_not_counted(self, tmp_path):
        """A third allele the ``ALT`` field declares but no sample carries leaves the site bi-allelic."""
        vcf = write_vcf(tmp_path / "declared.vcf", [
            ["1", "10", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/0", "0/1"],
            ["1", "11", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/1", "2/2"],
        ], ["s1", "s2"])

        assert verdicts(vcf, su.PolyAllelicFiltration) == [True, False]

    def test_filterer_keeps_the_declared_but_uncalled_allele(self, tmp_path):
        """The filterer without a samples restriction reaches the same verdict as the parser."""
        Settings.disable_pbar = True

        vcf = write_vcf(tmp_path / "declared.vcf", [
            ["1", "10", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/0", "0/1"],
            ["1", "11", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/1", "2/2"],
        ], ["s1", "s2"])

        out = str(tmp_path / "filtered.vcf")

        su.Filterer(source=vcf, output=out, filtrations=[su.PolyAllelicFiltration()]).filter()

        assert [v.POS for v in read(out)] == [10]

    def test_snp_filtration_judges_from_the_genotypes(self, tmp_path):
        """A record declaring an alternate allele no sample carries is not polymorphic."""
        vcf = write_vcf(tmp_path / "mono.vcf", [
            ["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0", "0/0"],
            ["1", "11", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0", "0/1"],
        ], ["s1", "s2"])

        assert verdicts(vcf, su.SNPFiltration) == [False, True]


class TestDeviantOutgroupAlleles:
    """
    A multi-character allele must be majority-counted as one allele on every backend (C12).
    """

    def _build(self):
        """
        The filtration under test.

        :return: The filtration.
        """
        return su.DeviantOutgroupFiltration(["s1"], ingroups=["s2", "s3", "s4"], retain_monomorphic=False)

    def test_mnp_counts_as_one_allele(self, tmp_path):
        """The outgroup's ``GC`` call weighs one allele per haplotype, not one per base."""
        vcf = write_vcf(tmp_path / "mnp.vcf", [
            ["1", "10", ".", "AT", "GC", ".", ".", "AA=AT", "GT", "1/1", "0/0", "0/0", "0/0"],
            ["1", "11", ".", "AT", "GC", ".", ".", "AA=AT", "GT", "1/1", "1/1", "1/1", "0/0"],
        ], MIXED_SAMPLES)

        assert verdicts(vcf, self._build) == [False, True]

    def test_agrees_across_backends_on_mnps_and_indels(self, tmp_path):
        """The verdicts of the VCF and of the store built from it match site by site."""
        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)
        vcz = to_vcz(vcf, str(tmp_path / "mixed.vcz"))

        from_vcf = verdicts(vcf, self._build)
        from_vcz = verdicts(vcz, self._build)

        assert from_vcf == from_vcz
        assert len(from_vcf) == len(MIXED_ROWS)

        # the haploid call of a later allele at a multi-character site is where the two used to part
        assert from_vcf[4] is False


class TestCrossBackendEquivalence:
    """
    Every masked and outgroup filtration must reach the same verdicts on the same data, whichever backend
    presents it.
    """

    @staticmethod
    def _builders(samples):
        """
        The filtrations compared, each as a zero-argument builder.

        :param samples: The sample names of the source.
        :return: The named builders.
        """
        half = list(samples[:max(1, len(samples) // 2)])

        return {
            "snp": su.SNPFiltration,
            "snp-masked": lambda: su.SNPFiltration(include_samples=half),
            "poly": su.PolyAllelicFiltration,
            "poly-masked": lambda: su.PolyAllelicFiltration(include_samples=half),
            "deviant": lambda: su.DeviantOutgroupFiltration([samples[0]], retain_monomorphic=False),
            "existing": lambda: su.ExistingOutgroupFiltration([samples[0], samples[-1]], n_missing=1),
        }

    def test_mixed_fixture_agrees_between_vcf_and_zarr(self, tmp_path):
        """Mixed ploidy, multi-allelic records, MNPs, indels and missing calls agree on both backends."""
        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)
        vcz = to_vcz(vcf, str(tmp_path / "mixed.vcz"))

        compared = 0
        for name, build in self._builders(MIXED_SAMPLES).items():
            from_vcf = verdicts(vcf, build)
            from_vcz = verdicts(vcz, build)

            assert from_vcf == from_vcz, name
            assert len(from_vcf) == len(MIXED_ROWS)

            compared += len(from_vcf)

        assert compared == 6 * len(MIXED_ROWS)

    def test_msprime_trio_agrees_on_every_backend(self):
        """The committed VCF, store and tree sequence hold the same data and must be judged alike."""
        available = [(path, name) for path, name in TRIO if os.path.exists(path)]

        if len(available) < 2:
            pytest.skip("the msprime fixtures are not available")

        samples = samples_of(available[0][0])
        reference = None

        for path, name in available:
            assert samples_of(path) == samples, name

            for label, build in self._builders(samples).items():
                got = verdicts(path, build)

                if reference is None:
                    reference = {}

                assert reference.setdefault(label, got) == got, f"{name} disagrees on {label}"

        assert sum(len(v) for v in reference.values()) > 1000


class TestMissingContig:
    """
    A contig the FASTA carries no sequence for must not abort the run (C13).
    """

    def _fasta(self, tmp_path):
        """
        A single-contig reference.

        :param tmp_path: The temporary directory.
        :return: The path to the FASTA.
        """
        path = tmp_path / "ref.fasta"
        path.write_text(">1\n" + "ACGT" * 25 + "\n")

        return str(path)

    def test_parse_survives_an_absent_contig(self, tmp_path, caplog):
        """The sites on the absent contig are kept and the rest of the parse completes."""
        Settings.disable_pbar = True

        vcf = write_vcf(tmp_path / "two_contigs.vcf", [
            ["1", "2", ".", "C", "T", ".", ".", "AA=C", "GT", "0/0", "0/1"],
            ["2", "2", ".", "C", "T", ".", ".", "AA=C", "GT", "0/0", "0/1"],
            ["2", "3", ".", "G", "A", ".", ".", "AA=G", "GT", "0/0", "0/1"],
        ], ["s1", "s2"])

        f = CpGFiltration()
        handler = MultiHandler(source=vcf, fasta=self._fasta(tmp_path))
        f._setup(handler)

        sites = read(vcf)

        # the reference reads ACGTACGT..., so position 2 of contig 1 is the only CpG site
        assert [f.filter_site(v) for v in sites] == [False, True, True]

        # the absent contig is warned about once, not once per site
        assert f._missing_contigs == {"2"}

        handler._reader.close()

    def test_parser_completes_over_an_absent_contig(self, tmp_path):
        """A whole parse runs through rather than being discarded by the one absent scaffold."""
        Settings.disable_pbar = True

        vcf = write_vcf(tmp_path / "two_contigs.vcf", [
            ["1", "5", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0", "0/1"],
            ["2", "2", ".", "C", "T", ".", ".", "AA=C", "GT", "0/0", "0/1"],
        ], ["s1", "s2"])

        sfs = su.Parser(source=vcf, n=2, fasta=self._fasta(tmp_path), filtrations=[CpGFiltration()]).parse()

        assert float(np.asarray(sfs.data).sum()) == 2


class TestMaxSites:
    """
    ``max_sites`` must bound the output rather than being reached only on an exact hit (C10).
    """

    def test_zero_is_rejected(self, tmp_path):
        """A non-positive bound would silently mean no bound at all."""
        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)

        for value in (0, -1):
            with pytest.raises(ValueError, match="max_sites must be positive"):
                su.Filterer(source=vcf, output=str(tmp_path / "out.vcf"), max_sites=value)

    def test_bound_is_honoured(self, tmp_path):
        """The output stops at the bound."""
        Settings.disable_pbar = True

        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)
        out = str(tmp_path / "out.vcf")

        su.Filterer(source=vcf, output=out, max_sites=3, filtrations=[su.NoFiltration()]).filter()

        assert len(read(out)) == 3


class TestExistingOutgroupBatching:
    """
    The batched missing-outgroup count must reach the verdicts the per-outgroup count reached (P7).
    """

    @staticmethod
    def _reference(filtration, variant, n_missing):
        """
        Count the missing outgroups one call into the numeric view at a time.

        :param filtration: The filtration holding the outgroup rows.
        :param variant: The site.
        :param n_missing: The number of missing outgroups required to fail.
        :return: The verdict.
        """
        site = SiteAlleles.from_site(variant)
        rows = filtration._outgroup_rows

        return sum(site.n_called(rows[i:i + 1]) == 0 for i in range(len(rows))) < n_missing

    @pytest.mark.parametrize("n_missing", [1, 2, 3])
    def test_verdicts_are_unchanged(self, tmp_path, n_missing):
        """Every site of the mixed fixture reaches the verdict of the per-outgroup count."""
        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)

        f = setup(su.ExistingOutgroupFiltration(list(MIXED_SAMPLES), n_missing=n_missing), MIXED_SAMPLES)

        compared = 0
        for v in read(vcf):
            assert f.filter_site(v) == self._reference(f, v, n_missing)
            compared += 1

        assert compared == len(MIXED_ROWS)

    def test_batching_does_not_rebin_once_per_outgroup(self, monkeypatch):
        """The cost of a site no longer grows with a re-binning per outgroup.

        The binning passes are counted rather than timed: the filtration settles every outgroup in one pass over
        their rows, while asking the view sample by sample bins the whole site once per outgroup."""
        if not os.path.exists(TRIO[0][0]):
            pytest.skip("the msprime fixture is not available")

        sites = read(TRIO[0][0])
        samples = samples_of(TRIO[0][0])

        n_binned = 0
        original = SiteAlleles._count

        def counting(self, mask):
            nonlocal n_binned
            n_binned += 1

            return original(self, mask)

        monkeypatch.setattr(SiteAlleles, '_count', counting)

        f = setup(su.ExistingOutgroupFiltration(list(samples), n_missing=1), samples)

        n_sites = 0
        for v in sites:
            f.filter_site(v)
            n_sites += 1

        batched = n_binned
        n_binned = 0

        for v in sites:
            self._reference(f, v, 1)
        per_outgroup = n_binned

        assert n_sites > 0
        assert batched == 0  # no binning pass at all, the rows are read directly
        assert per_outgroup == n_sites * len(f._outgroup_rows)


HEADER_SHORT_CONTIG = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length=1000>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)


GFF = "##gff-version 3\n1\tx\tCDS\t1\t18\t.\t+\t0\tID=c1;Parent=t1\n"


GFF_NO_PHASE = "##gff-version 3\n1\tx\tCDS\t1\t18\t.\t+\t.\tID=c1;Parent=t1\n"


def write_vcf_sites_or_samples(path, rows, samples, header=HEADER_SHORT_CONTIG):
    """
    Write a minimal VCF holding the given data rows, omitting the genotype columns for no samples.

    :param path: The path to write to.
    :param rows: The data rows, each a sequence of the eight fixed columns followed by the genotypes.
    :param samples: The sample names, empty for a sites-only file.
    :param header: The header to write.
    :return: The path as a string.
    """
    columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]

    if samples:
        columns += ["FORMAT", *samples]

    path.write_text(header + "#" + "\t".join(columns) + "\n" + "".join("\t".join(r) + "\n" for r in rows))

    return str(path)


def written(path):
    """
    Read back the data lines of a VCF.

    :param path: The path to read.
    :return: The data lines, split on tabs.
    """
    with open(path) as fh:
        return [line.rstrip("\n").split("\t") for line in fh if not line.startswith("#")]


def degeneracies(path):
    """
    Read back the degeneracy annotation of every record of a VCF.

    :param path: The path to read.
    :return: The ``Degeneracy`` and ``Degeneracy_Info`` values per record.
    """
    return [
        tuple(f for f in record[7].split(";") if f.startswith("Degeneracy"))
        for record in written(path)
    ]


@pytest.fixture
def short_fasta(tmp_path):
    """
    A FASTA whose only record is far shorter than the contig length the VCF header declares.

    :param tmp_path: The temporary directory.
    :return: The path to the FASTA.
    """
    path = tmp_path / "short.fa"
    path.write_text(">1\nACGTACGTAC\n")

    return str(path)


class TestCpGShortContig:
    """
    A reference sequence not reaching a site must not abort the run, whatever the reference base is.
    """

    @pytest.mark.parametrize("ref", ["C", "G"])
    def test_site_beyond_end_of_contig_is_retained(self, tmp_path, short_fasta, ref, caplog):
        """
        Both reference bases have to survive a FASTA record that stops short of the site.
        """
        vcf = write_vcf_sites_or_samples(tmp_path / "in.vcf",
                        [["1", "500", ".", ref, "A", ".", ".", f"AA={ref}", "GT", "0/1"]], ["s1"])

        out = str(tmp_path / f"out_{ref}.vcf")

        f = su.Filterer(source=vcf, output=out, fasta=short_fasta, filtrations=[CpGFiltration()])

        with caplog.at_level(logging.WARNING, logger="sfsutils"):
            f.filter()

        assert f.n_filtered == 0
        assert len(written(out)) == 1
        assert "beyond the end of contig 1" in caplog.text

    def test_short_contig_warns_once(self, tmp_path, short_fasta, caplog):
        """
        The warning is per contig, not per site.
        """
        rows = [["1", str(pos), ".", "G", "A", ".", ".", "AA=G", "GT", "0/1"] for pos in (500, 600, 700)]
        vcf = write_vcf_sites_or_samples(tmp_path / "in.vcf", rows, ["s1"])

        out = str(tmp_path / "out.vcf")

        f = su.Filterer(source=vcf, output=out, fasta=short_fasta, filtrations=[CpGFiltration()])

        with caplog.at_level(logging.WARNING, logger="sfsutils"):
            f.filter()

        assert len(written(out)) == 3
        assert caplog.text.count("beyond the end of contig") == 1

    def test_cpg_context_within_the_sequence_is_unchanged(self, tmp_path):
        """
        The bounds check must not change the verdict where the sequence does reach the site. ``ACGTACGTAC``
        has a CpG at positions 2 (C) and 3 (G), and the terminal bases have no neighbour to be typed by.
        """
        fasta = tmp_path / "ref.fa"
        fasta.write_text(">1\nACGTACGTAC\n")

        assert CpGFiltration._is_cpg("ACGTACGTAC", 2, "C") is True
        assert CpGFiltration._is_cpg("ACGTACGTAC", 3, "G") is True
        assert CpGFiltration._is_cpg("ACGTACGTAC", 6, "C") is True
        assert CpGFiltration._is_cpg("ACGTACGTAC", 10, "C") is False
        assert CpGFiltration._is_cpg("GCGTACGTAC", 1, "G") is False
        assert CpGFiltration._is_cpg("ACGTACGTAC", 11, "G") is None
        assert CpGFiltration._is_cpg("ACGTACGTAC", 0, "C") is None


class TestSitesOnlyInput:
    """
    With no sample to carry an allele, the masked filtrations have to fall back to the declared alleles
    rather than judging every site monomorphic (dropping all of them) or bi-allelic (keeping all of them).
    """

    ROWS = [
        ["1", "10", ".", "A", "T", ".", ".", "AA=A"],
        ["1", "20", ".", "A", "T,G", ".", ".", "AA=A"],
        ["1", "30", ".", "A", ".", ".", ".", "AA=A"],
    ]

    def test_snp_filtration_keeps_declared_snps(self, tmp_path, caplog):
        """
        The two SNPs survive and the monomorphic site does not.
        """
        vcf = write_vcf_sites_or_samples(tmp_path / "sites.vcf", self.ROWS, [])
        out = str(tmp_path / "out.vcf")

        f = su.Filterer(source=vcf, output=out, filtrations=[SNPFiltration()])

        with caplog.at_level(logging.WARNING, logger="sfsutils"):
            f.filter()

        assert [r[1] for r in written(out)] == ["10", "20"]
        assert f.n_filtered == 1
        assert "falls back to the alleles declared" in caplog.text

    def test_poly_allelic_filtration_drops_declared_poly_allelic_sites(self, tmp_path):
        """
        The tri-allelic site is dropped rather than kept.
        """
        vcf = write_vcf_sites_or_samples(tmp_path / "sites.vcf", self.ROWS, [])
        out = str(tmp_path / "out.vcf")

        f = su.Filterer(source=vcf, output=out, filtrations=[PolyAllelicFiltration()])
        f.filter()

        assert [r[1] for r in written(out)] == ["10", "30"]
        assert f.n_filtered == 1

    def test_verdicts_match_the_same_data_with_samples(self, tmp_path):
        """
        The fallback has to reach the verdict samples carrying every declared allele would produce.
        """
        genotypes = [["0/0", "0/1"], ["0/1", "2/2"], ["0/0", "0/0"]]
        with_samples = [row + ["GT", *gt] for row, gt in zip(self.ROWS, genotypes)]

        typed = write_vcf_sites_or_samples(tmp_path / "typed.vcf", with_samples, ["s1", "s2"])
        sites = write_vcf_sites_or_samples(tmp_path / "sites.vcf", self.ROWS, [])

        for filtration in (SNPFiltration, PolyAllelicFiltration):
            kept = []

            for name, source in (("typed", typed), ("sites", sites)):
                out = str(tmp_path / f"{filtration.__name__}_{name}.vcf")

                su.Filterer(source=source, output=out, filtrations=[filtration()]).filter()

                kept.append([r[1] for r in written(out)])

            assert kept[0] == kept[1]

    def test_samples_are_still_used_where_there_are_any(self, tmp_path):
        """
        An alternate allele no sample carries must keep making a site monomorphic, as before.
        """
        rows = [["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0"]]
        vcf = write_vcf_sites_or_samples(tmp_path / "typed.vcf", rows, ["s1"])
        out = str(tmp_path / "out.vcf")

        f = su.Filterer(source=vcf, output=out, filtrations=[SNPFiltration()])
        f.filter()

        assert written(out) == []
        assert f.n_filtered == 1


class TestCodingSequenceFiltrationRewind:
    """
    The processed count belongs to one pass, as every other counter of the filtration does.
    """

    def test_n_processed_is_reset_on_rewind(self):
        """
        A rewind restores the count the warning about a mismatched GFF is guarded by.
        """
        f = CodingSequenceFiltration()
        f.n_processed = 17

        f._rewind()

        assert f.n_processed == 0

    def test_mismatched_gff_warns_on_every_pass(self, tmp_path, caplog):
        """
        A shared filtration must warn about a GFF whose contigs do not match on the second pass too.
        """
        gff = tmp_path / "other.gff"
        gff.write_text("##gff-version 3\nother\tx\tCDS\t1\t18\t.\t+\t0\tID=c1;Parent=t1\n")

        rows = [["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/1"]]
        vcf = write_vcf_sites_or_samples(tmp_path / "in.vcf", rows, ["s1"])

        filtration = CodingSequenceFiltration()

        for i in range(2):
            caplog.clear()

            out = str(tmp_path / f"out{i}.vcf")

            with caplog.at_level(logging.WARNING, logger="sfsutils"):
                su.Filterer(source=vcf, output=out, gff=str(gff), filtrations=[filtration]).filter()

            # the teardown rewinds the filtration, so the count is back at the start of the next pass
            assert filtration.n_processed == 0
            assert "No subsequent coding sequence found" in caplog.text


class TestRepeatedPasses:
    """
    A second pass over an input has to produce what the first one did, rather than a header-only file.
    """

    def test_filter_can_be_called_twice(self, tmp_path):
        """
        The second ``filter()`` writes the same records and reports its own count.
        """
        rows = [["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/1"],
                ["1", "20", ".", "A", "T,G", ".", ".", "AA=A", "GT", "1/2"]]

        vcf = write_vcf_sites_or_samples(tmp_path / "in.vcf", rows, ["s1"])

        f = su.Filterer(source=vcf, output=str(tmp_path / "out1.vcf"), filtrations=[PolyAllelicFiltration()])
        f.filter()

        first, n_first = written(str(tmp_path / "out1.vcf")), f.n_filtered

        f.output = str(tmp_path / "out2.vcf")
        f.filter()

        assert written(str(tmp_path / "out2.vcf")) == first
        assert f.n_filtered == n_first

    def test_filter_releases_the_reader_when_setup_fails(self, tmp_path):
        """
        Discarding the reader of a previous pass must release it rather than leak it, so that a setup
        failure still leaves nothing open.
        """

        class RecordingReader:
            """A stand-in reader that records whether it was closed."""

            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        rows = [["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/1"]]
        vcf = write_vcf_sites_or_samples(tmp_path / "in.vcf", rows, ["s1"])

        # a tree sequence cannot be written to, so the writer the setup opens raises
        f = su.Filterer(source=vcf, output=str(tmp_path / "out.trees"), filtrations=[])

        reader = RecordingReader()
        f.__dict__["_reader"] = reader

        with pytest.raises(ValueError):
            f.filter()

        assert reader.closed

    def test_filter_recovers_from_a_failed_pass(self, tmp_path):
        """
        A pass that raised must not stop the next one from reading the input from the start.
        """
        rows = [["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/1"]]
        vcf = write_vcf_sites_or_samples(tmp_path / "in.vcf", rows, ["s1"])

        f = su.Filterer(source=vcf, output=str(tmp_path / "out.trees"), filtrations=[])

        with pytest.raises(ValueError):
            f.filter()

        f.output = str(tmp_path / "out.vcf")
        f.filter()

        assert len(written(f.output)) == 1

    def test_annotate_can_be_called_twice(self, tmp_path):
        """
        The second ``annotate()`` writes the same records rather than a header and a raise.
        """
        gff = tmp_path / "cds.gff"
        gff.write_text(GFF)

        fasta = tmp_path / "ref.fa"
        fasta.write_text(">1\nATGGTTTTTCGTTATTAA\n")

        rows = [["1", "6", ".", "T", "C", ".", ".", "AA=T", "GT", "0/1"]]
        vcf = write_vcf_sites_or_samples(tmp_path / "in.vcf", rows, ["s1"])

        a = su.Annotator(source=vcf, output=str(tmp_path / "out1.vcf"), fasta=str(fasta), gff=str(gff),
                         annotations=[su.DegeneracyAnnotation()])
        a.annotate()

        first = degeneracies(str(tmp_path / "out1.vcf"))

        a.output = str(tmp_path / "out2.vcf")
        a.annotate()

        assert degeneracies(str(tmp_path / "out2.vcf")) == first
        assert first == [("Degeneracy=4", "Degeneracy_Info=2,+,GTT")]


class TestSharedAnnotationAcrossInputs:
    """
    An annotation reused across inputs must annotate each of them against its own reference.
    """

    #: The two genomes differ only in the second codon, ``GTT`` (4-fold at its third position) against
    #: ``ATT`` (2-fold), so a stale reference sequence books a selected site as a neutral one
    GENOMES = {"A": ("ATGGTTTTTCGTTATTAA", ("Degeneracy=4", "Degeneracy_Info=2,+,GTT")),
               "B": ("ATGATTTTTCGTTATTAA", ("Degeneracy=2", "Degeneracy_Info=2,+,ATT"))}

    def _annotate(self, tmp_path, annotation, name, tag):
        """
        Annotate one of the two genomes with the given annotation.

        :param tmp_path: The temporary directory.
        :param annotation: The annotation to apply.
        :param name: The genome to annotate.
        :param tag: A tag distinguishing the output of this call.
        :return: The degeneracy annotation of the single record.
        """
        gff = tmp_path / "cds.gff"
        gff.write_text(GFF)

        fasta = tmp_path / f"{name}.fa"
        fasta.write_text(f">1\n{self.GENOMES[name][0]}\n")

        rows = [["1", "6", ".", "T", "C", ".", ".", "AA=T", "GT", "0/1"]]
        vcf = write_vcf_sites_or_samples(tmp_path / f"{name}.vcf", rows, ["s1"])

        out = str(tmp_path / f"{tag}{name}.vcf")

        su.Annotator(source=vcf, output=out, fasta=str(fasta), gff=str(gff),
                     annotations=[annotation]).annotate()

        return degeneracies(out)

    def test_reused_annotation_matches_fresh_ones(self, tmp_path):
        """
        Running one annotation over both genomes must give what two fresh annotations give.
        """
        shared = su.DegeneracyAnnotation()

        for name in ("A", "B"):
            expected = [self.GENOMES[name][1]]

            assert self._annotate(tmp_path, shared, name, "shared") == expected
            assert self._annotate(tmp_path, su.DegeneracyAnnotation(), name, "fresh") == expected

    def test_counts_do_not_accumulate_across_inputs(self, tmp_path):
        """
        The reported counts belong to the input just annotated.
        """
        shared = su.DegeneracyAnnotation()

        for name in ("A", "B"):
            self._annotate(tmp_path, shared, name, "shared")

            assert shared.n_annotated == 1
            assert shared.n_skipped == 0
            assert shared.mismatches == []
            assert shared.errors == []


class TestUndefinedPhase:
    """
    GFF3 allows a coding sequence to leave the phase undefined, which must not abort the annotation.
    """

    def test_phases_are_parsed_once(self):
        """
        An undefined phase is read as no offset into the first codon.
        """
        assert list(_CDSIndex._parse_phases(["0", "1", "2", ".", None])) == [0, 1, 2, 0, 0]

    def test_annotation_survives_an_undefined_phase(self, tmp_path, caplog):
        """
        The site is annotated as it would be at phase 0, with a warning naming the coding sequences.
        """
        gff = tmp_path / "cds.gff"
        gff.write_text(GFF_NO_PHASE)

        fasta = tmp_path / "ref.fa"
        fasta.write_text(">1\nATGGTTTTTCGTTATTAA\n")

        rows = [["1", "6", ".", "T", "C", ".", ".", "AA=T", "GT", "0/1"]]
        vcf = write_vcf_sites_or_samples(tmp_path / "in.vcf", rows, ["s1"])

        out = str(tmp_path / "out.vcf")

        with caplog.at_level(logging.WARNING, logger="sfsutils"):
            su.Annotator(source=vcf, output=out, fasta=str(fasta), gff=str(gff),
                         annotations=[su.DegeneracyAnnotation()]).annotate()

        assert degeneracies(out) == [("Degeneracy=4", "Degeneracy_Info=2,+,GTT")]
        assert "leave the phase undefined" in caplog.text

    def test_debug_logging_survives_an_undefined_phase(self, tmp_path, caplog):
        """
        The debug message about the located coding sequence is what raised, so it is exercised too.
        """
        gff = tmp_path / "cds.gff"
        gff.write_text(GFF_NO_PHASE)

        fasta = tmp_path / "ref.fa"
        fasta.write_text(">1\nATGGTTTTTCGTTATTAA\n")

        rows = [["1", "6", ".", "T", "C", ".", ".", "AA=T", "GT", "0/1"]]
        vcf = write_vcf_sites_or_samples(tmp_path / "in.vcf", rows, ["s1"])

        out = str(tmp_path / "out.vcf")

        with caplog.at_level(logging.DEBUG, logger="sfsutils"):
            su.Annotator(source=vcf, output=out, fasta=str(fasta), gff=str(gff),
                         annotations=[su.DegeneracyAnnotation()]).annotate()

        assert "Found coding sequence: 1:1-18" in caplog.text
        assert degeneracies(out) == [("Degeneracy=4", "Degeneracy_Info=2,+,GTT")]


class TestPolyAllelicMNP:
    """
    ``PolyAllelicFiltration`` must reach the same verdict with and without a samples mask.
    """

    def test_biallelic_mnp_kept_with_samples_mask(self):
        from sfsutils import PolyAllelicFiltration

        f = PolyAllelicFiltration()
        f._samples_mask = np.array([True, True])

        variant = type('V', (), dict(
            ALT=['GC'],
            gt_bases=np.array(['AT|GC', 'AT|AT'], dtype=object)
        ))()

        assert f.filter_site(variant)


class TestSNPPolyAllelicSeparation:
    """
    ``SNPFiltration`` keeps every site that is polymorphic among the included samples, poly-allelic ones
    included; dropping those is ``PolyAllelicFiltration``'s job alone.
    """

    @staticmethod
    def _site(genotypes, alt):
        return type('V', (), dict(gt_bases=np.array(genotypes, dtype=object), ALT=alt, is_snp=True))()

    @pytest.mark.parametrize('genotypes,alt,keeps_snp,keeps_polyallelic', [
        (['A|C', 'G|G', 'A|A'], ['C', 'G'], True, False),  # tri-allelic among the included samples
        (['A|C', 'A|A', 'A|G'], ['C', 'G'], True, True),  # bi-allelic among the included samples
        (['A|A', 'A|A', 'A|G'], ['G'], False, True),  # monomorphic among the included samples
    ])
    def test_verdicts(self, genotypes, alt, keeps_snp, keeps_polyallelic):
        from sfsutils import PolyAllelicFiltration, SNPFiltration

        variant = self._site(genotypes, alt)
        mask = np.array([True, True, False])

        for filtration, expected in [(SNPFiltration(), keeps_snp), (PolyAllelicFiltration(), keeps_polyallelic)]:
            filtration._samples_mask = mask

            assert filtration.filter_site(variant) == expected


class TestOutgroupValidation:
    """
    A sample name that is absent from the input must raise rather than silently change the outcome.
    """

    def test_deviant_rejects_unknown_ingroup(self):
        from sfsutils import DeviantOutgroupFiltration

        f = DeviantOutgroupFiltration(outgroups=['out'], ingroups=['nope'])
        f.samples = np.array(['in1', 'in2', 'out'])

        with pytest.raises(ValueError, match='ingroup'):
            f._create_masks()

    def test_existing_rejects_unknown_outgroup(self):
        from sfsutils import ExistingOutgroupFiltration

        f = ExistingOutgroupFiltration(outgroups=['nope'])
        f.samples = np.array(['in1', 'in2', 'out'])

        with pytest.raises(ValueError, match='outgroup'):
            f._create_mask()


class TestSNPFiltrationFastPath:
    """
    The numeric gt_types shortcut in SNPFiltration must agree with decoding the bases on every site, including
    the multi-allelic ones it deliberately falls through on.
    """

    @staticmethod
    def _reference(variant, mask):
        """Decide from the called bases, the implementation the shortcut replaces."""
        from sfsutils.io_handlers import get_distinct_called_bases

        if not variant.is_snp:
            return False

        return len(get_distinct_called_bases(variant.gt_bases[mask])) > 1

    @pytest.mark.parametrize('keep', [None, 'half', 'one'])
    def test_agrees_with_the_base_comparison(self, keep):
        import os

        from cyvcf2 import VCF

        from sfsutils import SNPFiltration

        vcf = "resources/msprime/two_epoch.vcf"
        if not os.path.exists(vcf):
            pytest.skip("the VCF fixture is absent")

        reader = VCF(vcf)
        n = len(reader.samples)
        mask = {None: np.ones(n, bool),
                'half': np.array([i % 2 == 0 for i in range(n)]),
                'one': np.array([i == 0 for i in range(n)])}[keep]

        f = SNPFiltration()
        f._samples_mask = mask

        compared = 0
        for variant in reader:
            assert f.filter_site(variant) == self._reference(variant, mask), f"disagreement at {variant.POS}"
            compared += 1

        assert compared > 100

    def test_multiallelic_homozygous_alt_falls_through(self):
        """Two samples homozygous for *different* ALT alleles are polymorphic, but their gt_types are identical,
        so the shortcut must defer to the bases rather than call the site monomorphic."""
        from sfsutils import SNPFiltration

        variant = type('V', (), dict(
            is_snp=True,
            gt_types=np.array([3, 3]),
            gt_bases=np.array(['C|C', 'G|G'], dtype=object),
        ))()

        f = SNPFiltration()
        f._samples_mask = np.array([True, True])

        assert f.filter_site(variant)


def test_contig_filtration_matches_through_aliases():
    """ContigFiltration must match a site whose contig is an alias of a requested contig, not only an
    exact string match."""
    Settings.disable_pbar = True

    vcf = tmp_vcf = None
    import tempfile, os
    tmp = tempfile.mkdtemp()
    vcf = os.path.join(tmp, "v.vcf")
    with open(vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr1,length=100>\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n')
        fh.write("#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "s1"]) + "\n")
        fh.write("\t".join(["chr1", "10", ".", "A", "T", ".", ".", ".", "GT", "0/1"]) + "\n")

    # request contig '1'; the input names it 'chr1' -> only matches if the alias is honoured
    out = os.path.join(tmp, "out.vcf")
    su.Filterer(source=vcf, output=out, filtrations=[su.ContigFiltration(contigs=["1"])],
                aliases={"chr1": ["1"]}).filter()

    from cyvcf2 import VCF
    assert len(list(VCF(out))) == 1


def test_snp_filtration_drops_indels():
    """SNPFiltration must reject an indel even when its genotype characters look polymorphic; the
    samples-mask branch counts bases, so it needs the is_snp gate."""
    import numpy as np
    f = su.SNPFiltration()
    f._samples_mask = np.array([True])

    assert f.filter_site(Variant(ref="A", pos=1, chrom="1", alt=["AT"], gt_bases=["A/AT"], is_snp=False)) is False
    assert f.filter_site(Variant(ref="A", pos=2, chrom="1", alt=["T"], gt_bases=["A/T"], is_snp=True)) is True
    # an SNP that is monomorphic among the included samples is still dropped
    assert f.filter_site(Variant(ref="A", pos=3, chrom="1", alt=["T"], gt_bases=["A/A"], is_snp=True)) is False
