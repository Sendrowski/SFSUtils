import logging

import sfsutils as su
from sfsutils.io_handlers import DummyVariant
import numpy as np
import pandas as pd
import pytest
from sfsutils.io_handlers import get_called_bases
from unittest.mock import Mock
from testing import TestCase, requires, requires_network
import types
from sfsutils.io_handlers import VCFHandler
from sfsutils.settings import Settings
from sfsutils.spectrum import Spectra
import random
import time
from sfsutils.filtration import Filtration, SNPFiltration
import gzip
from sfsutils.filtration import SNPFiltration
from sfsutils.parser import _snapshot_state, _restore_state
import argparse
import os
from sfsutils.cli import _check_output_distinct_from_input, _parse_pops, _sample_size, main
from sfsutils.cli import run
from sfsutils.io_handlers import Variant, DummyVariant
from sfsutils.json_handlers import DataframeHandler

@pytest.mark.slow
class ParserTestCase(TestCase):
    """
    Test parser.
    """


    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz')
    @staticmethod
    def test_degeneracy_stratification():
        """
        Test the degeneracy stratification.
        """
        p = su.Parser(
            source='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[su.DegeneracyStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz')
    @staticmethod
    def test_contig_stratification_dataset():
        """
        Test the degeneracy stratification.
        """
        p = su.Parser(
            source='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[su.ContigStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert np.round(sfs.all.data.sum()) == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    def test_contig_stratification(self):
        """
        Test the contig stratification.
        """
        s = su.ContigStratification(['contig1', 'contig2'])

        self.assertEqual(s.get_types(), ['contig1', 'contig2'])
        self.assertNotEqual(s.get_types(), ['contig1', 'contig3'])
        self.assertEqual(s.get_type(Mock(CHROM='contig1')), 'contig1')
        self.assertNotEqual(s.get_type(Mock(CHROM='contig1')), 'contig2')

    def test_random_stratification(self):
        """
        Test the RandomStratification class.
        """
        # Test with 3 bins and fixed seed
        s = su.RandomStratification(n_bins=3, seed=42)

        # Ensure all bin types are generated correctly
        self.assertEqual(s.get_types(), ['bin0', 'bin1', 'bin2'])

        # Ensure random assignment produces valid bins
        mock_variant = Mock()
        bin = s.get_type(mock_variant)
        self.assertIn(bin, ['bin0', 'bin1', 'bin2'])

        # Test reproducibility: two instances with the same seed should match
        s2 = su.RandomStratification(n_bins=3, seed=42)
        self.assertEqual(bin, s2.get_type(mock_variant))

        # Test with only 1 bin (should always return "bin1")
        s_single_bin = su.RandomStratification(n_bins=1, seed=42)
        self.assertEqual(s_single_bin.get_type(mock_variant), 'bin0')

        # Test invalid num_bins (should raise ValueError)
        with self.assertRaises(ValueError):
            su.RandomStratification(n_bins=0)

    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz')
    @staticmethod
    def test_chunked_stratification():
        """
        Test the degeneracy stratification.
        """
        n_chunks = 7
        s = su.ChunkedStratification(n_chunks=n_chunks)

        p = su.Parser(
            source='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[s]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert np.round(sfs.all.data.sum()) == 10000 - p.n_skipped

        assert s.n_valid == 10000 - p.n_skipped

        assert len(sfs.types) == n_chunks

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(s.get_types()))

    @requires('results/vcf/sapiens/chr21.vep.vcf.gz')
    @pytest.mark.slow
    @pytest.mark.very_slow
    def test_vep_stratification(self):
        """
        Test the VEP for human chr21.
        """
        p = su.Parser(
            source='snakemake/results/vcf/sapiens/chr21.vep.vcf.gz',
            n=20,
            stratifications=[su.VEPStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz')
    def test_vep_stratification_subset(self):
        """
        Test the synonymy stratification for a small subset of Betula spp.
        """
        p = su.Parser(
            source='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            max_sites=1000,
            stratifications=[su.VEPStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert that we have all types
        self.assertEqual(set(sfs.types), set(p.stratifications[0].get_types()))

    @requires('results/vcf/sapiens/chr21.snpeff.vcf.gz')
    @pytest.mark.slow
    @pytest.mark.very_slow
    def test_snpeff_stratification(self):
        """
        Test the synonymy stratification against SNPEFF for human chr21.
        """
        p = su.Parser(
            source='snakemake/results/vcf/sapiens/chr21.snpeff.vcf.gz',
            n=20,
            stratifications=[su.SnpEffStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @requires('resources/genome/betula/all.polarized.subset.10000.vcf.gz')
    @staticmethod
    def test_base_transition_stratification():
        """
        Test the base transition stratification.
        """
        p = su.Parser(
            source='resources/genome/betula/all.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[su.BaseTransitionStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @requires('resources/genome/betula/all.polarized.subset.10000.vcf.gz')
    @staticmethod
    def test_transition_transversion_stratification():
        """
        Test the transition transversion stratification.
        """
        p = su.Parser(
            source='resources/genome/betula/all.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[su.TransitionTransversionStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert np.round(sfs.all.data.sum()) == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz', 'resources/genome/betula/genome.subset.20.fasta')
    @staticmethod
    def test_base_context_stratification():
        """
        Test the base context stratification.
        """
        p = su.Parser(
            source='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[su.BaseContextStratification(fasta='resources/genome/betula/genome.subset.20.fasta')]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz')
    @staticmethod
    def test_reference_base_stratification():
        """
        Test the reference base stratification.
        """
        p = su.Parser(
            source='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[su.AncestralBaseStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert np.round(sfs.all.data.sum()) == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @requires('resources/genome/sapiens/chr21_test.vcf.gz', 'resources/genome/sapiens/hg38.sorted.gtf.gz')
    @pytest.mark.very_slow
    def test_parse_vcf_chr21_test(self):
        """
        Parse human chr21 test VCF file.
        """
        p = su.Parser(
            source="resources/genome/sapiens/chr21_test.vcf.gz",
            gff="resources/genome/sapiens/hg38.sorted.gtf.gz",
            fasta="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            n=20,
            annotations=[
                su.DegeneracyAnnotation(),
                su.MaximumParsimonyAncestralAnnotation()
            ],
            filtrations=[
                su.CodingSequenceFiltration()
            ],
            stratifications=[su.DegeneracyStratification()],
            max_sites=100000
        )

        sfs = p.parse()

        self.assertEqual(np.round(sfs.all.data.sum()), 6)


        # assert fixed number of target sites
        # self.assertAlmostEqual(sfs['neutral'].n_sites, 18897.233850, places=5)
        # self.assertAlmostEqual(sfs['selected'].n_sites, 81102.766149, places=5)

    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz')
    def test_filter_out_all_raises_warning(self):
        """
        Test that filtering out all sites logs a warning.
        """
        p = su.Parser(
            source="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
            n=20,
            filtrations=[su.AllFiltration()]
        )

        with self.assertLogs(level="WARNING", logger=logging.getLogger('sfsutils')):
            p.parse()

    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz')
    @staticmethod
    def test_parser_no_stratifications():
        """
        Test that filtering out all sites logs a warning.
        """
        p = su.Parser(
            source="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
            n=20,
            stratifications=[]
        )

        sfs = p.parse()

        assert 'all' in sfs.types

    @requires('resources/genome/betula/all.polarized.subset.10000.vcf.gz', 'resources/genome/betula/genome.gff.gz', 'resources/genome/betula/genome.subset.20.fasta')
    @staticmethod
    def test_parse_betula_vcf():
        """
        Parse the VCF file of Betula spp.
        """
        p = su.Parser(
            source="resources/genome/betula/all.polarized.subset.10000.vcf.gz",
            fasta="resources/genome/betula/genome.subset.20.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            n=20,
            annotations=[
                su.DegeneracyAnnotation(),
                su.MaximumParsimonyAncestralAnnotation()
            ],
            filtrations=[
                su.CodingSequenceFiltration()
            ],
            stratifications=[su.DegeneracyStratification()]
        )

        sfs = p.parse()

        pass

    @requires('resources/genome/betula/all.polarized.subset.10000.vcf.gz', 'resources/genome/betula/genome.gff.gz', 'resources/genome/betula/genome.subset.20.fasta')
    def test_parse_betula_vcf_degeneracy_vs_synonymy(self):
        """
        Parse the VCF file of Betula spp.
        """
        p = su.Parser(
            source="resources/genome/betula/all.polarized.subset.10000.vcf.gz",
            fasta="resources/genome/betula/genome.subset.20.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            n=20,
            annotations=[
                su.DegeneracyAnnotation(),
                su.SynonymyAnnotation()
            ],
            filtrations=[
                su.CodingSequenceFiltration()
            ],
            stratifications=[
                su.DegeneracyStratification(),
                su.SynonymyStratification()
            ]
        )

        sfs = p.parse()

        # make sure we only have equivalent types
        self.assertEqual(set(sfs.data.columns), {'neutral.neutral', 'selected.selected'})

    @requires('resources/genome/betula/biallelic.polarized.vcf.gz', 'resources/genome/betula/genome.fasta', 'resources/genome/betula/genome.gff.gz')
    @pytest.mark.slow
    @pytest.mark.very_slow
    def test_parse_betula_complete_vcf_biallelic_synonymy(self):
        """
        Parse the VCF file of Betula spp.
        """
        p = su.Parser(
            source="resources/genome/betula/biallelic.polarized.vcf.gz",
            fasta="resources/genome/betula/genome.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            n=10,
            annotations=[
                su.SynonymyAnnotation()
            ],
            filtrations=[
                su.CodingSequenceFiltration()
            ],
            stratifications=[su.SynonymyStratification()]
        )

        sfs = p.parse()

        sfs.plot()


    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz')
    def test_target_site_counter_no_fasta(self):
        """
        Make sure an error is raised when not FASTA file is specified
        """
        p = su.Parser(
            n=10,
            source="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
            target_site_counter=su.TargetSiteCounter(
                n_target_sites=40000
            ),
            max_sites=10
        )

        with self.assertRaises(ValueError):
            p.parse()

        pass

    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz', 'resources/genome/betula/genome.gff.gz', 'resources/genome/betula/genome.subset.20.fasta')
    def test_target_site_counter_betula(self):
        """
        Test whether the monomorphic site counter works on the Betula data.
        """
        p = su.Parser(
            source="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
            fasta="resources/genome/betula/genome.subset.20.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            max_sites=10000,
            n=10,
            target_site_counter=su.TargetSiteCounter(
                n_target_sites=40000,
                n_samples=10000
            ),
            annotations=[
                su.DegeneracyAnnotation()
            ],
            stratifications=[su.DegeneracyStratification()]
        )

        # set log level to DEBUG
        p.target_site_counter._logger.setLevel(logging.DEBUG)

        sfs = p.parse()

        # make sure that the sum of the target sites is correct
        self.assertEqual(sfs.n_sites.sum(), p.target_site_counter.n_target_sites)

        # assert that 3 contigs were parsed
        self.assertEqual(3, len(p._contig_bounds))

    def test_target_site_counter_update_target_sites_target_sites_lower_than_polymorphic_raises_warning(self):
        """
        Test updating the target sites for different spectra.
        """
        c = su.TargetSiteCounter(
            n_target_sites=1000,
            n_samples=10000
        )

        # assign a polymorphic SFS to the target site counter
        c._sfs_polymorphic = su.Spectra(dict(
            neutral=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            selected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ))

        with self.assertLogs(level="WARNING", logger=logging.getLogger('sfsutils.TargetSiteCounter')) as warning:
            c._update_target_sites(su.Spectra(dict(
                # an SFS, decreasing sequence
                neutral=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
                selected=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]
            )))

            print(warning[1][0])

    def test_target_site_counter_update_target_sites_target_sites_no_monomorphic_raises_warning(self):
        """
        Test updating the target sites for different spectra.
        """
        c = su.TargetSiteCounter(
            n_target_sites=100000,
            n_samples=10000
        )

        # assign a polymorphic SFS to the target site counter
        c._sfs_polymorphic = su.Spectra(dict(
            neutral=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            selected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ))

        with self.assertLogs(level="WARNING", logger=logging.getLogger('sfsutils.TargetSiteCounter')) as warning:
            c._update_target_sites(su.Spectra(dict(
                # an SFS, decreasing sequence
                neutral=[0, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
                selected=[0, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]
            )))

            print(warning[1][0])

    def test_target_site_counter_update_target_sites_sum_coincides_with_given_target_sites(self):
        """
        Test updating the target sites for different spectra.
        """
        c = su.TargetSiteCounter(
            n_target_sites=100000,
            n_samples=10000
        )

        # assign a polymorphic SFS to the target site counter
        c._sfs_polymorphic = su.Spectra(dict(
            neutral=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            selected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ))

        sfs1 = su.Spectra(dict(
            neutral=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
            selected=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]
        ))

        sfs2 = c._update_target_sites(sfs1)

        # a type whose observed sites exceed its share of the target has its monomorphic count
        # clipped to zero, which adds the shortfall back, so the total is at least the target
        self.assertGreaterEqual(sfs2.n_sites.sum(), 100000)

        # make sure ratio of neutral to selected is the same
        self.assertAlmostEqual(
            sfs1.data.loc[0, 'neutral'] / sfs1.data.loc[0, 'selected'],
            sfs2.data.loc[0, 'neutral'] / sfs2.data.loc[0, 'selected']
        )

    def test_target_site_counter_update_target_sites_more_entries_sum_coincides_with_given_target_sites(self):
        """
        Test updating the target sites for different spectra.
        """
        c = su.TargetSiteCounter(
            n_target_sites=100000,
            n_samples=10000
        )

        # assign a polymorphic SFS to the target site counter
        c._sfs_polymorphic = su.Spectra({
            'type1.neutral': [0, 0, 0, 0, 0, 0],
            'type1.selected': [0, 0, 0, 0, 0, 0],
            'type2.neutral': [0, 0, 0, 0, 0, 0],
            'type2.selected': [0, 0, 0, 0, 0, 0]
        })

        sfs1 = su.Spectra({
            'type1.neutral': [177130, 997, 441, 228, 156, 117],
            'type1.selected': [797939, 1329, 499, 265, 162, 104],
            'type2.neutral': [144430, 114, 83, 105, 109, 652],
            'type2.selected': [797939, 117, 90, 94, 119, 794]
        })

        sfs2 = c._update_target_sites(sfs1)

        # a type whose observed sites exceed its share of the target has its monomorphic count
        # clipped to zero, which adds the shortfall back, so the total is at least the target
        self.assertGreaterEqual(sfs2.n_sites.sum(), 100000)

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_parser_betula_include_samples(self):
        """
        Test that the parser includes only the samples that are given in the include_samples parameter.
        """
        p = su.Parser(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20,
            include_samples=['ASP01', 'ASP02', 'ASP03']
        )

        p._setup()

        self.assertEqual(np.sum(p._samples_mask), 3)

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_parser_betula_include_all_samples(self):
        """
        Test that an unrestricted parse leaves the samples mask None, which means every sample is included.
        """
        p = su.Parser(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20
        )

        p._setup()

        self.assertIsNone(p._samples_mask)

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_parser_betula_exclude_two_samples(self):
        """
        Test that the parser excludes the samples that are given in the exclude_samples parameter.
        """
        p = su.Parser(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20,
            exclude_samples=['ASP01', 'ASP02']
        )

        p._setup()

        self.assertEqual(np.sum(p._samples_mask), 375)

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_parser_betula_include_exclude(self):
        """
        Test that both include and exclude samples work together.
        """
        p = su.Parser(
            source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20,
            include_samples=['ASP01', 'ASP02', 'ASP03'],
            exclude_samples=['ASP02']
        )

        p._setup()

        self.assertEqual(np.sum(p._samples_mask), 2)

    @staticmethod
    def test_get_called_genotypes():
        """
        Test the get_called_genotypes function.
        """
        result = get_called_bases(["A|T", "C/T", ".|G"])

        expected = np.array(['A', 'T', 'C', 'T', 'G'])

        np.testing.assert_array_equal(result, expected)

    @requires('resources/genome/betula/biallelic.with_outgroups.subset.10000.vcf.gz',
              'resources/genome/betula/genome.subset.20.fasta',
              'resources/genome/betula/genome.gff.gz')
    @staticmethod
    @pytest.mark.slow
    def test_manuscript_example():
        """
        Test the example from the manuscript.
        """
        # instantiate parser
        p = su.Parser(
            n=8,  # SFS sample size
            source="resources/genome/betula/biallelic.with_outgroups.subset.10000.vcf.gz",
            fasta="resources/genome/betula/genome.subset.20.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            target_site_counter=su.TargetSiteCounter(
                n_target_sites=350000  # total number of target sites
            ),
            annotations=[
                su.DegeneracyAnnotation(),  # determine degeneracy
                su.MaximumLikelihoodAncestralAnnotation(
                    outgroups=["ERR2103730"]  # use one outgroup
                )
            ],
            stratifications=[su.DegeneracyStratification()]
        )

        # obtain SFS
        spectra: su.Spectra = p.parse()

        spectra.plot()

    @requires('resources/genome/betula/genome.gff.gz')
    def test_count_target_sites_remove_overlaps(self):
        """
        Test the count_target_sites function with removing overlaps.
        """
        sites_overlaps = su.Annotation.count_target_sites('resources/genome/betula/genome.gff.gz', remove_overlaps=True)
        sites = su.Annotation.count_target_sites('resources/genome/betula/genome.gff.gz', remove_overlaps=False)

        for config in sites.keys():
            self.assertLessEqual(sites_overlaps[config], sites[config])

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_invalid_subsample_model_raises_value_error(self):
        """
        Test that an invalid subsample model raises a ValueError.
        """
        with self.assertRaises(ValueError):
            su.Parser(
                source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
                n=20,
                subsample_mode='invalid'
            )

    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz')
    def test_probabilistic_polarization_no_aa_prob_tags_same_result_random_subsampling(self):
        """
        Make sure that probabilistic polarization without AA probability tags yields the same result as without.
        The used VCF files don't contain AA probability tags.
        """
        for n in [9, 10]:
            p1 = su.Parser(
                source="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
                polarize_probabilistically=True,
                subsample_mode='random',
                max_sites=1000,
                n=n
            )

            sfs_prob = p1.parse()

            p2 = su.Parser(
                source="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
                polarize_probabilistically=False,
                subsample_mode='random',
                max_sites=1000,
                n=n
            )

            sfs_fixed = p2.parse()

            spectra = su.Spectra(dict(
                prob=sfs_prob.all,
                fixed=sfs_fixed.all
            ))

            spectra.plot()

            self.assertGreater(sfs_prob.all.data.sum(), 0)

            np.testing.assert_array_equal(sfs_prob.all.data, sfs_fixed.all.data)

    @requires('resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz')
    def test_probabilistic_polarization_no_aa_prob_tags_same_result_probabilistic_subsampling(self):
        """
        Make sure that probabilistic polarization without AA probability tags yields the same result as without.
        The used VCF files don't contain AA probability tags.
        """
        for n in [9, 10]:
            p1 = su.Parser(
                source="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
                polarize_probabilistically=True,
                subsample_mode='probabilistic',
                max_sites=100,
                n=n
            )

            sfs_prob = p1.parse()

            p2 = su.Parser(
                source="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
                polarize_probabilistically=False,
                subsample_mode='probabilistic',
                max_sites=100,
                n=n
            )

            sfs_fixed = p2.parse()

            spectra = su.Spectra(dict(
                prob=sfs_prob.all,
                fixed=sfs_fixed.all
            ))

            spectra.plot()

            self.assertGreater(sfs_prob.all.data.sum(), 0)

            np.testing.assert_array_equal(sfs_prob.all.data, sfs_fixed.all.data)

    @requires('resources/genome/sapiens/hgdp.anc.deg.vcf.gz')
    @pytest.mark.very_slow
    def test_compare_probabilistic_polarization_vs_fixed_random_subsampling(self):
        """
        Compare probabilistic polarization with fixed polarization.
        """
        for n in [19, 20]:
            p1 = su.Parser(
                source="resources/genome/sapiens/hgdp.anc.deg.vcf.gz",
                polarize_probabilistically=True,
                subsample_mode='random',
                max_sites=10000,
                n=n
            )

            sfs_prob = p1.parse()

            p2 = su.Parser(
                source="resources/genome/sapiens/hgdp.anc.deg.vcf.gz",
                polarize_probabilistically=False,
                subsample_mode='random',
                max_sites=10000,
                n=n
            )

            sfs_fixed = p2.parse()

            spectra = su.Spectra(dict(
                prob=sfs_prob.all,
                fixed=sfs_fixed.all
            ))

            spectra.plot()

            # mean relative difference much lower than threshold for most bins
            self.assertLess(np.abs((sfs_prob.all.data - sfs_fixed.all.data) / sfs_fixed.all.data).mean(), 0.3)

    @requires('resources/genome/sapiens/hgdp.anc.deg.vcf.gz')
    @pytest.mark.very_slow
    def test_compare_probabilistic_polarization_vs_fixed_probabilistic_subsampling(self):
        """
        Compare probabilistic polarization with fixed polarization.
        """
        for n in [19, 20]:
            p1 = su.Parser(
                source="resources/genome/sapiens/hgdp.anc.deg.vcf.gz",
                polarize_probabilistically=True,
                max_sites=10000,
                n=n
            )

            sfs_prob = p1.parse()

            p2 = su.Parser(
                source="resources/genome/sapiens/hgdp.anc.deg.vcf.gz",
                polarize_probabilistically=False,
                max_sites=10000,
                n=n
            )

            sfs_fixed = p2.parse()

            spectra = su.Spectra(dict(
                prob=sfs_prob.all,
                fixed=sfs_fixed.all
            ))

            spectra.plot()

            # mean relative difference much lower than threshold for most bins
            self.assertLess(np.abs((sfs_prob.all.data - sfs_fixed.all.data) / sfs_fixed.all.data).mean(), 0.12)

class FastParserTestCase(TestCase):
    """
    Fast-tier parser coverage. Reuses the committed betula VCF but caps ``max_sites`` so only a
    handful of records are read (the parse loop short-circuits), exercising the stratification and
    SFS-assembly code paths in milliseconds rather than seconds.
    """

    vcf = 'resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz'
    fasta = 'resources/genome/betula/genome.subset.20.fasta'

    def _parse(self, stratifications, max_sites=200, **kwargs):
        sfs = su.Parser(
            source=self.vcf,
            n=20,
            stratifications=stratifications,
            max_sites=max_sites,
            **kwargs
        ).parse()

        # parse() always returns a Spectra; some stratifications skip every site in a tiny slice
        # (sparse INFO fields), which still exercises the parse/skip paths
        self.assertIsInstance(sfs, su.Spectra)
        return sfs

    def test_no_stratification(self):
        """A bare parse (no stratification), both subsample modes, yields a full SFS."""
        for sfs in (self._parse([]), self._parse([], subsample_mode='random', seed=1)):
            self.assertEqual(sfs.all.n, 20)
            self.assertGreater(sfs.all.data.sum(), 0)

    def test_stratifications_vcf_only(self):
        """Stratifications that read only the VCF / its INFO fields."""
        for strat in [
            su.DegeneracyStratification(),
            su.TransitionTransversionStratification(),
            su.BaseTransitionStratification(),
            su.AncestralBaseStratification(),
            su.RandomStratification(n_bins=3, seed=42),
            su.ContigStratification(),
            su.ChunkedStratification(n_chunks=2),
        ]:
            with self.subTest(stratification=type(strat).__name__):
                sfs = self._parse([strat])
                if sfs.types:
                    self.assertTrue(set(sfs.types).issubset(set(strat.get_types())))

    def test_base_context_stratification_with_fasta(self):
        """The FASTA-backed base-context stratification (tiny committed genome subset)."""
        self._parse([su.BaseContextStratification(fasta=self.fasta)])

    def test_filtrations(self):
        """Parse with VCF-only filtrations applied."""
        self._parse([], filtrations=[su.SNPFiltration()])
        self._parse([], filtrations=[su.SNPFiltration(), su.PolyAllelicFiltration()])

    def test_options(self):
        """The random subsample mode with an explicit seed."""
        self._parse([], subsample_mode='random', seed=3)

    @pytest.mark.slow
    @pytest.mark.very_slow
    @requires('resources/genome/betula/all.subset.100000.vcf.gz', 'resources/genome/betula/genome.gff.gz')
    def test_inline_annotation_and_stratification(self):
        """An inline degeneracy annotation + stratification during the parse (FASTA + GFF)."""
        sfs = su.Parser(
            source='resources/genome/betula/all.subset.100000.vcf.gz',
            fasta=self.fasta,
            gff='resources/genome/betula/genome.gff.gz',
            n=20,
            max_sites=200,
            annotations=[su.DegeneracyAnnotation()],
            stratifications=[su.DegeneracyStratification()],
        ).parse()

        self.assertIsInstance(sfs, su.Spectra)

    def test_target_site_counter(self):
        """
        Sampling monomorphic target sites from the FASTA via TargetSiteCounter (the parser is fed a
        SNP-only VCF and reconstructs the monomorphic counts from the reference). A small
        ``n_samples`` keeps it in the millisecond range while still exercising the count/update path.
        """
        sfs = su.Parser(
            source=self.vcf,
            fasta=self.fasta,
            n=20,
            max_sites=200,
            filtrations=[su.SNPFiltration()],
            target_site_counter=su.TargetSiteCounter(n_target_sites=100000, n_samples=200),
        ).parse()

        self.assertIsInstance(sfs, su.Spectra)
        # monomorphic counts were filled in from the reference, so the SFS is non-empty
        self.assertGreater(sfs.all.data.sum(), 0)

    @pytest.mark.slow
    @pytest.mark.very_slow
    @requires('resources/genome/betula/all.subset.100000.vcf.gz', 'resources/genome/betula/genome.gff.gz')
    def test_inline_synonymy_annotation_and_stratification(self):
        """
        Inline SynonymyAnnotation adds the ``Synonymy`` info tag on-the-fly, which
        SynonymyStratification then reads to split neutral/selected — exercising the synonymy
        stratification path without a pre-annotated (VEP/snpEff) VCF.
        """
        sfs = su.Parser(
            source='resources/genome/betula/all.subset.100000.vcf.gz',
            fasta=self.fasta,
            gff='resources/genome/betula/genome.gff.gz',
            n=20,
            max_sites=200,
            annotations=[su.SynonymyAnnotation()],
            stratifications=[su.SynonymyStratification()],
        ).parse()

        self.assertIsInstance(sfs, su.Spectra)
        if sfs.types:
            self.assertTrue(set(sfs.types).issubset({'neutral', 'selected'}))


# ---------------------------------------------------------------------------------------------------------------------
# Regression tests for scan-found edge cases
# ---------------------------------------------------------------------------------------------------------------------

def test_chunked_stratification_rewind_resets_counter_and_no_overshoot():
    """``ChunkedStratification._rewind`` resets the counter, and typing more sites than the first
    pass (as the TargetSiteCounter sampling pass does) falls back to the last chunk instead of
    raising ``StopIteration``."""
    s = su.ChunkedStratification(n_chunks=3)
    s.chunk_sizes = [2, 2, 2]          # as if _setup ran on 6 sites
    s.n_valid, s.counter = 5, 6        # state after a full first pass

    s._rewind()
    assert s.counter == 0 and s.n_valid == 0

    types = [s.get_type(DummyVariant("A", i + 1, "1")) for i in range(9)]  # 9 > sum(chunk_sizes) == 6
    assert types[:6] == ['chunk0', 'chunk0', 'chunk1', 'chunk1', 'chunk2', 'chunk2']
    assert all(t == 'chunk2' for t in types[6:])  # overshoot -> last chunk, no StopIteration


def test_ml_ancestral_zero_width_contig_bounds_no_crash():
    """``_sample_mono_allelic_sites`` returns gracefully when every parsed contig spans a single
    position (previously produced NaN sampling probabilities and raised ``ValueError``)."""
    ann = su.MaximumLikelihoodAncestralAnnotation(outgroups=["OG"], n_ingroups=2, n_target_sites=100)
    ann._logger = logging.getLogger('test')
    ann.n_sites = 2
    ann.n_samples_target_sites = 50
    ann._contig_bounds = {"1": (10, 10), "2": (20, 20)}  # all zero-width
    ann.rng = np.random.default_rng(0)

    ann._sample_mono_allelic_sites()  # must return via the guard, not raise


def test_ml_ancestral_all_masked_window_terminates(monkeypatch):
    """``_sample_mono_allelic_sites`` terminates when a contig's parsed interval is entirely
    non-ACGT (previously the sampling loop could spin forever)."""
    from sfsutils.settings import Settings
    from sfsutils.io_handlers import FASTAHandler

    ann = su.MaximumLikelihoodAncestralAnnotation(outgroups=["OG"], n_ingroups=2, n_target_sites=100)
    ann._logger = logging.getLogger('test')
    ann.n_sites = 2
    ann.n_samples_target_sites = 10
    ann.adjust_target_sites = False
    ann._contig_bounds = {"1": (10, 100)}  # non-zero width -> enters the sampling loop
    ann.rng = np.random.default_rng(0)

    class _Rec:  # all-N reference so no draw is A/C/G/T
        seq = "N" * 200

    handler = Mock()
    handler.get_aliases.return_value = ["1"]
    handler.get_contig.return_value = _Rec()
    ann._handler = handler

    Settings.disable_pbar = True
    # neutralise the trailing FASTA rewind + target-site extrapolation (out of scope here)
    monkeypatch.setattr(FASTAHandler, "_rewind", staticmethod(lambda h: None), raising=False)
    monkeypatch.setattr(type(ann), "_get_n_target_sites_adjusted", lambda self: self.n_target_sites, raising=False)

    ann._sample_mono_allelic_sites()  # must return (bounded loop), not hang
    assert ann._monomorphic_samples is not None


HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length=1000>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##INFO=<ID=AA_prob,Number=1,Type=Float,Description="ancestral allele probability">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)


def _write_vcf(path, rows, samples=("s1", "s2")):
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


def _write_fasta(path, length=1000, contig="1"):
    """
    Write a single-contig FASTA of a constant base.

    :param path: The path to write to.
    :param length: The contig length.
    :param contig: The contig name.
    :return: The path as a string.
    """
    path.write_text(f">{contig}\n" + "A" * length + "\n")

    return str(path)


def test_probabilistic_polarization_applies_to_fixed_derived_sites(tmp_path):
    """A site fixed for the derived allele shows a single observed base but is bi-allelic, so its mass
    must be split by the ancestral allele probability like any other site."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "fixed_derived.vcf", [
        ["1", "10", ".", "A", "T", ".", ".", "AA=A;AA_prob=0.7", "GT", "1|1", "1|1"],
        ["1", "20", ".", "A", "T", ".", ".", "AA=A;AA_prob=0.7", "GT", "0|1", "1|1"],
    ])

    for mode in ["random", "probabilistic"]:
        sfs = su.Parser(source=vcf, n=4, polarize_probabilistically=True, subsample_mode=mode).parse()["all"]

        # the fixed-derived site contributes 0.7 to the divergence bin and 0.3 to the monomorphic bin
        assert sfs.to_list() == pytest.approx([0.3, 0.3, 0.0, 0.7, 0.7])
        assert sfs.n_div == pytest.approx(0.7)


def test_probabilistic_polarization_applies_to_fixed_derived_sites_joint(tmp_path):
    """The joint projection reflects a fixed-derived site on all axes, moving mass off the all-derived
    corner and onto the origin."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "fixed_derived_joint.vcf", [
        ["1", "10", ".", "A", "T", ".", ".", "AA=A;AA_prob=0.7", "GT", "1|1", "1|1"],
    ], samples=("s1", "s2"))

    sfs = np.asarray(su.Parser(
        source=vcf,
        pops={"A": ["s1"], "B": ["s2"]},
        n={"A": 2, "B": 2},
        polarize_probabilistically=True,
        subsample_mode="probabilistic",
    ).parse()["all"])

    assert sfs[2, 2] == pytest.approx(0.7)
    assert sfs[0, 0] == pytest.approx(0.3)


def test_subsample_modes_agree_at_fixed_derived_site(tmp_path):
    """Both subsample modes must place the same mass at a site with a single observed base, as the
    down-projection there is deterministic."""
    Settings.disable_pbar = True

    vcf = _write_vcf(tmp_path / "modes.vcf", [
        ["1", "10", ".", "A", "T", ".", ".", "AA=A;AA_prob=0.8", "GT", "1|1", "1|1"],
        ["1", "20", ".", "A", "T", ".", ".", "AA=A;AA_prob=0.8", "GT", "0|0", "0|0"],
    ])

    def parse(mode):
        return su.Parser(source=vcf, n=4, polarize_probabilistically=True,
                         subsample_mode=mode).parse()["all"].to_list()

    assert parse("random") == pytest.approx(parse("probabilistic"))


def _joint_counter(n_target_sites, sampled, polymorphic, shape=(4, 4)):
    """
    Build a target site counter primed with a synthetic joint spectrum.

    :param n_target_sites: The number of target sites.
    :param sampled: The monomorphic mass sampled from the FASTA file per type.
    :param polymorphic: The polymorphic mass per type.
    :param shape: The shape of the joint spectrum.
    :return: The counter and the per-type joint SFS after sampling.
    """
    counter = su.TargetSiteCounter(n_samples=int(sum(sampled.values())), n_target_sites=n_target_sites)
    counter.parser = types.SimpleNamespace(_joint_shape=shape)

    before = {}
    for t, n in polymorphic.items():
        arr = np.zeros(shape)
        arr[1, 1] = n
        before[t] = arr

    counter._sfs_polymorphic = before

    sfs = {}
    for t, arr in before.items():
        after = arr.copy()
        after[(0,) * len(shape)] += sampled[t]
        sfs[t] = after

    return counter, sfs


def _one_dimensional_totals(n_target_sites, sampled, polymorphic):
    """
    Run the one-dimensional target-site accounting on the same numbers.

    :param n_target_sites: The number of target sites.
    :param sampled: The monomorphic mass sampled from the FASTA file per type.
    :param polymorphic: The polymorphic mass per type.
    :return: The total number of sites per type.
    """
    counter = su.TargetSiteCounter(n_samples=int(sum(sampled.values())), n_target_sites=n_target_sites)
    counter._sfs_polymorphic = Spectra({t: [0.0, polymorphic[t], 0.0, 0.0, 0.0] for t in polymorphic})
    after = Spectra({t: [sampled[t], polymorphic[t], 0.0, 0.0, 0.0] for t in polymorphic})

    return counter._update_target_sites(after).data.sum().to_dict()


@pytest.mark.parametrize("n_target_sites", [1000000, 20000])
def test_joint_target_sites_match_one_dimensional_accounting(n_target_sites):
    """The sites sampled from the FASTA file estimate the composition of all sites, not of the monomorphic
    ones, so each type is scaled to its share of the target sites just as in the one-dimensional path."""
    sampled = {"a": 3000.0, "b": 7000.0}
    polymorphic = {"a": 6000.0, "b": 4000.0}

    counter, sfs = _joint_counter(n_target_sites, sampled, polymorphic)
    joint = {t: float(arr.sum()) for t, arr in counter._update_target_sites_joint(sfs).items()}

    assert joint == pytest.approx(_one_dimensional_totals(n_target_sites, sampled, polymorphic))
    assert sum(joint.values()) == pytest.approx(n_target_sites)

    # the shares follow the sampled composition, not the composition of the monomorphic sites alone
    assert joint["a"] / joint["b"] == pytest.approx(sampled["a"] / sampled["b"])


def test_joint_target_sites_clip_negative_monomorphic_counts():
    """A type whose observed sites outnumber its share of the target sites is clipped to zero monomorphic
    sites rather than left with a negative mutational opportunity."""
    sampled = {"a": 1000.0, "b": 9000.0}
    polymorphic = {"a": 8000.0, "b": 1000.0}

    counter, sfs = _joint_counter(20000, sampled, polymorphic)
    updated = counter._update_target_sites_joint(sfs)

    # type 'a' is entitled to 2000 sites but was observed at 8000
    assert updated["a"][0, 0] == 0
    assert updated["a"][1, 1] == 8000
    assert updated["b"][0, 0] == pytest.approx(9000 * 2 - 1000)


def test_joint_target_sites_preserve_divergence_corner():
    """The fixed-derived corner is monomorphic but observed, so it survives the update and consumes
    target-site budget."""
    counter = su.TargetSiteCounter(n_samples=100, n_target_sites=10000)
    counter.parser = types.SimpleNamespace(_joint_shape=(3, 3))

    before = np.zeros((3, 3))
    before[1, 1] = 400.0
    before[2, 2] = 100.0
    counter._sfs_polymorphic = {"all": before}

    after = before.copy()
    after[0, 0] = 100.0

    updated = counter._update_target_sites_joint({"all": after})["all"]

    assert updated[2, 2] == 100.0
    assert updated[0, 0] == pytest.approx(10000 - 500)
    assert updated.sum() == pytest.approx(10000)


def test_chunked_stratification_assigns_sampled_sites_by_position(tmp_path):
    """The target-site sampling pass visits more sites than the first pass, so chunks are assigned by
    genomic position; the monomorphic mass then follows each chunk's genomic span."""
    Settings.disable_pbar = True

    # ten dense variants followed by ten sparse ones, so the two chunks span very different lengths
    positions = list(range(1, 11)) + list(range(500, 1000, 50))
    rows = [["1", str(pos), ".", "A", "T", ".", ".", ".", "GT", "0|1", "0|0"] for pos in positions]

    vcf = _write_vcf(tmp_path / "chunks.vcf", rows)
    fasta = _write_fasta(tmp_path / "chunks.fasta")

    strat = su.ChunkedStratification(2)

    spectra = su.Parser(
        source=vcf,
        fasta=fasta,
        n=4,
        seed=0,
        skip_non_polarized=False,
        subsample_mode="random",
        stratifications=[strat],
        filtrations=[su.SNPFiltration()],
        target_site_counter=su.TargetSiteCounter(n_samples=2000, n_target_sites=100000),
    ).parse()

    # the first pass split the variants evenly, so the boundary sits at the eleventh one
    assert strat._chunk_starts == [(0, 1), (0, 500)]

    monomorphic = spectra.data.iloc[0]
    span = {"chunk0": 500 - 1, "chunk1": 950 - 500}

    for t, expected in span.items():
        assert monomorphic[t] / monomorphic.sum() == pytest.approx(expected / sum(span.values()), abs=0.05)


def test_chunked_stratification_counts_sites_on_the_first_pass(tmp_path):
    """Without a second pass the chunks still hold equal numbers of sites, which is what the chunk sizes
    are computed for."""
    Settings.disable_pbar = True

    rows = [["1", str(pos), ".", "A", "T", ".", ".", ".", "GT", "0|1", "0|0"] for pos in range(1, 21)]
    vcf = _write_vcf(tmp_path / "counted.vcf", rows)

    spectra = su.Parser(
        source=vcf,
        n=4,
        skip_non_polarized=False,
        subsample_mode="random",
        stratifications=[su.ChunkedStratification(4)],
    ).parse()

    assert spectra.data.sum().to_dict() == {f"chunk{i}": 5.0 for i in range(4)}


def test_two_sfs_target_site_counter_does_not_reopen_the_source(tmp_path, monkeypatch):
    """The region length is read before the reader is closed, so the source is opened exactly once and no
    reader is left behind on the parser."""
    Settings.disable_pbar = True

    rows = [["1", str(pos), ".", "A", "T", ".", ".", ".", "GT", "0|1", "0|0"] for pos in range(1, 200, 10)]
    vcf = _write_vcf(tmp_path / "two_sfs.vcf", rows)

    opens = []
    original = VCFHandler._open_reader

    def counting_open(self, *args, **kwargs):
        opens.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VCFHandler, "_open_reader", counting_open)

    parser = su.Parser(
        source=vcf,
        n=4,
        two_sfs=True,
        d=50,
        skip_non_polarized=False,
        subsample_mode="random",
        filtrations=[su.SNPFiltration()],
        target_site_counter=su.TargetSiteCounter(n_samples=10, n_target_sites=10000),
    )

    parser.parse()

    assert len(opens) == 1
    assert "_reader" not in parser.__dict__


HEADER_LONG_CONTIG = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length=5000>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##INFO=<ID=AA_prob,Number=1,Type=Float,Description="ancestral allele probability">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)


SAMPLES = ("s0", "s1", "s2", "s3", "s4")


COMPLEMENT = {"A": "G", "G": "A", "C": "T", "T": "C"}


def _write_vcf_five_samples(path, rows, samples=SAMPLES):
    """
    Write a minimal VCF holding the given data rows.

    :param path: The path to write to.
    :param rows: The data rows, each a sequence of the nine fixed columns followed by the genotypes.
    :param samples: The sample names.
    :return: The path as a string.
    """
    columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples]

    path.write_text(HEADER_LONG_CONTIG + "#" + "\t".join(columns) + "\n" + "".join("\t".join(r) + "\n" for r in rows))

    return str(path)


def _write_fasta_from_sequence(path, seq, contig="1"):
    """
    Write a single-contig FASTA.

    :param path: The path to write to.
    :param seq: The sequence.
    :param contig: The contig name.
    :return: The path as a string.
    """
    path.write_text(f">{contig}\n{seq}\n")

    return str(path)


def _snp_rows(positions, seq=None):
    """
    Build one bi-allelic row per position, singleton in the first sample and polarized to the reference.

    :param positions: The positions.
    :param seq: The reference sequence the positions index into, or ``None`` for an all-``A`` reference.
    :return: The rows.
    """
    rows = []

    for pos in positions:
        ref = seq[pos - 1] if seq is not None else "A"
        rows.append(["1", str(pos), ".", ref, COMPLEMENT[ref], ".", ".", f"AA={ref}", "GT",
                     "0|1", "0|0", "0|0", "0|0", "0|0"])

    return rows


class _RaisingFiltration(Filtration):
    """
    Filtration that raises once it has seen a given number of sites, to interrupt a pass part-way through.
    """

    def __init__(self, at: int):
        """
        Create instance.

        :param at: The number of the site at which to raise.
        """
        super().__init__()

        self.at: int = at
        self.n_seen: int = 0

    def filter_site(self, variant) -> bool:
        """
        Pass every site until the configured one.

        :param variant: The site.
        :return: Always ``True``.
        """
        self.n_seen += 1

        if self.n_seen == self.at:
            raise RuntimeError("interrupted")

        return True


def test_joint_target_sites_keep_types_first_seen_when_sampling(tmp_path):
    """A stratification type that only the sampled monomorphic sites carry must appear in the joint
    spectra, and the joint total must match the one-dimensional total."""
    Settings.disable_pbar = True

    rng = random.Random(1)
    seq = "".join(rng.choice("ACGT") for _ in range(5000))
    fasta = _write_fasta_from_sequence(tmp_path / "ref.fasta", seq)

    # the variants sit in a short stretch, so many base contexts appear only among the sampled sites
    vcf = _write_vcf_five_samples(tmp_path / "joint_types.vcf", _snp_rows(list(range(2, 60)) + [4900], seq))

    def parse(pops, sampling=True):
        kwargs = dict(
            source=vcf,
            n=4,
            fasta=fasta,
            seed=7,
            stratifications=[su.BaseContextStratification(fasta=fasta, n_flanking=1)],
            filtrations=[SNPFiltration()],
        )

        if sampling:
            kwargs['target_site_counter'] = su.TargetSiteCounter(n_target_sites=40000, n_samples=2000)

        if pops:
            kwargs['pops'] = {"p1": list(SAMPLES[:3]), "p2": list(SAMPLES[3:])}

        return su.Parser(**kwargs).parse()

    spectra = parse(False)
    joint = parse(True)

    # the fixture only tests the sampled-only strata while some strata are in fact sampled-only
    assert set(spectra.types) - set(parse(False, sampling=False).types)

    assert set(joint.types) == set(spectra.types)
    assert sum(float(np.asarray(joint[t]).sum()) for t in joint.types) == pytest.approx(40000)


def test_fixed_derived_invariant_site_lands_in_divergence_bin(tmp_path):
    """A site without an alternate allele whose ancestral allele differs from the reference is a fixed
    difference, so its mass belongs in the divergence bin rather than the monomorphic one."""
    Settings.disable_pbar = True

    vcf = _write_vcf_five_samples(tmp_path / "fixed_derived_invariant.vcf", [
        ["1", "1", ".", "A", ".", ".", ".", "AA=C", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
        ["1", "2", ".", "A", ".", ".", ".", "AA=A", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
        ["1", "3", ".", "A", "G", ".", ".", "AA=A", "GT", "0|1", "0|0", "0|0", "0|0", "0|0"],
    ])

    sfs = su.Parser(source=vcf, n=4, skip_non_polarized=False).parse()["all"]

    assert sfs.to_list() == pytest.approx([1.6, 0.4, 0.0, 0.0, 1.0])
    assert sfs.n_div == pytest.approx(1.0)


def test_fixed_derived_invariant_site_matches_alt_encoding(tmp_path):
    """The two encodings of a fixed difference, all haplotypes carrying the alternate allele and no
    alternate allele at all, describe the same biology and must give the same spectrum."""
    Settings.disable_pbar = True

    no_alt = _write_vcf_five_samples(tmp_path / "no_alt.vcf", [
        ["1", "1", ".", "A", ".", ".", ".", "AA=C", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
    ])

    with_alt = _write_vcf_five_samples(tmp_path / "with_alt.vcf", [
        ["1", "1", ".", "C", "A", ".", ".", "AA=C", "GT", "1|1", "1|1", "1|1", "1|1", "1|1"],
    ])

    left = su.Parser(source=no_alt, n=4, skip_non_polarized=False).parse()["all"]
    right = su.Parser(source=with_alt, n=4, skip_non_polarized=False).parse()["all"]

    assert left.to_list() == pytest.approx(right.to_list())


def test_fixed_derived_invariant_site_polarized_probabilistically(tmp_path):
    """The ancestral allele probability splits the mass of an invariant site between the two
    monomorphic bins, as it does at a fixed-derived site carrying an alternate allele."""
    Settings.disable_pbar = True

    vcf = _write_vcf_five_samples(tmp_path / "prob_invariant.vcf", [
        ["1", "1", ".", "A", ".", ".", ".", "AA=C;AA_prob=0.7", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
        ["1", "2", ".", "A", ".", ".", ".", "AA=A;AA_prob=0.7", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
    ])

    sfs = su.Parser(source=vcf, n=4, polarize_probabilistically=True, skip_non_polarized=False).parse()["all"]

    assert sfs.to_list() == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0])


def test_fixed_derived_invariant_site_joint(tmp_path):
    """In joint mode a fixed difference without an alternate allele belongs in the all-derived corner,
    where every population carries the derived allele."""
    Settings.disable_pbar = True

    vcf = _write_vcf_five_samples(tmp_path / "fixed_derived_joint.vcf", [
        ["1", "1", ".", "A", ".", ".", ".", "AA=C", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
        ["1", "2", ".", "A", ".", ".", ".", "AA=A", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
    ])

    sfs = np.asarray(su.Parser(
        source=vcf,
        n=2,
        pops={"p1": list(SAMPLES[:3]), "p2": list(SAMPLES[3:])},
        skip_non_polarized=False,
    ).parse()["all"])

    assert sfs[-1, -1] == pytest.approx(1.0)
    assert sfs[0, 0] == pytest.approx(1.0)
    assert sfs.sum() == pytest.approx(2.0)


def test_invariant_site_without_ancestral_allele_stays_monomorphic(tmp_path):
    """An invariant site carrying no ancestral allele has no polarization information of its own, so the
    reference allele remains the ancestral one and the site keeps counting as a target site."""
    Settings.disable_pbar = True

    vcf = _write_vcf_five_samples(tmp_path / "no_aa.vcf", [
        ["1", "1", ".", "A", ".", ".", ".", ".", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
        ["1", "2", ".", "A", ".", ".", ".", "AA=N", "GT", "0|0", "0|0", "0|0", "0|0", "0|0"],
    ])

    for skip in [True, False]:
        sfs = su.Parser(source=vcf, n=4, skip_non_polarized=skip).parse()["all"]

        assert sfs.to_list() == pytest.approx([2.0, 0.0, 0.0, 0.0, 0.0])


def test_parse_rewinds_the_reader_after_an_interrupted_pass(tmp_path):
    """A parse that raises part-way through leaves the reader mid-input, so the next parse must rewind
    it and see every record again."""
    Settings.disable_pbar = True

    vcf = _write_vcf_five_samples(tmp_path / "rewind.vcf", _snp_rows(range(1, 201)))

    parser = su.Parser(source=vcf, n=4, filtrations=[_RaisingFiltration(at=100)])

    with pytest.raises(RuntimeError):
        parser.parse()

    parser.filtrations = []

    retried = parser.parse()["all"].to_list()
    fresh = su.Parser(source=vcf, n=4).parse()["all"].to_list()

    assert retried == pytest.approx(fresh)


def test_random_stratification_is_reproducible_across_passes(tmp_path):
    """The random stratification is re-seeded per pass, so parsing the same input twice with the same
    parser assigns the same sites to the same bins."""
    Settings.disable_pbar = True

    vcf = _write_vcf_five_samples(tmp_path / "random_strat.vcf", _snp_rows(range(1, 301)))

    parser = su.Parser(source=vcf, n=4, stratifications=[su.RandomStratification(n_bins=3, seed=42)])

    first = parser.parse().data.sum().to_dict()
    second = parser.parse().data.sum().to_dict()

    assert first == second

    # and the same as a parser that never saw the input before
    other = su.Parser(source=vcf, n=4, stratifications=[su.RandomStratification(n_bins=3, seed=42)])

    assert other.parse().data.sum().to_dict() == first


def test_target_site_counter_restores_filtrations_when_sampling_raises(tmp_path):
    """The counter suspends the SNP filtration while sampling, and must hand it back to the parser even
    when the sampling pass raises."""
    Settings.disable_pbar = True

    rng = random.Random(2)
    seq = "".join(rng.choice("ACGT") for _ in range(1000))
    fasta = _write_fasta_from_sequence(tmp_path / "ref.fasta", seq)
    vcf = _write_vcf_five_samples(tmp_path / "restore.vcf", _snp_rows(range(1, 51), seq))

    # the filtration survives the variant pass and raises in the sampling pass that follows it
    parser = su.Parser(
        source=vcf,
        n=4,
        fasta=fasta,
        filtrations=[SNPFiltration(), _RaisingFiltration(at=80)],
        target_site_counter=su.TargetSiteCounter(n_target_sites=1000, n_samples=200),
    )

    with pytest.raises(RuntimeError):
        parser.parse()

    assert [type(f) for f in parser.filtrations] == [SNPFiltration, _RaisingFiltration]


def test_target_site_counter_handles_fasta_shorter_than_variants(tmp_path):
    """A FASTA record that does not span the parsed variants must not send the sampling pass past its
    end; the sampling is confined to the part backed by the reference."""
    Settings.disable_pbar = True

    rng = random.Random(3)
    seq = "".join(rng.choice("ACGT") for _ in range(500))
    fasta = _write_fasta_from_sequence(tmp_path / "short.fasta", seq)
    vcf = _write_vcf_five_samples(tmp_path / "short_ref.vcf", _snp_rows([1, 50, 300, 900]))

    sfs = su.Parser(
        source=vcf,
        n=4,
        fasta=fasta,
        filtrations=[SNPFiltration()],
        target_site_counter=su.TargetSiteCounter(n_target_sites=1000, n_samples=50),
    ).parse()["all"]

    assert sfs.data.sum() == pytest.approx(1000)


def test_chunked_stratification_scales_with_the_number_of_chunks(tmp_path):
    """Locating a site's chunk is a lookup among the cumulative boundaries, so many chunks cost no more
    than few."""
    Settings.disable_pbar = True

    vcf = _write_vcf_five_samples(tmp_path / "chunks.vcf", _snp_rows(range(1, 20001)))

    def parse(n_chunks):
        strat = [su.ChunkedStratification(n_chunks=n_chunks)] if n_chunks else []

        start = time.process_time()
        spectra = su.Parser(source=vcf, n=4, stratifications=strat).parse()

        return time.process_time() - start, spectra

    few, spectra_few = parse(10)
    many, spectra_many = parse(1000)

    # the totals are unaffected by how the sites are split up
    assert spectra_many.data.to_numpy().sum() == pytest.approx(spectra_few.data.to_numpy().sum())

    # a hundredfold more chunks used to cost close to a hundredfold more time
    assert many < 3 * few


def test_chunked_stratification_assigns_the_same_chunks_as_a_running_total():
    """The cumulative boundaries locate the same chunk as summing the sizes site by site does."""
    strat = su.ChunkedStratification(n_chunks=4)
    strat.chunk_sizes = [3, 2, 0, 2]

    types = [strat.get_type(su.io_handlers.DummyVariant("A", i + 1, "1")) for i in range(9)]

    assert types == ['chunk0', 'chunk0', 'chunk0', 'chunk1', 'chunk1', 'chunk3', 'chunk3', 'chunk3', 'chunk3']


Settings.disable_pbar = True


HEADER_INFO = (
    "##fileformat=VCFv4.2\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)


def _write_vcf_multi_contig(path, contigs, rows, samples):
    """
    Write a minimal VCF holding the given data rows.

    :param path: The path to write to.
    :param contigs: The contigs, as ``(name, length)`` pairs.
    :param rows: The data rows, each a sequence of the nine fixed columns followed by the genotypes.
    :param samples: The sample names.
    :return: The path as a string.
    """
    with open(path, "w") as fh:
        fh.write(HEADER_INFO)

        for name, length in contigs:
            fh.write(f"##contig=<ID={name},length={length}>\n")

        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")

        for row in rows:
            fh.write("\t".join(str(x) for x in row) + "\n")

    return str(path)


def _write_fasta_gzipped(path, sequences):
    """
    Write a gzipped FASTA file.

    :param path: The path to write to.
    :param sequences: Mapping of contig name to sequence.
    :return: The path as a string.
    """
    with gzip.open(path, "wt") as fh:
        for name, seq in sequences.items():
            fh.write(f">{name}\n")

            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")

    return str(path)


def _polyploid_input(tmp_path, ploidy, n_individuals=5, n_snps=99, length=1000):
    """
    Write a SNP-only VCF of the given ploidy together with the reference it was drawn from.

    :param tmp_path: The directory to write to.
    :param ploidy: The ploidy of each sample.
    :param n_individuals: The number of samples.
    :param n_snps: The number of segregating sites.
    :param length: The length of the contig.
    :return: The VCF path and the FASTA path.
    """
    rng = np.random.default_rng(0)
    n_hap = n_individuals * ploidy
    ref = "".join(rng.choice(list("ACGT"), size=length))
    samples = [f"s{i}" for i in range(n_individuals)]
    rows = []

    for pos in sorted(rng.choice(np.arange(1, length + 1), size=n_snps, replace=False)):
        haplotypes = np.array([1] * int(rng.integers(1, n_hap)) + [0] * n_hap)[:n_hap]
        rng.shuffle(haplotypes)
        base = ref[pos - 1]
        alt = next(b for b in "ACGT" if b != base)
        genotypes = ["|".join(map(str, g)) for g in haplotypes.reshape(-1, ploidy)]
        rows.append(["1", pos, ".", base, alt, ".", "PASS", f"AA={base}", "GT"] + genotypes)

    return (_write_vcf_multi_contig(tmp_path / "poly.vcf", [("1", length)], rows, samples),
            _write_fasta_gzipped(tmp_path / "poly.fasta.gz", {"1": ref}))


@pytest.mark.parametrize("ploidy", [2, 4, 6])
def test_target_site_counter_extrapolates_above_ploidy_two(tmp_path, ploidy):
    """The sites the counter samples from the FASTA stand for fully covered sites, so the coverage gate must not
    reject them when the requested sample size exceeds twice the number of samples, which would silently turn the
    extrapolation into a no-op."""
    vcf, fasta = _polyploid_input(tmp_path, ploidy)

    n_target_sites = 1000
    spectra = su.Parser(
        source=vcf,
        fasta=fasta,
        n=5 * ploidy,
        skip_non_polarized=False,
        filtrations=[SNPFiltration()],
        target_site_counter=su.TargetSiteCounter(n_target_sites=n_target_sites, n_samples=500),
    ).parse()

    data = spectra.data

    assert float(data.values.sum()) == pytest.approx(n_target_sites)
    assert float(data.iloc[0].sum()) > 0


def test_target_site_counter_skips_contig_missing_from_fasta(tmp_path):
    """A contig the FASTA does not cover contributes no target sites, rather than aborting the run with a
    ``LookupError`` and discarding the whole first pass."""
    samples = ["s0", "s1", "s2", "s3", "s4"]
    rng = np.random.default_rng(1)
    ref = "".join(rng.choice(list("ACGT"), size=2000))
    rows = []

    for contig in ("1", "2"):
        for pos in range(10, 1000, 10):
            base = ref[pos - 1] if contig == "1" else "A"
            alt = next(b for b in "ACGT" if b != base)
            rows.append([contig, pos, ".", base, alt, ".", "PASS", f"AA={base}", "GT",
                         "0|1", "0|0", "0|0", "0|0", "0|0"])

    vcf = _write_vcf_multi_contig(tmp_path / "two_contigs.vcf", [("1", 2000), ("2", 2000)], rows, samples)
    fasta = _write_fasta_gzipped(tmp_path / "one_contig.fasta.gz", {"1": ref})

    n_target_sites = 100_000
    spectra = su.Parser(
        source=vcf,
        fasta=fasta,
        n=10,
        skip_non_polarized=False,
        filtrations=[SNPFiltration()],
        target_site_counter=su.TargetSiteCounter(n_target_sites=n_target_sites, n_samples=2000),
    ).parse()

    assert float(spectra.data.values.sum()) == pytest.approx(n_target_sites)


def test_two_sfs_extrapolation_populates_the_divergence_row():
    """A divergence site (bin ``n``) pairs with the sites missing from the input just as a segregating site does,
    so the divergence row and column must be populated rather than zeroed and their mass booked into ``(0, 0)``."""
    marginal = np.array([0.0, 10.0, 5.0, 20.0])
    n_target_sites, region_length, distance = 1000, 1000.0, 10

    extrapolated = su.TargetSiteCounter(n_target_sites=n_target_sites)._extrapolate_two_sfs(
        np.zeros((4, 4)), marginal, region_length, distance)

    # each observed site pairs with the missing ones, which number the target sites less the observed ones
    rho = (n_target_sites - marginal.sum()) / region_length

    np.testing.assert_allclose(extrapolated[0, 1:], marginal[1:] * rho * distance)
    np.testing.assert_allclose(extrapolated[1:, 0], marginal[1:] * rho * distance)
    assert extrapolated[0, -1] > 0
    assert extrapolated[0, 0] == pytest.approx((n_target_sites - marginal.sum()) * rho * distance)


def test_two_sfs_extrapolation_does_not_double_count_observed_monomorphic_sites():
    """Only the sites missing from the input are extrapolated: an input whose monomorphic sites are all present
    leaves the two-SFS untouched, rather than adding a second copy of their pairs."""
    marginal = np.array([900.0, 10.0, 5.0, 85.0])
    observed = np.full((4, 4), 7.0)

    extrapolated = su.TargetSiteCounter(n_target_sites=int(marginal.sum()))._extrapolate_two_sfs(
        observed.copy(), marginal, region_length=1000.0, distance=10)

    np.testing.assert_allclose(extrapolated, observed)


def test_two_sfs_extrapolation_matches_all_sites_ground_truth_with_divergence(tmp_path):
    """Ground truth: parse an all-sites VCF, which counts every pair for real, and compare against the SNP-only
    projection of the same data parsed with a target-site counter. Divergence sites (fixed for the derived
    allele) are present in both inputs, so their extrapolated row and column must reproduce the real ones."""
    length, n_hap, distance = 4000, 6, 50
    rng = np.random.default_rng(1)

    # 3% segregating, 2% fixed for the derived allele, the rest all-ancestral and absent from the SNP-only input
    u = rng.random(length)
    derived = np.where(u < 0.03, rng.integers(1, n_hap, size=length), np.where(u < 0.05, n_hap, 0))

    samples = [f"s{i}" for i in range(n_hap // 2)]
    all_rows, snp_rows = [], []

    for pos, k in enumerate(derived, start=1):
        haplotypes = np.array([1] * int(k) + [0] * (n_hap - int(k)))
        rng.shuffle(haplotypes)
        row = ["1", pos, ".", "A", "T" if k else ".", ".", "PASS", "AA=A", "GT"] + \
              [f"{a}|{b}" for a, b in haplotypes.reshape(-1, 2)]
        all_rows.append(row)

        if k:
            snp_rows.append(row)

    contigs = [("1", length)]
    all_sites = _write_vcf_multi_contig(tmp_path / "all.vcf", contigs, all_rows, samples)
    snps = _write_vcf_multi_contig(tmp_path / "snp.vcf", contigs, snp_rows, samples)

    kw = dict(n=n_hap, two_sfs=True, d=distance, skip_non_polarized=False, subsample_mode="random")
    truth = np.asarray(su.Parser(source=all_sites, **kw).parse()["all"].data)
    extrapolated = np.asarray(su.Parser(source=snps, **kw, target_site_counter=su.TargetSiteCounter(
        n_target_sites=length)).parse()["all"].data)

    # the pairs among observed sites are counted directly and must match exactly
    np.testing.assert_allclose(extrapolated[1:, 1:], truth[1:, 1:])

    # the sites near the contig edges have fewer partners than the uniform density assumes, hence the tolerance
    assert extrapolated[0, 0] == pytest.approx(truth[0, 0], rel=0.03)
    assert extrapolated[0, -1] == pytest.approx(truth[0, -1], rel=0.03)
    assert extrapolated[-1, 0] == pytest.approx(truth[-1, 0], rel=0.03)
    assert extrapolated[0, 1:-1].sum() == pytest.approx(truth[0, 1:-1].sum(), rel=0.03)


def test_two_sfs_with_counter_leaves_all_sites_input_unchanged():
    """An all-sites input carries its own monomorphic pairs, so adding a target-site counter that does not exceed
    the observed sites must not scale the ``(0, 0)`` corner, which broke the correlation matrix."""
    source = "resources/msprime/two_sfs_kingman.all.vcf.gz"

    if not __import__("os").path.exists(source):
        pytest.skip("the all-sites msprime fixture is absent")

    kw = dict(source=source, n=10, two_sfs=True, d=1000, skip_non_polarized=False)

    plain = su.Parser(**kw).parse()["all"]
    with_counter = su.Parser(**kw, target_site_counter=su.TargetSiteCounter(n_target_sites=600_000)).parse()["all"]

    np.testing.assert_allclose(np.asarray(with_counter.data), np.asarray(plain.data))
    np.testing.assert_allclose(np.diag(with_counter.corr())[1:-1], 1.0)


def test_snapshot_state_covers_counters_and_copies_lists():
    """The per-pass state is discovered by type rather than by a name list, so a component gaining a counter does
    not fall out of the snapshot, and the list diagnostics are copied rather than aliased."""

    class Component:
        def __init__(self):
            self.n_valid = 3
            self.n_filtered = 4
            self.mismatches = ["a"]
            self.use_parser = True
            self.label = "x"

    component = Component()
    state = _snapshot_state(component)

    assert state == {"n_valid": 3, "n_filtered": 4, "mismatches": ["a"]}

    component.n_valid, component.n_filtered = 99, 99
    component.mismatches.append("b")

    _restore_state(component, state)

    assert (component.n_valid, component.n_filtered, component.mismatches) == (3, 4, ["a"])


def test_target_site_counter_restores_component_state():
    """The sampling pass re-runs every component, whose counters must keep describing the sites that produced the
    spectra rather than the sampled ones."""
    vcf, fasta = "resources/msprime/two_epoch.vcf", "resources/msprime/two_epoch.ref.fasta.gz"

    if not (__import__("os").path.exists(vcf) and __import__("os").path.exists(fasta)):
        pytest.skip("the msprime VCF / reference FASTA fixtures are absent")

    def parse(counter):
        stratification = su.AncestralBaseStratification()
        filtration = SNPFiltration()

        su.Parser(source=vcf, fasta=fasta, n=10, skip_non_polarized=False,
                  stratifications=[stratification], filtrations=[filtration],
                  target_site_counter=counter).parse()

        return stratification.n_valid, filtration.n_filtered

    without = parse(None)
    with_counter = parse(su.TargetSiteCounter(n_target_sites=500_000, n_samples=2000))

    assert with_counter == without


def test_max_sites_is_positive():
    """A cap of zero parsed the whole input, as the site count it is compared against is itself capped at it."""
    with pytest.raises(ValueError, match="max_sites"):
        su.Parser(source="resources/msprime/two_epoch.vcf", n=10, max_sites=0)


def test_max_sites_caps_the_number_of_parsed_sites():
    """A positive cap stops the parse at that many sites."""
    if not __import__("os").path.exists("resources/msprime/two_epoch.vcf"):
        pytest.skip("the msprime VCF fixture is absent")

    parser = su.Parser(source="resources/msprime/two_epoch.vcf", n=10, max_sites=3, skip_non_polarized=False)

    assert float(parser.parse().data.values.sum()) == pytest.approx(3)


def test_no_samples_mask_installed_without_a_restriction():
    """Without a sample restriction no mask is installed, so no site copies the whole genotype array; a
    restriction still installs one, and the spectrum is the same either way."""
    if not __import__("os").path.exists("resources/msprime/two_epoch.vcf"):
        pytest.skip("the msprime VCF fixture is absent")

    kw = dict(source="resources/msprime/two_epoch.vcf", n=6, skip_non_polarized=False)

    unrestricted = su.Parser(**kw)
    unrestricted._setup()
    assert unrestricted._samples_mask is None

    restricted = su.Parser(**kw, include_samples=[f"tsk_{i}" for i in range(3)])
    restricted._setup()
    assert restricted._samples_mask is not None and restricted._samples_mask.sum() == 3

    # every sample selected explicitly reaches the same spectrum as no restriction at all
    everything = su.Parser(**kw, include_samples=[f"tsk_{i}" for i in range(10)])

    np.testing.assert_allclose(su.Parser(**kw).parse().data.values, everything.parse().data.values)


VCF = "resources/msprime/two_epoch.vcf"


def args_for(source, output):
    """
    Build the parsed arguments the output guard reads.

    :param source: The input source path.
    :param output: The output path.
    :return: The namespace.
    """
    return argparse.Namespace(vcf=source, zarr=None, trees=None, output=output)


class TestOutputGuard:
    """
    An output resolving to the input destroys the input, so it must be rejected however it is spelled (C1).
    """

    def test_hard_link_to_the_input_is_rejected(self, tmp_path):
        """A hard link shares the inode but not the path, which a comparison of the resolved paths misses."""
        source, link = tmp_path / "h1.vcf", tmp_path / "h2.vcf"
        source.write_text("##fileformat=VCFv4.2\n")
        os.link(source, link)

        with pytest.raises(SystemExit, match="input source"):
            _check_output_distinct_from_input(args_for(str(source), str(link)))

        assert source.read_text() == "##fileformat=VCFv4.2\n"

    def test_case_insensitive_spelling_of_the_input_is_rejected(self, tmp_path):
        """On a case-insensitive filesystem the upper-cased path is the very same store."""
        store = tmp_path / "z.vcz"
        store.mkdir()
        upper = tmp_path / "Z.VCZ"

        if not upper.exists():
            pytest.skip("the filesystem is case-sensitive")

        with pytest.raises(SystemExit, match="input source"):
            _check_output_distinct_from_input(args_for(str(store), str(upper)))

    def test_output_inside_the_input_store_is_rejected(self, tmp_path):
        """A zarr output is opened for writing and empties the directory holding the input."""
        store = tmp_path / "in.vcz"
        store.mkdir()

        with pytest.raises(SystemExit, match="input source"):
            _check_output_distinct_from_input(args_for(str(store), str(store / "out.vcz")))

    def test_output_directory_holding_the_input_is_rejected(self, tmp_path):
        """The other direction: an output directory is wiped whole, taking the input below it with it."""
        directory = tmp_path / "data"
        (directory / "nested").mkdir(parents=True)
        source = directory / "nested" / "in.vcz"
        source.mkdir()

        with pytest.raises(SystemExit, match="containing the input source"):
            _check_output_distinct_from_input(args_for(str(source), str(directory)))

    @pytest.mark.parametrize("output", ["out.vcf", "sub/out.vcf", "in.vcf.gz"])
    def test_distinct_outputs_are_allowed(self, tmp_path, output):
        """A distinct output beside the input, or in a directory that does not exist yet, passes."""
        source = tmp_path / "in.vcf"
        source.write_text("##fileformat=VCFv4.2\n")

        _check_output_distinct_from_input(args_for(str(source), str(tmp_path / output)))

    def test_a_missing_or_remote_input_is_not_checked(self, tmp_path):
        """Nothing to compare against, and a remote source is never written to."""
        _check_output_distinct_from_input(args_for(str(tmp_path / "absent.vcf"), str(tmp_path / "absent.vcf")))
        _check_output_distinct_from_input(args_for("https://example.org/in.vcf", str(tmp_path / "out.vcf")))


class TestPopulationSpec:
    """
    A repeated population name used to overwrite the earlier group, yielding a joint SFS of the wrong dimension
    that looks entirely valid downstream (C2).
    """

    def test_repeated_population_is_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="more than once"):
            _parse_pops("A=tsk_0,tsk_1;A=tsk_2,tsk_3")

    def test_empty_sample_list_is_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="no samples"):
            _parse_pops("A=tsk_0;B=")

    def test_empty_name_is_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="name is empty"):
            _parse_pops("=tsk_0")

    def test_missing_separator_is_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid population spec"):
            _parse_pops("A")

    def test_valid_spec_is_parsed(self):
        assert _parse_pops("A=tsk_0, tsk_1 ;B=tsk_2;") == {"A": ["tsk_0", "tsk_1"], "B": ["tsk_2"]}


class TestSampleSize:
    """
    A sample size below two makes bin 1 the divergence bin (``n == 1``), so every segregating site is booked as a
    fixed difference, or collapses the ancestral and the divergence bin onto one index (``n == 0``) (C15).
    """

    @pytest.mark.parametrize("n", [1, 0, -1, -3])
    def test_sample_size_below_two_is_rejected(self, n):
        with pytest.raises(ValueError, match="at least 2"):
            su.Parser(source=VCF, n=n)

    def test_missing_sample_size_is_rejected(self):
        with pytest.raises(ValueError, match="'n' must be given"):
            su.Parser(source=VCF)

    @pytest.mark.parametrize("n", [1, {"A": 4, "B": 1}, [4, 1]])
    def test_sample_size_below_two_is_rejected_per_population(self, n):
        with pytest.raises(ValueError, match="at least 2 for every population"):
            su.Parser(source=VCF, pops={"A": ["tsk_0", "tsk_1"], "B": ["tsk_2", "tsk_3"]}, n=n)

    def test_cli_rejects_a_sample_size_below_two(self):
        with pytest.raises(argparse.ArgumentTypeError, match="at least 2"):
            _sample_size("1")

        assert _sample_size("2") == 2

    def test_cli_parse_rejects_a_sample_size_below_two(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            main(["parse", "--vcf", VCF, "--output", str(tmp_path / "sfs.csv"), "--n", "1"])

        assert "at least 2" in capsys.readouterr().err


def write_sites(path, n_contigs, span, derived, n_hap, only_polymorphic=False):
    """
    Write a VCF holding one site per position of every contig.

    :param path: The path to write to.
    :param n_contigs: The number of contigs.
    :param span: The declared length of each contig, and the number of sites on it.
    :param derived: The derived counts, of shape ``(n_contigs, span)``.
    :param n_hap: The number of haplotypes.
    :param only_polymorphic: Whether to write the segregating sites alone.
    :return: The path as a string.
    """
    rng = np.random.default_rng(1)

    header = ('##fileformat=VCFv4.2\n'
              + ''.join(f'##contig=<ID=c{c},length={span}>\n' for c in range(n_contigs))
              + '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
              '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
              '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'
              + '\t'.join(f's{i}' for i in range(n_hap // 2)) + '\n')

    with open(path, 'w') as out:
        out.write(header)

        for contig in range(n_contigs):
            for pos, k in enumerate(derived[contig], start=1):
                if only_polymorphic and not k:
                    continue

                haplotypes = np.array([1] * int(k) + [0] * (n_hap - int(k)))
                rng.shuffle(haplotypes)

                out.write(f'c{contig}\t{pos}\t.\tA\t{"T" if k else "."}\t.\tPASS\tAA=A\tGT\t'
                          + '\t'.join(f'{a}|{b}' for a, b in haplotypes.reshape(-1, 2)) + '\n')

    return str(path)


class TestRegionLength:
    """
    The region length used to be the summed span of the observed variants, which for the SNP-only input a
    TargetSiteCounter exists for falls far short of the contigs and inflates the site density (C13).
    """

    def test_declared_contig_lengths_are_preferred_over_the_observed_span(self, tmp_path):
        """The variants cover the middle of each contig only, so their span underestimates the region."""
        n_contigs, span, n_hap = 5, 2000, 4

        derived = np.zeros((n_contigs, span), dtype=int)
        derived[:, span // 4:3 * span // 4:100] = 2

        vcf = write_sites(tmp_path / "snp.vcf", n_contigs, span, derived, n_hap, only_polymorphic=True)

        Settings.disable_pbar = True
        parser = su.Parser(source=vcf, n=n_hap, two_sfs=True, d=100, skip_non_polarized=False,
                           target_site_counter=su.TargetSiteCounter(n_target_sites=n_contigs * span))
        parser.parse()

        # the reader is closed after the parse, so the spans are captured while it is open
        assert parser._region_length() == pytest.approx(n_contigs * span)

    def test_the_observed_span_is_the_last_resort(self, tmp_path, caplog):
        """A source declaring no length falls back to the observed span, which is warned about."""
        parser = su.Parser(source=VCF, n=4)
        parser._contig_bounds.update({'c0': (100, 1100), 'c1': (10, 510)})
        parser._declared_contig_lengths = lambda: {}

        assert parser._region_length() == pytest.approx(1500)
        assert "no length" in caplog.text


class TestTwoSFSExtrapolation:
    """
    Pairs form within a contig only, so the number of partners a site has saturates once the distance approaches
    the contig span; the extrapolation used to apply a flat ``2 * rho * d`` to every site (C14).
    """

    @pytest.mark.parametrize("span, distance, expected", [
        (1000, 100, 0.95),  # the window fits, losing only the sites past the contig ends
        (100, 100, 0.5),  # the two expressions meet where the window is exactly the span
        (100, 1000, 0.05),  # saturated: the partners are the contig's own sites, not the window's
    ])
    def test_window_factor(self, span, distance, expected):
        assert su.TargetSiteCounter._window_factor([span], distance) == pytest.approx(expected)

    def test_window_factor_of_mixed_contigs_is_weighted_by_span(self):
        spans, distance = [100.0, 1000.0], 100
        mixed = su.TargetSiteCounter._window_factor(spans, distance)

        assert mixed == pytest.approx((100 * 0.5 + 1000 * 0.95) / 1100)

    def test_window_factor_degenerates_to_one(self):
        assert su.TargetSiteCounter._window_factor([], 100) == 1.0
        assert su.TargetSiteCounter._window_factor([1000.0], 0) == 1.0

    @pytest.mark.parametrize("n_contigs, span, distance", [(20, 200, 500), (5, 1000, 200)])
    def test_extrapolation_matches_all_sites_ground_truth_across_contigs(self, tmp_path, n_contigs, span, distance):
        """Ground truth: the same data parsed as an all-sites input, which counts the monomorphic-involving pairs
        for real. Before the fix the short contigs were overestimated by an order of magnitude."""
        n_hap = 4
        rng = np.random.default_rng(0)
        derived = np.where(rng.random((n_contigs, span)) < 0.05, rng.integers(1, n_hap, size=(n_contigs, span)), 0)

        all_sites = write_sites(tmp_path / "all.vcf", n_contigs, span, derived, n_hap)
        snps = write_sites(tmp_path / "snp.vcf", n_contigs, span, derived, n_hap, only_polymorphic=True)

        Settings.disable_pbar = True
        kw = dict(n=n_hap, two_sfs=True, d=distance, skip_non_polarized=False, subsample_mode="random")

        truth = np.asarray(su.Parser(source=all_sites, **kw).parse()["all"].data)
        extrapolated = np.asarray(su.Parser(
            source=snps, **kw,
            target_site_counter=su.TargetSiteCounter(n_target_sites=n_contigs * span)
        ).parse()["all"].data)

        # the polymorphic block is observed directly, so any difference elsewhere is the extrapolation's own
        np.testing.assert_allclose(extrapolated[1:-1, 1:-1], truth[1:-1, 1:-1])

        assert extrapolated.sum() == pytest.approx(truth.sum(), rel=0.03)
        assert extrapolated[0, 0] == pytest.approx(truth[0, 0], rel=0.03)
        assert extrapolated[0, 1:-1].sum() == pytest.approx(truth[0, 1:-1].sum(), rel=0.03)


VCF_PATH = "resources/msprime/two_epoch.vcf"


requires_vcf = pytest.mark.skipif(not os.path.exists(VCF_PATH), reason="msprime fixtures absent")


@requires_vcf
def test_parse_two_sfs_fails_when_nothing_was_included(tmp_path):
    """The fixture is unpolarized, so every site is skipped; the two-SFS parse must not write an all-zero
    spectrum and report success."""
    out = str(tmp_path / 'two.json')

    assert run(['-q', 'parse', '--vcf', VCF_PATH, '--n', '8', '--two-sfs', '--output', out]) == 1
    assert not os.path.exists(out)


@requires_vcf
def test_parse_two_sfs_succeeds_when_sites_were_included(tmp_path):
    """The guard must not fire on a parse that did include sites."""
    out = str(tmp_path / 'two.json')

    assert run(['-q', 'parse', '--vcf', VCF_PATH, '--n', '8', '--two-sfs',
                '--no-skip-non-polarized', '--output', out]) == 0
    assert not su.TwoSpectra.from_file(out).is_empty


def test_target_site_counter_single_position_does_not_crash(tmp_path):
    """A TargetSiteCounter on input whose every contig spans a single position must skip monomorphic
    sampling rather than dividing by a zero range span and raising in rng.multinomial."""
    Settings.disable_pbar = True
    fasta = tmp_path / "g.fasta"
    fasta.write_text(">1\nACGTACGTAC\n")
    vcf = tmp_path / "one.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=1,length=10>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
        "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "s1", "s2"]) + "\n"
        + "\t".join(["1", "5", ".", "A", "T", ".", ".", ".", "GT", "0/1", "0/0"]) + "\n")

    # a single polymorphic site -> its contig spans a single position (range 0)
    spectra = su.Parser(source=str(vcf), n=4, skip_non_polarized=False, fasta=str(fasta),
                        target_site_counter=su.TargetSiteCounter(n_samples=100, n_target_sites=1000)).parse()
    assert spectra["all"].n_polymorphic == 1


def _prob_vcf(tmp_path):
    """A small VCF carrying AA + a Float AA_prob tag."""
    vcf = tmp_path / "prob.vcf"
    header = [
        "##fileformat=VCFv4.2", "##contig=<ID=1,length=100>",
        '##INFO=<ID=AA,Number=1,Type=String,Description="aa">',
        '##INFO=<ID=AA_prob,Number=1,Type=Float,Description="p">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">',
        "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "a", "b"]),
    ]
    # numeric GT allele indices (0=REF A, 1=ALT T), as htslib requires for a real VCF
    rows = ["\t".join(["1", str(p), ".", "A", "T", ".", ".", "AA=A;AA_prob=0.9", "GT", "0|1", "1|1"])
            for p in (10, 20, 30)]
    vcf.write_text("\n".join(header + rows) + "\n")
    return str(vcf)


def _prob_vcz(tmp_path):
    """The same sites as a VCF-Zarr store (INFO stored as strings)."""
    from sfsutils.io_handlers import ZarrVariantWriter
    out = str(tmp_path / "prob.vcz")
    w = ZarrVariantWriter(out, samples=["a", "b"], seqnames=["1"], info_ancestral="AA")
    for p in (10, 20, 30):
        w.write(Variant(ref="A", pos=p, chrom="1", gt_bases=["A|T", "T|T"], alt=["T"], is_snp=True,
                        info={"AA": "A", "AA_prob": 0.9}))
    w.close()
    return out


def test_probabilistic_polarization_agrees_across_vcf_and_zarr(tmp_path):
    """AA_prob is typed by cyvcf2 but a string from the Zarr backend; probabilistic polarization must
    cast it and give the same spectrum from either source (previously the Zarr path raised on '0.9'*array)."""
    Settings.disable_pbar = True

    def spectrum(source):
        return su.Parser(source=source, n=4, skip_non_polarized=True,
                         polarize_probabilistically=True).parse()["all"].to_list()

    from_vcf = spectrum(_prob_vcf(tmp_path))
    from_vcz = spectrum(_prob_vcz(tmp_path))
    # equal within float precision (cyvcf2 returns AA_prob as float32, the Zarr string casts to float64)
    np.testing.assert_allclose(from_vcf, from_vcz, atol=1e-6)
    assert sum(from_vcf) > 0  # sites were actually kept and polarized


def test_ancestral_prob_sentinels_treated_as_unpolarized():
    """Empty / '.' AA_prob values are treated as certain (probability 1), like a missing tag."""
    from sfsutils.io_handlers import Variant as V
    p = su.Parser(source=None, vcf="x", n=4, polarize_probabilistically=True) if False else None
    # exercise _get_ancestral_prob directly against the sentinels
    parser = su.Parser.__new__(su.Parser)
    parser.polarize_probabilistically = True
    parser.info_ancestral_prob = "AA_prob"
    parser.n_aa_prob = 0
    for sentinel in ("", ".", None):
        v = V(ref="A", pos=1, chrom="1", alt=["T"], is_snp=True,
              info={} if sentinel is None else {"AA_prob": sentinel})
        assert parser._get_ancestral_prob(v) == 1.0
    # a real string value is cast to float
    v = V(ref="A", pos=1, chrom="1", alt=["T"], is_snp=True, info={"AA_prob": "0.75"})
    assert parser._get_ancestral_prob(v) == 0.75


def test_zarr_degeneracy_stratification_end_to_end(tmp_path):
    """The headline workflow the '.' regression broke: a VCF-Zarr store carrying Degeneracy (ints for
    coding sites, '.' for non-coding) parses stratified into a non-empty neutral/selected SFS."""
    from sfsutils.io_handlers import ZarrVariantWriter
    Settings.disable_pbar = True

    out = str(tmp_path / "s.vcz")
    w = ZarrVariantWriter(out, samples=["a", "b"], seqnames=["1"], info_ancestral="AA")
    sites = [(10, 4), (20, 4), (30, 4), (40, 0), (50, 0), (60, ".")]  # 3 neutral, 2 selected, 1 non-coding
    for pos, deg in sites:
        w.write(Variant(ref="A", pos=pos, chrom="1", gt_bases=["A|T", "T|T"], alt=["T"], is_snp=True,
                        info={"AA": "A", "Degeneracy": deg}))
    w.close()

    spectra = su.Parser(source=out, n=4, skip_non_polarized=True,
                        stratifications=[su.DegeneracyStratification()]).parse()
    assert spectra["neutral"].n_polymorphic == 3
    assert spectra["selected"].n_polymorphic == 2


def test_base_context_stratification_uppercases_soft_masked(tmp_path):
    """BaseContextStratification must upper-case soft-masked (lowercase) flanking bases so they match the
    upper-case contexts, and skip a site whose context contains a non-ACGT base (e.g. N)."""
    Settings.disable_pbar = True
    import gzip
    # soft-masked reference: lowercase repeat bases around an upper-case site, plus an N
    fasta = tmp_path / "g.fasta.gz"
    with gzip.open(fasta, "wt") as fh:
        fh.write(">1\nacgTacgNtac\n")  # positions (1-based): 1..11
    vcf = tmp_path / "v.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=1,length=11>\n"
        '##INFO=<ID=AA,Number=1,Type=String,Description="aa">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">\n'
        "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "s1"]) + "\n"
        # a SNP at pos 4 (T, flanked by lowercase c/a -> context CTA); one at pos 8 (t, next to N -> skipped)
        + "\t".join(["1", "4", ".", "T", "A", ".", ".", "AA=T", "GT", "0/1"]) + "\n"
        + "\t".join(["1", "8", ".", "T", "A", ".", ".", "AA=T", "GT", "0/1"]) + "\n")

    spectra = su.Parser(source=str(vcf), n=2, skip_non_polarized=True,
                        stratifications=[su.BaseContextStratification(n_flanking=1, fasta=str(fasta))]).parse()
    # the valid site's context is upper-case ACGT (not a mixed-case 'cTa'); the N-flanked site is skipped
    assert all(t == t.upper() and set(t) <= set("ACGT") for t in spectra.types)
    assert any(spectra[t].n_polymorphic > 0 for t in spectra.types)


def test_absent_ancestral_allele_site_is_skipped(tmp_path):
    """A segregating biallelic SNP whose AA names a base absent from the genotypes is effectively
    multi-allelic and must be skipped, not polarised into the all-derived bin."""
    Settings.disable_pbar = True
    def sfs(aa):
        vcf = tmp_path / f"v_{aa}.vcf"
        vcf.write_text(
            "##fileformat=VCFv4.2\n##contig=<ID=1,length=100>\n"
            '##INFO=<ID=AA,Number=1,Type=String,Description="aa">\n'
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">\n'
            "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT",
                             "s1", "s2", "s3", "s4", "s5"]) + "\n"
            + "\t".join(["1", "10", ".", "C", "G", ".", ".", f"AA={aa}", "GT",
                         "0|0", "0|1", "1|1", "0|1", "0|0"]) + "\n")
        return su.Parser(source=str(vcf), n=10, skip_non_polarized=True).parse()
    assert list(sfs("A").types) == []              # AA absent from genotypes -> site skipped
    assert sfs("C")["all"].to_list()[4] == 1        # AA=C valid -> 4 derived G at bin 4


def test_target_site_counter_with_gt_reading_filtration(tmp_path):
    """A TargetSiteCounter's synthetic DummyVariant sites carry genotypes, so a gt_bases-reading
    filtration that is not removed during counting (e.g. SNVFiltration) does not crash on them."""
    Settings.disable_pbar = True
    (tmp_path / "g.fasta").write_text(">1\n" + "ACGT" * 50 + "\n")
    vcf = tmp_path / "v.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=1,length=200>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">\n'
        "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "s1", "s2"]) + "\n"
        + "\t".join(["1", "10", ".", "A", "T", ".", ".", ".", "GT", "0|1", "0|0"]) + "\n"
        + "\t".join(["1", "190", ".", "C", "G", ".", ".", ".", "GT", "0|1", "1|1"]) + "\n")

    spectra = su.Parser(source=str(vcf), n=4, skip_non_polarized=False, fasta=str(tmp_path / "g.fasta"),
                        filtrations=[su.SNVFiltration()],
                        target_site_counter=su.TargetSiteCounter(n_samples=2000, n_target_sites=2000)).parse()
    assert spectra["all"].n_polymorphic == 2  # did not crash on the dummy sites

    # the genotypes the filtration reads are the ones the dummy site carries, one per sample
    dummy = DummyVariant(ref="A", pos=1, chrom="1", n_samples=2, ploidy=2)
    assert list(dummy.gt_bases) == ["A/A", "A/A"]


def test_low_coverage_monomorphic_site_is_skipped(tmp_path):
    """A monomorphic site with fewer than n called genotypes is skipped like a low-coverage SNP, so the
    monomorphic:polymorphic ratio is not inflated (a full-coverage monomorphic site is still kept)."""
    Settings.disable_pbar = True
    def parse(gt):
        vcf = tmp_path / "v.vcf"
        vcf.write_text(
            "##fileformat=VCFv4.2\n##contig=<ID=1,length=100>\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">\n'
            "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT",
                             "s1", "s2", "s3"]) + "\n"
            + "\t".join(["1", "10", ".", "A", ".", ".", ".", ".", "GT"] + gt) + "\n")
        return su.Parser(source=str(vcf), n=4, skip_non_polarized=False).parse()

    assert parse(["0|0", "0|0", "0|0"])["all"].n_monomorphic == 1   # 6 haplotypes >= 4 -> kept
    assert list(parse(["0|0", ".|.", ".|."]).types) == []           # 2 haplotypes < 4 -> skipped


def test_target_site_counter_no_nan_for_sampling_only_strata(tmp_path):
    """Stratification types that first appear among the sampled monomorphic sites must not get NaN in
    their monomorphic bins: the pre-sampling snapshot is aligned onto the post-sampling types with
    zeros (as the joint path already does), so the whole target-site budget is allocated."""
    Settings.disable_pbar = True
    fasta = tmp_path / "g.fasta"
    fasta.write_text(">1\n" + "ACGTACGTAC" * 10 + "\n")
    vcf = tmp_path / "v.vcf"
    header = ("##fileformat=VCFv4.2\n##contig=<ID=1,length=100>\n"
              '##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">\n'
              "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO",
                               "FORMAT", "s1", "s2"]) + "\n")
    rows = "".join("\t".join(["1", str(p), ".", "T", "A", ".", ".", ".", "GT", "0|1", "0|0"]) + "\n"
                   for p in (4, 8, 12))
    vcf.write_text(header + rows)

    spectra = su.Parser(source=str(vcf), n=4, skip_non_polarized=False, fasta=str(fasta),
                        stratifications=[su.BaseContextStratification(n_flanking=1, fasta=str(fasta))],
                        filtrations=[su.SNPFiltration()],
                        target_site_counter=su.TargetSiteCounter(n_samples=200, n_target_sites=10000)).parse()

    assert not spectra.data.isna().any().any()          # no silent NaN in a returned spectrum
    assert (spectra.data >= 0).all().all()              # and no negative mutational opportunity

    # the whole target-site budget is allocated, up to the types whose observed sites outnumber their share of
    # it: those are clipped to zero monomorphic sites rather than left negative, which adds back the shortfall
    assert spectra.data.sum().sum() == pytest.approx(10000, rel=1e-3)
    assert spectra.data.sum().sum() >= 10000


def test_subsample_modes_agree_on_non_biallelic_site(tmp_path):
    """The probabilistic-polarization flip applies only to bi-allelic sites, so both subsample modes
    return the same spectrum for a site that is monomorphic among the included samples."""
    Settings.disable_pbar = True
    vcf = tmp_path / "v.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=1,length=100>\n"
        '##INFO=<ID=AA,Number=1,Type=String,Description="aa">\n'
        '##INFO=<ID=AA_prob,Number=1,Type=Float,Description="p">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">\n'
        "#" + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT",
                         "s1", "s2"]) + "\n"
        + "\t".join(["1", "10", ".", "A", "T", ".", ".", "AA=A;AA_prob=0.8", "GT", "0|0", "0|0"]) + "\n")

    def sfs(mode):
        return su.Parser(source=str(vcf), n=4, skip_non_polarized=True, polarize_probabilistically=True,
                         subsample_mode=mode).parse()["all"].to_list()

    assert sfs("random") == pytest.approx(sfs("probabilistic"))
