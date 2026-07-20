import logging

import sfsutils as su
import numpy as np
import pandas as pd
import pytest
from sfsutils.io_handlers import get_called_bases
from unittest.mock import Mock
from testing import TestCase, requires, requires_network

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
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
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
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
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
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
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
            vcf='snakemake/results/vcf/sapiens/chr21.vep.vcf.gz',
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
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
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
            vcf='snakemake/results/vcf/sapiens/chr21.snpeff.vcf.gz',
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
            vcf='resources/genome/betula/all.polarized.subset.10000.vcf.gz',
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
            vcf='resources/genome/betula/all.polarized.subset.10000.vcf.gz',
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
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
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
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
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
            vcf="resources/genome/sapiens/chr21_test.vcf.gz",
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
            vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
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
            vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
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
            vcf="resources/genome/betula/all.polarized.subset.10000.vcf.gz",
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
            vcf="resources/genome/betula/all.polarized.subset.10000.vcf.gz",
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
            vcf="resources/genome/betula/biallelic.polarized.vcf.gz",
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
            vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
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
            vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
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

        # make sure that the sum of the target sites is the same
        self.assertEqual(sfs2.n_sites.sum(), 100000)

        # make sure ratio of neutral to selected is the same
        self.assertEqual(
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

        # make sure that the sum of the target sites is the same
        self.assertEqual(sfs2.n_sites.sum(), 100000)

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_parser_betula_include_samples(self):
        """
        Test that the parser includes only the samples that are given in the include_samples parameter.
        """
        p = su.Parser(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20,
            include_samples=['ASP01', 'ASP02', 'ASP03']
        )

        p._setup()

        self.assertEqual(np.sum(p._samples_mask), 3)

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_parser_betula_include_all_samples(self):
        """
        Test that the parser includes all samples if the include_samples parameter is not given.
        """
        p = su.Parser(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20
        )

        p._setup()

        self.assertEqual(np.sum(p._samples_mask), 377)

    @requires('resources/genome/betula/biallelic.subset.10000.vcf.gz')
    def test_parser_betula_exclude_two_samples(self):
        """
        Test that the parser excludes the samples that are given in the exclude_samples parameter.
        """
        p = su.Parser(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
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
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
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
            vcf="resources/genome/betula/biallelic.with_outgroups.subset.10000.vcf.gz",
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
                vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
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
                vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
                polarize_probabilistically=True,
                subsample_mode='random',
                max_sites=1000,
                n=n
            )

            sfs_prob = p1.parse()

            p2 = su.Parser(
                vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
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
                vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
                polarize_probabilistically=True,
                subsample_mode='probabilistic',
                max_sites=100,
                n=n
            )

            sfs_prob = p1.parse()

            p2 = su.Parser(
                vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
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
                vcf="resources/genome/sapiens/hgdp.anc.deg.vcf.gz",
                polarize_probabilistically=True,
                subsample_mode='random',
                max_sites=10000,
                n=n
            )

            sfs_prob = p1.parse()

            p2 = su.Parser(
                vcf="resources/genome/sapiens/hgdp.anc.deg.vcf.gz",
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
                vcf="resources/genome/sapiens/hgdp.anc.deg.vcf.gz",
                polarize_probabilistically=True,
                max_sites=10000,
                n=n
            )

            sfs_prob = p1.parse()

            p2 = su.Parser(
                vcf="resources/genome/sapiens/hgdp.anc.deg.vcf.gz",
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
            vcf=self.vcf,
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
            vcf='resources/genome/betula/all.subset.100000.vcf.gz',
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
            vcf=self.vcf,
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
            vcf='resources/genome/betula/all.subset.100000.vcf.gz',
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

    types = [s.get_type(None) for _ in range(9)]  # 9 > sum(chunk_sizes) == 6
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
