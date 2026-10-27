"""
Smoke tests replaying the sfsutils code published in the fastDFE manuscript and the book chapter.

Each test reruns a published listing or a VCF script of the book chapter's workflow (``fastDFEBookChapter``) with
small site counts, written against the classes fastDFE re-exports from sfsutils. The book chapter's VCF pipeline runs
on the betula subset in place of the HGDP data, with the two betula outgroups standing in for the three great apes,
and its parsing scripts read a betula subset annotated beforehand. The tests check that the code runs, not the values
it returns.
"""
import shutil
import tempfile
from typing import List, Union

import numpy as np
import pandas as pd
import pytest

import sfsutils as su
from sfsutils.io_handlers import DummyVariant, NoTypeException, MultiHandler
from testing import TestCase

BETULA = "resources/genome/betula/"
FASTA = BETULA + "genome.subset.20.fasta"
OUTGROUPS = ['ERR2103730', 'ERR2103731']

#: Betula subset with ancestral alleles and degeneracy annotated, read by the parsing scripts
POLARIZED = BETULA + "all.polarized.deg.subset.10000.vcf.gz"


class RecombinationIntensityAnnotation(su.Annotation):
    """
    Recombination intensity annotation using a genetic map, as defined in the book chapter's ``utils.py``.
    """

    def __init__(self, map_file: str, bins_file: str = None, n_bins: int = 10):
        """
        Create a new annotation instance.
        """
        super().__init__()

        self.bins_file = bins_file

        self.n_bins = n_bins

        self.map: pd.DataFrame = pd.read_csv(map_file, comment='#', sep='\t')

        self.chr_cache = {chr_: df for chr_, df in self.map.groupby("Chr")}

        self.values: List[float] = []

        self.current: pd.Series = self.map.iloc[0]

    def _setup(self, handler: MultiHandler):
        """
        Provide context to the annotator.
        """
        super()._setup(handler)

        handler._reader.add_info_to_header({
            'ID': 'Recombination_intensity',
            'Number': '.',
            'Type': 'Float',
            'Description': 'Recombination intensity in cM/Mb.'
        })

    def _teardown(self):
        """
        Finalize the annotation. Called after all sites have been annotated.
        """
        super()._teardown()

        if self.bins_file is not None:
            pd.DataFrame(np.quantile(self.values, np.linspace(0, 1, self.n_bins + 1))).to_csv(self.bins_file)

    def annotate_site(self, variant: Union['cyvcf2.Variant', DummyVariant]):
        """
        Annotate site.
        """
        if (self.current is not None and variant.CHROM == self.current.Chr and
                self.current.Begin <= variant.POS < self.current.End):
            variant.INFO['Recombination_intensity'] = float(self.current.cMperMb)
            self.values.append(self.current.cMperMb)
            self.n_annotated += 1
            return

        chr_map = self.chr_cache.get(variant.CHROM)
        if chr_map is not None:
            mask = (chr_map.Begin <= variant.POS) & (variant.POS < chr_map.End)
            if mask.any():
                self.current = chr_map[mask].iloc[0]
                self.annotate_site(variant)


class RecombinationIntensityStratification(su.Stratification):
    """
    Recombination intensity stratification, as defined in the book chapter's ``utils.py``.
    """

    def __init__(self, bins_file: str):
        """
        Initialize stratification.
        """
        super().__init__()

        self.bins: pd.Series = pd.read_csv(bins_file).iloc[:, 1]

    def get_types(self) -> List[str]:
        """
        Get all possible types.
        """
        return [f"bin{i}" for i in range(len(self.bins) - 1)]

    def get_type(self, variant: Union['cyvcf2.Variant', DummyVariant]) -> str:
        """
        Get type.
        """
        try:
            rho = variant.INFO['Recombination_intensity']
            return f"bin{sum(rho >= self.bins) - 1}"
        except KeyError:
            raise NoTypeException()


@pytest.mark.slow
class ManuscriptTestCase(TestCase):
    """
    The sfsutils code listing of the manuscript.
    """

    def test_parser(self):
        """
        Listing 3: parse a degeneracy-stratified SFS with ancestral alleles from one outgroup.
        """
        p = su.Parser(
            n=8,
            vcf=BETULA + "biallelic.with_outgroups.subset.10000.vcf.gz",
            fasta=FASTA,
            gff=BETULA + "genome.gff.gz",
            target_site_counter=su.TargetSiteCounter(
                n_target_sites=350000
            ),
            annotations=[
                su.DegeneracyAnnotation(),
                su.MaximumLikelihoodAncestralAnnotation(
                    outgroups=["ERR2103730"],
                    max_sites=300
                )
            ],
            stratifications=[su.DegeneracyStratification()]
        )

        spectra: su.Spectra = p.parse()

        spectra.plot(show=False)

        assert set(spectra.types) == {'neutral', 'selected'}

@pytest.mark.slow
class BookChapterVCFTestCase(TestCase):
    """
    The VCF scripts of the book chapter's workflow, run on the betula subset.
    """

    #: Temporary directory holding the outputs and the recombination files
    tmp: str

    #: Ingroup samples, in the chapter's ``samples.csv`` layout
    samples: pd.DataFrame

    @classmethod
    def setUpClass(cls):
        """
        Write the sample table, a genetic map and its recombination intensity bins.
        """
        cls.tmp = tempfile.mkdtemp()

        import cyvcf2

        reader = cyvcf2.Reader(BETULA + "biallelic.with_outgroups.subset.10000.vcf.gz")
        ingroups = [s for s in reader.samples if s not in OUTGROUPS]
        reader.close()

        cls.samples = pd.DataFrame(dict(id=ingroups, species='betula_pendula'))

        rho = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]

        pd.DataFrame(dict(
            Chr=['Contig0', 'Contig1', 'Contig15'] * 2,
            Begin=[0] * 3 + [10 ** 5] * 3,
            End=[10 ** 5] * 3 + [10 ** 9] * 3,
            cMperMb=rho
        )).to_csv(f"{cls.tmp}/rho_map.csv", sep='\t', index=False)

        pd.DataFrame(np.quantile(rho, np.linspace(0, 1, 4))).to_csv(f"{cls.tmp}/rho_bins.csv")

    @classmethod
    def tearDownClass(cls):
        """
        Remove the temporary directory.
        """
        shutil.rmtree(cls.tmp)

    def test_annotate_and_filter(self):
        """
        ``annotate_ancestral_allele.py``, ``annotate_degeneracy.py``, ``annotate_recombination_intensity.py``,
        ``filter_vcf_snps.py`` and ``determine_ancestral_mismatches.py``, chained as in the workflow.
        """
        anc = su.MaximumLikelihoodAncestralAnnotation(
            model=su.K2SubstitutionModel(
                fix_transition_transversion_ratio=True
            ),
            outgroups=OUTGROUPS,
            ingroups=self.samples,
            max_sites=300,
            n_target_sites=3e6,
            prior=su.AdaptivePolarizationPrior()
        )

        su.Annotator(
            annotations=[anc],
            vcf=BETULA + "biallelic.with_outgroups.subset.10000.vcf.gz",
            fasta=FASTA,
            output=f"{self.tmp}/anc.vcf.gz",
        ).annotate()

        anc.to_file(f"{self.tmp}/anc.json")
        anc.get_observed_transition_transversion_ratio()
        anc.get_outgroup_divergence()

        su.Annotator(
            annotations=[su.DegeneracyAnnotation()],
            vcf=f"{self.tmp}/anc.vcf.gz",
            gff=BETULA + "genome.gff.gz",
            fasta=FASTA,
            output=f"{self.tmp}/anc.deg.vcf.gz"
        ).annotate()

        su.Annotator(
            annotations=[RecombinationIntensityAnnotation(f"{self.tmp}/rho_map.csv", f"{self.tmp}/rho_bins.out.csv")],
            vcf=f"{self.tmp}/anc.deg.vcf.gz",
            gff=BETULA + "genome.gff.gz",
            fasta=FASTA,
            output=f"{self.tmp}/anc.deg.rho.vcf.gz"
        ).annotate()

        su.Filterer(
            vcf=f"{self.tmp}/anc.vcf.gz",
            filtrations=[
                su.ExistingOutgroupFiltration(OUTGROUPS),
                su.SNPFiltration(include_samples=self.samples)
            ],
            output=f"{self.tmp}/anc.filtered.vcf.gz"
        ).filter()

    def test_parse_plain(self):
        """
        ``parse_sfs_plain.py``: folded SFS including non-polarized sites.
        """
        sfs: su.Spectra = su.Parser(
            n=20,
            vcf=BETULA + "biallelic.with_outgroups.subset.10000.vcf.gz",
            skip_non_polarized=False,
            include_samples=self.samples.id
        ).parse()

        sfs.fold().plot(show=False)
        sfs.to_file(f"{self.tmp}/sfs.plain.csv")

    def test_parse_unfolded(self):
        """
        ``parse_sfs_unfolded.py`` and ``parse_sfs_unfolded_deg.py``.
        """
        for stratifications in [[], [su.DegeneracyStratification()]]:
            sfs: su.Spectra = su.Parser(
                n=20,
                vcf=POLARIZED,
                include_samples=self.samples.id,
                polarize_probabilistically=True,
                stratifications=stratifications
            ).parse()

            sfs.plot(show=False)

    def test_parse_complete_and_by_contig(self):
        """
        ``parse_sfs_complete.py`` and ``parse_sfs_chr.py``: monomorphic sites, optionally stratified by contig.
        """
        for contigs in [None, ['Contig0', 'Contig1']]:
            p = su.Parser(
                n=20,
                vcf=POLARIZED,
                fasta=FASTA,
                gff=BETULA + "genome.gff.gz",
                include_samples=self.samples.id,
                stratifications=[su.DegeneracyStratification()] + (
                    [su.ContigStratification(contigs)] if contigs else []
                ),
                filtrations=([su.ContigFiltration(contigs)] if contigs else []) + [
                    su.SNPFiltration(),
                    su.ExistingOutgroupFiltration(OUTGROUPS)
                ],
                target_site_counter=su.TargetSiteCounter(
                    n_target_sites=2e7,
                    n_samples=1e4
                ),
                annotations=[su.DegeneracyAnnotation()],
                polarize_probabilistically=True
            )

            sfs: su.Spectra = p.parse()
            sfs.plot(show=False)

    def test_parse_random_and_recombination(self):
        """
        ``parse_sfs_random.py`` and ``parse_sfs_recombination.py``: random and custom recombination strata.
        """
        for stratification in [
            su.RandomStratification(n_bins=10),
            RecombinationIntensityStratification(f"{self.tmp}/rho_bins.csv")
        ]:
            p = su.Parser(
                n=20,
                vcf=POLARIZED,
                fasta=FASTA,
                gff=BETULA + "genome.gff.gz",
                include_samples=self.samples.id,
                stratifications=[
                    su.DegeneracyStratification(),
                    stratification
                ],
                filtrations=[
                    su.SNPFiltration(),
                    su.ExistingOutgroupFiltration(OUTGROUPS)
                ],
                target_site_counter=su.TargetSiteCounter(
                    n_target_sites=2e7,
                    n_samples=1e4
                ),
                annotations=[
                    RecombinationIntensityAnnotation(f"{self.tmp}/rho_map.csv"),
                    su.DegeneracyAnnotation()
                ],
                polarize_probabilistically=True
            )

            spectra: su.Spectra = p.parse()
            spectra.plot(show=False)
