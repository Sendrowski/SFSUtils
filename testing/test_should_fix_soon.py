"""
Regressions for the medium-severity findings of the release-readiness scan.
"""
import numpy as np
import pandas as pd
import pytest

from sfsutils.io_handlers import get_called_alleles
from sfsutils.spectrum import Spectrum, Spectra


class TestCalledAlleles:
    """
    Multi-character alleles must count as one allele per haplotype.
    """

    def test_mnp_counts_as_two_alleles(self):
        """A bi-allelic MNP has two alleles, not four bases."""
        assert list(get_called_alleles(['AT|GC', 'AT|AT'])) == ['AT', 'GC']

    def test_missing_calls_ignored(self):
        """Missing calls do not contribute an allele."""
        assert list(get_called_alleles(['./.', 'A|T'])) == ['A', 'T']


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


class TestNumpyScalarArithmetic:
    """
    A numpy scalar on the left must defer to the reflected operator instead of broadcasting.
    """

    def test_spectrum_rmul(self):
        assert (np.float64(2) * Spectrum([1, 2, 3])).data.tolist() == [2, 4, 6]

    def test_spectra_rmul(self):
        spectra = Spectra(dict(a=[1, 2, 3]))

        assert (np.float64(2) * spectra).data['a'].tolist() == [2, 4, 6]


class TestMultiIndexRoundTrip:
    """
    ``MultiIndex`` axes must survive serialization.
    """

    def test_multiindex_columns_restored(self):
        from sfsutils.json_handlers import DataframeHandler

        df = pd.DataFrame([[1, 2], [3, 4]], columns=pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y')]))
        handler = DataframeHandler.__new__(DataframeHandler)

        restored = handler.restore(handler.flatten(df, {}))

        assert isinstance(restored.columns, pd.MultiIndex)
        pd.testing.assert_frame_equal(restored, df, check_dtype=False)


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


class TestCLIWiring:
    """
    ``--contigs`` reaches the contig stratification, and a malformed ``--pops`` exits cleanly.
    """

    def test_contigs_reach_stratification(self):
        from sfsutils.cli import _build_stratifications

        assert _build_stratifications(['contig'], ['chr1'])[0].contigs == ['chr1']

    def test_malformed_pops_exits(self):
        from sfsutils.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(['parse', '--source', 'x.vcf', '--n', '10', '--out', 'o.csv',
                                       '--pops', 'nonsense'])
