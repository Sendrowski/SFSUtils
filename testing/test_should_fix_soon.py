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


class TestFoldedFlag:
    """
    Folding is recorded explicitly, so a sparse unfolded spectrum is not mistaken for a folded one.
    """

    def test_sparse_unfolded_spectrum_not_reported_folded(self):
        sfs = Spectrum([10, 3, 2, 0, 0, 0, 5], folded=False)

        assert not sfs.is_folded()
        # subsampling must not fold it
        assert not sfs.subsample(4).is_folded()

    def test_fold_records_the_flag(self):
        assert Spectrum([10, 3, 2, 1, 5]).fold().is_folded()

    def test_inference_is_the_default(self):
        assert Spectrum([10, 3, 0, 0, 0]).is_folded()
        assert not Spectrum([10, 3, 2, 1, 5]).is_folded()


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
