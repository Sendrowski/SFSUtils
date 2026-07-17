"""
Fast synthetic unit tests targeting otherwise-uncovered branches: TwoSFS/JointSFS arithmetic,
folding, interior/covariance/correlation, and plotting; the CLI helper parsers and dispatch; the
Visualization helpers; and the site-counting utility. All inputs are constructed in-process (small
numpy matrices), so nothing here touches a VCF/FASTA/GFF fixture and every test runs in the fast
tier under the Agg backend.
"""
import logging

import numpy as np
import pytest

import sfsutils as su
from sfsutils import TwoSFS, JointSFS
from sfsutils.io_handlers import count_sites
from sfsutils.visualization import Visualization


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test so figures never leak between the plot tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# --- TwoSFS arithmetic ------------------------------------------------------------------------

def _two_sfs(n=5, fill=1.0):
    return TwoSFS(np.full((n, n), fill))


def test_twosfs_arithmetic_with_twosfs_operand():
    """The TwoSFS-operand branch of *, //, / and the scalar ** branch."""
    a = _two_sfs()
    b = TwoSFS(np.full((5, 5), 2.0))

    assert np.allclose((a * b).data, 2.0)
    assert np.allclose((a // b).data, 0.0)
    assert np.allclose((b / a).data, 2.0)
    assert np.allclose((b ** 2).data, 4.0)


def test_twosfs_arithmetic_with_scalar_operand():
    """The scalar/array branch of //, /."""
    a = TwoSFS(np.full((5, 5), 6.0))

    assert np.allclose((a // 4).data, 1.0)
    assert np.allclose((a / 2).data, 3.0)


def test_twosfs_is_iterable():
    """AbstractSpectrum.__iter__ yields the rows of the matrix."""
    rows = list(iter(_two_sfs(n=4)))
    assert len(rows) == 4


# --- TwoSFS interior / covariance / correlation ------------------------------------------------

def test_twosfs_interior_raw_and_normalized():
    a = _two_sfs()
    raw = a.interior(normalize=False)
    assert raw.shape == (3, 3)

    norm = a.interior(normalize=True)
    assert norm.sum() == pytest.approx(1.0)


def test_twosfs_interior_normalize_empty_raises():
    """Interior all-zero (mass only on the monomorphic border) cannot be normalized."""
    data = np.zeros((5, 5))
    data[0, :] = 1.0
    data[-1, :] = 1.0
    with pytest.raises(ValueError, match="interior .* is empty"):
        TwoSFS(data).interior(normalize=True)


def test_twosfs_cov_and_corr():
    """cov()/corr() over a spectrum with monomorphic-border mass return full-size TwoSFS."""
    a = _two_sfs()
    cov = a.cov()
    corr = a.corr()
    assert isinstance(cov, TwoSFS) and cov.data.shape == (5, 5)
    assert isinstance(corr, TwoSFS)
    # correlation entries are clipped to [-1, 1]
    interior = corr.data[1:-1, 1:-1]
    assert np.all(np.abs(interior) <= 1.0 + 1e-9)


def test_twosfs_cov_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        TwoSFS(np.zeros((5, 5))).cov()


def test_twosfs_cov_non_finite_raises():
    data = np.ones((5, 5))
    data[2, 2] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        TwoSFS(data).cov()


def test_twosfs_cov_no_monomorphic_raises():
    """Mass only in the interior (no monomorphic pairs) leaves the covariance undefined."""
    data = np.zeros((5, 5))
    data[1:-1, 1:-1] = 1.0
    with pytest.raises(ValueError, match="no monomorphic-involving pairs"):
        TwoSFS(data).cov()


# --- TwoSFS folding / masking / plotting -------------------------------------------------------

def test_twosfs_fold_symmetrize_masks():
    a = _two_sfs()
    assert a.fold().data.shape == (5, 5)
    assert np.allclose(a.symmetrize().data, a.data)
    assert np.isnan(a.mask_diagonal().data).any()
    assert np.isnan(a.mask_upper().data).any()
    assert np.isnan(a.fill_monomorphic().data).any()
    assert a.get_max_abs() == pytest.approx(1.0)


def test_twosfs_plot():
    _two_sfs(n=6).plot(title="t", log_scale=True, show=True)


def test_twosfs_plot_surface():
    _two_sfs(n=6).plot_surface(title="t", show=True)


def test_twosfs_plot_folded_branch():
    """A folded 2-SFS triggers the truncation branch of plot()."""
    folded = _two_sfs(n=6).fold()
    assert folded.is_folded()
    folded.plot(show=False)


def test_twosfs_plot_surface_folded_branch():
    """A folded 2-SFS triggers the truncation branch of plot_surface()."""
    _two_sfs(n=6).fold().plot_surface(show=False)


def test_twosfs_plot_too_small_warns(caplog):
    small = TwoSFS(np.ones((2, 2)))
    with caplog.at_level(logging.WARNING, logger="sfsutils"):
        small.plot(show=False)
        small.plot_surface(show=False)
    assert "Nothing to plot" in caplog.text


# --- JointSFS ----------------------------------------------------------------------------------

def _joint(n=3):
    return JointSFS(np.ones((n, n)))


def test_jointsfs_zero_dim_raises():
    with pytest.raises(ValueError, match="at least 1-dimensional"):
        JointSFS(np.array(5.0))


def test_jointsfs_truediv():
    a = _joint()
    assert np.allclose((a / a).data, 1.0)
    assert np.allclose((a / 2).data, 0.5)


def test_jointsfs_plot():
    _joint().plot(title="t", show=True)


def test_jointsfs_plot_surface():
    _joint().plot_surface(title="t", show=True)


def test_jointsfs_marginalize_out_of_range_raises():
    with pytest.raises(ValueError, match="Population indices"):
        _joint().marginalize([0, 5])


# --- CLI helpers ------------------------------------------------------------------------------

def test_cli_parse_pops_skips_empty_groups():
    from sfsutils.cli import _parse_pops
    pops = _parse_pops("a=s1,s2;;b=s3;")
    assert pops == {"a": ["s1", "s2"], "b": ["s3"]}


def test_cli_configure_logging_levels():
    from sfsutils.cli import _configure_logging, logger
    _configure_logging(verbose=0, quiet=True)
    assert logger.level == logging.WARNING
    _configure_logging(verbose=1, quiet=False)
    assert logger.level == logging.DEBUG
    _configure_logging(verbose=0, quiet=False)
    assert logger.level == logging.INFO


def test_cli_run_without_subcommand_errors():
    from sfsutils.cli import run
    with pytest.raises(SystemExit):
        run([])


def test_cli_main_help_exits():
    from sfsutils.cli import main
    with pytest.raises(SystemExit):
        main(["--help"])


# --- Visualization helpers --------------------------------------------------------------------

def test_visualization_change_default_figsize():
    import matplotlib.pyplot as plt
    original = list(plt.rcParams["figure.figsize"])
    try:
        Visualization.change_default_figsize(2.0)
        assert list(plt.rcParams["figure.figsize"]) == [2.0 * original[0], 2.0 * original[1]]
    finally:
        plt.rcParams["figure.figsize"] = original


def test_visualization_show_and_save_writes_file(tmp_path):
    import matplotlib.pyplot as plt
    plt.plot([0, 1], [0, 1])
    out = tmp_path / "fig.png"
    Visualization.show_and_save(file=str(out), show=False)
    assert out.exists()


def test_visualization_plot_scatter_log_scale():
    Visualization.plot_scatter(values=[1, 2, 3], file=None, show=False, scale="log")


# --- jsonpickle handlers for raw numpy / pandas -----------------------------------------------

def test_numpy_array_handler_roundtrip():
    """The np.ndarray handler flatten/restore path (registered globally by the package import)."""
    import jsonpickle
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    back = jsonpickle.decode(jsonpickle.encode(arr))
    np.testing.assert_array_equal(back, arr)


def test_dataframe_handler_roundtrip():
    """The pandas DataFrame handler flatten/restore path."""
    import jsonpickle
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    back = jsonpickle.decode(jsonpickle.encode(df))
    assert list(back.columns) == list(df.columns)
    np.testing.assert_array_equal(back.values, df.values)


# --- stratification guards --------------------------------------------------------------------

def test_contig_stratification_filters_unlisted_contig():
    """ContigStratification.get_type raises when the contig is not in the requested list."""
    from sfsutils.parser import NoTypeException
    from sfsutils.io_handlers import DummyVariant
    strat = su.ContigStratification(contigs=["chrZ"])
    with pytest.raises(NoTypeException):
        strat.get_type(DummyVariant("A", 1, "chr1"))


def test_contig_stratification_returns_contig():
    from sfsutils.io_handlers import DummyVariant
    strat = su.ContigStratification()
    assert strat.get_type(DummyVariant("A", 1, "chr1")) == "chr1"


def test_base_transition_stratification_non_snp_raises():
    from sfsutils.parser import NoTypeException
    from sfsutils.io_handlers import Variant
    with pytest.raises(NoTypeException, match="not a SNP"):
        su.BaseTransitionStratification().get_type(Variant("A", 1, "chr1", is_snp=False))


def test_transition_transversion_invalid_alt_raises():
    from sfsutils.parser import NoTypeException
    from sfsutils.io_handlers import Variant
    v = Variant("A", 1, "chr1", alt=["N"], is_snp=True)
    with pytest.raises(NoTypeException, match="Invalid alternate allele"):
        su.TransitionTransversionStratification().get_type(v)


def test_synonymy_stratification_missing_tag_raises():
    from sfsutils.parser import NoTypeException
    from sfsutils.io_handlers import Variant
    with pytest.raises(NoTypeException, match="No synonymy tag"):
        su.SynonymyStratification().get_type(Variant("A", 1, "chr1", alt=["T"], is_snp=True))


# --- filtration dummy-variant branches --------------------------------------------------------

def test_deviant_outgroup_filtration_drops_dummy():
    """A monomorphic dummy site is dropped when monomorphic sites are not retained."""
    from sfsutils.io_handlers import DummyVariant
    f = su.DeviantOutgroupFiltration(outgroups=["o1"], retain_monomorphic=False)
    assert f.filter_site(DummyVariant("A", 1, "chr1")) is False


def test_existing_outgroup_filtration_keeps_dummy():
    from sfsutils.io_handlers import DummyVariant
    f = su.ExistingOutgroupFiltration(outgroups=["o1"])
    assert f.filter_site(DummyVariant("A", 1, "chr1")) is True


# --- io_handlers utility ----------------------------------------------------------------------

def test_count_sites_from_iterable():
    assert count_sites([object(), object(), object()]) == 3
