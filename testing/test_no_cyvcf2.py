"""
Ensure sfsutils imports and its non-VCF functionality works when the optional ``cyvcf2`` dependency is
absent. VCF reading and writing are gated behind a lazy import, so only the actual VCF code paths may
require cyvcf2; importing the package and working with in-memory spectra must not.

The test hides ``cyvcf2`` from the import machinery in a subprocess so it does not disturb the rest of the
suite (which does use cyvcf2), and so a genuinely installed cyvcf2 cannot mask the failure.
"""
import subprocess
import sys
import textwrap


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True)


def test_import_and_spectra_without_cyvcf2():
    """With cyvcf2 blocked, importing sfsutils and using the spectrum containers must still work."""
    result = _run(
        """
        import builtins
        _real_import = builtins.__import__
        def _blocked(name, *a, **k):
            if name == 'cyvcf2' or name.startswith('cyvcf2.'):
                raise ImportError('cyvcf2 hidden for test')
            return _real_import(name, *a, **k)
        builtins.__import__ = _blocked

        import numpy as np
        import sfsutils as su

        # the spectrum containers need no VCF backend
        assert su.Spectrum([10, 5, 3, 2]).n_polymorphic > 0
        assert su.TwoSFS(np.ones((3, 3))).n == 3
        assert su.JointSFS(np.arange(6).reshape(2, 3), pop_names=['A', 'B']).n_pops == 2
        print('OK')
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


def _run_blocking(module: str, body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a subprocess with ``module`` hidden from the import machinery."""
    return _run(
        f"""
        import builtins
        _real_import = builtins.__import__
        def _blocked(name, *a, **k):
            if name == {module!r} or name.startswith({module + '.'!r}):
                raise ImportError('{module} hidden for test')
            return _real_import(name, *a, **k)
        builtins.__import__ = _blocked

        import sfsutils as su
        from sfsutils.settings import Settings
        Settings.disable_pbar = True
        {body}
        """
    )


def test_import_without_tskit_or_zarr():
    """sfsutils must import with neither tskit nor zarr installed."""
    for module in ("tskit", "zarr"):
        result = _run_blocking(module, "print('OK')")
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout


def test_tree_sequence_without_tskit_raises_clear_error():
    """Parsing a .trees path with tskit blocked must raise a clear ImportError naming the extra."""
    result = _run_blocking(
        "tskit",
        """
        try:
            su.Parser(source='x.trees', n=4).parse()
        except ImportError as e:
            assert 'tskit' in str(e)
            print('RAISED')
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "RAISED" in result.stdout


def test_vcf_zarr_without_zarr_raises_clear_error():
    """Parsing a .vcz path with zarr blocked must raise a clear ImportError naming the extra."""
    result = _run_blocking(
        "zarr",
        """
        try:
            su.Parser(source='x.vcz', n=4).parse()
        except ImportError as e:
            assert 'zarr' in str(e)
            print('RAISED')
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "RAISED" in result.stdout


def test_vcf_parsing_without_cyvcf2_raises_clear_error():
    """A VCF parse with cyvcf2 blocked must raise a clear ImportError naming the optional extra."""
    result = _run(
        """
        import builtins
        _real_import = builtins.__import__
        def _blocked(name, *a, **k):
            if name == 'cyvcf2' or name.startswith('cyvcf2.'):
                raise ImportError('cyvcf2 hidden for test')
            return _real_import(name, *a, **k)
        builtins.__import__ = _blocked

        import sfsutils as su
        from sfsutils.settings import Settings
        Settings.disable_pbar = True
        try:
            su.Parser(source='does_not_matter.vcf', n=4).parse()
        except ImportError as e:
            assert 'cyvcf2' in str(e)
            print('RAISED')
        """
    )
    assert result.returncode == 0, result.stderr
    assert "RAISED" in result.stdout
