"""
Execute an R (IRkernel) reference notebook in place so that Python-side logging emitted through
reticulate becomes visible in the rendered cell outputs.

reticulate does not passively forward the embedded Python interpreter's stdout/stderr to IRkernel
cell output under nbconvert, so ``sfsutils`` log lines would otherwise be lost. Redirecting the
Python streams does not help here: under IRkernel reticulate routes Python stderr to the process's
real file descriptor 2, which neither ``reticulate::py_capture_output`` (a Python ``sys.stderr``
swap) nor an ``os.dup2`` redirect reliably intercepts. This driver therefore captures at the
logging layer instead, which is independent of stream routing: it attaches a buffered
``logging.StreamHandler`` to the ``sfsutils`` logger for the duration of each cell and redirects the
tqdm progress bar into the same buffer, then emits the buffered text to R's stderr (via
``message()``, the only R stream nbconvert captures here) and tears the capture down. The buffered
progress bar is coalesced afterwards like the Python notebooks (docs/coalesce_streams.py).
``Settings.parallelize = False`` keeps the annotation classes that parallelize (currently
``MaximumLikelihoodAncestralAnnotation`` and ``AdaptivePolarizationPrior``) in the parent process,
so that every log record is emitted where the handler sees it; records from multiprocessing workers
would not propagate to it. Results are rendered as text/plain (``jupyter.rich_display = FALSE``) so a
scalar value shows in a proper output box rather than IRkernel's bare inline HTML.

The original (clean) source is restored before the notebook is written back, so the captured output
lands in the cell's outputs while the persisted source stays unwrapped. Cells that bind the Python
interpreter (``use_condaenv``, ``load_sfsutils``, ``library(reticulate)``) are left unwrapped:
calling into Python there would initialise it before the intended conda environment is selected.

Usage:  python execute_r_notebook.py <notebook.ipynb> [kernel_name] [timeout_seconds]
"""

import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from coalesce_streams import coalesce

# cells binding the Python interpreter must run before any capture call
_SKIP_MARKERS = ("use_condaenv", "load_sfsutils", "library(reticulate)")

# attach a buffered handler to the sfsutils logger (matching its own colored format) and redirect
# the tqdm progress bar into the same buffer, so both land in the captured output
_LOG_OPEN = (
    "import io, logging, sfsutils, tqdm\n"
    "_sfs_buf = io.StringIO()\n"
    "_sfs_h = logging.StreamHandler(_sfs_buf)\n"
    "_sfs_h.setFormatter(sfsutils.ColoredFormatter('%(levelname)s:%(name)s: %(message)s'))\n"
    "logging.getLogger('sfsutils').addHandler(_sfs_h)\n"
    "_sfs_tqdm_init = tqdm.std.tqdm.__init__\n"
    "def _sfs_patched_init(self, *a, **k):\n"
    "    k['file'] = _sfs_buf\n"
    # force unicode block glyphs; tqdm falls back to ASCII '#' when writing to a StringIO
    "    k.setdefault('ascii', False)\n"
    "    _sfs_tqdm_init(self, *a, **k)\n"
    "tqdm.std.tqdm.__init__ = _sfs_patched_init\n"
)

# restore tqdm, detach the handler, and read back what was buffered
_LOG_CLOSE = (
    "tqdm.std.tqdm.__init__ = _sfs_tqdm_init\n"
    "logging.getLogger('sfsutils').removeHandler(_sfs_h)\n"
    "_sfs_h.flush()\n"
    "_sfs_captured = _sfs_buf.getvalue()\n"
    "_sfs_buf.close()\n"
)


def _wrap(src: str) -> str:
    """
    Wrap a cell body so the ``sfsutils`` log records emitted during its evaluation are captured via a
    temporary logging handler and emitted to R's stderr, while preserving the block's own (visible)
    value for auto-printing. The parser is switched to serial execution so all its log records stay
    in the parent process where the handler sees them.

    :param src: The original R cell source.
    :return: The wrapped source.
    """
    return (
        '.sfsmod <- reticulate::import("sfsutils")\n'
        ".sfsmod$Settings$parallelize <- FALSE\n"
        # render results as text/plain (a proper output box), not IRkernel's bare text/html repr
        "options(jupyter.rich_display = FALSE)\n"
        f'reticulate::py_run_string("{_LOG_OPEN}")\n'
        ".res <- withVisible({\n"
        f"{src}\n"
        "})\n"
        f'reticulate::py_run_string("{_LOG_CLOSE}")\n'
        ".pyout <- reticulate::py$`_sfs_captured`\n"
        # message() reaches the IRkernel stderr stream; cat(file=stderr()) does not under nbconvert
        "if (!is.null(.pyout) && nzchar(.pyout)) message(.pyout, appendLF = FALSE)\n"
        "if (.res$visible) .res$value else invisible(.res$value)\n"
    )


class _CapturingExecutePreprocessor(ExecutePreprocessor):
    """
    An :class:`ExecutePreprocessor` that captures reticulate's Python output per cell and restores
    the original cell source afterwards.
    """

    def preprocess_cell(self, cell, resources, index):
        original = None

        if cell.cell_type == "code" and cell.source.strip() \
                and not any(m in cell.source for m in _SKIP_MARKERS):
            original = cell.source
            cell.source = _wrap(original)

        cell, resources = super().preprocess_cell(cell, resources, index)

        if original is not None:
            cell.source = original

        return cell, resources


def main(path: str, kernel: str = "ir", timeout: int = 1200) -> None:
    """
    Execute the notebook in place with per-cell Python-output capture.

    :param path: Path to the notebook.
    :param kernel: Jupyter kernel name.
    :param timeout: Per-cell execution timeout in seconds.
    """
    nb = nbformat.read(path, as_version=4)

    ep = _CapturingExecutePreprocessor(timeout=int(timeout), kernel_name=kernel)
    ep.preprocess(nb, {"metadata": {"path": "."}})

    # collapse the buffered progress bar to its final state here as well as in the snakemake rule, so
    # that running this driver directly cannot leave the intermediate tqdm frames in the outputs.
    # coalesce() rebuilds the outputs as plain dicts, so convert back before writing.
    coalesce(nb)
    nb = nbformat.from_dict(nb)

    nbformat.write(nb, path)


if __name__ == "__main__":
    main(*sys.argv[1:])
