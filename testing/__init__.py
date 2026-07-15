"""
Initialization for the testing module.
"""
import logging
import os
import sys
from pathlib import Path
from unittest import TestCase as BaseTestCase

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt


def prioritize_installed_packages():
    """
    This function prioritizes installed packages over local packages.
    """
    # Get the current working directory
    cwd = str(Path().resolve())

    # Check if the current working directory is in sys.path
    if cwd in sys.path:
        # Remove the current working directory from sys.path
        sys.path = [p for p in sys.path if p != cwd]
        # Append the current working directory to the end of sys.path
        sys.path.append(cwd)


# run before importing sfsutils
prioritize_installed_packages()

import sfsutils

logger = logging.getLogger('sfsutils')

logger.info(sys.version)
logger.info(f"Running tests for {sfsutils.__file__}")
logger.info(f"sfsutils version: {sfsutils.__version__}")

# only show plots when running in PyCharm
if 'PYCHARM_HOSTED' not in os.environ:
    matplotlib.use('Agg')
else:
    logger.setLevel(logging.INFO)

# check for PARALLELIZE environment variable
if 'PARALLELIZE' in os.environ and os.environ['PARALLELIZE'].lower() == 'false':
    sfsutils.Settings.parallelize = False
    logger.info("Parallelization disabled.")

# create scratch directory if it doesn't exist
if not os.path.exists('scratch'):
    os.makedirs('scratch')


class TestCase(BaseTestCase):
    @pytest.fixture(autouse=True)
    def cleanup(self):
        """

        """
        yield
        plt.close('all')

    @staticmethod
    def rel_diff(a, b, eps=1e-12):
        """
        Compute the relative difference between a and b.
        """
        return np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)


def requires(*paths):
    """Skip a test if any required fixture file is absent (large/uncommitted data)."""
    import os as _os, pytest as _pytest
    missing = [p for p in paths if not _os.path.exists(p)]
    return _pytest.mark.skipif(bool(missing), reason=f"missing fixture(s): {missing}")


def requires_network():
    """Skip a test that downloads large remote data unless SFSUTILS_NETWORK_TESTS is set."""
    import os as _os, pytest as _pytest
    return _pytest.mark.skipif(
        not _os.environ.get("SFSUTILS_NETWORK_TESTS"),
        reason="requires network download of large remote data (set SFSUTILS_NETWORK_TESTS=1 to run)"
    )
