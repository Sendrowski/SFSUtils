# SFSUtils  <img align="right" width="100" src="https://raw.githubusercontent.com/Sendrowski/SFSUtils/master/docs/logo.png">
[![codecov](https://codecov.io/gh/Sendrowski/SFSUtils/branch/master/graph/badge.svg)](https://codecov.io/gh/Sendrowski/SFSUtils)
[![Documentation Status](https://readthedocs.org/projects/sfsutils/badge/?version=latest)](https://sfsutils.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/sfsutils.svg)](https://badge.fury.io/py/sfsutils)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.1093/molbev/msae070-blue)](https://doi.org/10.1093/molbev/msae070)

``sfsutils`` is a package for parsing site frequency spectra (SFS) from VCF files, with support for versatile stratification, ancestral allele and site-degeneracy annotation, and filtering. It provides the ``Spectrum``/``Spectra`` containers and a configurable VCF-to-SFS ``Parser``. It was factored out of [``fastdfe``](https://github.com/Sendrowski/fastDFE), which uses it for the data-preparation stage of DFE inference. VCF support is an optional dependency.

Please see the [documentation](https://sfsutils.readthedocs.io/en/latest/) for all the details.
