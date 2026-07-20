# SFSUtils  <img align="right" width="100" src="https://raw.githubusercontent.com/Sendrowski/SFSUtils/master/docs/logo.png">
[![codecov](https://codecov.io/gh/Sendrowski/SFSUtils/branch/master/graph/badge.svg)](https://codecov.io/gh/Sendrowski/SFSUtils)
[![Documentation Status](https://readthedocs.org/projects/sfsutils/badge/?version=latest)](https://sfsutils.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/sfsutils-popgen.svg)](https://badge.fury.io/py/sfsutils-popgen)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/sfsutils.svg)](https://anaconda.org/conda-forge/sfsutils)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Downloads](https://static.pepy.tech/badge/sfsutils-popgen)](https://pepy.tech/project/sfsutils-popgen)
[![DOI](https://img.shields.io/badge/DOI-10.1093/molbev/msae070-blue)](https://doi.org/10.1093/molbev/msae070)

``sfsutils`` is a package for parsing site frequency spectra (SFS), with support for versatile stratification, ancestral allele and site-degeneracy annotation, and filtering. Beyond the one-dimensional spectrum it also derives the joint SFS across several populations and the two-site SFS of linked pairs of sites, and reads from VCF files, VCF-Zarr stores, and tskit tree sequences (ARGs).

## Installation

```bash
pip install "sfsutils-popgen[vcf,zarr,arg]"
```

```bash
conda install -c conda-forge sfsutils
```

The distribution is `sfsutils-popgen` on PyPI, but the import name is `sfsutils`. The `vcf`, `zarr`, and `arg` extras pull in the optional backends for VCF, VCF-Zarr, and tskit tree-sequence input.

## Usage

Parse a spectrum from a polarised VCF and plot it:

```python
import sfsutils as su

sfs = su.Parser(n=10, source="biallelic.polarized.subset.vcf.gz").parse()
sfs.plot()
```

Please see the [documentation](https://sfsutils.readthedocs.io/en/latest/) for all the details.
