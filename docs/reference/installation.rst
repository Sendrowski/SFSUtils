.. _reference.installation:

Installation
============

.. tab-set::
   :sync-group: language
   :class: code-tabs

   .. tab-item:: :fab:`python` Python
      :sync: python

      .. rubric:: PyPI

      ``sfsutils`` is distributed on PyPI as ``sfsutils-popgen`` and on conda-forge as ``sfsutils``, and is imported as ``sfsutils`` in both cases. It is installed with ``pip``:

      .. code-block:: bash

         pip install sfsutils-popgen

      ``sfsutils`` is compatible with Python 3.11 through 3.13.

      The backends for the different input sources are optional extras: ``vcf`` (the :mod:`cyvcf2 <cyvcf2.cyvcf2>` dependency, for VCF files), ``zarr`` (the :mod:`zarr` dependency, for VCF-Zarr stores) and ``arg`` (the :mod:`tskit` dependency, for tree sequences). All of them are installed with:

      .. code-block:: bash

         pip install sfsutils-popgen[vcf,zarr,arg]

      .. rubric:: Conda

      To avoid potential conflicts with other packages, it is recommended to install ``sfsutils`` in an isolated environment. The easiest way to do this is with ``conda`` or ``mamba``:

      .. code-block:: bash

         mamba create -n sfsutils -c conda-forge sfsutils
         mamba activate sfsutils

      The optional input backends are not installed automatically with conda. :mod:`zarr` and :mod:`tskit` are available on conda-forge, while ``cyvcf2`` is available on bioconda, so both channels are required:

      .. code-block:: bash

         mamba create -n sfsutils -c conda-forge -c bioconda sfsutils cyvcf2 zarr tskit

      Alternatively, for reproducibility, the environment can be defined in a file ``environment.yml``:

      .. code-block:: yaml

        name: sfsutils
        channels:
          - conda-forge
          - bioconda
        dependencies:
          - sfsutils
          - cyvcf2
          - zarr
          - tskit

      The environment is then created and activated with:

      .. code-block:: bash

        mamba env create -f environment.yml
        mamba activate sfsutils

      ``sfsutils`` is then imported with:

      .. code-block:: python

          import sfsutils as su

   .. tab-item:: :fab:`r-project` R
      :sync: r

      The ``sfsutils`` R package is installed from GitHub with:

      .. code-block:: r

         devtools::install_github("Sendrowski/SFSUtils")

      Once the installation has completed, the package is loaded in an R session with:

      .. code-block:: r

         library(sfsutils)

      The ``sfsutils`` R package serves as a wrapper around the Python library, and draws its figures with ``ggplot2``. Loading the R package declares the Python requirement, which ``reticulate`` resolves into a suitable environment the first time the module is loaded:

      .. code-block:: r

         su <- load_sfsutils()

      ``sfsutils`` is compatible with Python 3.11 through 3.13, and the declared requirement resolves against Python 3.11.

      The backends for the different input sources are optional extras: ``vcf`` (the :mod:`cyvcf2 <cyvcf2.cyvcf2>` dependency, for VCF files), ``zarr`` (the :mod:`zarr` dependency, for VCF-Zarr stores) and ``arg`` (the :mod:`tskit` dependency, for tree sequences). Only ``vcf`` is declared by default. Additional backends are declared by calling ``install_sfsutils()`` before the module is loaded:

      .. code-block:: r

         install_sfsutils(extras = c("vcf", "zarr", "arg"))
         su <- load_sfsutils()

      An existing Python installation can be used instead by installing ``sfsutils`` as described under the Python tab and selecting its environment before loading the module:

      .. code-block:: r

         reticulate::use_condaenv("~/miniforge3/envs/sfsutils", required = TRUE)
         su <- load_sfsutils()

      The R package documentation describes the available functions in more detail.
