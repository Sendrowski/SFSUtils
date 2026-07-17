.. _reference.r.installation:

Installation
============

To install the ``sfsutils`` package in R, execute the following command:

.. code-block:: r

   devtools::install_github("Sendrowski/SFSUtils")

Once the installation is successfully completed, initiate the package within your R session using:

.. code-block:: r

   library(sfsutils)

The ``sfsutils`` R package serves as a wrapper around the Python library but re-implements visualization through ggplot2. Because of this, the Python package must be installed separately. This can be accomplished with:

.. code-block:: r

   install_sfsutils()

``sfsutils`` is compatible with Python 3.10 through 3.13.

.. note::

   The backends for the different input sources are optional extras: ``vcf`` (the ``cyvcf2`` dependency, for VCF
   files), ``zarr`` (the ``zarr`` dependency, for VCF-Zarr stores) and ``arg`` (the ``tskit`` dependency, for
   tree sequences / ARGs). To enable the ones you need, run the following command in R **before** calling
   ``install_sfsutils()``, for example for all of them:

   .. code-block:: r

      reticulate::py_install("sfsutils[vcf,zarr,arg]", pip = TRUE)

.. note::

   ``sfsutils`` is also available on **conda-forge**. As ``install_sfsutils()`` installs the Python package into a
   conda environment via reticulate, you can instead provision that environment directly. ``zarr`` and ``tskit``
   are on **conda-forge**, while ``cyvcf2`` (for VCF handling) is on **bioconda**, so add both channels:

   .. code-block:: bash

      mamba create -n sfsutils -c conda-forge -c bioconda sfsutils cyvcf2 zarr tskit

   Then point reticulate at that environment before loading the package:

   .. code-block:: r

      reticulate::use_condaenv("sfsutils", required = TRUE)

Alternatively, you can also follow the instructions in the `Python installation guide <../Python/installation.html>`_ to install the Python package.

Once installed, the ``sfsutils`` wrapper module can be loaded into your R environment using the following command:

.. code-block:: r

   su <- load_sfsutils()

See the R package documentation for more information on the available functions.
