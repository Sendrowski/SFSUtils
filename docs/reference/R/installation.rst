.. _reference.r.installation:

Installation
============

To install the ``sfsutils`` package in R, execute the following command:

.. code-block:: r

   devtools::install_github("Sendrowski/SFSUtils")

Once the installation is successfully completed, initiate the package within your R session using:

.. code-block:: r

   library(sfsutils)

The ``sfsutils`` R package serves as a wrapper around the Python library but re-implements visualization through ggplot2. Loading the R package declares the Python requirement, which reticulate resolves into a suitable environment the first time the module is loaded:

.. code-block:: r

   su <- load_sfsutils()

``sfsutils`` is compatible with Python 3.11 through 3.13, and the declared requirement resolves against Python 3.11.

.. note::

   The backends for the different input sources are optional extras: ``vcf`` (the :mod:`cyvcf2 <cyvcf2.cyvcf2>` dependency, for VCF
   files), ``zarr`` (the :mod:`zarr` dependency, for VCF-Zarr stores) and ``arg`` (the :mod:`tskit` dependency, for
   tree sequences / ARGs). Only ``vcf`` is declared by default. Additional backends are declared by calling
   ``install_sfsutils()`` before the module is loaded:

   .. code-block:: r

      install_sfsutils(extras = c("vcf", "zarr", "arg"))
      su <- load_sfsutils()

To use an existing Python installation instead, follow the `Python installation guide <../Python/installation.html>`_ and select the environment before loading the module:

.. code-block:: r

   reticulate::use_condaenv("~/miniforge3/envs/sfsutils", required = TRUE)
   su <- load_sfsutils()

See the R package documentation for more information on the available functions.
