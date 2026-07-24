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

``sfsutils`` is compatible with Python 3.11 through 3.13.

.. note::

   The backends for the different input sources are optional extras: ``vcf`` (the :mod:`cyvcf2 <cyvcf2.cyvcf2>` dependency, for VCF
   files), ``zarr`` (the :mod:`zarr` dependency, for VCF-Zarr stores) and ``arg`` (the :mod:`tskit` dependency, for
   tree sequences / ARGs). Only ``vcf`` is installed by default; pass ``extras`` to change that:

   .. code-block:: r

      install_sfsutils(extras = c("vcf", "zarr", "arg"))

Alternatively, you can also follow the instructions in the `Python installation guide <../Python/installation.html>`_ to install the Python package.

Once installed, the ``sfsutils`` wrapper module can be loaded into your R environment using the following command:

.. code-block:: r

   su <- load_sfsutils()

See the R package documentation for more information on the available functions.
