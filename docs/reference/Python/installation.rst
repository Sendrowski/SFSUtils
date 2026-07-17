.. _reference.python.installation:

Installation
============

PyPI
^^^^
``sfsutils`` is distributed on PyPI as ``sfsutils-popgen`` and on conda-forge as ``sfsutils`` (the import name stays ``sfsutils`` throughout). To install it via pip:

.. code-block:: bash

   pip install sfsutils-popgen

``sfsutils`` is compatible with Python 3.10 through 3.13.

.. note::

   The backends for the different input sources are optional extras: ``vcf`` (the ``cyvcf2`` dependency, for VCF
   files), ``zarr`` (the ``zarr`` dependency, for VCF-Zarr stores) and ``arg`` (the ``tskit`` dependency, for
   tree sequences / ARGs). Install the ones you need, for example all of them:

   .. code-block:: bash

      pip install sfsutils-popgen[vcf,zarr,arg]

Conda
^^^^^
``sfsutils`` is also available on **conda-forge**. To install it:

.. code-block:: bash

   mamba create -n sfsutils -c conda-forge sfsutils
   mamba activate sfsutils

.. note::

   The optional input backends are not pulled in automatically via conda. ``zarr`` and ``tskit`` are on
   **conda-forge**, while ``cyvcf2`` (for VCF handling) is on **bioconda**, so add both channels:

   .. code-block:: bash

      mamba create -n sfsutils -c conda-forge -c bioconda sfsutils cyvcf2 zarr tskit

   Alternatively, to ensure reproducibility, create a file ``environment.yml``:

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

   Then create and activate the environment:

   .. code-block:: bash

     mamba env create -f environment.yml
     mamba activate sfsutils

You are now ready to use ``sfsutils``:

.. code-block:: python

    import sfsutils as su
