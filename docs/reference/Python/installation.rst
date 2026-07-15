.. _reference.python.installation:

Installation
============

PyPI
^^^^
To install the ``sfsutils`` package via pip:

.. code-block:: bash

   pip install sfsutils

``sfsutils`` is compatible with Python 3.10 through 3.12.

.. note::

   The ``cyvcf2`` dependency, which is required for VCF handling, is optional.
   To enable VCF support, install with the ``vcf`` extra:

   .. code-block:: bash

      pip install sfsutils[vcf]

You are now ready to use ``sfsutils``:

.. code-block:: python

    import sfsutils as sf
