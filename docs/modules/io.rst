.. _modules.io:

Input and output
----------------

The parser, filterer and annotator read variants through a single streamed site interface and write them back
in the format implied by the output file's extension. Any input backend (VCF, VCF-Zarr store, or tskit tree
sequence) is exposed to downstream code as a :class:`~sfsutils.io_handlers.Site`, and any output format is
written through a :class:`~sfsutils.io_handlers.VariantWriter`.

Site interface
~~~~~~~~~~~~~~

.. autoclass:: sfsutils.io_handlers.Site
   :members:

.. autoclass:: sfsutils.io_handlers.Variant
   :members:

Input sources
~~~~~~~~~~~~~

.. autoclass:: sfsutils.io_handlers.VariantReader
   :members:

.. autoclass:: sfsutils.io_handlers.TskitVariantReader
   :members:

.. autoclass:: sfsutils.io_handlers.ZarrVariantReader
   :members:

Output sinks
~~~~~~~~~~~~

.. autofunction:: sfsutils.io_handlers.open_writer

.. autoclass:: sfsutils.io_handlers.VariantWriter
   :members:

.. autoclass:: sfsutils.io_handlers.VCFVariantWriter
   :members:

.. autoclass:: sfsutils.io_handlers.ZarrVariantWriter
   :members:

.. autoclass:: sfsutils.io_handlers.TskitVariantWriter
   :members:
