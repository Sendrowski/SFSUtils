.. _modules.io:

Input and output
----------------

The :class:`~sfsutils.parser.Parser`, :class:`~sfsutils.filtration.Filterer` and :class:`~sfsutils.annotation.Annotator` read variants through a single streamed site interface and write them back in the format implied by the output file's extension. Any input backend (VCF, VCF-Zarr store, or tskit tree sequence) is exposed to downstream code as a :class:`~sfsutils.io_handlers.Site`, and any output format is written through a :class:`~sfsutils.io_handlers.VariantWriter`.

.. autosummary::
   :nosignatures:

   ~sfsutils.io_handlers.Site
   ~sfsutils.io_handlers.NoTypeException
   ~sfsutils.io_handlers.Variant
   ~sfsutils.io_handlers.VariantReader
   ~sfsutils.io_handlers.TskitVariantReader
   ~sfsutils.io_handlers.ZarrVariantReader
   ~sfsutils.io_handlers.VariantWriter
   ~sfsutils.io_handlers.VCFVariantWriter
   ~sfsutils.io_handlers.ZarrVariantWriter
   ~sfsutils.io_handlers.TskitVariantWriter
   ~sfsutils.io_handlers.FileHandler
   ~sfsutils.io_handlers.VCFHandler
   ~sfsutils.io_handlers.FASTAHandler
   ~sfsutils.io_handlers.GFFHandler

.. autoclass:: sfsutils.io_handlers.Site
   :members:

.. autoclass:: sfsutils.io_handlers.Variant
   :members:

.. autoclass:: sfsutils.io_handlers.VariantReader
   :members:

.. autoclass:: sfsutils.io_handlers.TskitVariantReader
   :members:

.. autoclass:: sfsutils.io_handlers.ZarrVariantReader
   :members:

.. autoclass:: sfsutils.io_handlers.VariantWriter
   :members:

.. autoclass:: sfsutils.io_handlers.VCFVariantWriter
   :members:

.. autoclass:: sfsutils.io_handlers.ZarrVariantWriter
   :members:

.. autoclass:: sfsutils.io_handlers.TskitVariantWriter
   :members:

.. autoclass:: sfsutils.io_handlers.NoTypeException
   :members:

.. autoclass:: sfsutils.io_handlers.FileHandler
   :members:

.. autoclass:: sfsutils.io_handlers.VCFHandler
   :members:

.. autoclass:: sfsutils.io_handlers.FASTAHandler
   :members:

.. autoclass:: sfsutils.io_handlers.GFFHandler
   :members:
