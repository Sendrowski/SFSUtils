.. _modules.io:

Input and output
----------------

The :class:`~sfsutils.parser.Parser`, :class:`~sfsutils.filtration.Filterer` and :class:`~sfsutils.annotation.Annotator` read variants through a single streamed site interface and write them back in the format implied by the output file's extension. Any input backend (VCF, VCF-Zarr store, or tskit tree sequence) is exposed to downstream code as a :class:`~sfsutils.io_handlers.Site`, and any output format is written through a :class:`~sfsutils.io_handlers.VariantWriter`.

.. autosummary::
   :nosignatures:

   ~sfsutils.io_handlers.Site
   ~sfsutils.io_handlers.Variant
   ~sfsutils.io_handlers.VariantReader
   ~sfsutils.io_handlers.TskitVariantReader
   ~sfsutils.io_handlers.ZarrVariantReader
   ~sfsutils.io_handlers.VariantWriter
   ~sfsutils.io_handlers.VCFVariantWriter
   ~sfsutils.io_handlers.ZarrVariantWriter
   ~sfsutils.io_handlers.TskitVariantWriter
   ~sfsutils.io_handlers.open_writer

Site
~~~~

.. autoclass:: sfsutils.io_handlers.Site
   :members:

Variant
~~~~~~~

.. autoclass:: sfsutils.io_handlers.Variant
   :members:

VariantReader
~~~~~~~~~~~~~

.. autoclass:: sfsutils.io_handlers.VariantReader
   :members:

TskitVariantReader
~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.io_handlers.TskitVariantReader
   :members:

ZarrVariantReader
~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.io_handlers.ZarrVariantReader
   :members:

VariantWriter
~~~~~~~~~~~~~

.. autoclass:: sfsutils.io_handlers.VariantWriter
   :members:

VCFVariantWriter
~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.io_handlers.VCFVariantWriter
   :members:

ZarrVariantWriter
~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.io_handlers.ZarrVariantWriter
   :members:

TskitVariantWriter
~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.io_handlers.TskitVariantWriter
   :members:

open_writer
~~~~~~~~~~~

.. autofunction:: sfsutils.io_handlers.open_writer
