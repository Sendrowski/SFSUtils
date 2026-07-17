.. _modules.filtration:

VCF filtration
--------------

The :class:`~sfsutils.filtration.Filterer` streams a VCF through a sequence of :class:`~sfsutils.filtration.Filtration` passes, writing out only the sites that pass every filter.

.. autosummary::
   :nosignatures:

   ~sfsutils.filtration.Filterer
   ~sfsutils.filtration.Filtration
   ~sfsutils.filtration.MaskedFiltration
   ~sfsutils.filtration.SNPFiltration
   ~sfsutils.filtration.SNVFiltration
   ~sfsutils.filtration.PolyAllelicFiltration
   ~sfsutils.filtration.AllFiltration
   ~sfsutils.filtration.NoFiltration
   ~sfsutils.filtration.CodingSequenceFiltration
   ~sfsutils.filtration.DeviantOutgroupFiltration
   ~sfsutils.filtration.ExistingOutgroupFiltration
   ~sfsutils.filtration.BiasedGCConversionFiltration
   ~sfsutils.filtration.CpGFiltration
   ~sfsutils.filtration.ContigFiltration

Filterer
~~~~~~~~

.. autoclass:: sfsutils.filtration.Filterer

Filtration
~~~~~~~~~~

.. autoclass:: sfsutils.filtration.Filtration

MaskedFiltration
~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.MaskedFiltration

SNPFiltration
~~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.SNPFiltration

SNVFiltration
~~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.SNVFiltration

PolyAllelicFiltration
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.PolyAllelicFiltration

AllFiltration
~~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.AllFiltration

NoFiltration
~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.NoFiltration

CodingSequenceFiltration
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.CodingSequenceFiltration

DeviantOutgroupFiltration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.DeviantOutgroupFiltration

ExistingOutgroupFiltration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.ExistingOutgroupFiltration

BiasedGCConversionFiltration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.BiasedGCConversionFiltration

CpGFiltration
~~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.CpGFiltration

ContigFiltration
~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.filtration.ContigFiltration
