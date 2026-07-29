.. _modules.filtration:

Site filtration
---------------

A :class:`~sfsutils.filtration.Filtration` drops sites that violate downstream modelling assumptions. Filtrations can be applied on the fly by the :class:`~sfsutils.parser.Parser` while it builds a spectrum, or run through the :class:`~sfsutils.filtration.Filterer` to write the retained sites to a file.

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

.. autoclass:: sfsutils.filtration.Filterer

.. autoclass:: sfsutils.filtration.Filtration

.. autoclass:: sfsutils.filtration.MaskedFiltration

.. autoclass:: sfsutils.filtration.SNPFiltration

.. autoclass:: sfsutils.filtration.SNVFiltration

.. autoclass:: sfsutils.filtration.PolyAllelicFiltration

.. autoclass:: sfsutils.filtration.AllFiltration

.. autoclass:: sfsutils.filtration.NoFiltration

.. autoclass:: sfsutils.filtration.CodingSequenceFiltration

.. autoclass:: sfsutils.filtration.DeviantOutgroupFiltration

.. autoclass:: sfsutils.filtration.ExistingOutgroupFiltration

.. autoclass:: sfsutils.filtration.BiasedGCConversionFiltration

.. autoclass:: sfsutils.filtration.CpGFiltration

.. autoclass:: sfsutils.filtration.ContigFiltration
