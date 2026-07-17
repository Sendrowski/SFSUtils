.. _modules.parser:

Parsing
-------

The :class:`~sfsutils.parser.Parser` reads variants into a site-frequency spectrum. When the input contains no monomorphic sites, a :class:`~sfsutils.parser.TargetSiteCounter` recovers the number of mutational target sites so the monomorphic bins, and hence the overall scale of the spectrum, are correct. Sites can be split into categories with a :class:`~sfsutils.parser.Stratification` (see :doc:`stratification`).

.. autosummary::
   :nosignatures:

   ~sfsutils.parser.Parser
   ~sfsutils.parser.TargetSiteCounter

Parser
~~~~~~

.. autoclass:: sfsutils.parser.Parser

TargetSiteCounter
~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.TargetSiteCounter
