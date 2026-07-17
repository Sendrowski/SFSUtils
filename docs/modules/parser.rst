.. _modules.parser:

VCF parsing
-----------

The :class:`~sfsutils.parser.Parser` reads variants into a site-frequency spectrum, optionally stratifying sites through a :class:`~sfsutils.parser.Stratification` and rescaling the spectrum with a :class:`~sfsutils.parser.TargetSiteCounter`.

.. autosummary::
   :nosignatures:

   ~sfsutils.parser.Parser
   ~sfsutils.parser.Stratification
   ~sfsutils.parser.SNPStratification
   ~sfsutils.parser.BaseContextStratification
   ~sfsutils.parser.BaseTransitionStratification
   ~sfsutils.parser.TransitionTransversionStratification
   ~sfsutils.parser.AncestralBaseStratification
   ~sfsutils.parser.DegeneracyStratification
   ~sfsutils.parser.SynonymyStratification
   ~sfsutils.parser.VEPStratification
   ~sfsutils.parser.SnpEffStratification
   ~sfsutils.parser.GenomePositionDependentStratification
   ~sfsutils.parser.ContigStratification
   ~sfsutils.parser.ChunkedStratification
   ~sfsutils.parser.RandomStratification
   ~sfsutils.parser.TargetSiteCounter

Parser
~~~~~~

.. autoclass:: sfsutils.parser.Parser

Stratification
~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.Stratification

SNPStratification
~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.SNPStratification

BaseContextStratification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.BaseContextStratification

BaseTransitionStratification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.BaseTransitionStratification

TransitionTransversionStratification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.TransitionTransversionStratification

AncestralBaseStratification
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.AncestralBaseStratification

DegeneracyStratification
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.DegeneracyStratification

SynonymyStratification
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.SynonymyStratification

VEPStratification
~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.VEPStratification

SnpEffStratification
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.SnpEffStratification

GenomePositionDependentStratification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.GenomePositionDependentStratification

ContigStratification
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.ContigStratification

ChunkedStratification
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.ChunkedStratification

RandomStratification
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.RandomStratification

TargetSiteCounter
~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.parser.TargetSiteCounter
