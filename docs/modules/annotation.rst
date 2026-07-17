.. _modules.annotation:

VCF annotation
--------------

The :class:`~sfsutils.annotation.Annotator` applies one or more :class:`~sfsutils.annotation.Annotation` passes to a VCF, adding degeneracy, synonymy and ancestral-allele information used downstream by the parser.

.. autosummary::
   :nosignatures:

   ~sfsutils.annotation.Annotator
   ~sfsutils.annotation.Annotation
   ~sfsutils.annotation.DegeneracyAnnotation
   ~sfsutils.annotation.SynonymyAnnotation
   ~sfsutils.annotation.AncestralAlleleAnnotation
   ~sfsutils.annotation.MaximumParsimonyAncestralAnnotation
   ~sfsutils.annotation.SubstitutionModel
   ~sfsutils.annotation.JCSubstitutionModel
   ~sfsutils.annotation.K2SubstitutionModel
   ~sfsutils.annotation.SiteConfig
   ~sfsutils.annotation.SiteInfo
   ~sfsutils.annotation.BaseType
   ~sfsutils.annotation.PolarizationPrior
   ~sfsutils.annotation.KingmanPolarizationPrior
   ~sfsutils.annotation.AdaptivePolarizationPrior
   ~sfsutils.annotation.MaximumLikelihoodAncestralAnnotation
   ~sfsutils.annotation.AdHocAncestralAnnotation

Annotator
~~~~~~~~~

.. autoclass:: sfsutils.annotation.Annotator

Annotation
~~~~~~~~~~

.. autoclass:: sfsutils.annotation.Annotation

DegeneracyAnnotation
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.DegeneracyAnnotation

SynonymyAnnotation
~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.SynonymyAnnotation

AncestralAlleleAnnotation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.AncestralAlleleAnnotation

MaximumParsimonyAncestralAnnotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.MaximumParsimonyAncestralAnnotation

SubstitutionModel
~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.SubstitutionModel

JCSubstitutionModel
~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.JCSubstitutionModel

K2SubstitutionModel
~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.K2SubstitutionModel

SiteConfig
~~~~~~~~~~

.. autoclass:: sfsutils.annotation.SiteConfig

SiteInfo
~~~~~~~~

.. autoclass:: sfsutils.annotation.SiteInfo

BaseType
~~~~~~~~

.. autoclass:: sfsutils.annotation.BaseType

PolarizationPrior
~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.PolarizationPrior

KingmanPolarizationPrior
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.KingmanPolarizationPrior

AdaptivePolarizationPrior
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.AdaptivePolarizationPrior

MaximumLikelihoodAncestralAnnotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.MaximumLikelihoodAncestralAnnotation

AdHocAncestralAnnotation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.annotation.AdHocAncestralAnnotation
