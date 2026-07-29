.. _modules.annotation:

Site annotation
---------------

An :class:`~sfsutils.annotation.Annotation` adds site-level information such as degeneracy, synonymy or the ancestral allele. Annotations can be applied on the fly by the :class:`~sfsutils.parser.Parser` while it builds a spectrum, or run through the :class:`~sfsutils.annotation.Annotator` to write the annotated variants to a file.

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

.. autoclass:: sfsutils.annotation.Annotator

.. autoclass:: sfsutils.annotation.Annotation

.. autoclass:: sfsutils.annotation.DegeneracyAnnotation

.. autoclass:: sfsutils.annotation.SynonymyAnnotation

.. autoclass:: sfsutils.annotation.AncestralAlleleAnnotation

.. autoclass:: sfsutils.annotation.MaximumParsimonyAncestralAnnotation

.. autoclass:: sfsutils.annotation.SubstitutionModel

.. autoclass:: sfsutils.annotation.JCSubstitutionModel

.. autoclass:: sfsutils.annotation.K2SubstitutionModel

.. autoclass:: sfsutils.annotation.SiteConfig

.. autoclass:: sfsutils.annotation.SiteInfo

.. autoclass:: sfsutils.annotation.BaseType

.. autoclass:: sfsutils.annotation.PolarizationPrior

.. autoclass:: sfsutils.annotation.KingmanPolarizationPrior

.. autoclass:: sfsutils.annotation.AdaptivePolarizationPrior

.. autoclass:: sfsutils.annotation.MaximumLikelihoodAncestralAnnotation

.. autoclass:: sfsutils.annotation.AdHocAncestralAnnotation
