.. _modules.settings:

Settings
--------

:class:`~sfsutils.settings.Settings` holds package-wide defaults. Its attributes are set on the class
itself, so a change applies to every subsequent operation::

    import sfsutils

    sfsutils.Settings.disable_pbar = True

``parallelize`` acts as a global override rather than a default: setting it to ``False`` disables
parallel execution everywhere, even where a call asks for it. Left at ``None`` the decision is
deferred to the ``parallelize`` argument of the classes that expose one,
:class:`~sfsutils.annotation.MaximumLikelihoodAncestralAnnotation` and
:class:`~sfsutils.annotation.AdaptivePolarizationPrior`.

.. autosummary::
   :nosignatures:

   ~sfsutils.settings.Settings

Settings
~~~~~~~~

.. autoclass:: sfsutils.settings.Settings
   :members:
