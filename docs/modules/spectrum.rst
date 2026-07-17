.. _modules.spectrum:

Spectrum classes
----------------

The spectrum classes and their named-collection types, all defined in :mod:`sfsutils.spectrum`. The :class:`~sfsutils.spectrum.TwoSFS` is the two-dimensional spectrum of linked pairs of sites, whose departures from the outer product of the one-dimensional spectrum reflect linkage and non-Kingman genealogies. The :class:`~sfsutils.spectrum.JointSFS` is the multi-population spectrum whose entries count sites by their derived-allele count in each population.

.. autosummary::
   :nosignatures:

   ~sfsutils.spectrum.Spectrum
   ~sfsutils.spectrum.Spectra
   ~sfsutils.spectrum.JointSFS
   ~sfsutils.spectrum.JointSpectra
   ~sfsutils.spectrum.TwoSFS
   ~sfsutils.spectrum.TwoLocusSFS
   ~sfsutils.spectrum.TwoSpectra
   ~sfsutils.spectrum.AbstractSpectrum
   ~sfsutils.spectrum.AbstractSpectra

Spectrum
~~~~~~~~

.. autoclass:: sfsutils.spectrum.Spectrum

Spectra
~~~~~~~

.. autoclass:: sfsutils.spectrum.Spectra

JointSFS
~~~~~~~~

.. autoclass:: sfsutils.spectrum.JointSFS

JointSpectra
~~~~~~~~~~~~

.. autoclass:: sfsutils.spectrum.JointSpectra

TwoSFS
~~~~~~

.. autoclass:: sfsutils.spectrum.TwoSFS

TwoLocusSFS
~~~~~~~~~~~

.. autoclass:: sfsutils.spectrum.TwoLocusSFS

TwoSpectra
~~~~~~~~~~

.. autoclass:: sfsutils.spectrum.TwoSpectra

AbstractSpectrum
~~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.spectrum.AbstractSpectrum

AbstractSpectra
~~~~~~~~~~~~~~~

.. autoclass:: sfsutils.spectrum.AbstractSpectra
