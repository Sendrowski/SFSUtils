.. _modules.changelog:

Changelog
=========

[Unreleased]
^^^^^^^^^^^^
- Joint (multi-population) site-frequency spectra via the ``pops`` argument of :class:`~sfsutils.parser.Parser`, returned as a :class:`~sfsutils.spectrum.JointSpectra` of :class:`~sfsutils.spectrum.JointSFS`.
- Two-site (two-locus) site-frequency spectra via ``two_sfs``, returned as a :class:`~sfsutils.spectrum.TwoSFS`.
- Abstract base classes :class:`~sfsutils.spectrum.AbstractSpectrum` and :class:`~sfsutils.spectrum.AbstractSpectra` for the spectrum and spectra containers.
- Command-line interface: ``sfsutils parse`` / ``filter`` / ``annotate``.
- R plotting for :class:`~sfsutils.spectrum.TwoSFS` and :class:`~sfsutils.spectrum.JointSFS`, and a mirrored R documentation reference.

[1.0.0] - 2026-07-15
^^^^^^^^^^^^^^^^^^^^
- Initial release. SFS parsing, stratification, filtration and ancestral-allele/site-degeneracy annotation, factored out of `fastdfe <https://github.com/Sendrowski/fastDFE>`_.
