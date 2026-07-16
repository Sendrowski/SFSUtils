.. _modules.changelog:

Changelog
=========

[Unreleased]
^^^^^^^^^^^^
- Joint (multi-population) site-frequency spectra via the ``pops`` argument of :class:`~sfsutils.parser.Parser`, returned as a :class:`~sfsutils.spectrum.JointSpectra` of :class:`~sfsutils.spectrum.JointSFS`.
- Two-site (two-locus) site-frequency spectra via ``two_sfs``, returned as a :class:`~sfsutils.spectrum.TwoSFS`. Stratified two-site parsing counts only within-stratum pairs, returned as a :class:`~sfsutils.spectrum.TwoSpectra`.
- :meth:`~sfsutils.spectrum.TwoSFS.covariance` and :meth:`~sfsutils.spectrum.TwoSFS.correlation` of the derived-allele frequency between two linked polymorphic sites, computed from the segregating interior (:meth:`~sfsutils.spectrum.TwoSFS.interior`) and therefore independent of the monomorphic sites and of any target-site count.
- Reading variants from VCF-Zarr stores (optional ``zarr`` extra) and tskit tree sequences / ARGs (optional ``tskit`` extra), in addition to VCF, through a common streamed site interface exposed as the :class:`~sfsutils.io_handlers.Site` protocol. The ``parse`` CLI subcommand accepts ``--vcf`` / ``--zarr`` / ``--trees``.
- Writing filtered or annotated data to a VCF-Zarr store or (from a tree-sequence input) a tree sequence, chosen by the output file's extension. The ``filter`` and ``annotate`` CLI subcommands accept the same ``--vcf`` / ``--zarr`` / ``--trees`` inputs.
- :class:`~sfsutils.parser.TargetSiteCounter` support for the joint SFS (scaling the all-ancestral corner to the target-site count) and the two-SFS (extrapolating the monomorphic-involving pairs analytically from the target-site count, apportioned across strata by sampling when stratifications are used).
- Abstract base classes :class:`~sfsutils.spectrum.AbstractSpectrum` and :class:`~sfsutils.spectrum.AbstractSpectra` for the spectrum and spectra containers.
- Command-line interface: ``sfsutils parse`` / ``filter`` / ``annotate``.
- R plotting for :class:`~sfsutils.spectrum.TwoSFS` and :class:`~sfsutils.spectrum.JointSFS`, and a mirrored R documentation reference.

[1.0.0] - 2026-07-15
^^^^^^^^^^^^^^^^^^^^
- Initial release. SFS parsing, stratification, filtration and ancestral-allele/site-degeneracy annotation, factored out of `fastdfe <https://github.com/Sendrowski/fastDFE>`_.
