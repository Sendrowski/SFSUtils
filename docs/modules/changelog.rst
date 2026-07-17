.. _modules.changelog:

Changelog
=========

[0.1.0b1] - 2026-07-17
^^^^^^^^^^^^^^^^^^^^^^
- First (beta) release: SFS parsing, stratification, filtration and ancestral-allele / site-degeneracy annotation, factored out of `fastdfe <https://github.com/Sendrowski/fastDFE>`_.
- Joint (multi-population) site-frequency spectra via the ``pops`` argument of :class:`~sfsutils.parser.Parser`, returned as a :class:`~sfsutils.spectrum.JointSpectra` of :class:`~sfsutils.spectrum.JointSFS`.
- Two-site (two-locus) site-frequency spectra via ``two_sfs``, returned as a :class:`~sfsutils.spectrum.TwoSFS`. Stratified two-site parsing counts only within-stratum pairs, returned as a :class:`~sfsutils.spectrum.TwoSpectra`.
- Class-resolved branch-length covariance :meth:`~sfsutils.spectrum.TwoSFS.cov` and correlation :meth:`~sfsutils.spectrum.TwoSFS.corr` of two linked sites, ``Cov(L_i, L_j)``, each returned as a :class:`~sfsutils.spectrum.TwoSFS` over the segregating interior. They are normalized over the full spectrum, so they require the monomorphic sites (parse an all-sites input); this matches PhaseGen's ``sfs2.mean - outer(sfs.mean)`` and reproduces the multiple-merger signal. A polymorphic-only spectrum raises.
- Frequency pointwise mutual information :meth:`~sfsutils.spectrum.TwoSFS.fpmi` of two linked sites (Fenton et al. 2025), a ratio statistic over the polymorphic interior that needs no monomorphic sites and is exactly invariant to them, validated against PhaseGen.
- Reading variants from VCF-Zarr stores (optional ``zarr`` extra) and tskit tree sequences / ARGs (optional ``arg`` extra), in addition to VCF, through a common streamed site interface exposed as the :class:`~sfsutils.io_handlers.Site` protocol. The ``parse`` CLI subcommand accepts ``--vcf`` / ``--zarr`` / ``--trees``.
- Writing filtered or annotated data to a VCF-Zarr store or (from a tree-sequence input) a tree sequence, chosen by the output file's extension. The ``filter`` and ``annotate`` CLI subcommands accept the same ``--vcf`` / ``--zarr`` / ``--trees`` inputs.
- :class:`~sfsutils.parser.TargetSiteCounter` support for the joint SFS and the unstratified two-SFS, scaling the monomorphic sites to the target-site count.
- Abstract base classes :class:`~sfsutils.spectrum.AbstractSpectrum` and :class:`~sfsutils.spectrum.AbstractSpectra` for the spectrum and spectra containers.
- Command-line interface: ``sfsutils parse`` / ``filter`` / ``annotate``.
- R plotting for :class:`~sfsutils.spectrum.TwoSFS` and :class:`~sfsutils.spectrum.JointSFS`, and a mirrored R documentation reference.
