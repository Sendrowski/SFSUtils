.. _modules.changelog:

Changelog
=========

[1.0.0] - 2026-07-20
^^^^^^^^^^^^^^^^^^^^
- First stable release, adding correctness and robustness fixes and documentation polish to the beta. Several
  of the fixes change the numbers the beta produced, most notably the two-SFS target-site extrapolation and
  the folding of the joint SFS, so results computed with the beta should be regenerated.

[0.1.0b2] - 2026-07-19
^^^^^^^^^^^^^^^^^^^^^^
- Generalize the parser input to a single ``source`` argument (VCF, VCF-Zarr, or tree sequence), add ``JointSFS.fold``, plus bug fixes and documentation improvements.

[0.1.0b1] - 2026-07-17
^^^^^^^^^^^^^^^^^^^^^^
- First (beta) release: SFS parsing, stratification, filtration and ancestral-allele / site-degeneracy annotation, factored out of ``fastdfe``.
