.. _modules.changelog:

Changelog
=========

[1.0.0] - 2026-07-20
^^^^^^^^^^^^^^^^^^^^
First stable release, consolidating extensive correctness, performance and robustness work on the beta.

- **Input sources.** Parse from VCF files, VCF-Zarr stores and tskit tree sequences (ARGs) interchangeably through a single ``source`` argument.
- **Spectra.** Derive the one-dimensional SFS, the joint (multi-population) SFS, and the two-site SFS (2-SFS), each with its own folding, covariance and plotting support.
- **Command line.** A ``sfsutils`` command-line tool with ``parse``, ``filter`` and ``annotate`` subcommands.
- **Performance.** Significant speedup across parsing and annotation, with indexed coding-sequence and FASTA lookups so annotation no longer scales with the annotation file.
- **Memory.** The VCF-Zarr writer streams to the store in chunks, keeping its footprint flat regardless of the number of sites.
- **Correctness.** Numerous fixes to the numbers produced, including the two-site SFS extrapolation, joint SFS folding, and consistency across the VCF, VCF-Zarr and tree-sequence backends.
- **Compatibility.** Requires Python 3.11 or newer.

[0.1.0b2] - 2026-07-19
^^^^^^^^^^^^^^^^^^^^^^
- Generalize the parser input to a single ``source`` argument (VCF, VCF-Zarr, or tree sequence), add ``JointSFS.fold``, plus bug fixes and documentation improvements.

[0.1.0b1] - 2026-07-17
^^^^^^^^^^^^^^^^^^^^^^
- First (beta) release: SFS parsing, stratification, filtration and ancestral-allele / site-degeneracy annotation, factored out of ``fastdfe``.
