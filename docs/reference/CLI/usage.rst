.. _reference.cli.usage:

Command-line interface
======================

Installing ``sfsutils`` provides the ``sfsutils`` command, a thin wrapper around the Python API with three
subcommands: ``parse`` derives a spectrum from a VCF, a VCF-Zarr store, or a tskit tree sequence, while
``filter`` and ``annotate`` write a transformed dataset whose format follows the output file's extension (a
VCF, a VCF-Zarr store, or, from a tree-sequence input, a tree sequence).

.. code-block:: bash

   sfsutils --help
   sfsutils parse --help

Global options ``-v``/``--verbose`` and ``-q``/``--quiet`` adjust logging; ``--version`` prints the version.

parse
-----

Derive a one-dimensional, joint (multi-population), or two-site SFS from a VCF, a VCF-Zarr store, or a tskit
tree sequence. The output format follows the spectrum: a single-population SFS is written as CSV, a joint or
two-site SFS as JSON.

.. code-block:: bash

   # one-dimensional SFS, projected to 20 haplotypes
   sfsutils parse --vcf variants.vcf.gz --n 20 --out sfs.csv

   # the same, reading a VCF-Zarr store or a tree sequence instead
   sfsutils parse --zarr variants.vcz --n 20 --out sfs.csv
   sfsutils parse --trees ancestry.trees --n 20 --out sfs.csv

   # neutral vs selected SFS, annotating and stratifying by degeneracy
   sfsutils parse --vcf variants.vcf.gz --n 20 \
       --fasta genome.fasta --gff genome.gff.gz \
       --annotate degeneracy --stratify degeneracy \
       --filter snp --out sfs.csv

   # joint SFS across two populations
   sfsutils parse --vcf variants.vcf.gz --n 10 \
       --pops "CEU=NA06984,NA06985;YRI=NA18486,NA18487" --out jsfs.json

   # two-site SFS, pairing sites within 1 kb
   sfsutils parse --vcf variants.vcf.gz --n 20 --two-sfs --two-sfs-distance 1000 --out two_sfs.json

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Option
     - Description
   * - ``--vcf`` / ``--zarr`` / ``--trees``
     - Input source: a VCF (gzipped or a URL), a VCF-Zarr store (``.vcz`` / ``.zarr``), or a tskit tree sequence (``.trees``). Exactly one is required.
   * - ``--out``
     - Output spectrum (CSV for one population, JSON for a joint or two-site SFS). Required.
   * - ``--n``
     - Sample size to project to (per population for a joint SFS). Required.
   * - ``--pops``
     - Population spec ``name=sample1,sample2;...`` for a joint SFS.
   * - ``--fasta`` / ``--gff``
     - Reference files required by some annotations and filtrations.
   * - ``--stratify``
     - Comma-separated stratifications (``degeneracy``, ``synonymy``, ``base-transition``, ``transition-transversion``, ``ancestral-base``, ``contig``).
   * - ``--annotate``
     - Comma-separated on-the-fly annotations (``degeneracy``, ``maximum-likelihood-ancestral``).
   * - ``--filter``
     - Comma-separated filtrations (``snp``, ``snv``, ``poly-allelic``, ``coding-sequence``, ``cpg``, ``contig``, ``no``, ``all``). Default ``poly-allelic``.
   * - ``--contigs``
     - Contigs to keep (required by the ``contig`` stratification and filtration).
   * - ``--two-sfs`` / ``--two-sfs-distance`` / ``--two-sfs-offset``
     - Parse the two-site SFS, pairing sites separated by ``(offset, offset + distance]`` base pairs (distance default 1000, offset default 0).
   * - ``--no-skip-non-polarized``
     - Use the reference allele as ancestral where no ancestral tag is available.
   * - ``--subsample-mode``
     - ``random`` or ``probabilistic`` (default) down-sampling.
   * - ``--outgroups`` / ``--n-ingroups``
     - Outgroup samples and minimum ingroup count for the maximum-likelihood ancestral annotation.
   * - ``--info-ancestral``
     - INFO tag holding the ancestral allele (default ``AA``).

filter
------

Write only the sites that pass the given filtrations. The output format follows the ``--out`` extension: a VCF
(``.vcf``/``.vcf.gz``), a VCF-Zarr store (``.vcz``/``.zarr``), or a tree sequence (``.trees``). A VCF-Zarr store
can be written from any input; a tree sequence only from a tree-sequence input (the surviving sites are kept via
``delete_sites``), since a genealogy cannot be reconstructed from genotype data.

.. code-block:: bash

   # keep only biallelic SNPs in coding sequences
   sfsutils filter --vcf variants.vcf.gz --filter snp,coding-sequence \
       --gff genome.gff.gz --out coding.vcf.gz

   # the same, writing a VCF-Zarr store instead
   sfsutils filter --vcf variants.vcf.gz --filter snp,coding-sequence \
       --gff genome.gff.gz --out coding.vcz

   # subset a tree sequence to its SNP sites, keeping the genealogy
   sfsutils filter --trees ancestry.trees --filter snp --out coding.trees

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Option
     - Description
   * - ``--vcf`` / ``--zarr`` / ``--trees``
     - Input source (VCF, VCF-Zarr store, or tree sequence). Exactly one is required.
   * - ``--out``
     - Output path; its extension selects the format (``.vcf``/``.vcf.gz``, ``.vcz``/``.zarr``, ``.trees``). Required.
   * - ``--filter``
     - Comma-separated filtrations (see ``parse``). Required.
   * - ``--fasta`` / ``--gff``
     - References required by the ``cpg`` and ``coding-sequence`` filtrations.
   * - ``--contigs``
     - Contigs to keep (for the ``contig`` filtration).

annotate
--------

Write a VCF with added site-level information: site degeneracy from a reference, or the ancestral allele inferred
from outgroups under a maximum-likelihood substitution model.

.. code-block:: bash

   # annotate site degeneracy
   sfsutils annotate --vcf variants.vcf.gz --annotation degeneracy \
       --fasta genome.fasta --gff genome.gff.gz --out degeneracy.vcf.gz

   # infer the ancestral allele from two outgroups
   sfsutils annotate --vcf variants.with_outgroups.vcf.gz \
       --annotation maximum-likelihood-ancestral \
       --outgroups ERR2103730,ERR2103731 --n-ingroups 15 --out polarized.vcf.gz

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Option
     - Description
   * - ``--vcf`` / ``--out``
     - Input VCF and output VCF (gzipped supported). Required.
   * - ``--annotation``
     - Comma-separated annotations (``degeneracy``, ``maximum-likelihood-ancestral``). Required.
   * - ``--fasta`` / ``--gff``
     - References required by the ``degeneracy`` annotation.
   * - ``--outgroups`` / ``--n-ingroups``
     - Outgroup samples and minimum ingroup count for the maximum-likelihood ancestral annotation.

See also
--------

The subcommands wrap :class:`~sfsutils.parser.Parser`, :class:`~sfsutils.filtration.Filterer`, and
:class:`~sfsutils.annotation.Annotator`; see the Python reference for the full set of options.
