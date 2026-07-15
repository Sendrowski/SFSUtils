.. _reference.cli.usage:

Command-line interface
======================

Installing ``sfsutils`` provides the ``sfsutils`` command, a thin wrapper around the Python API with three
subcommands: ``parse`` derives a spectrum from a VCF, while ``filter`` and ``annotate`` write a transformed VCF.

.. code-block:: bash

   sfsutils --help
   sfsutils parse --help

Global options ``-v``/``--verbose`` and ``-q``/``--quiet`` adjust logging; ``--version`` prints the version.

parse
-----

Derive a one-dimensional, joint (multi-population), or two-site SFS from a VCF. The output format follows the
spectrum: a single-population SFS is written as CSV, a joint or two-site SFS as JSON.

.. code-block:: bash

   # one-dimensional SFS, projected to 20 haplotypes
   sfsutils parse --vcf variants.vcf.gz --n 20 --out sfs.csv

   # neutral vs selected SFS, annotating and stratifying by degeneracy
   sfsutils parse --vcf variants.vcf.gz --n 20 \
       --fasta genome.fasta --gff genome.gff.gz \
       --annotate degeneracy --stratify degeneracy \
       --filter snp --out sfs.csv

   # joint SFS across two populations
   sfsutils parse --vcf variants.vcf.gz --n 10 \
       --pops "CEU=NA06984,NA06985;YRI=NA18486,NA18487" --out jsfs.json

   # two-site SFS, pairing sites within 1 kb
   sfsutils parse --vcf variants.vcf.gz --n 20 --two-sfs --two-sfs-distance 1000 --out sfs2.json

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Option
     - Description
   * - ``--vcf``
     - Input VCF (gzipped or a URL). Required.
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
   * - ``--two-sfs`` / ``--two-sfs-distance``
     - Parse the two-site SFS, pairing sites within the given number of base pairs (default 1000).
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

Write a VCF containing only the sites that pass the given filtrations.

.. code-block:: bash

   # keep only biallelic SNPs in coding sequences
   sfsutils filter --vcf variants.vcf.gz --filter snp,coding-sequence \
       --gff genome.gff.gz --out coding.vcf.gz

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Option
     - Description
   * - ``--vcf`` / ``--out``
     - Input VCF and output VCF (gzipped supported). Required.
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
