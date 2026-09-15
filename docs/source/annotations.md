# Site annotation
`sfsutils` can add site-level annotations such as the ancestral allele, site degeneracy, and synonymy. The {class}`~sfsutils.parser.Parser` applies them on the fly while it builds a spectrum, while the {class}`~sfsutils.annotation.Annotator` applies them and writes the annotated variants to a file. Polarising against an ancestral state is only needed for an unfolded spectrum: the {class}`~sfsutils.parser.Parser` reads it from the `AA` field by default (this can be customised), and otherwise the spectrum can be folded.

## Degeneracy Annotation
{class}`~sfsutils.annotation.DegeneracyAnnotation` annotates the SFS by the degeneracy of the site. This annotation requires information from a FASTA and GFF file and is useful for stratifying the SFS by 0-fold and 4-fold degenerate sites, a common way of contrasting putatively neutral and selected sites (see {class}`~sfsutils.parser.DegeneracyStratification`).

```{code-cell} python
import sfsutils as su

ann = su.Annotator(
    source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
    fasta="resources/genome/betula/genome.subset.20.fasta",
    gff="resources/genome/betula/genome.gff.gz",
    annotations=[su.DegeneracyAnnotation()],
    output="genome.deg.vcf.gz"
)

ann.annotate()
```

```{code-cell} python
:tags: [remove-cell]
import os

assert ann.annotations[0].n_annotated > 0 and os.path.getsize("genome.deg.vcf.gz") > 0
```

```{code-cell} r
library(sfsutils)
su <- load_sfsutils()

ann <- su$Annotator(
  source = "resources/genome/betula/biallelic.subset.10000.vcf.gz",
  fasta = "resources/genome/betula/genome.subset.20.fasta",
  gff = "resources/genome/betula/genome.gff.gz",
  annotations = list(su$DegeneracyAnnotation()),
  output = "genome.deg.vcf.gz"
)

ann$annotate()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(ann$annotations[[1]]$n_annotated > 0, file.size("genome.deg.vcf.gz") > 0)
```

+++
## Ancestral Allele Annotation
Currently, two ancestral allele annotations are available: {class}`~sfsutils.annotation.MaximumParsimonyAncestralAnnotation` and {class}`~sfsutils.annotation.MaximumLikelihoodAncestralAnnotation`. The former is error-prone and not recommended. Alternatively, if outgroups are missing, the spectra can be folded, though this discards information and yields a less informative spectrum. Ideally, we would like to use {class}`~sfsutils.annotation.MaximumLikelihoodAncestralAnnotation`, which is more sophisticated and requires one or several outgroups to be specified. Its underlying model is based on [`EST-SFS`](https://doi.org/10.1534/genetics.118.301120). The maximum-likelihood model estimates branch rates from monomorphic sites, so ideally these are present in the input. When there are few or none, {attr}`~sfsutils.annotation.MaximumLikelihoodAncestralAnnotation.n_target_sites` specifies the total number of sites (segregating and monomorphic) underlying the data, and a FASTA reference supplies the monomorphic sites to sample from.

```{code-cell} python
ann = su.Annotator(
    source="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
    fasta="resources/genome/betula/genome.subset.20.fasta",
    annotations=[su.MaximumLikelihoodAncestralAnnotation(
        outgroups=["ERR2103730"],
        n_ingroups=10,
        n_target_sites=200000
    )],
    output="genome.aa.vcf.gz"
)

ann.annotate()
```

```{code-cell} python
:tags: [remove-cell]
assert ann.annotations[0].n_annotated > 0 and os.path.getsize("genome.aa.vcf.gz") > 0
```

```{code-cell} r
ann <- su$Annotator(
  source = "resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
  fasta = "resources/genome/betula/genome.subset.20.fasta",
  annotations = list(su$MaximumLikelihoodAncestralAnnotation(
    outgroups = list("ERR2103730"),
    n_ingroups = 10,
    n_target_sites = 200000
  )),
  output = "genome.aa.vcf.gz"
)

ann$annotate()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(ann$annotations[[1]]$n_annotated > 0, file.size("genome.aa.vcf.gz") > 0)
```
