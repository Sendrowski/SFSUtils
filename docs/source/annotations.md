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
The unfolded SFS requires the ancestral allele at each site. It is inferred with [``ancestree``](https://ancestree.readthedocs.io), which by default writes it to the `AA` field and its posterior to `AA_post`, both read by {class}`~sfsutils.parser.Parser`. Here we use its local-tree inference ({meth}`Inference.from_local_tree() <ancestree.inference.Inference.from_local_tree>`), which infers the genealogies with a pairwise-coalescent HMM rather than taking a pre-built one. The HMM samples coalescent times between each pair of haplotypes along the genome, and each draw is clustered into a dated tree per window. The ancestral allele is read at the ingroup's most recent common ancestor and averaged over the sampled trees. The suffixes `_h0` and `_h1` denote the first and second haplotype of a diploid sample. See [``ancestree``](https://ancestree.readthedocs.io)'s [local-tree inference guide](https://ancestree.readthedocs.io/en/latest/reference/Python/local_tree_inference.html) for details.

```{code-cell} python
import ancestree as anc

inf = anc.Inference.from_local_tree(
    "resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
    sample_names=[f"ASP{i:02d}_h{h}" for i in range(1, 21) for h in (0, 1)] + ["ERR2103730_h0", "ERR2103731_h0"],
    outgroup_samples=["ERR2103730_h0", "ERR2103731_h0"],
    model=anc.JC69(),
    rec_rate=4e-8,
    mu=8e-9
)

inf.to_vcf("genome.aa.vcf.gz");
```

```{code-cell} python
:tags: [remove-cell]
import gzip
import re

with gzip.open("genome.aa.vcf.gz", "rt") as f:
    records = [line for line in f if not line.startswith("#")]

assert records and sum(bool(re.search(r"[\t;]AA=", r)) for r in records) > 0.9 * len(records)
```

```{code-cell} r
anc <- reticulate::import("ancestree")

inf <- anc$Inference$from_local_tree(
  "resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
  sample_names = c(paste0(rep(sprintf("ASP%02d", 1:20), each = 2), "_h", 0:1), "ERR2103730_h0", "ERR2103731_h0"),
  outgroup_samples = c("ERR2103730_h0", "ERR2103731_h0"),
  model = anc$JC69(),
  rec_rate = 4e-8,
  mu = 8e-9
)

invisible(inf$to_vcf("genome.aa.vcf.gz"))
```

```{code-cell} r
:tags: [remove-cell]
lines <- readLines(gzfile("genome.aa.vcf.gz"))
records <- lines[!startsWith(lines, "#")]
stopifnot(length(records) > 0, mean(grepl("[\t;]AA=", records)) > 0.9)
```
