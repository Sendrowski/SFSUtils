# Parsing the SFS
The {class}`~sfsutils.parser.Parser` reads variants and counts derived-allele frequencies into a site-frequency spectrum. For a sample of `n` haplotypes it returns counts in bins `0, 1, ..., n`, projecting down when more haplotypes are present. Polarisation relies on the ancestral allele: by default the `AA` info field is consulted, and sites where it is undefined are skipped (see {attr}`~sfsutils.parser.Parser.skip_non_polarized`). When no ancestral state is available the spectrum can instead be folded, at the cost of conflating minor and major allele counts.

A single spectrum is rarely enough on its own. More often we want to separate sites into classes that behave differently and read off one spectrum per class. This is expressed by passing {class}`~sfsutils.parser.Stratification` objects to the {class}`~sfsutils.parser.Parser`. The example below contrasts 0-fold and 4-fold degenerate sites using a VCF for `Betula spp.`

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = [4.4, 3.3]
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

```{code-cell} python
import sfsutils as su

parser = su.Parser(
    n=8,
    source="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
    fasta="resources/genome/betula/genome.subset.20.fasta",
    gff="resources/genome/betula/genome.gff.gz",
    annotations=[
        su.DegeneracyAnnotation()
    ],
    stratifications=[su.DegeneracyStratification()]
)

spectra: su.Spectra = parser.parse()
```

```{code-cell} python
:tags: [remove-cell]
assert sorted(spectra.types) == ['neutral', 'selected']
```

```{code-cell} python
spectra.plot();
```

```{code-cell} r
library(sfsutils)
su <- load_sfsutils()

parser <- su$Parser(
  n = 8,
  source = "resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
  fasta = "resources/genome/betula/genome.subset.20.fasta",
  gff = "resources/genome/betula/genome.gff.gz",
  annotations = list(
    su$DegeneracyAnnotation()
  ),
  stratifications = list(su$DegeneracyStratification())
)

spectra <- parser$parse()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(identical(sort(unlist(spectra$types)), c("neutral", "selected")))
```

```{code-cell} r
p <- spectra$plot()
```

+++
`sfsutils` relies here on VCF info tags to determine the degeneracy of a site, but this behaviour can be customised (cf. {class}`~sfsutils.parser.DegeneracyStratification`).

+++
## Stacked stratification
Stratifications split sites into categories, so the parser produces a separate spectrum for each, such as neutral versus selected sites. Several can be combined by passing a list: here we stratify the SFS by degeneracy as well as ancestral base. See the {doc}`Site stratification reference <../modules/stratification>` for the complete list of available stratifications.

```{code-cell} python
parser = su.Parser(
    n=10,
    source="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
    fasta="resources/genome/betula/genome.subset.20.fasta",
    gff="resources/genome/betula/genome.gff.gz",
    annotations=[
        su.DegeneracyAnnotation()
    ],
    stratifications=[
        su.DegeneracyStratification(),
        su.AncestralBaseStratification()
    ]
)

spectra: su.Spectra = parser.parse()
```

```{code-cell} python
:tags: [remove-cell]
assert sorted(spectra.types) == [f'{d}.{b}' for d in ['neutral', 'selected'] for b in 'ACGT']
```

```{code-cell} python
spectra.plot();
```

```{code-cell} r
parser <- su$Parser(
  n = 10,
  source = "resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
  fasta = "resources/genome/betula/genome.subset.20.fasta",
  gff = "resources/genome/betula/genome.gff.gz",
  annotations = list(
    su$DegeneracyAnnotation()
  ),
  stratifications = list(
    su$DegeneracyStratification(),
    su$AncestralBaseStratification()
  )
)

spectra <- parser$parse()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(identical(sort(unlist(spectra$types)), sort(outer(c("neutral", "selected"), c("A", "C", "G", "T"), paste, sep = "."))))
```

```{code-cell} r
p <- spectra$plot()
```
