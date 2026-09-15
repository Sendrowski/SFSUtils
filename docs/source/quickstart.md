# Quickstart
Parse a site-frequency spectrum from a polarised VCF and plot it. The example VCF for `Betula spp.` carries ancestral alleles in the `AA` info field, which the {class}`~sfsutils.parser.Parser` uses to polarise each site. `n` is the sample size (number of haplotypes) the spectrum is projected to.

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

vcf = "resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz"

sfs = su.Parser(n=10, source=vcf).parse()
```

```{code-cell} python
:tags: [remove-cell]
assert sfs.types == ['all'] and sfs.all.n == 10 and sfs.all.n_polymorphic > 0
```

```{code-cell} python
sfs.plot();
```

```{code-cell} r
library(sfsutils)
su <- load_sfsutils()

vcf <- "resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz"

sfs <- su$Parser(n = 10, source = vcf)$parse()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(identical(unlist(sfs$types), "all"), sfs$all$n == 10, sfs$all$n_polymorphic > 0)
```

```{code-cell} r
p <- sfs$plot()
```
