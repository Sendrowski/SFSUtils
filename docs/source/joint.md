# Joint SFS
Given several populations, the {class}`~sfsutils.parser.Parser` counts derived alleles jointly rather than pooling them, yielding the joint SFS: an array whose entry $(i, j, \dots)$ is the number of sites with $i$ derived alleles in the first population, $j$ in the second, and so on. Populations are supplied as a mapping from name to sample names via `pops`, and `n` sets the sample size each population is projected to. Parsing then returns a {class}`~sfsutils.spectrum.JointSpectra`, holding one {class}`~sfsutils.spectrum.JointSFS` per type.

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = [4.4, 3.3]
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

+++
The example VCF for `Betula spp.` carries samples from several localities, encoded in the sample names. We take two of them as populations.

```{code-cell} python
import sfsutils as su
from cyvcf2 import VCF

vcf = "resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz"
samples = VCF(vcf).samples

pops = {loc: [s for s in samples if s.startswith(loc)] for loc in ["ASP", "PIT"]}
```

```{code-cell} python
sfs = su.Parser(source=vcf, pops=pops, n=10).parse()
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np

assert sfs["all"].data.shape == (11, 11) and sfs["all"].data.sum() > 0
```

```{code-cell} r
library(sfsutils)
su <- load_sfsutils()

cyvcf2 <- reticulate::import("cyvcf2")

vcf <- "resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz"
samples <- cyvcf2$VCF(vcf)$samples

pops <- list()
for (loc in c("ASP", "PIT")) {
  pops[[loc]] <- samples[startsWith(samples, loc)]
}
```

```{code-cell} r
sfs <- su$Parser(source = vcf, pops = pops, n = 10)$parse()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(identical(dim(sfs["all"]$data), c(11L, 11L)), sum(sfs["all"]$data) > 0)
```

+++
For two populations the joint SFS is a matrix, plotted as a heatmap of derived-allele counts.

```{code-cell} python
sfs["all"].plot();
```

```{code-cell} r
p <- sfs["all"]$plot()
```

+++
When the ancestral state is unknown, the joint SFS is folded so that each allele count is combined with its complement.

```{code-cell} python
sfs["all"].fold().plot();
```

```{code-cell} python
:tags: [remove-cell]
assert np.isclose(sfs["all"].fold().data.sum(), sfs["all"].data.sum())
```

```{code-cell} r
p <- sfs["all"]$fold()$plot()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(isTRUE(all.equal(sum(sfs["all"]$fold()$data), sum(sfs["all"]$data))))
```

+++
Marginalizing over one population recovers the single-population SFS of the other, which lets the joint spectrum be reconciled with a one-dimensional parse.

```{code-cell} python
su.Spectrum(sfs["all"].marginalize([0]).data).plot();  # the SFS of population ASP
```

```{code-cell} python
:tags: [remove-cell]
assert np.allclose(sfs["all"].marginalize([0]).data, sfs["all"].data.sum(axis=1))
```

```{code-cell} r
p <- su$Spectrum(sfs["all"]$marginalize(list(0L))$data)$plot()  # the SFS of population ASP
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(isTRUE(all.equal(as.numeric(sfs["all"]$marginalize(list(0L))$data), rowSums(sfs["all"]$data))))
```
