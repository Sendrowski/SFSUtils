# Input and output formats

`sfsutils` reads variants from a VCF file, a VCF-Zarr store, or a tskit tree sequence through a single streamed site interface, so the same analysis code works for any input format. Writing follows the output file's extension.

The dataset here is one synthetic ARG, provided as a tree sequence together with the VCF and VCF-Zarr store converted from it. Since all three encode the same genotypes, they yield the same spectrum.

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
## Three input forms

```{code-cell} python
trees = "resources/msprime/two_epoch.trees"  # tskit tree sequence (the ARG)
vcf = "resources/msprime/two_epoch.vcf"  # VCF written from it
vcz = "resources/msprime/two_epoch.vcz"  # VCF-Zarr store converted from the VCF
```

```{code-cell} r
trees <- "resources/msprime/two_epoch.trees"  # tskit tree sequence (the ARG)
vcf <- "resources/msprime/two_epoch.vcf"  # VCF written from it
vcz <- "resources/msprime/two_epoch.vcz"  # VCF-Zarr store converted from the VCF
```

+++
## Reading

{class}`~sfsutils.parser.Parser` accepts any of the three as its `source` argument and infers the backend from the source. Reading a VCF-Zarr store needs the optional `zarr` package, a tree sequence the optional `tskit` package.

```{code-cell} python
import numpy as np
import sfsutils as su

sfs_trees = su.Parser(source=trees, n=10, skip_non_polarized=False).parse()
sfs_vcf = su.Parser(source=vcf, n=10, skip_non_polarized=False).parse()
sfs_vcz = su.Parser(source=vcz, n=10, skip_non_polarized=False).parse()
```

```{code-cell} python
# the three spectra are identical
np.array_equal(sfs_trees.all.data, sfs_vcf.all.data) and np.array_equal(sfs_vcf.all.data, sfs_vcz.all.data)
```

```{code-cell} python
:tags: [remove-cell]
assert np.array_equal(sfs_trees.all.data, sfs_vcf.all.data) and np.array_equal(sfs_vcf.all.data, sfs_vcz.all.data)
assert sfs_vcf.all.n_polymorphic > 0
```

```{code-cell} python
sfs_vcf.all.plot();
```

```{code-cell} r
library(sfsutils)
su <- load_sfsutils()

sfs_trees <- su$Parser(source = trees, n = 10, skip_non_polarized = FALSE)$parse()
sfs_vcf <- su$Parser(source = vcf, n = 10, skip_non_polarized = FALSE)$parse()
sfs_vcz <- su$Parser(source = vcz, n = 10, skip_non_polarized = FALSE)$parse()
```

```{code-cell} r
# the three spectra are identical
identical(sfs_trees$all$to_list(), sfs_vcf$all$to_list()) && identical(sfs_vcf$all$to_list(), sfs_vcz$all$to_list())
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(identical(sfs_trees$all$to_list(), sfs_vcf$all$to_list()), identical(sfs_vcf$all$to_list(), sfs_vcz$all$to_list()))
stopifnot(sfs_vcf$all$n_polymorphic > 0)
```

```{code-cell} r
p <- sfs_vcf$all$plot()
```

+++
## Writing

{class}`~sfsutils.filtration.Filterer` and {class}`~sfsutils.annotation.Annotator` pick the writer from the output file's extension: `.vcf`/`.vcf.gz` for a VCF, `.vcz`/`.zarr` for a VCF-Zarr store, and `.trees` for a tree sequence. A tree sequence can only be written from a tree-sequence input: filtering removes the discarded sites with `delete_sites`, leaving the genealogy intact. A genealogy cannot be reconstructed from genotype data, so writing a `.trees` from a VCF or VCF-Zarr store is rejected.

```{code-cell} python
import os
import tempfile

out = tempfile.mkdtemp()

# a VCF-Zarr store can be written from any input
su.Filterer(source=vcf, output=os.path.join(out, "snps.vcz"), filtrations=[su.SNPFiltration()]).filter()

# a VCF is written from a VCF input
su.Filterer(source=vcf, output=os.path.join(out, "snps.vcf"), filtrations=[su.SNPFiltration()]).filter()

# a tree sequence is written from a tree-sequence input
su.Filterer(source=trees, output=os.path.join(out, "snps.trees"), filtrations=[su.SNPFiltration()]).filter()
```

```{code-cell} r
out <- tempfile()
dir.create(out)

# a VCF-Zarr store can be written from any input
su$Filterer(source = vcf, output = file.path(out, "snps.vcz"), filtrations = list(su$SNPFiltration()))$filter()

# a VCF is written from a VCF input
su$Filterer(source = vcf, output = file.path(out, "snps.vcf"), filtrations = list(su$SNPFiltration()))$filter()

# a tree sequence is written from a tree-sequence input
su$Filterer(source = trees, output = file.path(out, "snps.trees"), filtrations = list(su$SNPFiltration()))$filter()
```

+++
The store and tree sequence written above parse back to the same spectrum as the VCF output.

```{code-cell} python
back = [
    su.Parser(source=os.path.join(out, f), n=10, skip_non_polarized=False).parse().all.data
    for f in ["snps.vcf", "snps.vcz", "snps.trees"]
]

all(np.array_equal(b, back[0]) for b in back)
```

```{code-cell} python
:tags: [remove-cell]
assert all(np.array_equal(b, back[0]) for b in back)
```

```{code-cell} r
back <- lapply(c("snps.vcf", "snps.vcz", "snps.trees"), function(f) {
  su$Parser(source = file.path(out, f), n = 10, skip_non_polarized = FALSE)$parse()$all$to_list()
})

all(vapply(back, identical, logical(1), back[[1]]))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(all(vapply(back, identical, logical(1), back[[1]])))
```
