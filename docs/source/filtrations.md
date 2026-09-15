# Site Filtration
`sfsutils` offers a number of filtrations that are either applied on the fly while parsing, or run through the {class}`~sfsutils.filtration.Filterer` class to filter an input and write the result to a file. Some useful filtrations include {class}`~sfsutils.filtration.DeviantOutgroupFiltration`, {class}`~sfsutils.filtration.CodingSequenceFiltration`, and {class}`~sfsutils.filtration.BiasedGCConversionFiltration`. For a complete list of available filtrations, refer to the {doc}`API reference <../modules/filtration>`.

```{code-cell} python
import sfsutils as su

f = su.Filterer(
    source="resources/genome/betula/biallelic.subset.10000.vcf.gz",
    filtrations=[su.BiasedGCConversionFiltration()],
    output="genome.gc.vcf.gz"
)

f.filter()
```

```{code-cell} python
:tags: [remove-cell]
import os

assert 0 < f.n_filtered < f.n_sites and os.path.getsize("genome.gc.vcf.gz") > 0
```

```{code-cell} r
library(sfsutils)
su <- load_sfsutils()

f <- su$Filterer(
  source = "resources/genome/betula/biallelic.subset.10000.vcf.gz",
  filtrations = list(su$BiasedGCConversionFiltration()),
  output = "genome.gc.vcf.gz"
)

f$filter()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(f$n_filtered > 0, f$n_filtered < f$n_sites, file.size("genome.gc.vcf.gz") > 0)
```

+++
All components can be customised by extending the corresponding base class.
