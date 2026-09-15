# Manipulating the SFS
Parsing yields {class}`~sfsutils.spectrum.Spectrum` and {class}`~sfsutils.spectrum.Spectra` objects. A {class}`~sfsutils.spectrum.Spectrum` holds a single site-frequency spectrum, and a {class}`~sfsutils.spectrum.Spectra` holds a named collection, one entry per type, such as the neutral and selected spectra produced by a stratified parse. Both expose the operations needed to inspect and reshape spectra, including indexing, grouping, folding, resampling, serialisation, and plotting. The examples below build spectra directly to illustrate these operations, but in practice they are usually returned by the {class}`~sfsutils.parser.Parser`.

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

# create spectra with two subtypes and two types
spectra = su.Spectra.from_spectra({
    "subtype1.type1": su.Spectrum.standard_kingman(10) * 1,
    "subtype1.type2": su.Spectrum.standard_kingman(10) * 2,
    "subtype2.type1": su.Spectrum.standard_kingman(10) * 3,
})

spectra.plot();
```

```{code-cell} r
library(sfsutils)
su <- load_sfsutils()

# create spectra with two subtypes and two types
spectra <- su$Spectra$from_spectra(list(
  "subtype1.type1" = su$Spectrum$standard_kingman(10) * 1,
  "subtype1.type2" = su$Spectrum$standard_kingman(10) * 2,
  "subtype2.type1" = su$Spectrum$standard_kingman(10) * 3
))

p <- spectra$plot()
```

+++
We access types by their index from which we obtain a {class}`~sfsutils.spectrum.Spectrum` object.

```{code-cell} python
sfs: su.Spectrum = spectra["subtype1.type1"]

sfs.plot();
```

```{code-cell} r
sfs <- spectra["subtype1.type1"]

p <- sfs$plot()
```

+++
We can also use wildcards to access multiple types at once.

```{code-cell} python
spectra["subtype1.*"].plot();
```

```{code-cell} python
:tags: [remove-cell]
assert spectra["subtype1.*"].types == ["subtype1.type1", "subtype1.type2"]
```

```{code-cell} r
p <- spectra["subtype1.*"]$plot()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(identical(unlist(spectra["subtype1.*"]$types), c("subtype1.type1", "subtype1.type2")))
```

+++
## Grouping
To get rid of the subtypes, we can merge the spectra over the specified number of groups.

```{code-cell} python
spectra.merge_groups(1).plot();
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np

merged = spectra.merge_groups(1)
assert merged.types == ["type1", "type2"]
assert np.allclose(merged["type1"].data, spectra["subtype1.type1"].data + spectra["subtype2.type1"].data)
```

```{code-cell} r
p <- spectra$merge_groups(1)$plot()
```

```{code-cell} r
:tags: [remove-cell]
merged <- spectra$merge_groups(1)
stopifnot(identical(unlist(merged$types), c("type1", "type2")))
stopifnot(isTRUE(all.equal(merged["type1"]$data, spectra["subtype1.type1"]$data + spectra["subtype2.type1"]$data)))
```

+++
All subtypes for each type are merged into a single spectrum by adding them up.

+++
## Serialization
We can also save the spectra to a file and restore them again.

```{code-cell} python
spectra.to_file("spectra.csv")

spectra2 = su.Spectra.from_file("spectra.csv")
```

```{code-cell} python
:tags: [remove-cell]
assert spectra2.types == spectra.types and np.allclose(spectra2.data, spectra.data)
```

```{code-cell} r
spectra$to_file("spectra.csv")

spectra2 <- su$Spectra$from_file("spectra.csv")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(identical(unlist(spectra2$types), unlist(spectra$types)))
stopifnot(isTRUE(all.equal(unname(as.matrix(spectra2$data)), unname(as.matrix(spectra$data)))))
```

+++
## Prefixing
Here we prefix the spectra with a string to distinguish them and then combine them into a single spectra object.

```{code-cell} python
spectra.prefix('original').combine(spectra2.prefix('restored')).plot();
```

```{code-cell} r
p <- spectra$prefix('original')$combine(spectra2$prefix('restored'))$plot()
```

+++
For a complete reference of the available methods and properties, see {class}`~sfsutils.spectrum.Spectra` and {class}`~sfsutils.spectrum.Spectrum`.

+++
## Folded spectra
{class}`~sfsutils.spectrum.Spectrum` and {class}`~sfsutils.spectrum.Spectra` objects can also be folded by collapsing the bins corresponding to the derived allele counts onto the bins corresponding to the ancestral allele counts. Folding discards information, which is particularly noticeable when beneficial mutations are present. However, folded spectra are easier to obtain, and are robust to misspecification of the ancestral state, which is often unknown. A spectrum reports whether it has been folded through its {meth}`~sfsutils.spectrum.Spectrum.is_folded` method.

```{code-cell} python
:tags: [full-width]
import matplotlib.pyplot as plt

_, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7, 3))

# fold spectra object
spectra.fold().plot(ax=ax1)

# fold spectrum object
sfs.fold().plot(ax=ax2);
```

```{code-cell} python
:tags: [remove-cell]
assert sfs.fold().is_folded() and not sfs.is_folded()
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3)
```

```{code-cell} r
:tags: [full-width]
# fold spectra object
p1 <- spectra$fold()$plot(show = FALSE)

# fold spectrum object
p2 <- sfs$fold()$plot(show = FALSE)

cowplot::plot_grid(p1, p2, ncol = 2)
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(sfs$fold()$is_folded(), !sfs$is_folded())
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```
