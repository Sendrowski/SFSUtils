# 2-Site SFS
The two-site (or two-locus) SFS summarises pairs of linked sites: its entry $(i, j)$ counts pairs of sites, separated by up to `d` base pairs, at which one site carries $i$ and the other $j$ derived alleles. Its departure from the outer product of the one-dimensional spectrum reflects linkage and the shape of the underlying genealogies, which makes it a sensitive probe of departures from the standard (Kingman) coalescent, such as the multiple mergers produced by strong skew in reproductive success (see {cite:t}`fenton2025`).

Enabling {attr}`~sfsutils.parser.Parser.two_sfs` makes the {class}`~sfsutils.parser.Parser` pair sites within `d` base pairs. As with the other parsing modes, {meth}`~sfsutils.parser.Parser.parse` returns a {class}`~sfsutils.spectrum.TwoSpectra` collection keyed by type. Without stratifications, the single unstratified two-site SFS is stored under the type `all`, a square {class}`~sfsutils.spectrum.TwoSFS`. The example uses a synthetic all-sites dataset, monomorphic sites included, simulated under the standard Kingman coalescent.

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

two_sfs = su.Parser(
    source="resources/msprime/two_sfs_kingman.all.vcf.gz",
    n=10, two_sfs=True, d=1000
).parse()["all"]
```

```{code-cell} r
library(sfsutils)
su <- load_sfsutils()

two_sfs <- su$Parser(
  source = "resources/msprime/two_sfs_kingman.all.vcf.gz",
  n = 10, two_sfs = TRUE, d = 1000
)$parse()[["all"]]
```

+++
The raw two-site SFS is the symmetric matrix of pair counts, shown here as a heatmap with the monomorphic rows and columns omitted.

```{code-cell} python
two_sfs.plot(title="pair counts");
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np

assert np.allclose(two_sfs.data, two_sfs.data.T)
```

```{code-cell} r
p <- two_sfs$plot(title = "pair counts")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(isTRUE(all.equal(two_sfs$data, t(two_sfs$data))))
```

+++
Beyond the raw pair counts, {meth}`~sfsutils.spectrum.TwoSFS.cov` and {meth}`~sfsutils.spectrum.TwoSFS.corr` give the class-resolved branch-length covariance and correlation $\mathrm{Cov}(L_i, L_j)$ of the two linked sites, the deviation of the joint class distribution from independence. Because they normalise over the whole spectrum so that the marginal is the site-frequency spectrum, they require the monomorphic sites, so the input must contain all sites.

```{code-cell} python
two_sfs.corr().plot(title="correlation");
```

```{code-cell} r
p <- two_sfs$corr()$plot(title = "correlation")
```

+++
The spectrum can also be folded when the ancestral state is unknown, and the diagonals masked, as for any {class}`~sfsutils.spectrum.TwoSFS`.

```{code-cell} python
two_sfs.fold().plot(title="folded");
```

```{code-cell} python
:tags: [remove-cell]
assert two_sfs.fold().is_folded()
```

```{code-cell} r
p <- two_sfs$fold()$plot(title = "folded")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(two_sfs$fold()$is_folded())
```

+++
## Detecting multiple mergers (MMCs)

Under the Kingman coalescent, mutations at different low frequencies are negatively correlated {cite:p}`fu1995`. Multiple-merger genealogies instead generate positive associations between low-frequency mutations {cite:p}`birkner2013`, the opposite sign. To see the contrast, we parse a second synthetic all-sites dataset, simulated under a Beta(1.3) multiple-merger coalescent, and compare its correlation with the Kingman one.

```{code-cell} python
beta = su.Parser(
    source="resources/msprime/two_sfs_beta.all.vcf.gz",
    n=10, two_sfs=True, d=1000
).parse()["all"]
```

```{code-cell} r
beta <- su$Parser(
  source = "resources/msprime/two_sfs_beta.all.vcf.gz",
  n = 10, two_sfs = TRUE, d = 1000
)$parse()[["all"]]
```

+++
The two correlations are shown side by side.

```{code-cell} python
:tags: [full-width]
import matplotlib.pyplot as plt

_, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7, 3))

two_sfs.corr().plot(ax=ax1, title="Kingman", show=False)
beta.corr().plot(ax=ax2, title="Beta(1.3)");
```

```{code-cell} python
:tags: [remove-cell]
# the off-diagonal correlations among the classes 1 to 4 are negative under Kingman and positive under Beta(1.3)
low = ~np.eye(4, dtype=bool)
assert (np.asarray(two_sfs.corr().data)[1:5, 1:5][low] < 0).all()
assert (np.asarray(beta.corr().data)[1:5, 1:5][low] > 0).all()
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3)
```

```{code-cell} r
:tags: [full-width]
p_kingman <- two_sfs$corr()$plot(title = "Kingman", show = FALSE)
p_beta <- beta$corr()$plot(title = "Beta(1.3)", show = FALSE)

cowplot::plot_grid(p_kingman, p_beta, ncol = 2)
```

```{code-cell} r
:tags: [remove-cell]
low <- !diag(4)
stopifnot(all(two_sfs$corr()$data[2:5, 2:5][low] < 0), all(beta$corr()$data[2:5, 2:5][low] > 0))
options(repr.plot.width = 4.4, repr.plot.height = 3.3)
```

+++
The contrast in the low-frequency block, negative under Kingman and positive under the multiple-merger coalescent, is the signal that two-site spectra exploit to detect departures from Kingman coalescence.

Because {meth}`~sfsutils.spectrum.TwoSFS.cov` and {meth}`~sfsutils.spectrum.TwoSFS.corr` normalise over the full spectrum so that the marginal is the site-frequency spectrum, they require the real monomorphic sites of an all-sites input. A polymorphic-only (SNP) spectrum leaves them undefined and raises an error. The ratio statistic {meth}`~sfsutils.spectrum.TwoSFS.fpmi` needs no monomorphic sites and remains available for polymorphic-only data.
