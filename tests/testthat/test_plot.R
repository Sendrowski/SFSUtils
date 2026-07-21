library(sfsutils)

# These tests exercise the R plotting overrides against the Python package via reticulate. They are
# skipped when sfsutils is not importable (e.g. reticulate has no configured Python with the package).
# Both call styles are checked: the explicit `su$Class$plot(obj)` form (as in the fastdfe wrapper) and
# the implicit `obj$plot()` form, which relies on reticulate binding the instance as `self`.

if (sfsutils_is_installed()) {

  su <- load_sfsutils()

  expect_ggplot <- function(p) expect_true(inherits(p, "ggplot"))

  test_that("Spectrum plots", {
    s <- su$Spectrum(c(100, 20, 10, 6, 4, 3, 8))
    expect_ggplot(su$Spectrum$plot(s, show = FALSE))
    expect_ggplot(s$plot(show = FALSE))
  })

  test_that("TwoSFS plots both call styles", {
    sfs2 <- su$TwoSFS(matrix(as.double(1:25), 5, 5))
    expect_ggplot(su$TwoSFS$plot(sfs2, show = FALSE))   # explicit self
    expect_ggplot(sfs2$plot(show = FALSE))              # implicit self (reticulate binds the instance)
  })

  test_that("JointSFS plots both call styles", {
    j <- su$JointSFS(matrix(as.double(0:8), 3, 3), pop_names = c("A", "B"))
    expect_ggplot(su$JointSFS$plot(j, show = FALSE))    # explicit self
    expect_ggplot(j$plot(show = FALSE))                 # implicit self
  })

  test_that("JointSFS honours the requested axis order", {
    j <- su$JointSFS(matrix(as.double(1:9), 3, 3), pop_names = c("A", "B"))

    p01 <- su$JointSFS$plot(j, pops = c(0, 1), show = FALSE)
    p10 <- su$JointSFS$plot(j, pops = c(1, 0), show = FALSE)

    # the swapped order transposes the spectrum along with the axis labels
    expect_false(identical(p01$data$value, p10$data$value))
    expect_equal(p01$labels$x, p10$labels$y)
    expect_equal(p01$labels$y, p10$labels$x)

    expect_error(su$JointSFS$plot(j, pops = c(0), show = FALSE), "two populations")
  })

  test_that("Parser derives a spectrum that plots", {
    vcf <- "../../resources/msprime/two_epoch.vcf"
    skip_if_not(file.exists(vcf), "msprime VCF fixture absent")
    spectra <- su$Parser(source = vcf, n = 20L, skip_non_polarized = FALSE,
                         subsample_mode = "random")$parse()
    expect_ggplot(spectra$all$plot(show = FALSE))
  })
}
