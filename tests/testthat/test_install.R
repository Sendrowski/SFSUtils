library(sfsutils)

test_that("install_sfsutils replaces the declared requirement and Python version", {
  skip_if(reticulate::py_available(initialize = FALSE), "Python is initialised, so requirements are fixed")

  requirement <- function() unclass(reticulate::py_require())[c("packages", "python_version")]
  declared <- function() grep("^sfsutils-popgen", requirement()$packages, value = TRUE)

  install_sfsutils(extras = NULL, python_version = "3.12", silent = TRUE)

  expect_equal(declared(), "sfsutils-popgen")
  expect_equal(requirement()$python_version, "3.12")

  install_sfsutils(version = "1.1.0", extras = c("vcf", "zarr"), silent = TRUE)

  expect_equal(declared(), "sfsutils-popgen[vcf,zarr]==1.1.0")
  expect_equal(requirement()$python_version, "3.11")

  # back to the requirement declared on load, for the tests that follow
  install_sfsutils(silent = TRUE)

  expect_equal(declared(), "sfsutils-popgen[vcf]")
})
