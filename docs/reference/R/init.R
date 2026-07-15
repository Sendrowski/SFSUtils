
# jupyter lab
# reticulate::py_last_error()

devtools::install_github("Sendrowski/SFSUtils")

sink(file = stderr(), type = "message")

library(sfsutils)

# install_sfsutils()

setwd("~/PycharmProjects/SFSUtils/")

sf <- load_sfsutils()

sf$Settings$disable_pbar <- TRUE
