
# jupyter lab
# reticulate::py_last_error()

devtools::install_github("Sendrowski/SFSUtils")

sink(file = stderr(), type = "message")

library(sfsutils)

# install_sfsutils()

setwd("~/PycharmProjects/SFSUtils/")

su <- load_sfsutils()

su$Settings$disable_pbar <- TRUE
