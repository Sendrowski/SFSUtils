pkgname <- "sfsutils"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('sfsutils')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("install_sfsutils")
### * install_sfsutils

flush(stderr()); flush(stdout())

### Name: install_sfsutils
### Title: Install the 'sfsutils' Python module
### Aliases: install_sfsutils

### ** Examples

## Not run: 
##D install_sfsutils()  # Installs the latest version of sfsutils
##D install_sfsutils("1.0.0")  # Installs version 1.0.0 of sfsutils
##D install_sfsutils(force = TRUE)  # Reinstalls the sfsutils module
## End(Not run)




cleanEx()
nameEx("load_sfsutils")
### * load_sfsutils

flush(stderr()); flush(stdout())

### Name: load_sfsutils
### Title: Load the sfsutils library and associated visualization functions
### Aliases: load_sfsutils

### ** Examples

## Not run: 
##D load_sfsutils(install = TRUE)
##D # now you can use sfsutils functionalities as per its API
## End(Not run)




cleanEx()
nameEx("sfsutils_is_installed")
### * sfsutils_is_installed

flush(stderr()); flush(stdout())

### Name: sfsutils_is_installed
### Title: Check if the 'sfsutils' Python module is installed
### Aliases: sfsutils_is_installed

### ** Examples

## Not run: 
##D is_installed()  # Returns TRUE or FALSE based on the installation status of sfsutils
## End(Not run)




### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
