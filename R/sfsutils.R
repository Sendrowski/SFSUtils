if (getRversion() >= "2.15.1") utils::globalVariables(c(".data"))

# vector of required packages
required_packages <- c("reticulate", "ggplot2", "cowplot", "RColorBrewer", "scales")

# install required R packages
for(package in required_packages){
  if(!package %in% installed.packages()[,"Package"]){
    install.packages(package)
  }
}

#' Check if the `sfsutils` Python module is installed
#'
#' This function uses the reticulate package to verify if the `sfsutils` Python
#' module is currently installed. An unrelated project of the same name exists on
#' PyPI, so the module is additionally checked for the `Parser` class it provides.
#'
#' @return Logical `TRUE` if the `sfsutils` Python module is installed, otherwise `FALSE`.
#'
#' @examples
#' \dontrun{
#' is_installed()  # Returns TRUE or FALSE based on the installation status of sfsutils
#' }
#'
#' @export
sfsutils_is_installed <- function() {

  # Check if sfsutils is installed
  if (!reticulate::py_module_available("sfsutils")) {
    return(FALSE)
  }

  # Check that the module found is this package and not a namesake
  installed <- tryCatch({
    sf <- reticulate::import("sfsutils", delay_load = FALSE)
    !is.null(sf$Parser)
  }, error = function(e) FALSE)

  return(installed)
}


#' Install the `sfsutils` Python module
#'
#' This function checks if the `sfsutils` Python module is available.
#' If not, or if the `force` argument is TRUE, it installs it via pip.
#' If the `silent` argument is set to TRUE, the function will not output a
#' message when the module is already installed.
#'
#' @param version A character string specifying the version of the `sfsutils` module
#'        to install. Default is `NULL` which will install the latest version.
#' @param extras A character vector of optional input backends to install alongside the module:
#'        `'vcf'` for VCF files, `'zarr'` for VCF-Zarr stores and `'arg'` for tree sequences.
#'        Default is `c("vcf")`; pass `NULL` to install none of them.
#' @param force Logical, if `TRUE` it will force the reinstallation of the `sfsutils` module
#'        even if it's already available. Default is `FALSE`.
#' @param silent Logical, if `TRUE` it will suppress the message about `sfsutils` being
#'        already installed. Default is `FALSE`.
#' @param python_version A character string specifying the Python version reticulate
#'        should provision the environment with. Default is `'3.11'`.
#'
#' @return Invisible `NULL`.
#'
#' @examples
#' \dontrun{
#' install_sfsutils()  # Installs the latest version of sfsutils with the vcf backend
#' install_sfsutils("1.0.0")  # Installs version 1.0.0 of sfsutils
#' install_sfsutils(extras = c("vcf", "zarr", "arg"))  # Installs all input backends
#' install_sfsutils(extras = NULL)  # Installs without any of the optional backends
#' install_sfsutils(force = TRUE)  # Reinstalls the sfsutils module
#' }
#'
#' @export
install_sfsutils <- function(version = NULL, extras = c("vcf"), force = FALSE, silent = FALSE, python_version = '3.11') {

  # Create the package string with the extras and version if specified. The distribution is named
  # 'sfsutils-popgen' on PyPI, where 'sfsutils' is an unrelated project
  package_name <- "sfsutils-popgen"
  if (length(extras) > 0) {
    package_name <- paste0(package_name, "[", paste(extras, collapse = ","), "]")
  }
  if (!is.null(version)) {
    package_name <- paste0(package_name, "==", version)
  }

  # Check if sfsutils is installed or if force is TRUE
  if (force || !sfsutils_is_installed()) {
    reticulate::py_install(
      package_name,
      method = "conda",
      pip = TRUE,
      python_version = python_version,
      ignore_installed = TRUE
   )
  } else {
    if (!silent) {
      message("The 'sfsutils' Python module is already installed.")
    }
  }

  invisible(NULL)
}

#' Load the sfsutils library and associated visualization functions
#'
#' This function imports the Python package 'sfsutils' using the reticulate package
#' and then configures it to work seamlessly with R, overriding some of the default
#' visualization functions with custom R-based ones. This function also ensures
#' that required R libraries are loaded for visualization.
#'
#' @param install A logical. If TRUE, the function will attempt to run install_sfsutils().
#'
#' @return A reference to the 'sfsutils' Python library loaded through reticulate.
#'         This reference can be used to access 'sfsutils' functionalities.
#'
#' @examples
#' \dontrun{
#' load_sfsutils(install = TRUE)
#' # now you can use sfsutils functionalities as per its API
#' }
#'
#' @seealso \link[reticulate]{import} for importing Python modules in R.
#'
#' @export
load_sfsutils <- function(install = FALSE) {

  # install if install flag is true
  if (install) {
    install_sfsutils(silent = TRUE)
  }

  # configure plot
  options(repr.plot.width = 4.6, repr.plot.height = 3.2)

  sf <- reticulate::import("sfsutils")

  # override python visualization functions
  viz <- sf$visualization$Visualization

  # Create a scatter plot.
  #
  # @param values List or matrix. Values to plot.
  # @param file Character. File path to save plot to. Default is NULL.
  # @param show Logical. Whether to show plot. Default is TRUE.
  # @param title Character. Title of plot.
  # @param scale Character. Scale of y-axis. One of 'lin', 'log'. Default is 'lin'.
  #
  # @return A ggplot object.
  viz$plot_scatter <- function(
    values,
    file = NULL,
    show = TRUE,
    title = NULL,
    scale = 'lin',
    ...
  ) {
    # Create data frame
    data <- data.frame(x = seq_along(values), y = unlist(values))

    # Create plot
    p <- ggplot2::ggplot(data, ggplot2::aes(x = .data$x, y = .data$y)) +
      ggplot2::geom_point() +
      ggplot2::labs(title = title, y = 'lnl')

    # Set y scale
    if (scale == 'log') {
      p <- p + ggplot2::scale_y_continuous(trans = 'log10')
    }

    # Display plot if 'show' is TRUE
    if (show) print(p)

    # Save plot to file if 'file' is provided
    if (!is.null(file)) ggplot2::ggsave(file, plot = p)

    return(p)
  }


  # Plot the given 1D spectra
  #
  # @param spectra List of lists of spectra or a 2D array in which each row
  #                is a spectrum in the same order as labels
  # @param labels Character vector. Labels for each spectrum
  # @param log_scale Logical. Whether to use logarithmic y-scale
  # @param use_subplots Logical. Whether to use subplots
  # @param show_monomorphic Logical. Whether to show monomorphic site counts
  # @param title Character. Title of plot
  # @param n_ticks Numeric. Number of x-ticks to use
  # @param file Character. File to save plot to
  # @param show Logical. Whether to show the plot
  #
  # @return ggplot object
  viz$plot_spectra <- function(
    spectra,
    labels = character(0),
    log_scale = FALSE,
    use_subplots = FALSE,
    show_monomorphic = FALSE,
    title = NULL,
    file = NULL,
    show = TRUE,
    ...
  ) {

    if (length(spectra) == 0) {
      warning('No spectra to plot.')
      return(NULL)
    }

    if (use_subplots) {
      # Creating a grid of plots
      plot_list <- lapply(1:length(spectra), function(i) {
        viz$plot_spectra(
          spectra = list(spectra[[i]]),
          labels = if (length(labels)) labels[i] else character(0),
          log_scale = log_scale,
          show_monomorphic = show_monomorphic,
          show = FALSE
        ) +
          ggplot2::labs(title = if (length(labels) >= i) labels[i] else '')
      })

      plot_grid <- cowplot::plot_grid(plotlist = plot_list)

      if (show) print(plot_grid)
      if (!is.null(file)) ggplot2::ggsave(file, plot = plot_grid)

      return(plot_grid)
    }

    if (length(labels) == 0) {
      labels <- as.character(1:length(spectra))
    }

    df <- data.frame()
    for (i in seq_along(spectra)) {
      indices <- if (show_monomorphic) seq_along(spectra[[i]]) else seq_along(spectra[[i]])[-c(1, length(spectra[[i]]))]
      heights <- if (show_monomorphic) unlist(spectra[[i]]) else unlist(spectra[[i]][-c(1, length(spectra[[i]]))])
      df_temp <- data.frame(indices = indices,
                            heights = heights,
                            group = rep(labels[i], length(indices)))
      df <- rbind(df, df_temp)
    }

    # Create a ggplot object
    p <- ggplot2::ggplot(df, ggplot2::aes(x = indices, y = heights, fill = .data$group)) +
      ggplot2::geom_bar(stat = "identity", position = "dodge",
                        width = 0.7, show.legend = length(spectra) > 1) +
      ggplot2::labs(x = "frequency", y = "", title = title, fill = NULL) +
      ggplot2::theme_bw() +
      ggplot2::theme(panel.grid.major = ggplot2::element_blank(),
                     panel.grid.minor = ggplot2::element_blank(),
                     # draw the legend inside the panel, so it does not eat into the plot width
                     legend.position = "inside",
                     legend.position.inside = c(0.98, 0.98),
                     legend.justification.inside = c(1, 1),
                     legend.background = ggplot2::element_rect(
                       fill = scales::alpha("white", 0.7), colour = NA
                     ),
                     legend.key.size = ggplot2::unit(0.9, "lines")) +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, .1)))

    if (log_scale) {
      p <- p + ggplot2::scale_y_log10()
    }

    # Adjust x-axis labels based on show_monomorphic
    if (show_monomorphic) {
      p <- p + ggplot2::scale_x_continuous(breaks = 0:(length(spectra[[1]]) + 1),
                                           labels = 0:(length(spectra[[1]]) + 1) - 1,
                                           expand = c(0, 0))
    } else {
      p <- p + ggplot2::scale_x_continuous(breaks = 1:length(spectra[[1]]),
                                           labels = 1:length(spectra[[1]]) - 1,
                                           expand = c(0, 0))
    }

    # Display or save the plot
    if (show) print(p)
    if (!is.null(file)) ggplot2::ggsave(file, plot = p)

    return(p)
  }


  # Convert a matrix to a long data frame with 1-based integer x (column) and
  # y (row) coordinates, suitable for ggplot2::geom_tile. The value column is
  # filled in column-major order to match R's own matrix layout.
  #
  # @param mat Numeric matrix.
  #
  # @return A data frame with columns x, y and value.
  matrix_to_long <- function(mat) {
    nr <- nrow(mat)
    nc <- ncol(mat)

    data.frame(
      y = rep(seq_len(nr), times = nc),
      x = rep(seq_len(nc), each = nr),
      value = as.vector(mat)
    )
  }


  # The two overrides below replace Python *instance* methods by assigning an R function to a class
  # attribute. Both call styles work (verified against the tests/testthat suite): `obj$plot()` binds the
  # instance as `self`, and `sf$TwoSFS$plot(obj)` passes it explicitly (as in the fastdfe wrapper).
  #
  # Plot a 2-SFS (TwoSFS) as a heatmap.
  #
  # Reimplements TwoSFS.plot using a ggplot2 geom_tile heatmap. The monomorphic first and
  # last rows and columns are dropped, and if the spectrum is folded only the folded half is
  # shown. As in the Python backend the colour scale depends on the spectrum: a sequential
  # log viridis scale for raw pair counts, a diverging symmetric-log PuOr_r scale for the
  # class-resolved results (cov / corr / fpmi).
  #
  # @param self The TwoSFS object (passed implicitly as the instance).
  # @param title Character. Title of the plot. Default is NULL.
  # @param log_scale Logical. Kept for signature compatibility with the Python
  #                  backend; currently ignored. Default is FALSE.
  # @param max_abs Numeric. Maximum absolute value for the diverging colour scale; ignored
  #                for raw pair counts. Default is NULL (inferred from the data).
  # @param show Logical. Whether to show the plot. Default is TRUE.
  # @param file Character. File path to save plot to. Default is NULL.
  # @param ... Additional arguments which are ignored.
  #
  # @return A ggplot object.
  sf$TwoSFS$plot <- function(
    self,
    title = NULL,
    log_scale = FALSE,
    max_abs = NULL,
    show = TRUE,
    file = NULL,
    ...
  ) {
    mat <- as.matrix(self$data)
    storage.mode(mat) <- "double"
    n <- nrow(mat)

    if (n < 3) {
      warning('Nothing to plot.')
      return(invisible(NULL))
    }

    # remove monomorphic first and last row and column
    d <- mat[2:(n - 1), 2:(n - 1), drop = FALSE]

    # truncate to the folded half if the spectrum is folded
    if (isTRUE(self$is_folded())) {
      w <- as.integer(self$w)
      d <- d[1:(w - 1), 1:(w - 1), drop = FALSE]
    }

    # a raw pair-count spectrum carries mass in the monomorphic bins (row/column 0 and n); the
    # class-resolved results (cov / corr / fpmi) are embedded with those bins zeroed. Use that to
    # choose the colour scale, as the Python backend does: a sequential log scale for the
    # heavy-tailed counts, a diverging symmetric-log scale centred at zero for the derived quantities
    border <- c(mat[c(1, n), ], mat[, c(1, n)])
    is_counts <- sum(abs(border), na.rm = TRUE) > 0

    if (is_counts) {
      # log10 maps zero counts to -Inf, which ggplot2 would drop to na.value; clamp them to the
      # smallest positive count instead, so they take the lowest colour as under matplotlib's LogNorm
      positive <- d[!is.na(d) & d > 0]
      if (length(positive)) d[!is.na(d) & d == 0] <- min(positive)

      fill_scale <- ggplot2::scale_fill_viridis_c(trans = 'log10', na.value = 'white')
    } else {
      # symmetric colour range around zero
      if (is.null(max_abs)) {
        max_abs <- max(abs(d), na.rm = TRUE)
        if (!is.finite(max_abs) || max_abs == 0) max_abs <- 1
      }

      fill_scale <- ggplot2::scale_fill_gradientn(
        colours = rev(RColorBrewer::brewer.pal(11, 'PuOr')),
        limits = c(-max_abs, max_abs),
        # approximates matplotlib's SymLogNorm(linthresh = max_abs / 10)
        trans = scales::pseudo_log_trans(sigma = max_abs / 10),
        na.value = 'white'
      )
    }

    df <- matrix_to_long(d)

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$y, fill = .data$value)) +
      ggplot2::geom_tile() +
      ggplot2::coord_fixed() +
      fill_scale +
      ggplot2::scale_x_continuous(expand = c(0, 0)) +
      ggplot2::scale_y_continuous(expand = c(0, 0)) +
      ggplot2::labs(x = '', y = '', title = title, fill = '') +
      ggplot2::theme_bw() +
      ggplot2::theme(panel.grid.major = ggplot2::element_blank(),
                     panel.grid.minor = ggplot2::element_blank())

    if (show) print(p)
    if (!is.null(file)) ggplot2::ggsave(file, plot = p)

    return(p)
  }


  # Plot a joint (multi-population) SFS (JointSFS) as a heatmap.
  #
  # Reimplements JointSFS.plot using a ggplot2 geom_tile heatmap with a viridis
  # palette. The joint SFS is marginalized onto the two requested populations,
  # which also puts their axes in the requested order. The monomorphic corners
  # are masked, and allele counts are shown on both axes with the origin at the
  # bottom left.
  #
  # @param self The JointSFS object (passed implicitly as the instance).
  # @param pops Numeric vector of length two. The (0-based) population indices to
  #             plot as (y-axis, x-axis). Default is c(0, 1).
  # @param title Character. Title of the plot. Default is NULL.
  # @param log_scale Logical. Whether to use a logarithmic colour scale.
  #                  Default is TRUE, since the joint SFS is heavily skewed
  #                  toward the low-frequency corner.
  # @param mask_monomorphic Logical. Whether to mask the monomorphic corners.
  #                         Default is TRUE.
  # @param show Logical. Whether to show the plot. Default is TRUE.
  # @param file Character. File path to save plot to. Default is NULL.
  # @param ... Additional arguments which are ignored.
  #
  # @return A ggplot object.
  sf$JointSFS$plot <- function(
    self,
    pops = c(0, 1),
    title = NULL,
    log_scale = TRUE,
    mask_monomorphic = TRUE,
    show = TRUE,
    file = NULL,
    ...
  ) {
    if (length(pops) != 2) {
      stop("Exactly two populations must be specified for a 2-dimensional plot.")
    }

    # marginalize onto the two requested populations, which also applies the requested axis order:
    # for pops = c(1, 0) the spectrum is transposed, as in Python
    jsfs <- self$marginalize(as.integer(pops))

    mat <- as.matrix(jsfs$data)
    storage.mode(mat) <- "double"
    pop_names <- unlist(jsfs$pop_names)

    # mask the monomorphic corners (all-ancestral and all-derived)
    if (mask_monomorphic) {
      mat[1, 1] <- NA
      mat[nrow(mat), ncol(mat)] <- NA
    }

    # log10 maps zero counts to -Inf, which ggplot2 would drop to na.value; clamp them to the
    # smallest positive count instead, so they take the lowest colour as under matplotlib's LogNorm
    if (log_scale) {
      positive <- mat[!is.na(mat) & mat > 0]
      if (length(positive)) mat[!is.na(mat) & mat == 0] <- min(positive)
    }

    # allele counts are 0-based
    df <- matrix_to_long(mat)
    df$x <- df$x - 1
    df$y <- df$y - 1

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$y, fill = .data$value)) +
      ggplot2::geom_tile() +
      ggplot2::coord_fixed() +
      ggplot2::labs(
        x = paste('allele count', pop_names[2]),
        y = paste('allele count', pop_names[1]),
        title = title,
        fill = ''
      ) +
      ggplot2::scale_x_continuous(expand = c(0, 0)) +
      ggplot2::scale_y_continuous(expand = c(0, 0)) +
      ggplot2::theme_bw() +
      ggplot2::theme(panel.grid.major = ggplot2::element_blank(),
                     panel.grid.minor = ggplot2::element_blank())

    if (log_scale) {
      p <- p + ggplot2::scale_fill_viridis_c(trans = 'log10', na.value = 'white')
    } else {
      p <- p + ggplot2::scale_fill_viridis_c(na.value = 'white')
    }

    if (show) print(p)
    if (!is.null(file)) ggplot2::ggsave(file, plot = p)

    return(p)
  }

  return(sf)
}
