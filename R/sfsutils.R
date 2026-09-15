if (getRversion() >= "2.15.1") utils::globalVariables(c(".data"))

# matplotlib's default colour cycle ('C0', 'C1', ...)
.tab10 <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

# ggplot2 theme of the plots: no grid lines and a centred title
.plot_theme <- function() {
  ggplot2::theme(panel.grid = ggplot2::element_blank(), plot.title = ggplot2::element_text(hjust = 0.5))
}

# colour bar spanning the height of the plot without tick marks, as the colour bars of seaborn's heatmaps
.colourbar <- function() {
  ggplot2::guide_colourbar(theme = ggplot2::theme(
    legend.key.height = ggplot2::unit(1, "null"),
    legend.ticks.length = ggplot2::unit(0, "pt")
  ))
}

# print the plot if 'show' is TRUE, save it to 'file' if one is given, and return it
.show_and_save <- function(p, show, file) {
  if (show) print(p)
  if (!is.null(file)) ggplot2::ggsave(file, plot = p)
  p
}

# scales transformation of matplotlib's symmetric log scale (base 10, linear scale 1), which is linear within
# [-linthresh, linthresh] and logarithmic outside
.symlog_trans <- function(linthresh) {
  adj <- 1 / (1 - 1 / 10)

  scales::trans_new(
    "symlog",
    transform = function(x) {
      ifelse(abs(x) <= linthresh, x * adj, sign(x) * linthresh * (adj + log10(abs(x) / linthresh)))
    },
    inverse = function(y) {
      ifelse(abs(y) <= linthresh * adj, y / adj, sign(y) * linthresh * 10^(abs(y) / linthresh - adj))
    }
  )
}

# breaks at the powers of ten within the limits of a log scale, and at their 1-3 or 1-2-5 multiples where the limits
# contain fewer than two powers
.log_breaks <- function(limits) {
  powers <- 10^(floor(log10(limits[1])):ceiling(log10(limits[2])))
  powers <- powers[powers >= limits[1] & powers <= limits[2]]

  if (length(powers) >= 2) powers else scales::breaks_log()(limits)
}

# breaks at 0 and the signed powers of ten within the limits of a symmetric log scale with linear threshold
# 'linthresh', and pretty breaks where the limits contain none of them
.symlog_breaks <- function(linthresh) {
  function(limits) {
    powers <- 10^(floor(log10(linthresh)):ceiling(log10(max(abs(limits), linthresh))))
    breaks <- c(-rev(powers), 0, powers)
    breaks <- breaks[breaks >= limits[1] & breaks <= limits[2]]

    if (length(breaks) > 0) breaks else scales::extended_breaks()(limits)
  }
}

# plotmath labels writing breaks that are all 0 or signed single-digit multiples of powers of ten as m %*% 10^k,
# and plain numbers otherwise
.power_labels <- function(breaks) {
  k <- floor(log10(abs(breaks)) + 1e-9)
  m <- round(abs(breaks) / 10^k, 8)
  is_power <- is.na(breaks) | breaks == 0 | m == round(m)

  if (!all(is_power)) {
    return(format(breaks, trim = TRUE, drop0trailing = TRUE))
  }

  text <- paste0(ifelse(breaks < 0, "-", ""), ifelse(m == 1, "", paste0(m, " %*% ")), "10^", k)
  parse(text = ifelse(is.na(breaks), "''", ifelse(breaks == 0, "0", text)))
}

# ggplot2 y scale that is linear ('lin'), log10 ('log') or matplotlib's symmetric log with its default linear
# threshold 2 ('symlog'), labelled at the powers of ten on the logarithmic scales
.scale_y <- function(scale, expand = ggplot2::waiver()) {
  switch(
    scale,
    lin = ggplot2::scale_y_continuous(expand = expand),
    log = ggplot2::scale_y_continuous(trans = "log10", breaks = .log_breaks, labels = .power_labels,
                                      expand = expand),
    symlog = ggplot2::scale_y_continuous(trans = .symlog_trans(2), breaks = .symlog_breaks(2),
                                         labels = .power_labels, expand = expand)
  )
}

# viridis fill scale, on a log10 scale labelled at the powers of ten if 'log_scale' is TRUE, leaving missing cells blank
.scale_fill_viridis <- function(log_scale) {
  if (log_scale) {
    ggplot2::scale_fill_viridis_c(trans = "log10", breaks = .log_breaks, labels = .power_labels,
                                  na.value = "white", guide = .colourbar())
  } else {
    ggplot2::scale_fill_viridis_c(na.value = "white", guide = .colourbar())
  }
}

# heatmap of a matrix with square cells and its first row at the bottom, the rows and columns numbered from 'origin'
.heatmap <- function(mat, origin, fill_scale, title, x = NULL, y = NULL) {
  df <- data.frame(
    x = rep(seq_len(ncol(mat)), each = nrow(mat)) + origin - 1,
    y = rep(seq_len(nrow(mat)), times = ncol(mat)) + origin - 1,
    value = as.vector(mat)
  )

  # integer cell numbers, at most twelve of them per axis
  breaks <- function(limits) seq(ceiling(limits[1]), floor(limits[2]), by = max(1, ceiling(diff(limits) / 12)))

  ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$y, fill = .data$value)) +
    ggplot2::geom_tile() +
    ggplot2::coord_fixed() +
    fill_scale +
    ggplot2::scale_x_continuous(breaks = breaks, expand = c(0, 0)) +
    ggplot2::scale_y_continuous(breaks = breaks, expand = c(0, 0)) +
    ggplot2::labs(x = x, y = y, title = title, fill = NULL) +
    .plot_theme()
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
#' sfsutils_is_installed()  # Returns TRUE or FALSE based on the installation status of sfsutils
#' }
#'
#' @export
sfsutils_is_installed <- function() {

  # An unbound session reports FALSE without touching Python, leaving the interpreter
  # for the declared requirements to select at the version they ask for
  if (!reticulate::py_available(initialize = FALSE)) {
    return(FALSE)
  }

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


# Requirement string for the Python distribution, carrying the optional input backends
# and a pinned version where one is given. The distribution is named 'sfsutils-popgen'
# on PyPI, where 'sfsutils' is an unrelated project
py_requirement <- function(version = NULL, extras = c("vcf")) {

  spec <- "sfsutils-popgen"

  if (length(extras) > 0) {
    spec <- paste0(spec, "[", paste(extras, collapse = ","), "]")
  }

  if (!is.null(version)) {
    spec <- paste0(spec, "==", version)
  }

  spec
}


.onLoad <- function(libname, pkgname) {
  reticulate::py_require(py_requirement(), python_version = "3.11")
}


#' Declare the `sfsutils` Python module requirement
#'
#' Loading the package declares `sfsutils` with the `vcf` backend. This function declares
#' a different set of backends, or a pinned version.
#' The requirement is resolved when Python is first initialised, at which point
#' reticulate provisions an environment satisfying it.
#'
#' @param version A character string specifying the version of the `sfsutils` module
#'        to require. Default is `NULL` which resolves to the latest version.
#' @param extras A character vector of optional input backends to require alongside the module:
#'        `'vcf'` for VCF files, `'zarr'` for VCF-Zarr stores and `'arg'` for tree sequences.
#'        Default is `c("vcf")`; pass `NULL` to require none of them.
#' @param force Logical, has no effect. Default is `FALSE`.
#' @param silent Logical, if `TRUE` it will suppress the message naming the declared
#'        requirement. Default is `FALSE`.
#' @param python_version A character string specifying the Python version reticulate
#'        should provision the environment with. Default is `'3.11'`.
#'
#' @return Invisible `NULL`.
#'
#' @examples
#' \dontrun{
#' install_sfsutils()  # Requires the latest version of sfsutils with the vcf backend
#' install_sfsutils(extras = c("vcf", "zarr", "arg"))  # Requires all input backends
#' install_sfsutils(extras = NULL)  # Requires none of the optional backends
#' }
#'
#' @export
install_sfsutils <- function(version = NULL, extras = c("vcf"), force = FALSE, silent = FALSE, python_version = '3.11') {

  if (force) {
    warning("'force' has no effect.", call. = FALSE)
  }

  spec <- py_requirement(version, extras)

  reticulate::py_require(spec, python_version = python_version)

  if (!silent) {
    message("Declared Python requirement '", spec, "' on Python ", python_version, ".")
  }

  invisible(NULL)
}

#' Load the sfsutils library and associated visualization functions
#'
#' This function imports the Python package 'sfsutils' using the reticulate package
#' and then configures it to work seamlessly with R, overriding some of the default
#' visualization functions with custom R-based ones.
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

  forward_python_output()

  sf <- reticulate::import("sfsutils")

  # override python visualization functions
  viz <- sf$visualization$Visualization

  # Create a scatter plot.
  #
  # @param values List or numeric vector. Values to plot.
  # @param file Character. File path to save plot to. Default is NULL.
  # @param show Logical. Whether to show plot. Default is TRUE.
  # @param title Character. Title of plot.
  # @param scale Character. Scale of y-axis. One of 'lin', 'log', where 'log' is a symmetric log scale.
  #              Default is 'lin'.
  # @param ylabel Character. Label of the y-axis. Default is 'lnl'.
  # @param ... Additional arguments which are ignored.
  #
  # @return A ggplot object.
  viz$plot_scatter <- function(
    values,
    file = NULL,
    show = TRUE,
    title = NULL,
    scale = 'lin',
    ylabel = 'lnl',
    ...
  ) {
    df <- data.frame(x = seq_along(values) - 1, y = unlist(values))

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$y)) +
      ggplot2::geom_point(colour = .tab10[1], size = 2) +
      .scale_y(if (scale == 'log') 'symlog' else 'lin') +
      ggplot2::labs(x = NULL, y = ylabel, title = title) +
      .plot_theme()

    .show_and_save(p, show, file)
  }


  # Plot the given 1D spectra as bars, dodged within each allele count and coloured by matplotlib's colour cycle
  # in the order of the spectra.
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
  # @param ... Additional arguments which are ignored.
  #
  # @return ggplot object
  plot_spectra <- function(
    spectra,
    labels = character(0),
    log_scale = FALSE,
    use_subplots = FALSE,
    show_monomorphic = FALSE,
    title = NULL,
    n_ticks = 10,
    file = NULL,
    show = TRUE,
    ...
  ) {
    if (length(spectra) == 0) {
      warning('No spectra to plot.')
      return(NULL)
    }

    labels <- as.character(unlist(labels))

    if (use_subplots) {
      # one plot per spectrum on a square grid, titled by its label
      n_cols <- ceiling(sqrt(length(spectra)))

      plot_list <- lapply(seq_along(spectra), function(i) {
        label <- if (length(labels) >= i) labels[i] else character(0)

        plot_spectra(
          spectra = list(spectra[[i]]),
          labels = label,
          log_scale = log_scale,
          show_monomorphic = show_monomorphic,
          title = if (length(label)) label else NULL,
          n_ticks = 15 %/% min(2, n_cols),
          show = FALSE
        )
      })

      return(.show_and_save(cowplot::plot_grid(plotlist = plot_list, nrow = n_cols, ncol = n_cols), show, file))
    }

    if (length(labels) == 0) {
      labels <- as.character(seq_along(spectra))
    }

    # allele counts of the bars, each spectrum taking an equal share of the width 0.9 per allele count
    n <- length(spectra[[1]]) - 1
    x <- if (show_monomorphic) 0:n else seq_len(n - 1)
    width <- 0.9 / length(spectra)

    df <- data.frame(
      xmin = unlist(lapply(seq_along(spectra), function(i) x - 0.45 + (i - 1) * width)),
      y = unlist(lapply(spectra, function(sfs) unlist(sfs)[x + 1])),
      group = factor(rep(labels, each = length(x)), levels = unique(labels))
    )
    df$xmax <- df$xmin + width

    # on the log scale, bars rise from the power of ten below the smallest positive count, which is the bottom of the axis
    if (log_scale) {
      df <- df[df$y > 0, ]
      df$ymin <- 10^floor(log10(min(df$y)))
    } else {
      df$ymin <- 0
    }

    # label every allele count, or every k-th starting at 1 where there are more than 'n_ticks'
    breaks <- if (n > n_ticks) x[x %% ceiling(n / n_ticks) == 1] else x

    p <- ggplot2::ggplot(df, ggplot2::aes(xmin = .data$xmin, xmax = .data$xmax, ymin = .data$ymin, ymax = .data$y,
                                          fill = .data$group)) +
      ggplot2::geom_rect(show.legend = length(spectra) > 1) +
      ggplot2::scale_fill_manual(values = rep_len(.tab10, nlevels(df$group))) +
      ggplot2::scale_x_continuous(breaks = breaks, expand = c(0, 0)) +
      .scale_y(if (log_scale) 'log' else 'lin', expand = ggplot2::expansion(mult = c(0, 0.05))) +
      ggplot2::labs(x = "allele count", y = NULL, title = title) +
      .plot_theme() +
      # legend inside the panel (top-right) in a frame over a semi-transparent background
      ggplot2::theme(
        legend.position = "inside",
        legend.position.inside = c(0.97, 0.97),
        legend.justification = c(1, 1),
        legend.title = ggplot2::element_blank(),
        legend.text = ggplot2::element_text(size = 8),
        legend.key.size = ggplot2::unit(0.8, "lines"),
        legend.margin = ggplot2::margin(4, 4, 4, 4),
        legend.background = ggplot2::element_rect(fill = scales::alpha("white", 0.8), colour = "grey80"),
        legend.key = ggplot2::element_rect(fill = NA, colour = NA)
      )

    .show_and_save(p, show, file)
  }

  viz$plot_spectra <- plot_spectra


  # The heatmaps below override instance methods of the Python classes, so `obj$plot()` and
  # `sf$TwoSFS$plot(obj)` both pass the object as `self`.
  #
  # Plot a 2-SFS (TwoSFS) as a heatmap of its segregating interior, restricted to the folded half if the
  # spectrum is folded. Raw pair counts use a sequential log viridis scale, the class-resolved results
  # (cov / corr / fpmi) a diverging symmetric log PuOr_r scale centred at zero, as in the Python package.
  #
  # @param self The TwoSFS object (passed implicitly as the instance).
  # @param title Character. Title of the plot. Default is NULL.
  # @param log_scale Logical. Ignored. Default is FALSE.
  # @param max_abs Numeric. Maximum absolute value of the diverging colour scale; ignored
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
    if (self$n < 3) {
      warning('Nothing to plot.')
      return(invisible(NULL))
    }

    mat <- as.matrix(self$data)
    storage.mode(mat) <- "double"
    n <- nrow(mat)

    # remove monomorphic first and last row and column
    d <- mat[2:(n - 1), 2:(n - 1), drop = FALSE]

    # truncate to the folded half if the spectrum is folded
    if (isTRUE(self$is_folded())) {
      w <- as.integer(self$w)
      d <- d[1:(w - 1), 1:(w - 1), drop = FALSE]
    }

    # a raw pair-count spectrum carries mass in the monomorphic bins (row/column 0 and n), whereas the
    # class-resolved results are embedded with those bins zeroed
    is_counts <- sum(abs(c(mat[c(1, n), ], mat[, c(1, n)])), na.rm = TRUE) > 0

    if (is_counts) {
      # zero counts have no logarithm and are left blank, as under matplotlib's LogNorm
      d[!is.na(d) & d <= 0] <- NA

      fill_scale <- .scale_fill_viridis(log_scale = TRUE)
    } else {
      if (is.null(max_abs)) {
        max_abs <- max(abs(d), na.rm = TRUE)
        if (!is.finite(max_abs) || max_abs == 0) max_abs <- 1
      }

      # matplotlib's SymLogNorm(linthresh = max_abs / 10) over a symmetric range, clipping values outside it
      fill_scale <- ggplot2::scale_fill_gradientn(
        colours = rev(RColorBrewer::brewer.pal(11, 'PuOr')),
        limits = c(-max_abs, max_abs),
        trans = .symlog_trans(max_abs / 10),
        breaks = .symlog_breaks(max_abs / 10),
        labels = .power_labels,
        oob = scales::squish,
        na.value = 'white',
        guide = .colourbar()
      )
    }

    .show_and_save(.heatmap(d, 1, fill_scale, title), show, file)
  }


  # Plot a joint (multi-population) SFS (JointSFS) as a heatmap.
  #
  # The joint SFS is marginalized onto the two requested populations, which also puts their
  # axes in the requested order. The monomorphic corners are masked, and allele counts are
  # shown on both axes with the origin at the bottom left.
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

    # for pops = c(1, 0) the marginalized spectrum is transposed
    mat <- as.matrix(self$marginalize(as.integer(pops))$data)
    storage.mode(mat) <- "double"
    pop_names <- unlist(self$`_names`())[pops + 1]

    # mask the monomorphic corners (all-ancestral and all-derived)
    if (mask_monomorphic) {
      mat[1, 1] <- NA
      mat[nrow(mat), ncol(mat)] <- NA
    }

    # zero counts have no logarithm and are left blank, as under matplotlib's LogNorm
    if (log_scale) {
      mat[!is.na(mat) & mat <= 0] <- NA
    }

    p <- .heatmap(mat, 0, .scale_fill_viridis(log_scale), title,
                  x = paste('allele count', pop_names[2]), y = paste('allele count', pop_names[1]))

    .show_and_save(p, show, file)
  }

  return(sf)
}


# In a Jupyter kernel, write Python's standard output and error through R's output and message streams, which the kernel
# captures, so log messages and progress bars reach the cell output.
forward_python_output <- function() {

  if (!isTRUE(getOption("jupyter.in_kernel"))) {
    return(invisible(NULL))
  }

  # the kernel ends every message with a line break, so the error stream is passed on as complete lines without one
  streams <- reticulate::py_run_string("
import io

class RStream(io.TextIOBase):
    encoding = 'utf-8'

    def __init__(self, write, lines=False):
        super().__init__()
        self._write = write
        self._lines = lines
        self._buffer = ''

    def writable(self):
        return True

    def write(self, text):
        if not self._lines:
            self._write(text)
            return len(text)

        *complete, partial = (self._buffer + text).split('\\n')
        for line in complete:
            self._write(line.split('\\r')[-1])

        # a carriage return starts the line over, as a progress bar redraws itself
        self._buffer = partial.split('\\r')[-1]

        return len(text)
", local = TRUE, convert = FALSE)

  sys <- reticulate::import("sys", convert = FALSE)
  sys$stdout <- streams$RStream(function(text) cat(reticulate::py_to_r(text)))
  sys$stderr <- streams$RStream(
    function(line) message(reticulate::py_to_r(line), appendLF = FALSE),
    lines = TRUE
  )

  invisible(NULL)
}
