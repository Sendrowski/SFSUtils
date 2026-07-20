"""
Visualization module for SFS-related plots.

Contains the plotting utilities used by :class:`~sfsutils.spectrum.Spectrum`,
:class:`~sfsutils.spectrum.Spectra` and the polarization diagnostics in
:mod:`sfsutils.annotation`.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import functools
import logging
from typing import Callable, List, Literal, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.container import BarContainer
from matplotlib.ticker import MaxNLocator

# get logger
logger = logging.getLogger('sfsutils').getChild('Visualization')


class Visualization:
    """
    Visualization class for SFS-related plots.
    """

    @staticmethod
    def clear_show_save(func: Callable) -> Callable:
        """
        Decorator for clearing current figure in the beginning
        and showing or saving produced plot subsequently.

        :param func: Function to decorate
        :return: Wrapper function
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> plt.Axes:
            """
            Wrapper function.

            :param args: Positional arguments
            :param kwargs: Keyword arguments
            :return: Axes
            """

            # add axes if not given
            if 'ax' not in kwargs or ('ax' in kwargs and kwargs['ax'] is None):
                # clear current figure
                plt.close()

                kwargs['ax'] = plt.gca()

            # execute function
            func(*args, **kwargs)

            # make layout tight
            plt.tight_layout()

            # show or save
            # show by default here
            return Visualization.show_and_save(
                file=kwargs['file'] if 'file' in kwargs else None,
                show=kwargs['show'] if 'show' in kwargs else True
            )

        return wrapper

    @staticmethod
    def show_and_save(file: str = None, show: bool = True, pad: float = 1.08) -> plt.Axes:
        """
        Show and save plot.

        :param file: File path to save plot to
        :param show: Whether to show plot
        :param pad: Padding for tight layout
        :return: Axes

        """
        plt.tight_layout(pad=pad)

        # save figure if file path given
        if file is not None:
            plt.savefig(file, dpi=200, bbox_inches='tight', pad_inches=0.1)

        # show figure if specified and if not in interactive mode
        if show and not plt.isinteractive():
            plt.show()

        # return current axes
        return plt.gca()

    @staticmethod
    def get_hatch(i: int, labels: List[str] = None) -> str | None:
        """
        Get hatch style for specified index i.

        :param labels: List of labels
        :param i: Index
        :return: Hatch style
        """

        # determine whether hatch style should be used
        if labels is None or len(labels) < 1 or '.' not in labels[i]:
            return

        # determine unique prefixes
        prefixes = set([label.split('.')[0] for label in labels if '.' in label])
        hatch_styles = ['/////', '\\\\\\\\\\', '***', 'ooo', 'xxx', '...']

        prefix = labels[i].split('.')[0]
        prefix_index = list(prefixes).index(prefix)

        return hatch_styles[prefix_index % len(hatch_styles)]

    @staticmethod
    def plot_spectra(
            ax: plt.Axes,
            spectra: List[List[float]] | np.ndarray,
            labels: List[str] | np.ndarray = [],
            colors: List[str] | np.ndarray = None,
            log_scale: bool = False,
            use_subplots: bool = False,
            show_monomorphic: bool = False,
            title: str = None,
            n_ticks: int = 10,
            file: str = None,
            show: bool = True,
            kwargs_legend: dict = dict(prop=dict(size=8))
    ) -> plt.Axes:
        """
        Plot the given 1D spectra.

        :param show_monomorphic: Whether to show monomorphic site counts
        :param n_ticks: Number of x-ticks to use
        :param ax: Axes to plot on. Only for Python visualization backend and if ``use_subplots`` is ``False``.
        :param title: Title of plot
        :param spectra: List of lists of spectra or a 2D array in which each row is a spectrum in the
            same order as ``labels``
        :param colors: List of colors for each spectrum.
        :param labels: List of labels for each spectrum
        :param log_scale: Whether to use logarithmic y-scale
        :param use_subplots: Whether to use subplots
        :param file: File to save plot to
        :param show: Whether to show the plot
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`.
        :return: Axes
        """
        if len(spectra) == 0:
            logger.warning('No spectra to plot.')
            return ax

        if use_subplots:

            # clear current figure
            plt.close()

            n_plots = len(spectra)
            n_rows = int(np.ceil(np.sqrt(n_plots)))
            n_cols = int(np.ceil(np.sqrt(n_plots)))

            fig = plt.figure(figsize=(6.4 * n_cols ** (1 / 3), 4.8 * n_rows ** (1 / 3)))
            axes = fig.subplots(ncols=n_cols, nrows=n_rows, squeeze=False).flatten()

            # plot spectra individually
            for i in range(n_plots):
                Visualization.plot_spectra(
                    spectra=[spectra[i]],
                    labels=[labels[i]] if len(labels) else [],
                    colors=[colors[i]] if colors else None,
                    ax=axes[i],
                    n_ticks=15 // min(2, n_cols),
                    log_scale=log_scale,
                    show_monomorphic=show_monomorphic,
                    show=False
                )

                # set title
                axes[i].set_title(labels[i] if i < len(labels) else '')

            # make empty plots invisible
            [ax.set_visible(False) for ax in axes[n_plots:]]

            # show and save plot
            return Visualization.show_and_save(file, show)

        if ax is None:
            plt.close()
            _, ax = plt.subplots()

        # determine sample size and width
        n = len(spectra[0]) - 1
        width_total = 0.9
        width = width_total / len(spectra)

        x = np.arange(n + 1) if show_monomorphic else np.arange(1, n)

        # iterator over spectra and draw bars
        for i, sfs in enumerate(spectra):
            bars = ax.bar(
                x=x + i * width,
                height=sfs if show_monomorphic else sfs[1:-1],
                width=width,
                label=labels[i] if len(labels) else None,
                color=colors[i] if colors else None,
                linewidth=0,
                hatch=Visualization.get_hatch(i, labels)
            )

            Visualization.darken_edge_colors(bars)

        # adjust ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        indices_ticks = x

        # filter ticks
        if n > n_ticks:
            indices_ticks = indices_ticks[indices_ticks % max(int(np.ceil(n / n_ticks)), 1) == 1]

        ax.set_xticks([i + (width_total - width) / 2 for i in indices_ticks], indices_ticks)

        ax.set_xlabel('allele count')

        # remove x-margins
        ax.autoscale(tight=True, axis='x')

        if log_scale:
            ax.set_yscale('log')

        # set title
        ax.set_title(title)

        # show legend if more than one label
        if len(labels) > 1:
            ax.legend(**kwargs_legend)

        # show and save plot
        return Visualization.show_and_save(file, show)

    @staticmethod
    def darken_edge_colors(bars: BarContainer):
        """
        Darken the edge color of the given bars.

        :param bars: Bars to darken
        """
        for bar in bars:
            color = bar.get_facecolor()
            edge_color = Visualization.darken_color(color, amount=0.75)
            bar.set_edgecolor(edge_color)

    @staticmethod
    def darken_color(color, amount=0.5) -> tuple:
        """
        Darken a color.

        :param color: Color to darken
        :param amount: Amount to darken
        :return: Darkened color as tuple
        """
        c = mcolors.to_rgba(color)

        return c[0] * amount, c[1] * amount, c[2] * amount, c[3]

    @staticmethod
    @clear_show_save
    def plot_scatter(
            values: Sequence,
            file: str,
            show: bool,
            ax: plt.Axes,
            title: str | None = None,
            scale: Literal['lin', 'log', 'symlog'] = 'lin',
            ylabel: str = 'lnl'
    ) -> plt.Axes:
        """
        A scatter plot.

        :param scale: Scale of y-axis
        :param values: Values to plot
        :param file: File to save plot to
        :param show: Whether to show plot
        :param title: Title of plot
        :param ax: Axes to plot on.
        :param ylabel: Label of y-axis
        :return: Axes
        """
        # plot
        sns.scatterplot(x=range(len(values)), y=values, ax=ax)

        ax.set(ylabel=ylabel)

        # set title
        ax.set_title(title)

        if scale == 'log':
            ax.set_yscale('symlog')

        return ax
