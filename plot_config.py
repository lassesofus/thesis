"""
Shared plotting configuration for consistent visualization across all thesis plots.

This module provides centralized plotting parameters to ensure visual consistency
across all figures in the thesis report.
"""

import matplotlib.pyplot as plt

# Central plotting configuration
PLOT_PARAMS = {
    "figsize_with_images": (10, 14),
    "figsize_plots_only": (12, 10),
    "title_size": 28,
    "subtitle_size": 20,
    "label_size": 20,
    "legend_size": 12,
    "grid_alpha": 0.3,
    "euclid_linewidth": 2.0,
    "euclid_markersize": 5,
    "euclid_capsize": 3,
    "repr_linewidth": 2.0,
    "repr_markersize": 5,
    "repr_capsize": 3,
    "threshold_linewidth": 2,
    "tick_label_size": 12,
    "tick_length": 4,
    "y_tick_step": 0.05,
}


def apply_plot_params(ax=None):
    """
    Apply consistent plot parameters to a matplotlib axis.

    Args:
        ax: matplotlib axis object. If None, applies to current axis.
    """
    if ax is None:
        ax = plt.gca()

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=PLOT_PARAMS["tick_label_size"],
        length=PLOT_PARAMS["tick_length"],
    )
    ax.grid(True, alpha=PLOT_PARAMS["grid_alpha"])

    return ax


def configure_axis(ax, xlabel, ylabel, title):
    """
    Configure axis with consistent font sizes.

    Args:
        ax: matplotlib axis object
        xlabel: x-axis label
        ylabel: y-axis label
        title: plot title
    """
    ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS["label_size"])
    ax.set_ylabel(ylabel, fontsize=PLOT_PARAMS["label_size"])
    ax.set_title(title, fontsize=PLOT_PARAMS["title_size"])
    apply_plot_params(ax)

    return ax
