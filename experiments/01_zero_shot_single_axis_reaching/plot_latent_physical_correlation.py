#!/usr/bin/env python3
"""
Plot latent-physical distance correlation from pre-computed data.

This script loads the pre-computed correlation data and generates plots
showing the relationship between Euclidean distance to goal and mean L1 distance
in the V-JEPA 2 encoder's latent space for each axis.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add thesis directory for plot config
sys.path.insert(0, "/home/s185927/thesis")
from plot_config import PLOT_PARAMS, configure_axis


def main():
    output_dir = Path("/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching")

    # Load existing data
    data = np.load(output_dir / "latent_physical_correlation_data.npz")

    # Detect available axes
    axes = []
    for key in data.keys():
        if key.endswith("_euclidean"):
            axis = key.replace("_euclidean", "")
            axes.append(axis)

    print(f"Found axes: {axes}")

    colors = {'x': 'tab:red', 'y': 'tab:green', 'z': 'tab:blue'}

    # Create comparison plot (1 row, N columns)
    n_axes = len(axes)
    fig, ax_list = plt.subplots(1, n_axes, figsize=(5 * n_axes, 5))
    if n_axes == 1:
        ax_list = [ax_list]

    for idx, axis in enumerate(axes):
        ax = ax_list[idx]
        euclidean = data[f"{axis}_euclidean"]
        latent = data[f"{axis}_latent"]
        corr = float(data[f"{axis}_correlation"])

        print(f"{axis.upper()}-axis: {len(euclidean)} points, correlation: {corr:.4f}")

        ax.scatter(euclidean, latent, alpha=0.3, s=20, color=colors.get(axis, 'tab:gray'))

        # Add trend line
        z = np.polyfit(euclidean, latent, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(euclidean), max(euclidean), 100)
        ax.plot(x_trend, p(x_trend), "k--", linewidth=2)

        # Add Pearson correlation as text annotation
        ax.text(
            0.95, 0.05, f"Pearson r = {corr:.2f}",
            transform=ax.transAxes,
            fontsize=PLOT_PARAMS["legend_size"] + 2,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        configure_axis(
            ax,
            xlabel="Euclidean Distance to Goal (m)",
            ylabel="Mean L1 Distance to Goal",
            title=f"{axis.upper()}-Axis Movement",
        )

    plt.tight_layout()

    output_path = output_dir / "latent_physical_correlation_by_axis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_path}")

    # Create combined plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for axis in axes:
        euclidean = data[f"{axis}_euclidean"]
        latent = data[f"{axis}_latent"]
        corr = float(data[f"{axis}_correlation"])

        ax.scatter(
            euclidean,
            latent,
            alpha=0.3,
            s=20,
            color=colors.get(axis, 'tab:gray'),
            label=f"{axis.upper()}-axis (r={corr:.2f})",
        )

    ax.legend(loc='lower right', fontsize=PLOT_PARAMS["legend_size"])
    configure_axis(
        ax,
        xlabel="Euclidean Distance to Goal (m)",
        ylabel="Mean L1 Distance to Goal",
        title="Latent-Physical Correlation by Axis (Zero-Shot)",
    )

    plt.tight_layout()
    combined_path = output_dir / "latent_physical_correlation_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {combined_path}")


if __name__ == "__main__":
    main()
