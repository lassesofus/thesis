#!/usr/bin/env python3
"""
Plot latent-physical distance correlation from pre-computed data.

This script loads the pre-computed correlation data and generates a single plot
showing the relationship between Euclidean distance to goal and mean L1 distance
in the V-JEPA 2 encoder's latent space.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add thesis directory for plot config
sys.path.insert(0, "/home/s185927/thesis")
from plot_config import PLOT_PARAMS, configure_axis


def main():
    output_dir = Path("/home/s185927/thesis/experiments/sim_training_different_fractions_x_axis_finetune")

    # Load existing data
    data = np.load(output_dir / "latent_physical_correlation_data.npz")

    # Get the first available key prefix (handles different naming conventions)
    keys = list(data.keys())
    prefix = keys[0].rsplit("_euclidean", 1)[0] if "_euclidean" in keys[0] else keys[0].rsplit("_", 1)[0]

    euclidean = data[f"{prefix}_euclidean"]
    latent = data[f"{prefix}_latent"]
    corr = float(data[f"{prefix}_correlation"])

    print(f"Loaded {len(euclidean)} points, correlation: {corr:.4f}")

    # Create single plot using consistent figure size
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(euclidean, latent, alpha=0.3, s=20, color="tab:blue")

    # Add trend line
    z = np.polyfit(euclidean, latent, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(euclidean), max(euclidean), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label="Linear fit")

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
        title="Latent-Physical Distance Correlation",
    )

    plt.tight_layout()

    output_path = output_dir / "latent_physical_correlation_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
