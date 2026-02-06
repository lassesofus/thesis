#!/usr/bin/env python3
"""
Plot latent-physical distance correlation from pre-computed data.

This script loads the pre-computed correlation data and generates plots
showing the relationship between Euclidean distance to goal and mean L1 distance
in the V-JEPA 2 encoder's latent space for each axis.

Episodes are plotted with distinct colors from a colormap, with lines connecting
consecutive frames within each trajectory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

    # Match titles from consolidated plots with panel labels
    titles = {'x': '(a) Reach along x', 'y': '(b) Reach along y', 'z': '(c) Reach along z'}

    # Compute shared axis limits across all axes for easier comparison
    all_euclidean = []
    all_latent = []
    for axis in axes:
        all_euclidean.extend(data[f"{axis}_euclidean"])
        all_latent.extend(data[f"{axis}_latent"])

    x_min, x_max = min(all_euclidean), max(all_euclidean)
    y_min, y_max = min(all_latent), max(all_latent)
    # Add small padding
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    # Create comparison plot (1 row, 3 columns)
    fig, ax_list = plt.subplots(1, 3, figsize=(15, 5))

    for idx, axis in enumerate(axes):
        ax = ax_list[idx]
        euclidean = data[f"{axis}_euclidean"]
        latent = data[f"{axis}_latent"]
        corr = float(data[f"{axis}_correlation"])

        # Check if episode indices are available
        episode_key = f"{axis}_episode_idx"
        if episode_key in data:
            episode_idx = data[episode_key].astype(int)
            unique_episodes = np.unique(episode_idx)
            n_episodes = len(unique_episodes)

            # Get colors from viridis colormap at equidistant points
            cmap = cm.get_cmap('viridis')
            episode_colors = [cmap(i / (n_episodes - 1)) if n_episodes > 1 else cmap(0.5)
                             for i in range(n_episodes)]

            print(f"{axis.upper()}-axis: {len(euclidean)} points, {n_episodes} episodes, correlation: {corr:.4f}")

            # Plot each episode with its own color (no label - shared legend added later)
            for ep_i, ep in enumerate(unique_episodes):
                mask = episode_idx == ep
                ep_euclidean = euclidean[mask]
                ep_latent = latent[mask]
                color = episode_colors[ep_i]

                # Plot markers without label (shared legend handles episodes)
                ax.scatter(ep_latent, ep_euclidean, s=25, color=color, alpha=0.9,
                          edgecolors='white', linewidths=0.3)
        else:
            # Fallback: no episode info, plot as before
            print(f"{axis.upper()}-axis: {len(euclidean)} points, correlation: {corr:.4f}")
            ax.scatter(latent, euclidean, alpha=0.8, s=20, color='#1f77b4')

        # Add trend line (red dashed) with slope and intercept in label
        z = np.polyfit(latent, euclidean, 1)
        slope, intercept = z[0], z[1]
        p = np.poly1d(z)
        x_trend = np.linspace(min(latent), max(latent), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2,
                label=f"{slope:.2f}x + {intercept:.2f}")

        # Add Pearson correlation as text annotation (top left)
        ax.text(
            0.05, 0.95, f"Pearson r = {corr:.2f}",
            transform=ax.transAxes,
            fontsize=PLOT_PARAMS["legend_size"] + 2,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Set matching axis limits for easier comparison (swapped for flipped axes)
        ax.set_xlim(y_min - y_padding, y_max + y_padding)
        ax.set_ylim(x_min - x_padding, x_max + x_padding)

        # Configure axis - only show y-axis label on first subplot
        configure_axis(
            ax,
            xlabel=r"$\frac{1}{TD}\|z_k - z_g\|_1$",
            ylabel=r"$\|p_k - p_g\|_2$ (m)" if idx == 0 else "",
            title=titles[axis],
        )

        # Add legend for linear fit only (per subplot, since slope varies)
        ax.legend(loc='lower right', fontsize=PLOT_PARAMS["legend_size"], framealpha=0.9)

    # Create shared legend for episodes below the subplots
    # Use dummy scatter plots for the legend handles
    cmap = cm.get_cmap('viridis')
    episode_handles = []
    for i in range(5):  # 5 episodes
        color = cmap(i / 4)
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                           markersize=8, label=f"Episode {i + 1}")
        episode_handles.append(handle)

    fig.legend(handles=episode_handles, loc='lower center', ncol=5,
               fontsize=PLOT_PARAMS["legend_size"], framealpha=0.9,
               bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Make room for shared legend at bottom

    output_path = output_dir / "latent_physical_correlation_by_axis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_path}")

    # Create combined plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, axis in enumerate(axes):
        euclidean = data[f"{axis}_euclidean"]
        latent = data[f"{axis}_latent"]
        corr = float(data[f"{axis}_correlation"])

        # Use different base colors for each axis in combined plot
        axis_colors = {'x': '#1f77b4', 'y': '#2ca02c', 'z': '#ff7f0e'}

        ax.scatter(
            latent,
            euclidean,
            alpha=0.3,
            s=20,
            color=axis_colors.get(axis, '#1f77b4'),
            label=f"{axis.upper()}-axis (r={corr:.2f})",
        )

    ax.legend(loc='lower right', fontsize=PLOT_PARAMS["legend_size"])
    configure_axis(
        ax,
        xlabel=r"$\frac{1}{TD}\|z_k - z_g\|_1$",
        ylabel=r"$\|p_k - p_g\|_2$ (m)",
        title="Latent-Physical Correlation by Axis (Zero-Shot)",
    )

    plt.tight_layout()
    combined_path = output_dir / "latent_physical_correlation_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {combined_path}")


if __name__ == "__main__":
    main()
