#!/usr/bin/env python3
"""
Plot 3x3 grid of cross-axis latent-physical distance correlation.

Rows: goal axis (x, y, z) — goal is 20cm offset along this axis.
Columns: movement axis (x, y, z) — the robot moves along this axis.
Diagonal: matching goal and movement axes (expected high correlation).
Off-diagonal: mismatched goal and movement (expected low/no correlation).

Each cell shows latent L1 distance (x-axis) vs L2 Euclidean distance to
goal (y-axis), colored by episode, with a linear trend line and Pearson r.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

sys.path.insert(0, "/home/s185927/thesis")
from plot_config import PLOT_PARAMS, apply_plot_params


def main():
    output_dir = Path(
        "/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/"
        "follow_up_experiments/latent_physical_distance_alignment"
    )

    data = np.load(output_dir / "cross_axis_correlation_data.npz")

    goal_axes = ["x", "y", "z"]
    movement_axes = ["x", "y", "z"]

    # --- Compute shared axis limits across all 9 cells ---
    all_latent = []
    all_euclidean = []
    for g in goal_axes:
        for m in movement_axes:
            key = f"{g}_{m}"
            all_latent.append(data[f"{key}_latent"])
            all_euclidean.append(data[f"{key}_euclidean"])
    all_latent = np.concatenate(all_latent)
    all_euclidean = np.concatenate(all_euclidean)

    latent_min, latent_max = all_latent.min(), all_latent.max()
    latent_pad = (latent_max - latent_min) * 0.05
    euc_min, euc_max = all_euclidean.min(), all_euclidean.max()
    euc_pad = (euc_max - euc_min) * 0.05

    # --- Create 3x3 figure ---
    fig, axs = plt.subplots(3, 3, figsize=(15, 14))

    for row_idx, goal_ax in enumerate(goal_axes):
        for col_idx, move_ax in enumerate(movement_axes):
            ax = axs[row_idx, col_idx]
            key = f"{goal_ax}_{move_ax}"

            latent = data[f"{key}_latent"]
            euclidean = data[f"{key}_euclidean"]
            episode_idx = data[f"{key}_episode_idx"].astype(int)
            r = float(data[f"{key}_correlation"])

            unique_eps = np.unique(episode_idx)
            n_eps = len(unique_eps)
            is_diagonal = row_idx == col_idx

            cmap = cm.get_cmap("viridis")
            episode_colors = [
                cmap(i / max(1, n_eps - 1)) for i in range(n_eps)
            ]

            # Scatter by episode
            for ep_i, ep in enumerate(unique_eps):
                mask = episode_idx == ep
                ax.scatter(
                    latent[mask],
                    euclidean[mask],
                    s=15,
                    color=episode_colors[ep_i],
                    alpha=0.8,
                    edgecolors="white",
                    linewidths=0.2,
                )

            # Trend line
            z = np.polyfit(latent, euclidean, 1)
            slope, intercept = z[0], z[1]
            p = np.poly1d(z)
            x_trend = np.linspace(latent.min(), latent.max(), 100)
            ax.plot(
                x_trend,
                p(x_trend),
                "r--",
                linewidth=2,
                label=f"{slope:.2f}x + {intercept:.2f}",
            )

            # Pearson r annotation
            ax.text(
                0.05,
                0.95,
                f"r = {r:.2f}",
                transform=ax.transAxes,
                fontsize=PLOT_PARAMS["legend_size"] + 2,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontweight="bold" if is_diagonal else "normal",
            )

            # Shared axis limits
            ax.set_xlim(latent_min - latent_pad, latent_max + latent_pad)
            ax.set_ylim(euc_min - euc_pad, euc_max + euc_pad)

            # Highlight diagonal
            if is_diagonal:
                ax.patch.set_facecolor("#f0f7ff")

            # Axis labels (only on edges)
            if row_idx == 2:
                ax.set_xlabel(
                    r"$\frac{1}{TD}\|z_k - z_g\|_1$",
                    fontsize=PLOT_PARAMS["label_size"],
                )
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel(
                    r"$\|p_k - p_g\|_2$ (m)",
                    fontsize=PLOT_PARAMS["label_size"],
                )
            else:
                ax.set_yticklabels([])

            # Column titles (top row)
            if row_idx == 0:
                ax.set_title(
                    f"{move_ax}-movement",
                    fontsize=PLOT_PARAMS["subtitle_size"],
                    pad=10,
                )

            ax.tick_params(
                axis="both",
                which="both",
                labelsize=PLOT_PARAMS["tick_label_size"],
                length=PLOT_PARAMS["tick_length"],
            )
            ax.grid(True, alpha=PLOT_PARAMS["grid_alpha"])

            # Linear fit legend
            ax.legend(
                loc="lower right",
                fontsize=PLOT_PARAMS["legend_size"] - 1,
                framealpha=0.9,
            )

    # Row labels
    for row_idx, goal_ax in enumerate(goal_axes):
        axs[row_idx, 0].annotate(
            f"{goal_ax}-axis goal",
            xy=(-0.38, 0.5),
            xycoords="axes fraction",
            fontsize=PLOT_PARAMS["subtitle_size"],
            rotation=90,
            va="center",
            ha="center",
            fontweight="bold",
        )

    # Shared episode legend at bottom
    cmap = cm.get_cmap("viridis")
    n_legend = min(5, n_eps)
    episode_handles = []
    for i in range(n_legend):
        color = cmap(i / max(1, n_legend - 1))
        handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=8,
            label=f"Episode {i + 1}",
        )
        episode_handles.append(handle)

    fig.legend(
        handles=episode_handles,
        loc="lower center",
        ncol=n_legend,
        fontsize=PLOT_PARAMS["legend_size"],
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0.06, 0.05, 1, 1])

    output_path = output_dir / "latent_physical_correlation_3x3.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
