#!/usr/bin/env python3
"""
Overlay CEM-planned trajectory points on the IK correlation curve.

Shows that IK (optimal) trajectories follow the latent-physical correlation
curve, while CEM-planned trajectories deviate — latent distance decreases
but physical distance does not decrease proportionally.

Layout: 1 row × 3 columns (one per axis), matching the original correlation plot.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

sys.path.insert(0, "/home/s185927/thesis")
from plot_config import PLOT_PARAMS, configure_axis


def main():
    base_dir = Path(
        "/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching"
    )
    corr_dir = base_dir / "follow_up_experiments" / "latent_physical_distance_alignment"
    cem_dir = base_dir / "left_cam_cem_optimized_teleport"

    # Load IK correlation data
    ik_data = np.load(corr_dir / "latent_physical_correlation_data.npz")

    axes = ["x", "y", "z"]
    titles = {
        "x": "(a) Reach along x",
        "y": "(b) Reach along y",
        "z": "(c) Reach along z",
    }

    # Compute shared axis limits from IK data
    all_latent = np.concatenate([ik_data[f"{a}_latent"] for a in axes])
    all_euclidean = np.concatenate([ik_data[f"{a}_euclidean"] for a in axes])
    x_min, x_max = all_latent.min(), all_latent.max()
    y_min, y_max = all_euclidean.min(), all_euclidean.max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.10

    fig, ax_list = plt.subplots(1, 3, figsize=(15, 5))

    for idx, axis in enumerate(axes):
        ax = ax_list[idx]

        # --- Single IK trajectory (episode 0) ---
        ik_latent = ik_data[f"{axis}_latent"]
        ik_euclidean = ik_data[f"{axis}_euclidean"]
        ik_corr = float(ik_data[f"{axis}_correlation"])
        ik_episode_idx = ik_data[f"{axis}_episode_idx"].astype(int)

        # Extract episode 0 only
        ik_ep0_mask = ik_episode_idx == 0
        ik_lat_ep0 = ik_latent[ik_ep0_mask]
        ik_euc_ep0 = ik_euclidean[ik_ep0_mask]

        # Plot as connected line with markers
        # Sort by latent distance so the line doesn't zigzag
        sort_idx = np.argsort(ik_lat_ep0)
        ax.plot(
            ik_lat_ep0[sort_idx],
            ik_euc_ep0[sort_idx],
            color="tab:blue",
            linewidth=2.5,
            alpha=0.9,
            zorder=2,
        )
        ax.scatter(
            ik_lat_ep0,
            ik_euc_ep0,
            s=20,
            color="tab:blue",
            alpha=0.6,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

        # IK Pearson r (computed on all episodes)
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {ik_corr:.2f}",
            transform=ax.transAxes,
            fontsize=PLOT_PARAMS["legend_size"] + 2,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # --- CEM overlay (mean trajectory ± std) ---
        cem_npz = np.load(
            cem_dir / f"reach_along_{axis}" / f"run_{axis}_distance_summary.npz",
            allow_pickle=True,
        )
        cem_phys = cem_npz["phase3_distances_per_episode"]  # (10, 6) object array
        cem_latent = cem_npz["phase3_repr_l1_distances_per_episode"]  # (10, 6)

        n_cem_episodes = cem_phys.shape[0]
        n_steps = cem_phys.shape[1]

        # Compute mean and std across episodes per step
        phys_all = np.array(
            [[float(cem_phys[i][j]) for j in range(n_steps)] for i in range(n_cem_episodes)]
        )  # (10, 6)
        latent_all = np.array(
            [[float(cem_latent[i][j]) for j in range(n_steps)] for i in range(n_cem_episodes)]
        )  # (10, 6)

        phys_mean = phys_all.mean(axis=0)
        phys_std = phys_all.std(axis=0)
        latent_mean = latent_all.mean(axis=0)
        latent_std = latent_all.std(axis=0)

        # Draw mean CEM trajectory with step numbers
        ax.plot(
            latent_mean,
            phys_mean,
            color="red",
            linewidth=2.5,
            alpha=0.9,
            zorder=4,
        )
        # Error bars at each step
        ax.errorbar(
            latent_mean,
            phys_mean,
            xerr=latent_std,
            yerr=phys_std,
            fmt="none",
            ecolor="red",
            elinewidth=1.0,
            capsize=3,
            alpha=0.5,
            zorder=3,
        )
        # Step markers with step numbers
        for step_i in range(n_steps):
            ax.scatter(
                latent_mean[step_i],
                phys_mean[step_i],
                s=50,
                color="red",
                edgecolors="darkred",
                linewidths=1.2,
                zorder=5,
                marker="o",
            )
            ax.annotate(
                f"k={step_i}",
                (latent_mean[step_i], phys_mean[step_i]),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=8,
                color="darkred",
                fontweight="bold",
                zorder=6,
            )

        # Axis limits (expand to include CEM data)
        all_cem_latent = latent_all.ravel()
        all_cem_phys = phys_all.ravel()
        cell_x_min = min(x_min, all_cem_latent.min())
        cell_x_max = max(x_max, all_cem_latent.max())
        cell_y_min = min(y_min, all_cem_phys.min())
        cell_y_max = max(y_max, all_cem_phys.max())
        ax.set_xlim(cell_x_min - x_pad, cell_x_max + x_pad)
        ax.set_ylim(cell_y_min - y_pad, cell_y_max + y_pad)

        # Labels
        configure_axis(
            ax,
            xlabel=r"$\frac{1}{TD}\|z_k - z_g\|_1$",
            ylabel=r"$\|p_k - p_g\|_2$ (m)" if idx == 0 else "",
            title=titles[axis],
        )

    # Shared legend at bottom
    handles = [
        plt.Line2D(
            [0], [0],
            marker="o", color="tab:blue", markerfacecolor="tab:blue",
            markersize=8, linewidth=2.5,
            label="IK trajectory (optimal)",
        ),
        plt.Line2D(
            [0], [0],
            marker="o", color="red", markerfacecolor="red",
            markeredgecolor="darkred", markersize=6, linewidth=2.5,
            label="CEM planning (mean ± std, N=10)",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=PLOT_PARAMS["legend_size"],
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    output_path = corr_dir / "latent_physical_correlation_cem_overlay.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
