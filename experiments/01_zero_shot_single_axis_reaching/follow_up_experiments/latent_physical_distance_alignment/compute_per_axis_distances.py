#!/usr/bin/env python3
"""
Compute per-axis physical distances from zero-shot trajectory data.

Loads raw trajectory.h5 files and computes per-axis (x, y, z) physical
distances for each frame, then pairs them with existing latent distances
from the pre-computed correlation data. No GPU required.

Output: per_axis_correlation_data.npz
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import pearsonr


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-axis physical distances from zero-shot trajectory data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/s185927/droid_sim/zero_shot_correlation",
        help="Base directory containing {x,y,z}_axis subdirectories",
    )
    parser.add_argument(
        "--existing_npz",
        type=str,
        default="/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/"
        "follow_up_experiments/latent_physical_distance_alignment/"
        "latent_physical_correlation_data.npz",
        help="Path to existing npz with latent distances",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/"
        "follow_up_experiments/latent_physical_distance_alignment",
        help="Output directory for the new npz",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    existing_npz_path = Path(args.existing_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing latent distance data
    existing = np.load(existing_npz_path)

    goal_axes = ["x", "y", "z"]
    component_names = {0: "x", 1: "y", 2: "z"}

    save_dict = {}

    for goal_axis in goal_axes:
        csv_path = data_dir / f"{goal_axis}_axis" / "test_trajectories.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping {goal_axis}-axis")
            continue

        with open(csv_path) as f:
            episodes = [line.strip() for line in f if line.strip()]

        # Determine how many episodes were used in the existing data
        existing_idx_key = f"{goal_axis}_episode_idx"
        if existing_idx_key in existing:
            n_existing_episodes = len(np.unique(existing[existing_idx_key].astype(int)))
        else:
            n_existing_episodes = len(episodes)

        # Only process the same episodes used in the existing analysis
        episodes = episodes[:n_existing_episodes]
        print(f"\n{goal_axis.upper()}-axis: processing {len(episodes)} episodes "
              f"(matching existing data)...")

        # Collect per-axis distances for all frames
        per_axis_dists = {0: [], 1: [], 2: []}
        euclidean_dists = []
        episode_indices = []

        for ep_idx, episode_path in enumerate(episodes):
            h5_path = Path(episode_path) / "trajectory.h5"
            if not h5_path.exists():
                print(f"  WARNING: {h5_path} not found, skipping")
                continue

            with h5py.File(h5_path, "r") as f:
                ee_pos = f["observation/robot_state/cartesian_position"][:, :3]

            goal_pos = ee_pos[-1]

            for i in range(len(ee_pos) - 1):
                diff = goal_pos - ee_pos[i]
                for comp in range(3):
                    per_axis_dists[comp].append(abs(diff[comp]))
                euclidean_dists.append(np.linalg.norm(diff))
                episode_indices.append(ep_idx)

        euclidean_dists = np.array(euclidean_dists)
        episode_indices = np.array(episode_indices)

        # Verify Euclidean distances match existing data
        existing_euc_key = f"{goal_axis}_euclidean"
        if existing_euc_key in existing:
            existing_euc = existing[existing_euc_key]
            if len(euclidean_dists) == len(existing_euc):
                max_diff = np.max(np.abs(euclidean_dists - existing_euc))
                print(f"  Euclidean verification: {len(euclidean_dists)} points, max diff = {max_diff:.2e}")
                if max_diff > 1e-4:
                    print("  WARNING: Euclidean distances don't match existing data!")
                    print("  This may indicate video subsampling differences.")
            else:
                print(
                    f"  WARNING: Length mismatch: computed {len(euclidean_dists)} "
                    f"vs existing {len(existing_euc)}"
                )

        # Get existing latent distances
        latent_key = f"{goal_axis}_latent"
        if latent_key not in existing:
            print(f"  WARNING: {latent_key} not found in existing npz, skipping")
            continue
        latent = existing[latent_key]

        # Save latent and euclidean distances
        save_dict[f"{goal_axis}_latent"] = latent
        save_dict[f"{goal_axis}_euclidean"] = euclidean_dists
        save_dict[f"{goal_axis}_episode_idx"] = episode_indices

        # Save per-axis distances and compute correlations
        for comp in range(3):
            comp_name = component_names[comp]
            comp_dists = np.array(per_axis_dists[comp])
            save_dict[f"{goal_axis}_d{comp_name}"] = comp_dists

            # Compute correlation between latent distance and per-axis physical distance
            if len(latent) == len(comp_dists):
                r, p_val = pearsonr(latent, comp_dists)
                save_dict[f"{goal_axis}_d{comp_name}_correlation"] = r
                print(f"  goal={goal_axis}, dist={comp_name}: r={r:.4f} (p={p_val:.2e})")
            else:
                print(
                    f"  WARNING: Cannot compute correlation for goal={goal_axis}, "
                    f"dist={comp_name}: length mismatch ({len(latent)} vs {len(comp_dists)})"
                )

    # Save
    output_path = output_dir / "per_axis_correlation_data.npz"
    np.savez(output_path, **save_dict)
    print(f"\nSaved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX (Pearson r: latent distance vs per-axis physical distance)")
    print("=" * 70)
    print(f"{'':>12s} {'Δx':>10s} {'Δy':>10s} {'Δz':>10s}")
    print("-" * 44)
    for goal_axis in goal_axes:
        row = f"{goal_axis}-goal:  "
        for comp_name in ["x", "y", "z"]:
            key = f"{goal_axis}_d{comp_name}_correlation"
            if key in save_dict:
                row += f"  {save_dict[key]:>8.4f}"
            else:
                row += f"  {'N/A':>8s}"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    main()
