#!/usr/bin/env python3
"""
Analyze results from the latent alignment evaluation experiment.

This script:
1. Consolidates per-episode metrics from all conditions into a single CSV
2. Generates comparison plots (final distance distribution, error vs step)
3. Computes success rates and statistical comparisons

Usage:
    python analyze_results.py --results_dir /path/to/results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add path for plot config
sys.path.insert(0, "/home/s185927/thesis")
from plot_config import PLOT_PARAMS, apply_plot_params


RESULTS_DIR = Path("/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/inference_time_latent_alignment/results")
OUTPUT_DIR = Path("/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/inference_time_latent_alignment/plots")


def load_axis_data(axis_dir: Path, axis: str) -> Dict:
    """Load data from axis summary JSON file."""
    summary_file = axis_dir / f"run_{axis}_distance_summary.json"
    if not summary_file.exists():
        return None

    with open(summary_file, "r") as f:
        data = json.load(f)

    return data


def consolidate_results(results_dir: Path) -> pd.DataFrame:
    """
    Consolidate results from all conditions into a DataFrame.

    Returns DataFrame with columns:
        - axis: x, y, or z
        - condition: none, mean, or coral
        - episode_id: episode number
        - step_0..step_5: Cartesian error at each step
        - final_cart_err: final Cartesian error
        - success_5cm: 1 if final_err < 0.05
        - success_10cm: 1 if final_err < 0.10
        - latent_dist_0..latent_dist_5: latent L1 distance at each step
    """
    records = []

    for condition in ["none", "mean", "coral"]:
        condition_dir = results_dir / condition
        if not condition_dir.exists():
            print(f"Warning: {condition_dir} not found, skipping")
            continue

        for axis in ["x", "y", "z"]:
            axis_dir = condition_dir / f"reach_along_{axis}"
            if not axis_dir.exists():
                print(f"Warning: {axis_dir} not found, skipping")
                continue

            data = load_axis_data(axis_dir, axis)
            if data is None:
                print(f"Warning: No summary file in {axis_dir}, skipping")
                continue

            # Extract per-episode data
            n_episodes = len(data.get("phase3_final_distance", []))
            episode_ids = data.get("episode_ids", list(range(n_episodes)))

            for ep_idx in range(n_episodes):
                record = {
                    "axis": axis,
                    "condition": condition,
                    "episode_id": episode_ids[ep_idx] if ep_idx < len(episode_ids) else ep_idx,
                }

                # Per-step Cartesian errors (phase3_vjepa_distances)
                if "phase3_vjepa_distances" in data and ep_idx < len(data["phase3_vjepa_distances"]):
                    cart_errors = data["phase3_vjepa_distances"][ep_idx]
                    for i, err in enumerate(cart_errors):
                        record[f"step_{i}"] = err

                # Final Cartesian error
                if "phase3_final_distance" in data and ep_idx < len(data["phase3_final_distance"]):
                    final_err = data["phase3_final_distance"][ep_idx]
                    record["final_cart_err"] = final_err
                    record["success_5cm"] = 1 if final_err < 0.05 else 0
                    record["success_10cm"] = 1 if final_err < 0.10 else 0

                # Per-step latent distances
                if "phase3_repr_l1_distances" in data and ep_idx < len(data["phase3_repr_l1_distances"]):
                    latent_dists = data["phase3_repr_l1_distances"][ep_idx]
                    for i, dist in enumerate(latent_dists):
                        record[f"latent_dist_{i}"] = dist

                records.append(record)

    df = pd.DataFrame(records)
    return df


def plot_final_distance_distribution(df: pd.DataFrame, output_dir: Path):
    """
    Create box/violin plot of final distance distribution for each axis and condition.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    conditions = ["none", "mean", "coral"]
    condition_labels = ["Baseline", "Mean-only", "CORAL"]
    colors = ["tab:blue", "tab:green", "tab:red"]

    for ax_idx, axis in enumerate(["x", "y", "z"]):
        ax = axes[ax_idx]

        data_to_plot = []
        labels = []

        for cond, label in zip(conditions, condition_labels):
            subset = df[(df["axis"] == axis) & (df["condition"] == cond)]
            if len(subset) > 0 and "final_cart_err" in subset.columns:
                vals = subset["final_cart_err"].dropna().values * 100  # Convert to cm
                if len(vals) > 0:
                    data_to_plot.append(vals)
                    labels.append(label)

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_ylabel("Final Cartesian Error (cm)" if ax_idx == 0 else "", fontsize=PLOT_PARAMS["label_size"])
        ax.set_title(f"{axis.upper()}-axis", fontsize=PLOT_PARAMS["subtitle_size"])
        ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='5cm threshold')
        ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5, label='10cm threshold')
        ax.set_ylim(0, 25)
        apply_plot_params(ax)

        if ax_idx == 2:
            ax.legend(loc='upper right', fontsize=PLOT_PARAMS["legend_size"])

    plt.tight_layout()
    plt.savefig(output_dir / "final_distance_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'final_distance_distribution.png'}")


def plot_error_vs_step(df: pd.DataFrame, output_dir: Path):
    """
    Plot Cartesian error vs planning step for each axis, comparing conditions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    conditions = ["none", "mean", "coral"]
    condition_labels = ["Baseline", "Mean-only", "CORAL"]
    colors = ["tab:blue", "tab:green", "tab:red"]
    markers = ["o", "s", "^"]

    step_cols = [f"step_{i}" for i in range(6)]

    for ax_idx, axis in enumerate(["x", "y", "z"]):
        ax = axes[ax_idx]

        for cond, label, color, marker in zip(conditions, condition_labels, colors, markers):
            subset = df[(df["axis"] == axis) & (df["condition"] == cond)]
            if len(subset) == 0:
                continue

            existing_steps = [col for col in step_cols if col in subset.columns]
            if not existing_steps:
                continue

            means = []
            stds = []
            steps = []

            for i, col in enumerate(existing_steps):
                vals = subset[col].dropna().values * 100  # Convert to cm
                if len(vals) > 0:
                    means.append(vals.mean())
                    stds.append(vals.std())
                    steps.append(i)

            if means:
                ax.errorbar(steps, means, yerr=stds, marker=marker, color=color,
                           label=label, capsize=3, linewidth=2, markersize=6)

        ax.set_xlabel("Planning Step", fontsize=PLOT_PARAMS["label_size"])
        ax.set_ylabel("Cartesian Error (cm)" if ax_idx == 0 else "", fontsize=PLOT_PARAMS["label_size"])
        ax.set_title(f"{axis.upper()}-axis", fontsize=PLOT_PARAMS["subtitle_size"])
        ax.set_xticks(range(6))
        ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylim(0, 25)
        apply_plot_params(ax)
        ax.legend(loc='upper right', fontsize=PLOT_PARAMS["legend_size"])

    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_step.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'error_vs_step.png'}")


def plot_latent_distance_vs_step(df: pd.DataFrame, output_dir: Path):
    """
    Plot latent L1 distance vs planning step for each axis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    conditions = ["none", "mean", "coral"]
    condition_labels = ["Baseline", "Mean-only", "CORAL"]
    colors = ["tab:blue", "tab:green", "tab:red"]
    markers = ["o", "s", "^"]

    latent_cols = [f"latent_dist_{i}" for i in range(7)]

    for ax_idx, axis in enumerate(["x", "y", "z"]):
        ax = axes[ax_idx]

        for cond, label, color, marker in zip(conditions, condition_labels, colors, markers):
            subset = df[(df["axis"] == axis) & (df["condition"] == cond)]
            if len(subset) == 0:
                continue

            existing_cols = [col for col in latent_cols if col in subset.columns]
            if not existing_cols:
                continue

            means = []
            stds = []
            steps = []

            for i, col in enumerate(existing_cols):
                vals = subset[col].dropna().values
                if len(vals) > 0:
                    means.append(vals.mean())
                    stds.append(vals.std())
                    steps.append(i)

            if means:
                ax.errorbar(steps, means, yerr=stds, marker=marker, color=color,
                           label=label, capsize=3, linewidth=2, markersize=6)

        ax.set_xlabel("Planning Step", fontsize=PLOT_PARAMS["label_size"])
        ax.set_ylabel("Latent L1 Distance" if ax_idx == 0 else "", fontsize=PLOT_PARAMS["label_size"])
        ax.set_title(f"{axis.upper()}-axis", fontsize=PLOT_PARAMS["subtitle_size"])
        apply_plot_params(ax)
        ax.legend(loc='upper right', fontsize=PLOT_PARAMS["legend_size"])

    plt.tight_layout()
    plt.savefig(output_dir / "latent_distance_vs_step.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'latent_distance_vs_step.png'}")


def compute_summary_statistics(df: pd.DataFrame) -> Dict:
    """Compute summary statistics for all conditions."""
    summary = {}

    for condition in ["none", "mean", "coral"]:
        cond_data = df[df["condition"] == condition]
        if len(cond_data) == 0:
            continue

        summary[condition] = {}

        for axis in ["x", "y", "z"]:
            axis_data = cond_data[cond_data["axis"] == axis]
            if len(axis_data) == 0:
                continue

            final_errors = axis_data["final_cart_err"].dropna() if "final_cart_err" in axis_data.columns else pd.Series()
            success_5cm = axis_data["success_5cm"].dropna() if "success_5cm" in axis_data.columns else pd.Series()
            success_10cm = axis_data["success_10cm"].dropna() if "success_10cm" in axis_data.columns else pd.Series()

            summary[condition][axis] = {
                "n_episodes": len(axis_data),
                "final_err_mean": float(final_errors.mean()) if len(final_errors) > 0 else None,
                "final_err_std": float(final_errors.std()) if len(final_errors) > 0 else None,
                "final_err_median": float(final_errors.median()) if len(final_errors) > 0 else None,
                "success_rate_5cm": float(success_5cm.mean()) if len(success_5cm) > 0 else None,
                "success_rate_10cm": float(success_10cm.mean()) if len(success_10cm) > 0 else None,
            }

        # Overall statistics
        overall_final = cond_data["final_cart_err"].dropna() if "final_cart_err" in cond_data.columns else pd.Series()
        overall_5cm = cond_data["success_5cm"].dropna() if "success_5cm" in cond_data.columns else pd.Series()
        overall_10cm = cond_data["success_10cm"].dropna() if "success_10cm" in cond_data.columns else pd.Series()

        summary[condition]["overall"] = {
            "n_episodes": len(cond_data),
            "final_err_mean": float(overall_final.mean()) if len(overall_final) > 0 else None,
            "final_err_std": float(overall_final.std()) if len(overall_final) > 0 else None,
            "success_rate_5cm": float(overall_5cm.mean()) if len(overall_5cm) > 0 else None,
            "success_rate_10cm": float(overall_10cm.mean()) if len(overall_10cm) > 0 else None,
        }

    return summary


def print_summary(summary: Dict):
    """Print a formatted summary of results."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for condition in ["none", "mean", "coral"]:
        if condition not in summary:
            continue

        cond_label = {"none": "Baseline", "mean": "Mean-only", "coral": "CORAL"}[condition]
        print(f"\n{cond_label}:")
        print("-" * 50)

        for axis in ["x", "y", "z", "overall"]:
            if axis not in summary[condition]:
                continue

            stats = summary[condition][axis]
            axis_label = axis.upper() if axis != "overall" else "Overall"

            if stats["final_err_mean"] is not None:
                print(f"  {axis_label}: "
                      f"err={stats['final_err_mean']*100:.1f}±{stats['final_err_std']*100:.1f}cm, "
                      f"5cm={stats['success_rate_5cm']*100:.0f}%, "
                      f"10cm={stats['success_rate_10cm']*100:.0f}% "
                      f"(n={stats['n_episodes']})")


def main():
    parser = argparse.ArgumentParser(description="Analyze latent alignment experiment results")
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR),
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory to save plots and analysis")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Consolidate results
    df = consolidate_results(results_dir)

    if len(df) == 0:
        print("ERROR: No data found!")
        return

    print(f"\nLoaded {len(df)} episode records")
    print(f"Conditions: {df['condition'].unique().tolist()}")
    print(f"Axes: {df['axis'].unique().tolist()}")

    # Save consolidated CSV
    csv_path = output_dir / "consolidated_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved consolidated CSV: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_final_distance_distribution(df, output_dir)
    plot_error_vs_step(df, output_dir)
    plot_latent_distance_vs_step(df, output_dir)

    # Compute and print summary
    summary = compute_summary_statistics(df)
    print_summary(summary)

    # Save summary JSON
    summary_path = output_dir / "summary_statistics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
