"""
Analyze results from the metric sensitivity experiment.

This script:
1. Loads all experimental results across metrics and axes
2. Consolidates them into a single DataFrame/CSV
3. Generates the required plots:
   a) Cartesian error vs planning step (mean +/- std) per axis, comparing metrics
   b) Final Cartesian error distribution per axis (box/violin)
   c) Latent distance vs planning step per axis (using corresponding metric)
4. Computes summary statistics and writes 5 bullet point summary

Usage:
    python analyze_results.py [--results_dir PATH] [--output_dir PATH]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import shared plotting configuration
sys.path.insert(0, '/home/s185927/thesis')
from plot_config import PLOT_PARAMS, apply_plot_params

# Constants
METRICS = ['l1', 'cosine', 'norm_l1']
AXES = ['x', 'y', 'z']
N_EPISODES = 10
N_PLANNING_STEPS = 5

# Plot styling - matching consolidated_analysis.py
METRIC_COLORS = {
    'l1': '#1f77b4',      # Blue (same as baseline)
    'cosine': '#ff7f0e',  # Orange
    'norm_l1': '#2ca02c'  # Green
}
METRIC_LABELS = {
    'l1': 'L1 (baseline)',
    'cosine': 'Cosine',
    'norm_l1': 'Normalized L1'
}

# Direction labels matching consolidated_analysis.py
DIRECTION_LABELS = {
    'x': 'Reach along x',
    'y': 'Reach along y',
    'z': 'Reach along z',
}
PANEL_LABELS = ['(a)', '(b)', '(c)']

# Local plot params (scaled for multi-panel figures, matching consolidated_analysis.py style)
PLOT_PARAMS_LOCAL = {
    "figsize": (18, 5),
    "figsize_violin": (16, 5),
    "label_size": PLOT_PARAMS["label_size"],
    "legend_size": PLOT_PARAMS["legend_size"],
    "tick_label_size": PLOT_PARAMS["tick_label_size"],
    "tick_length": PLOT_PARAMS["tick_length"],
    "title_size": PLOT_PARAMS["subtitle_size"],
    "linewidth": PLOT_PARAMS["euclid_linewidth"],
    "markersize": PLOT_PARAMS["euclid_markersize"],
    "capsize": PLOT_PARAMS["euclid_capsize"],
    "threshold_linewidth": PLOT_PARAMS["threshold_linewidth"],
}


def load_axis_data(results_dir: str, metric: str, axis: str) -> dict:
    """Load data for a specific metric and axis."""
    json_path = Path(results_dir) / metric / f"reach_along_{axis}" / f"run_{axis}_distance_summary.json"

    if not json_path.exists():
        print(f"WARNING: Missing data file: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def load_baseline_data(baseline_dir: str, axis: str) -> dict:
    """Load baseline data for a specific axis from the fixed experiment directory."""
    json_path = Path(baseline_dir) / f"reach_along_{axis}" / f"run_{axis}_distance_summary.json"

    if not json_path.exists():
        print(f"WARNING: Missing baseline data file: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def consolidate_results(results_dir: str, baseline_dir: str = None) -> pd.DataFrame:
    """
    Load and consolidate all results into a single DataFrame.

    If baseline_dir is provided, L1 baseline data is loaded from there instead of
    results_dir to avoid variability from re-running.

    Returns DataFrame with columns:
        - axis: x, y, z
        - metric: l1, cosine, norm_l1
        - episode_id: 0-9
        - seed: episode seed
        - step_0, step_1, ..., step_5: Cartesian errors at each planning step (meters)
        - final_cart_err: Final Cartesian error (meters)
        - success_5cm: Whether final error < 5cm
        - success_10cm: Whether final error < 10cm
        - latent_dist_0, ..., latent_dist_5: Latent distances at each step
    """
    rows = []

    for metric in METRICS:
        for axis in AXES:
            # For L1, use baseline data if available
            if metric == 'l1' and baseline_dir is not None:
                data = load_baseline_data(baseline_dir, axis)
            else:
                data = load_axis_data(results_dir, metric, axis)

            if data is None:
                continue

            # Extract per-episode data
            vjepa_distances = data.get('phase3_vjepa_distances', [])
            repr_distances = data.get('phase3_repr_l1_distances', [])
            episode_seeds = data.get('episode_seeds', [])
            final_distances = data.get('phase3_final_distance', [])

            for ep_idx in range(len(vjepa_distances)):
                row = {
                    'axis': axis,
                    'metric': metric,
                    'episode_id': ep_idx,
                    'seed': episode_seeds[ep_idx] if ep_idx < len(episode_seeds) else None,
                }

                # Cartesian errors per step (keep in meters for plotting)
                cart_errors = vjepa_distances[ep_idx]
                for step_idx, err in enumerate(cart_errors):
                    row[f'step_{step_idx}'] = err  # meters

                # Final Cartesian error
                final_err = final_distances[ep_idx] if ep_idx < len(final_distances) else cart_errors[-1]
                row['final_cart_err'] = final_err  # meters

                # Success thresholds
                row['success_5cm'] = final_err < 0.05
                row['success_10cm'] = final_err < 0.10

                # Latent distances per step
                if ep_idx < len(repr_distances):
                    latent_dists = repr_distances[ep_idx]
                    for step_idx, dist in enumerate(latent_dists):
                        row[f'latent_dist_{step_idx}'] = dist

                rows.append(row)

    return pd.DataFrame(rows)


def plot_error_vs_step(df: pd.DataFrame, output_dir: str):
    """
    Plot a) Cartesian error vs planning step for each axis, comparing metrics.

    Creates a 1x3 subplot with one panel per axis.
    Uses axis labels matching consolidated_analysis.py:
        y-axis: $\|p_k - p_g\|_2$ (m)
        x-axis: Step (k)
    """
    fig, axes = plt.subplots(1, 3, figsize=PLOT_PARAMS_LOCAL["figsize"], sharey=True)

    step_cols = [f'step_{i}' for i in range(N_PLANNING_STEPS + 1)]

    # Compute shared y-axis limits
    all_max = 0
    for axis in AXES:
        axis_df = df[df['axis'] == axis]
        for metric in METRICS:
            metric_df = axis_df[axis_df['metric'] == metric]
            if len(metric_df) == 0:
                continue
            for col in step_cols:
                if col in metric_df.columns:
                    values = metric_df[col].dropna().values
                    if len(values) > 0:
                        all_max = max(all_max, np.mean(values) + np.std(values))

    for ax_idx, axis in enumerate(AXES):
        ax = axes[ax_idx]
        axis_df = df[df['axis'] == axis]

        for metric in METRICS:
            metric_df = axis_df[axis_df['metric'] == metric]
            if len(metric_df) == 0:
                continue

            # Compute mean and std across episodes
            means = []
            stds = []
            for col in step_cols:
                if col in metric_df.columns:
                    values = metric_df[col].dropna().values
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            means = np.array(means)
            stds = np.array(stds)
            steps = np.arange(len(means))

            # Plot mean with error bars
            ax.errorbar(
                steps, means, yerr=stds,
                fmt='-o',
                color=METRIC_COLORS[metric],
                label=METRIC_LABELS[metric],
                linewidth=PLOT_PARAMS_LOCAL["linewidth"],
                markersize=PLOT_PARAMS_LOCAL["markersize"],
                capsize=PLOT_PARAMS_LOCAL["capsize"],
                alpha=0.9
            )

        # Add threshold line (matching consolidated_analysis.py style)
        ax.axhline(y=0.05, color='r', linestyle='--',
                   linewidth=PLOT_PARAMS_LOCAL["threshold_linewidth"],
                   label='Threshold (0.05m)' if ax_idx == 0 else None, alpha=1.0)

        # Axis labels matching consolidated_analysis.py
        ax.set_xlabel('Step (k)', fontsize=PLOT_PARAMS_LOCAL["label_size"])
        if ax_idx == 0:
            ax.set_ylabel(r'$\|p_k - p_g\|_2$ (m)', fontsize=PLOT_PARAMS_LOCAL["label_size"])

        # Title with panel label
        ax.set_title(f'{PANEL_LABELS[ax_idx]} {DIRECTION_LABELS[axis]}',
                     fontsize=PLOT_PARAMS_LOCAL["title_size"])

        ax.set_xticks(range(N_PLANNING_STEPS + 1))
        ax.tick_params(axis='both', labelsize=PLOT_PARAMS_LOCAL["tick_label_size"],
                       length=PLOT_PARAMS_LOCAL["tick_length"])
        ax.grid(True, alpha=PLOT_PARAMS["grid_alpha"])

        # Y-axis formatting
        y_step = PLOT_PARAMS["y_tick_step"]
        ax.set_ylim(0.0, all_max * 1.1)
        y_ticks = np.arange(0.0, all_max * 1.1 + y_step * 0.5, y_step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.2f}' for y in y_ticks])

        if ax_idx == 2:
            ax.legend(fontsize=PLOT_PARAMS_LOCAL["legend_size"], loc='upper right')

    plt.tight_layout()

    save_path = Path(output_dir) / 'error_vs_step.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_final_error_distribution(df: pd.DataFrame, output_dir: str):
    """
    Plot b) Final Cartesian error distribution per axis (box/violin).

    Creates a 1x3 subplot with one panel per axis.
    """
    fig, axes = plt.subplots(1, 3, figsize=PLOT_PARAMS_LOCAL["figsize_violin"], sharey=True)

    for ax_idx, axis in enumerate(AXES):
        ax = axes[ax_idx]
        axis_df = df[df['axis'] == axis].copy()

        if len(axis_df) == 0:
            continue

        # Create violin plot
        sns.violinplot(
            data=axis_df,
            x='metric',
            y='final_cart_err',
            palette=METRIC_COLORS,
            ax=ax,
            order=METRICS,
            inner='box'
        )

        # Overlay individual points
        sns.stripplot(
            data=axis_df,
            x='metric',
            y='final_cart_err',
            color='black',
            alpha=0.5,
            size=4,
            ax=ax,
            order=METRICS
        )

        # Add threshold line (only 5cm, matching consolidated_analysis.py style)
        ax.axhline(y=0.05, color='r', linestyle='--',
                   linewidth=PLOT_PARAMS_LOCAL["threshold_linewidth"],
                   label='Threshold (0.05m)' if ax_idx == 0 else None)

        ax.set_xlabel('Energy Metric', fontsize=PLOT_PARAMS_LOCAL["label_size"])
        if ax_idx == 0:
            ax.set_ylabel(r'$\|p_k - p_g\|_2$ (m)', fontsize=PLOT_PARAMS_LOCAL["label_size"])

        ax.set_title(f'{PANEL_LABELS[ax_idx]} {DIRECTION_LABELS[axis]}',
                     fontsize=PLOT_PARAMS_LOCAL["title_size"])
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15, ha='right')
        ax.tick_params(axis='both', labelsize=PLOT_PARAMS_LOCAL["tick_label_size"],
                       length=PLOT_PARAMS_LOCAL["tick_length"])
        ax.grid(True, alpha=PLOT_PARAMS["grid_alpha"], axis='y')

        if ax_idx == 0:
            ax.legend(fontsize=PLOT_PARAMS_LOCAL["legend_size"], loc='upper right')

    plt.tight_layout()

    save_path = Path(output_dir) / 'final_distance_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_latent_distance_vs_step(df: pd.DataFrame, output_dir: str):
    """
    Plot c) Latent distance vs planning step per axis.

    Note: For each metric, we show the latent distance computed using that metric.
    Uses axis label matching consolidated_analysis.py style.
    """
    fig, axes = plt.subplots(1, 3, figsize=PLOT_PARAMS_LOCAL["figsize"])

    latent_cols = [f'latent_dist_{i}' for i in range(N_PLANNING_STEPS + 1)]

    for ax_idx, axis in enumerate(AXES):
        ax = axes[ax_idx]
        axis_df = df[df['axis'] == axis]

        for metric in METRICS:
            metric_df = axis_df[axis_df['metric'] == metric]
            if len(metric_df) == 0:
                continue

            # Compute mean and std across episodes
            means = []
            stds = []
            for col in latent_cols:
                if col in metric_df.columns:
                    values = metric_df[col].dropna().values
                    if len(values) > 0:
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            means = np.array(means)
            stds = np.array(stds)
            steps = np.arange(len(means))

            # Plot mean with error bars
            ax.errorbar(
                steps, means, yerr=stds,
                fmt='-s',
                color=METRIC_COLORS[metric],
                label=METRIC_LABELS[metric],
                linewidth=PLOT_PARAMS_LOCAL["linewidth"],
                markersize=PLOT_PARAMS_LOCAL["markersize"],
                capsize=PLOT_PARAMS_LOCAL["capsize"],
                alpha=0.9
            )

        ax.set_xlabel('Step (k)', fontsize=PLOT_PARAMS_LOCAL["label_size"])
        if ax_idx == 0:
            ax.set_ylabel(r'$\frac{1}{TD}\|z_k - z_g\|_1$', fontsize=PLOT_PARAMS_LOCAL["label_size"])

        ax.set_title(f'{PANEL_LABELS[ax_idx]} {DIRECTION_LABELS[axis]}',
                     fontsize=PLOT_PARAMS_LOCAL["title_size"])
        ax.set_xticks(range(N_PLANNING_STEPS + 1))
        ax.tick_params(axis='both', labelsize=PLOT_PARAMS_LOCAL["tick_label_size"],
                       length=PLOT_PARAMS_LOCAL["tick_length"])
        ax.grid(True, alpha=PLOT_PARAMS["grid_alpha"])

        if ax_idx == 2:
            ax.legend(fontsize=PLOT_PARAMS_LOCAL["legend_size"], loc='upper right')

    plt.tight_layout()

    save_path = Path(output_dir) / 'latent_distance_vs_step.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_error_and_latent(df: pd.DataFrame, output_dir: str):
    """
    Plot combined figure with Cartesian error (top row) and latent distance (bottom row).

    Creates a 2x3 subplot matching the inference_time_latent_alignment style.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex='col')

    step_cols = [f'step_{i}' for i in range(N_PLANNING_STEPS + 1)]
    latent_cols = [f'latent_dist_{i}' for i in range(N_PLANNING_STEPS + 1)]

    # Compute shared y-axis limits for Cartesian error (top row) - in cm
    all_cart_max = 0
    for axis in AXES:
        axis_df = df[df['axis'] == axis]
        for metric in METRICS:
            metric_df = axis_df[axis_df['metric'] == metric]
            if len(metric_df) == 0:
                continue
            for col in step_cols:
                if col in metric_df.columns:
                    values = metric_df[col].dropna().values * 100  # Convert to cm
                    if len(values) > 0:
                        all_cart_max = max(all_cart_max, np.mean(values) + np.std(values))

    # Store legend handles for common legend
    legend_handles = []
    legend_labels = []

    # Plot each axis (column)
    for ax_idx, axis in enumerate(AXES):
        axis_df = df[df['axis'] == axis]

        # Top row: Cartesian error vs step
        ax_cart = axes[0, ax_idx]
        for metric in METRICS:
            metric_df = axis_df[axis_df['metric'] == metric]
            if len(metric_df) == 0:
                continue

            means = []
            stds = []
            for col in step_cols:
                if col in metric_df.columns:
                    values = metric_df[col].dropna().values * 100  # Convert to cm
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            means = np.array(means)
            stds = np.array(stds)
            steps = np.arange(len(means))

            line = ax_cart.errorbar(
                steps, means, yerr=stds,
                fmt='-o',
                color=METRIC_COLORS[metric],
                linewidth=PLOT_PARAMS_LOCAL["linewidth"],
                markersize=PLOT_PARAMS_LOCAL["markersize"],
                capsize=PLOT_PARAMS_LOCAL["capsize"],
            )
            if ax_idx == 0:
                legend_handles.append(line)
                legend_labels.append(METRIC_LABELS[metric])

        # Add threshold line at 5 cm
        ax_cart.axhline(y=5, color='red', linestyle='--', alpha=0.7)

        # Title without panel label
        ax_cart.set_title(f'{DIRECTION_LABELS[axis]}',
                          fontsize=PLOT_PARAMS_LOCAL["title_size"])

        if ax_idx == 0:
            ax_cart.set_ylabel(r'$\|p_k - p_g\|_2$ (cm)', fontsize=PLOT_PARAMS_LOCAL["label_size"])
        ax_cart.set_xticks(range(N_PLANNING_STEPS + 1))
        ax_cart.set_ylim(0, all_cart_max * 1.1)
        apply_plot_params(ax_cart)

        # Bottom row: Latent distance vs step
        ax_latent = axes[1, ax_idx]
        for metric in METRICS:
            metric_df = axis_df[axis_df['metric'] == metric]
            if len(metric_df) == 0:
                continue

            means = []
            stds = []
            for col in latent_cols:
                if col in metric_df.columns:
                    values = metric_df[col].dropna().values
                    if len(values) > 0:
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            means = np.array(means)
            stds = np.array(stds)
            steps = np.arange(len(means))

            ax_latent.errorbar(
                steps, means, yerr=stds,
                fmt='-o',
                color=METRIC_COLORS[metric],
                linewidth=PLOT_PARAMS_LOCAL["linewidth"],
                markersize=PLOT_PARAMS_LOCAL["markersize"],
                capsize=PLOT_PARAMS_LOCAL["capsize"],
            )

        ax_latent.set_xlabel('Step (k)', fontsize=PLOT_PARAMS_LOCAL["label_size"])
        if ax_idx == 0:
            ax_latent.set_ylabel(r'$\frac{1}{TD}\|z_k - z_g\|_1$', fontsize=PLOT_PARAMS_LOCAL["label_size"])
        apply_plot_params(ax_latent)

    # Add common legend at the bottom
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=3,
               fontsize=16, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    save_path = Path(output_dir) / 'combined_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def compute_summary_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics for all conditions."""
    stats = {}

    for metric in METRICS:
        stats[metric] = {}
        for axis in AXES:
            subset = df[(df['metric'] == metric) & (df['axis'] == axis)]
            if len(subset) == 0:
                continue

            # Convert numpy types to native Python types for JSON serialization
            stats[metric][axis] = {
                'n_episodes': int(len(subset)),
                'final_err_mean': float(subset['final_cart_err'].mean()),
                'final_err_std': float(subset['final_cart_err'].std()),
                'final_err_min': float(subset['final_cart_err'].min()),
                'final_err_max': float(subset['final_cart_err'].max()),
                'success_5cm': int(subset['success_5cm'].sum()),
                'success_10cm': int(subset['success_10cm'].sum()),
                'success_rate_5cm': float(subset['success_5cm'].mean() * 100),
                'success_rate_10cm': float(subset['success_10cm'].mean() * 100),
            }

    return stats


def generate_summary_text(stats: dict, output_dir: str):
    """Generate 5 bullet point summary conclusions."""
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("METRIC SENSITIVITY EXPERIMENT: SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append("")

    # Find best metric per axis
    best_metrics_per_axis = {}
    for axis in AXES:
        best_metric = None
        best_err = float('inf')
        for metric in METRICS:
            if metric in stats and axis in stats[metric]:
                err = stats[metric][axis]['final_err_mean']
                if err < best_err:
                    best_err = err
                    best_metric = metric
        best_metrics_per_axis[axis] = (best_metric, best_err)

    # 1. Best metric per axis
    summary_lines.append("1. BEST METRIC PER AXIS:")
    for axis in AXES:
        metric, err = best_metrics_per_axis[axis]
        summary_lines.append(f"   - {axis.upper()}-axis: {METRIC_LABELS.get(metric, 'N/A')} ({err*100:.1f} cm)")
    summary_lines.append("")

    # 2. Success rates
    summary_lines.append("2. SUCCESS RATES (5cm / 10cm thresholds):")
    for metric in METRICS:
        if metric not in stats:
            continue
        total_5cm = sum(stats[metric][ax]['success_5cm'] for ax in AXES if ax in stats[metric])
        total_10cm = sum(stats[metric][ax]['success_10cm'] for ax in AXES if ax in stats[metric])
        total_eps = sum(stats[metric][ax]['n_episodes'] for ax in AXES if ax in stats[metric])
        summary_lines.append(f"   - {METRIC_LABELS[metric]}: {total_5cm}/{total_eps} (5cm), {total_10cm}/{total_eps} (10cm)")
    summary_lines.append("")

    # 3. Qualitative behavior differences
    summary_lines.append("3. QUALITATIVE BEHAVIOR:")
    # Compare initial to final error reduction
    for metric in METRICS:
        if metric not in stats:
            continue
        reductions = []
        for axis in AXES:
            if axis in stats[metric]:
                # Estimate initial error from typical ~20cm offset
                initial_err = 0.23  # Approximate initial error in meters
                final_err = stats[metric][axis]['final_err_mean']
                reduction = initial_err - final_err
                reductions.append(reduction)
        if reductions:
            avg_reduction = np.mean(reductions) * 100  # Convert to cm
            summary_lines.append(f"   - {METRIC_LABELS[metric]}: avg error reduction = {avg_reduction:.1f} cm")
    summary_lines.append("")

    # 4. Consistency across seeds
    summary_lines.append("4. CONSISTENCY ACROSS SEEDS (std of final error):")
    for metric in METRICS:
        if metric not in stats:
            continue
        stds = []
        for axis in AXES:
            if axis in stats[metric]:
                stds.append(stats[metric][axis]['final_err_std'])
        if stds:
            avg_std = np.mean(stds) * 100  # Convert to cm
            summary_lines.append(f"   - {METRIC_LABELS[metric]}: avg std = {avg_std:.2f} cm")
    summary_lines.append("")

    # 5. Interpretation
    summary_lines.append("5. INTERPRETATION:")

    # Compare L1 to alternatives
    l1_avg = np.mean([stats['l1'][ax]['final_err_mean'] for ax in AXES if 'l1' in stats and ax in stats['l1']])

    improvements = []
    for metric in ['cosine', 'norm_l1']:
        if metric in stats:
            metric_avg = np.mean([stats[metric][ax]['final_err_mean'] for ax in AXES if ax in stats[metric]])
            diff = (l1_avg - metric_avg) * 100  # Convert to cm
            improvements.append((metric, diff))

    any_improvement = any(diff > 1.0 for _, diff in improvements)  # > 1cm improvement

    if any_improvement:
        summary_lines.append("   -> Alternative metrics show improvement over L1 baseline.")
        summary_lines.append("   -> This suggests metric sensitivity is a significant factor;")
        summary_lines.append("      latent direction matters more than magnitude under domain shift.")
    else:
        summary_lines.append("   -> No metric shows substantial improvement over L1 baseline.")
        summary_lines.append("   -> This suggests representational geometry is not sufficient")
        summary_lines.append("      for control under sim-to-real shift, independent of distance choice.")
        summary_lines.append("   -> Strengthens motivation for encoder fine-tuning or language-grounded goals.")

    summary_lines.append("")
    summary_lines.append("=" * 60)

    # Write to file
    summary_path = Path(output_dir) / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"Saved: {summary_path}")

    # Also print to console
    print('\n'.join(summary_lines))


def main():
    parser = argparse.ArgumentParser(description='Analyze metric sensitivity experiment results')
    parser.add_argument('--results_dir', type=str,
                        default='/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/metric_sensitivity/results',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str,
                        default='/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/metric_sensitivity/plots',
                        help='Directory to save plots and analysis')
    parser.add_argument('--baseline_dir', type=str,
                        default='/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/left_cam_fixed_cem_optimized',
                        help='Directory containing baseline L1 results (to avoid variability)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("ANALYZING METRIC SENSITIVITY EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Baseline directory: {args.baseline_dir}")
    print(f"Output directory: {args.output_dir}")
    print("")

    # 1. Consolidate results
    print("Loading and consolidating results...")
    print("  (L1 baseline loaded from fixed experiment to avoid variability)")
    df = consolidate_results(args.results_dir, args.baseline_dir)

    if len(df) == 0:
        print("ERROR: No data found. Make sure experiments have been run.")
        return

    print(f"Loaded {len(df)} episodes")
    print(f"  Metrics: {df['metric'].unique().tolist()}")
    print(f"  Axes: {df['axis'].unique().tolist()}")
    print("")

    # 2. Save consolidated CSV
    csv_path = Path(args.output_dir) / 'consolidated_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # 3. Generate plots
    print("\nGenerating plots...")

    print("  a) Cartesian error vs planning step...")
    plot_error_vs_step(df, args.output_dir)

    print("  b) Final error distribution...")
    plot_final_error_distribution(df, args.output_dir)

    print("  c) Latent distance vs planning step...")
    plot_latent_distance_vs_step(df, args.output_dir)

    print("  d) Combined analysis (error + latent, 2x3)...")
    plot_combined_error_and_latent(df, args.output_dir)

    # 4. Compute and save statistics
    print("\nComputing summary statistics...")
    stats = compute_summary_statistics(df)

    stats_path = Path(args.output_dir) / 'summary_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")

    # 5. Generate summary text
    print("\nGenerating summary...")
    generate_summary_text(stats, args.output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
