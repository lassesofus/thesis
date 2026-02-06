#!/usr/bin/env python3
"""
Plot V-JEPA model evaluation results for X-Axis Finetune experiment.

Usage:
    python plot_eval_results.py /home/s185927/thesis/experiments/sim_training_different_fractions_x_axis_finetune/eval_results
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# Import shared plotting configuration
sys.path.insert(0, '/home/s185927/thesis')
from plot_config import PLOT_PARAMS


def load_results(eval_dir):
    """Load evaluation results from directory."""
    eval_path = Path(eval_dir)

    # Load individual results (n50 files, falling back to n49 for 025pct)
    individual_results = {}
    for result_file in eval_path.glob("*_n4*.json"):
        model_name = result_file.stem.replace("_results_n50", "").replace("_results_n49", "")
        with open(result_file, 'r') as f:
            individual_results[model_name] = json.load(f)
    for result_file in eval_path.glob("*_n50.json"):
        model_name = result_file.stem.replace("_results_n50", "")
        with open(result_file, 'r') as f:
            individual_results[model_name] = json.load(f)

    # Generate summary from individual results
    summary = {}
    for model_name, results in individual_results.items():
        final_distances = [r['final_distance'] for r in results]
        successes = [r['success'] for r in results]

        summary[model_name] = {
            'mean_final_distance': np.mean(final_distances),
            'std_final_distance': np.std(final_distances),
            'median_final_distance': np.median(final_distances),
            'success_rate': np.mean(successes),
            'num_samples': len(results)
        }

    return summary, individual_results


def extract_fraction(model_name):
    """Extract training data fraction from model name."""
    if 'meta' in model_name.lower():
        return 'Meta'
    elif '025pct' in model_name:
        return '25%'
    elif '050pct' in model_name:
        return '50%'
    elif '075pct' in model_name:
        return '75%'
    elif '100pct' in model_name:
        return '100%'
    return model_name


def plot_performance_comparison(summary, save_path=None):
    """Plot mean final distance comparison across models."""
    # Sort models by fraction
    fraction_order = ['25%', '50%', '75%', '100%', 'Meta']
    model_data = {}

    for model_name, stats in summary.items():
        fraction = extract_fraction(model_name)
        model_data[fraction] = stats

    # Create ordered lists
    fractions = [f for f in fraction_order if f in model_data]
    means = [model_data[f]['mean_final_distance'] for f in fractions]
    stds = [model_data[f]['std_final_distance'] for f in fractions]
    success_rates = [model_data[f]['success_rate'] * 100 for f in fractions]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot: Mean final distance
    x_pos = np.arange(len(fractions))

    bars = ax.bar(x_pos, means, yerr=stds, capsize=PLOT_PARAMS["euclid_capsize"],
                  color='#2E86AB', alpha=0.7, width=0.6)
    ax.set_xlabel('Training set fraction', fontsize=PLOT_PARAMS["label_size"])
    ax.set_ylabel('Mean Final Distance (m)', fontsize=PLOT_PARAMS["label_size"])
    ax.set_title('Planning Performance by Training Data Amount', fontsize=PLOT_PARAMS["title_size"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fractions)
    ax.grid(True, alpha=PLOT_PARAMS["grid_alpha"], axis='y')
    ax.tick_params(labelsize=PLOT_PARAMS["tick_label_size"], length=PLOT_PARAMS["tick_length"])

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}m',
                ha='center', va='bottom', fontsize=PLOT_PARAMS["tick_label_size"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_distance_distributions(individual_results, save_path=None):
    """Plot distribution of final distances for each model."""
    fraction_order = ['25%', '50%', '75%', '100%', 'Meta']
    model_data = {}

    for model_name, results in individual_results.items():
        fraction = extract_fraction(model_name)
        final_distances = [r['final_distance'] for r in results]
        model_data[fraction] = final_distances

    # Create ordered lists
    fractions = [f for f in fraction_order if f in model_data]
    data = [model_data[f] for f in fractions]
    n_samples = len(data[0]) if data else 0

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot with mean markers
    bp = ax.boxplot(data, tick_labels=fractions, patch_artist=True,
                    notch=False, widths=0.5, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=6))

    # Uniform color for all boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#2E86AB')
        patch.set_alpha(0.7)

    # Add horizontal line at success threshold
    ax.axhline(y=0.05, color='r', linestyle='--', linewidth=PLOT_PARAMS["threshold_linewidth"],
               label='Success Threshold (0.05m)', alpha=0.7)

    # Add mean marker to legend
    mean_marker = plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
                              markeredgecolor='black', markersize=6, label='Mean', linestyle='None')

    ax.set_xlabel('Training set fraction', fontsize=PLOT_PARAMS["label_size"])
    ax.set_ylabel('Final Distance to Goal (m)', fontsize=PLOT_PARAMS["label_size"])
    ax.set_title(f'Distribution of Final Distances (n=50)', fontsize=PLOT_PARAMS["title_size"])

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mean_marker)
    ax.legend(handles=handles, fontsize=PLOT_PARAMS["legend_size"])
    ax.grid(True, alpha=PLOT_PARAMS["grid_alpha"], axis='y')
    ax.tick_params(labelsize=PLOT_PARAMS["tick_label_size"], length=PLOT_PARAMS["tick_length"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_planning_trajectories(individual_results, save_path=None):
    """Plot how distance changes over planning steps for each model."""
    fraction_order = ['25%', '50%', '75%', '100%', 'Meta']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#888888']

    fig, ax = plt.subplots(figsize=(10, 6))

    n_samples = None  # Track sample size for legend
    for fraction, color in zip(fraction_order, colors):
        # Find matching model
        model_data = None
        for model_name, results in individual_results.items():
            if extract_fraction(model_name) == fraction:
                model_data = results
                break

        if model_data is None:
            continue

        # Collect all distance trajectories
        all_trajectories = []
        for result in model_data:
            distances = result['distances_per_step']
            all_trajectories.append(distances)

        if n_samples is None:
            n_samples = len(all_trajectories)

        # Compute mean and std at each step
        max_len = max(len(traj) for traj in all_trajectories)
        mean_traj = []
        std_traj = []

        for step in range(max_len):
            step_distances = [traj[step] for traj in all_trajectories if step < len(traj)]
            mean_traj.append(np.mean(step_distances))
            std_traj.append(np.std(step_distances))

        steps = np.arange(len(mean_traj))
        mean_traj = np.array(mean_traj)
        std_traj = np.array(std_traj)

        # Plot
        ax.plot(steps, mean_traj, marker='o',
                linewidth=PLOT_PARAMS["euclid_linewidth"],
                markersize=PLOT_PARAMS["euclid_markersize"],
                label=fraction, color=color, alpha=0.8)
        ax.fill_between(steps, mean_traj - std_traj, mean_traj + std_traj,
                        color=color, alpha=0.2)

    # Success threshold
    ax.axhline(y=0.05, color='r', linestyle='--',
               linewidth=PLOT_PARAMS["threshold_linewidth"],
               label='Success Threshold', alpha=0.7)

    ax.set_xlabel('Planning Step', fontsize=PLOT_PARAMS["label_size"])
    ax.set_ylabel('Distance to Goal (m)', fontsize=PLOT_PARAMS["label_size"])
    ax.set_title('Distance Reduction Over Planning Steps', fontsize=PLOT_PARAMS["title_size"])

    # Add legend entry for shaded std region
    std_label = '± 1 std (N=50)'
    std_patch = Patch(facecolor='gray', alpha=0.2, label=std_label)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(std_patch)
    ax.legend(handles=handles, fontsize=PLOT_PARAMS["legend_size"], loc='best')
    ax.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
    ax.tick_params(labelsize=PLOT_PARAMS["tick_label_size"], length=PLOT_PARAMS["tick_length"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def print_summary_table(summary):
    """Print formatted summary table."""
    fraction_order = ['25%', '50%', '75%', '100%', 'Meta']

    print("\n" + "="*90)
    print("EVALUATION SUMMARY (X-Axis Finetune)")
    print("="*90)
    print(f"{'Model':<12} {'Mean Dist (m)':<15} {'Median (m)':<12} {'Success Rate':<15} {'N Samples':<10}")
    print("-"*90)

    for fraction in fraction_order:
        for model_name, stats in summary.items():
            if extract_fraction(model_name) == fraction:
                print(f"{fraction:<12} "
                      f"{stats['mean_final_distance']:.4f} ± {stats['std_final_distance']:.4f}   "
                      f"{stats['median_final_distance']:.4f}       "
                      f"{stats['success_rate']*100:5.1f}%          "
                      f"{stats['num_samples']:<10}")
                break

    print("="*90)
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_eval_results.py <eval_results_dir>")
        print("\nExample:")
        print("  python plot_eval_results.py /home/s185927/thesis/experiments/sim_training_different_fractions_x_axis_finetune/eval_results")
        sys.exit(1)

    eval_dir = sys.argv[1]
    eval_path = Path(eval_dir)

    if not eval_path.exists():
        print(f"Error: Directory not found: {eval_dir}")
        sys.exit(1)

    print(f"Loading results from: {eval_dir}")

    # Load data
    summary, individual_results = load_results(eval_dir)

    # Print summary table
    print_summary_table(summary)

    # Generate plots
    print("Generating plots...")

    plot_distance_distributions(
        individual_results,
        save_path=eval_path / "distance_distributions.png"
    )

    plot_planning_trajectories(
        individual_results,
        save_path=eval_path / "planning_trajectories.png"
    )

    print(f"\nDone! Plots saved to: {eval_dir}")
    print("  - distance_distributions.png")
    print("  - planning_trajectories.png")

    plt.show()


if __name__ == "__main__":
    main()
