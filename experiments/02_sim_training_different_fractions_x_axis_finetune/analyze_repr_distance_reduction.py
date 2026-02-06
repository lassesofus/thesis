#!/usr/bin/env python3
"""
Part 1: Representation Distance Analysis

Analyze whether CEM is reducing latent L1 distance during planning
even when physical (Euclidean) distance doesn't improve.

This script uses existing eval results JSON files which contain:
- distances_per_step: Euclidean distance to goal at each planning step
- repr_distances_per_step: Latent L1 distance to goal at each planning step
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "/home/s185927/thesis")
from plot_config import PLOT_PARAMS, configure_axis


# Model configurations - order matters for plotting
MODELS = {
    "Meta Baseline": "meta_baseline_results_n10.json",
    "25% Finetuned": "x_axis_finetune_025pct_results_n10.json",
    "50% Finetuned": "x_axis_finetune_050pct_results_n10.json",
    "75% Finetuned": "x_axis_finetune_075pct_results_n10.json",
    "100% Finetuned": "x_axis_finetune_100pct_results_n10.json",
}


def load_eval_results(eval_dir):
    """Load all evaluation results from JSON files."""
    results = {}
    for model_name, filename in MODELS.items():
        filepath = eval_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                results[model_name] = json.load(f)
            print(f"Loaded {model_name}: {len(results[model_name])} samples")
        else:
            print(f"Warning: {filepath} not found")
    return results


def compute_step_statistics(results):
    """
    Compute mean and std of distances per step for each model.

    Returns dict with structure:
    {
        model_name: {
            'euclidean': {'mean': [...], 'std': [...]},
            'latent': {'mean': [...], 'std': [...]}
        }
    }
    """
    stats = {}

    for model_name, samples in results.items():
        # Collect distances per step across all samples
        all_euclidean = []
        all_latent = []

        for sample in samples:
            euclidean = sample.get('distances_per_step', [])
            latent = sample.get('repr_distances_per_step', [])

            if euclidean and latent:
                all_euclidean.append(euclidean)
                all_latent.append(latent)

        if not all_euclidean:
            continue

        # Convert to arrays (samples might have different lengths, use max)
        max_euclidean_steps = max(len(e) for e in all_euclidean)
        max_latent_steps = max(len(l) for l in all_latent)

        # Pad shorter sequences with NaN
        euclidean_arr = np.full((len(all_euclidean), max_euclidean_steps), np.nan)
        latent_arr = np.full((len(all_latent), max_latent_steps), np.nan)

        for i, e in enumerate(all_euclidean):
            euclidean_arr[i, :len(e)] = e
        for i, l in enumerate(all_latent):
            latent_arr[i, :len(l)] = l

        # Compute statistics (ignoring NaN)
        stats[model_name] = {
            'euclidean': {
                'mean': np.nanmean(euclidean_arr, axis=0),
                'std': np.nanstd(euclidean_arr, axis=0),
            },
            'latent': {
                'mean': np.nanmean(latent_arr, axis=0),
                'std': np.nanstd(latent_arr, axis=0),
            },
            'n_samples': len(all_euclidean),
        }

    return stats


def plot_distance_reduction(stats, output_dir):
    """
    Create plots showing distance reduction over planning steps.

    Creates two subplots:
    1. Euclidean distance vs planning step
    2. Latent L1 distance vs planning step
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Colors for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(stats)))

    # Plot 1: Euclidean distance
    ax1 = axes[0]
    for idx, (model_name, model_stats) in enumerate(stats.items()):
        mean = model_stats['euclidean']['mean']
        std = model_stats['euclidean']['std']
        steps = np.arange(len(mean))

        ax1.plot(steps, mean, 'o-', label=model_name, color=colors[idx],
                 linewidth=PLOT_PARAMS["euclid_linewidth"],
                 markersize=PLOT_PARAMS["euclid_markersize"])
        ax1.fill_between(steps, mean - std, mean + std, alpha=0.2, color=colors[idx])

    configure_axis(ax1,
                   xlabel='Planning Step',
                   ylabel='Euclidean Distance to Goal (m)',
                   title='Physical Distance During Planning')
    ax1.legend(fontsize=PLOT_PARAMS["legend_size"])

    # Plot 2: Latent L1 distance
    ax2 = axes[1]
    for idx, (model_name, model_stats) in enumerate(stats.items()):
        mean = model_stats['latent']['mean']
        std = model_stats['latent']['std']
        steps = np.arange(len(mean))

        ax2.plot(steps, mean, 'o-', label=model_name, color=colors[idx],
                 linewidth=PLOT_PARAMS["repr_linewidth"],
                 markersize=PLOT_PARAMS["repr_markersize"])
        ax2.fill_between(steps, mean - std, mean + std, alpha=0.2, color=colors[idx])

    configure_axis(ax2,
                   xlabel='Planning Step',
                   ylabel='Latent L1 Distance to Goal',
                   title='Representation Distance During Planning')
    ax2.legend(fontsize=PLOT_PARAMS["legend_size"])

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / 'repr_distance_reduction_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()

    return plot_path


def plot_distance_reduction_ratio(stats, output_dir):
    """
    Plot the ratio of final to initial distance for both metrics.

    This shows how much CEM is reducing distances relative to the starting point.
    """
    fig, ax = plt.subplots(figsize=PLOT_PARAMS["figsize_plots_only"])

    model_names = list(stats.keys())
    x = np.arange(len(model_names))
    width = 0.35

    euclidean_ratios = []
    latent_ratios = []

    for model_name in model_names:
        model_stats = stats[model_name]

        # Euclidean: ratio of final to initial distance
        eucl_mean = model_stats['euclidean']['mean']
        eucl_ratio = eucl_mean[-1] / eucl_mean[0] if eucl_mean[0] > 0 else 1.0
        euclidean_ratios.append(eucl_ratio)

        # Latent: ratio of final to initial distance
        lat_mean = model_stats['latent']['mean']
        lat_ratio = lat_mean[-1] / lat_mean[0] if lat_mean[0] > 0 else 1.0
        latent_ratios.append(lat_ratio)

    bars1 = ax.bar(x - width/2, euclidean_ratios, width, label='Euclidean Distance', color='steelblue')
    bars2 = ax.bar(x + width/2, latent_ratios, width, label='Latent L1 Distance', color='coral')

    # Add reference line at 1.0 (no change)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='No Change')

    # Add value labels on bars
    for bar, ratio in zip(bars1, euclidean_ratios):
        height = bar.get_height()
        ax.annotate(f'{ratio:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    for bar, ratio in zip(bars2, latent_ratios):
        height = bar.get_height()
        ax.annotate(f'{ratio:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')

    configure_axis(ax,
                   xlabel='Model',
                   ylabel='Final / Initial Distance Ratio',
                   title='Distance Reduction During Planning\n(Lower is Better)')
    ax.legend(fontsize=PLOT_PARAMS["legend_size"])

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / 'repr_distance_reduction_ratio.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved ratio plot to: {plot_path}")
    plt.close()

    return plot_path


def print_summary(stats):
    """Print a summary of the distance reduction analysis."""
    print("\n" + "=" * 80)
    print("SUMMARY: Distance Reduction During Planning")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Eucl. Start':<12} {'Eucl. End':<12} {'Eucl. Δ':<12} "
          f"{'Lat. Start':<12} {'Lat. End':<12} {'Lat. Δ':<12}")
    print("-" * 80)

    for model_name, model_stats in stats.items():
        eucl_start = model_stats['euclidean']['mean'][0]
        eucl_end = model_stats['euclidean']['mean'][-1]
        eucl_delta = eucl_end - eucl_start

        lat_start = model_stats['latent']['mean'][0]
        lat_end = model_stats['latent']['mean'][-1]
        lat_delta = lat_end - lat_start

        print(f"{model_name:<20} {eucl_start:<12.4f} {eucl_end:<12.4f} {eucl_delta:<+12.4f} "
              f"{lat_start:<12.4f} {lat_end:<12.4f} {lat_delta:<+12.4f}")

    print("=" * 80)

    # Key diagnostic
    print("\nKEY DIAGNOSTIC:")
    print("If latent distance decreases but Euclidean distance doesn't improve,")
    print("the learned representation doesn't align with physical space.")
    print()


def main():
    # Paths
    script_dir = Path(__file__).parent
    eval_dir = script_dir / "eval_results"
    output_dir = script_dir

    print("Part 1: Representation Distance Reduction Analysis")
    print("=" * 60)

    # Load results
    print("\nLoading evaluation results...")
    results = load_eval_results(eval_dir)

    if not results:
        print("Error: No evaluation results found!")
        return

    # Compute statistics
    print("\nComputing step-wise statistics...")
    stats = compute_step_statistics(results)

    # Create plots
    print("\nCreating plots...")
    plot_distance_reduction(stats, output_dir)
    plot_distance_reduction_ratio(stats, output_dir)

    # Print summary
    print_summary(stats)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
