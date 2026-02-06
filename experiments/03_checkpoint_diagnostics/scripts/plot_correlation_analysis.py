#!/usr/bin/env python3
"""
Plot correlation analysis results from Probe A Extended.

Creates:
1. Correlation vs epoch plot (Spearman and Pearson)
2. Scatter plots of E(a) vs D(a) for selected epochs
3. Summary table with all metrics
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(output_dir, epochs):
    """Load results from all epoch files."""
    results = {}
    for epoch in epochs:
        filepath = os.path.join(output_dir, f"epoch_{epoch}_correlation.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results[epoch] = json.load(f)
        else:
            print(f"Warning: Missing results for epoch {epoch}")
    return results


def plot_correlation_vs_epoch(results, output_path):
    """Plot correlation metrics across epochs."""
    epochs = sorted(results.keys())

    spearman_means = [results[e]['aggregated']['spearman_corr_mean'] for e in epochs]
    spearman_stds = [results[e]['aggregated']['spearman_corr_std'] for e in epochs]
    pearson_means = [results[e]['aggregated']['pearson_corr_mean'] for e in epochs]
    pearson_stds = [results[e]['aggregated']['pearson_corr_std'] for e in epochs]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(epochs, spearman_means, yerr=spearman_stds,
                marker='o', capsize=3, label='Spearman (rank)', linewidth=2)
    ax.errorbar(epochs, pearson_means, yerr=pearson_stds,
                marker='s', capsize=3, label='Pearson (linear)', linewidth=2)

    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Correlation(E(a), D(a))', fontsize=12)
    ax.set_title('Energy-Distance Correlation vs Training Epoch', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Add horizontal line at 0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scatter_grid(results, output_path, epochs_to_plot=None):
    """Plot E(a) vs D(a) scatter for selected epochs."""
    if epochs_to_plot is None:
        # Default: first, middle, last
        all_epochs = sorted(results.keys())
        epochs_to_plot = [all_epochs[0], all_epochs[len(all_epochs)//2], all_epochs[-1]]

    epochs_to_plot = [e for e in epochs_to_plot if e in results]
    n_epochs = len(epochs_to_plot)

    fig, axes = plt.subplots(1, n_epochs, figsize=(5*n_epochs, 5))
    if n_epochs == 1:
        axes = [axes]

    for ax, epoch in zip(axes, epochs_to_plot):
        data = results[epoch]['per_probe']

        # Concatenate all probes for this epoch
        all_E = []
        all_D = []
        for i in range(len(data['energies'])):
            all_E.extend(data['energies'][i])
            all_D.extend(data['true_distances'][i])

        all_E = np.array(all_E)
        all_D = np.array(all_D)

        ax.scatter(all_E, all_D, alpha=0.3, s=10)

        # Fit line
        z = np.polyfit(all_E, all_D, 1)
        p = np.poly1d(z)
        E_range = np.linspace(all_E.min(), all_E.max(), 100)
        ax.plot(E_range, p(E_range), 'r-', linewidth=2,
                label=f'r={results[epoch]["aggregated"]["pearson_corr_mean"]:.2f}')

        ax.set_xlabel('Energy E(a)', fontsize=11)
        ax.set_ylabel('True Distance D(a)', fontsize=11)
        ax.set_title(f'Epoch {epoch}', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Energy vs True Distance Across Training', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_metrics(results, output_path):
    """Plot correlation alongside sharpness metrics."""
    epochs = sorted(results.keys())

    spearman = [results[e]['aggregated']['spearman_corr_mean'] for e in epochs]
    stdE = [results[e]['aggregated']['sharpness_stdE'] for e in epochs]
    rangeE = [results[e]['aggregated']['sharpness_rangeE'] for e in epochs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Spearman correlation
    axes[0].plot(epochs, spearman, marker='o', linewidth=2, color='tab:blue')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Spearman Correlation', fontsize=11)
    axes[0].set_title('Rank Correlation (Signal Quality)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)

    # Plot 2: Energy std (sharpness)
    axes[1].plot(epochs, stdE, marker='s', linewidth=2, color='tab:orange')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('σ(E)', fontsize=11)
    axes[1].set_title('Energy Spread (Landscape Sharpness)', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Both normalized
    spearman_norm = np.array(spearman) / max(spearman) if max(spearman) > 0 else spearman
    stdE_norm = np.array(stdE) / max(stdE) if max(stdE) > 0 else stdE

    axes[2].plot(epochs, spearman_norm, marker='o', linewidth=2, label='Correlation (norm)')
    axes[2].plot(epochs, stdE_norm, marker='s', linewidth=2, label='σ(E) (norm)')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Normalized Value', fontsize=11)
    axes[2].set_title('Signal Quality vs Spread', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(results, output_path):
    """Create summary CSV with all metrics."""
    epochs = sorted(results.keys())

    data = {
        'epoch': epochs,
        'spearman_corr': [results[e]['aggregated']['spearman_corr_mean'] for e in epochs],
        'spearman_std': [results[e]['aggregated']['spearman_corr_std'] for e in epochs],
        'pearson_corr': [results[e]['aggregated']['pearson_corr_mean'] for e in epochs],
        'pearson_std': [results[e]['aggregated']['pearson_corr_std'] for e in epochs],
        'stdE': [results[e]['aggregated']['sharpness_stdE'] for e in epochs],
        'rangeE': [results[e]['aggregated']['sharpness_rangeE'] for e in epochs],
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"Saved: {output_path}")

    # Also print to console
    print("\nSummary Table:")
    print(df.to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description='Plot correlation analysis results')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing correlation JSON files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: same as input)')
    parser.add_argument('--epochs', type=int, nargs='+',
                        default=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
                        help='Epochs to process')

    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results = load_results(args.input_dir, args.epochs)

    if not results:
        print("No results found!")
        return

    print(f"Loaded results for epochs: {sorted(results.keys())}")

    # Generate plots
    plot_correlation_vs_epoch(
        results,
        os.path.join(output_dir, 'correlation_vs_epoch.png')
    )

    plot_scatter_grid(
        results,
        os.path.join(output_dir, 'energy_distance_scatter.png'),
        epochs_to_plot=[0, 25, 55]  # Early, middle, late
    )

    plot_combined_metrics(
        results,
        os.path.join(output_dir, 'combined_metrics.png')
    )

    create_summary_table(
        results,
        os.path.join(output_dir, 'correlation_summary.csv')
    )


if __name__ == '__main__':
    main()
