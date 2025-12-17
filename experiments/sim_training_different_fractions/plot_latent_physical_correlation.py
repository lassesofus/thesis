#!/usr/bin/env python3
"""
Plot correlation between Euclidean distance and latent L1 distance from saved data.

This script loads pre-computed data from an NPZ file and creates a scatter plot.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add thesis directory for plot config
sys.path.insert(0, "/home/s185927/thesis")

from plot_config import PLOT_PARAMS, configure_axis


def main():
    parser = argparse.ArgumentParser(
        description='Plot correlation between Euclidean distance and latent distance from NPZ file'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='/home/s185927/thesis/experiments/sim_training_different_fractions/latent_physical_correlation.npz',
        help='Path to NPZ data file',
    )
    parser.add_argument(
        '--output_plot',
        type=str,
        default='/home/s185927/thesis/experiments/sim_training_different_fractions/latent_physical_correlation.png',
        help='Output plot path (defaults to same location as data file with .png extension)',
    )

    args = parser.parse_args()

    # Load data
    data_path = Path(args.data_path)
    print(f"Loading data from: {data_path}")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    data = np.load(data_path)
    all_euclidean_dist = data['euclidean_dist']
    all_latent_dist = data['latent_dist']
    correlation = float(data['correlation'])

    print(f"Loaded {len(all_euclidean_dist)} data points")
    print(f"Correlation coefficient: {correlation:.4f}")

    # Create scatter plot
    print("Creating scatter plot...")
    fig, ax = plt.subplots(figsize=PLOT_PARAMS["figsize_plots_only"])

    # Scatter plot with transparency
    ax.scatter(all_euclidean_dist, all_latent_dist, alpha=0.3, s=20)

    # Configure axis with consistent styling
    configure_axis(
        ax,
        xlabel='Euclidean Distance to Goal (m)',
        ylabel='Latent L1 Distance to Goal',
        title=f'Correlation between Physical Distance and Latent Distance\n'
              f'(Correlation: {correlation:.4f}, N={len(all_euclidean_dist)} points)'
    )

    # Add trend line
    z = np.polyfit(all_euclidean_dist, all_latent_dist, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(all_euclidean_dist), max(all_euclidean_dist), 100)
    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')

    ax.legend(fontsize=PLOT_PARAMS["legend_size"])

    plt.tight_layout()

    # Determine output path
    if args.output_plot:
        output_path = Path(args.output_plot)
    else:
        output_path = data_path.with_suffix('.png')

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    print("\nPlotting complete!")


if __name__ == '__main__':
    main()
