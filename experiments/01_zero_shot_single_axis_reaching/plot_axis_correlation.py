#!/usr/bin/env python3
"""
Plot zero-shot axis correlation from pre-computed data.

This script quickly regenerates the correlation plot from saved .npz data
without needing to re-run the encoder analysis.

Usage:
    python plot_axis_correlation.py
    python plot_axis_correlation.py --data_path /path/to/data.npz --output_path /path/to/plot.png
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
        description='Plot zero-shot axis correlation from saved data'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/zero_shot_axis_correlation_data.npz',
        help='Path to the .npz data file',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/zero_shot_axis_correlation.png',
        help='Path to save the output plot',
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.data_path}")
    data = np.load(args.data_path, allow_pickle=True)

    # Extract data for each axis
    axes_data = {}
    for axis in ['X_axis', 'Y_axis', 'Z_axis']:
        axes_data[axis] = {
            'euclidean_dist': data[f'{axis}_euclidean'],
            'latent_dist': data[f'{axis}_latent'],
            'correlation': float(data[f'{axis}_correlation']),
        }

    # Create 1x3 comparison plot with shared axes labels
    print("Creating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axis_names = ['X-axis', 'Y-axis', 'Z-axis']

    for idx, (axis_key, axis_name) in enumerate(zip(['X_axis', 'Y_axis', 'Z_axis'], axis_names)):
        ax = axes[idx]
        d = axes_data[axis_key]

        # Scatter plot with transparency - use same color for all axes
        ax.scatter(
            d['euclidean_dist'],
            d['latent_dist'],
            alpha=0.3,
            s=20,
            color='#1f77b4',
        )

        # Add trend line
        z = np.polyfit(d['euclidean_dist'], d['latent_dist'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(d['euclidean_dist']), max(d['euclidean_dist']), 100)
        ax.plot(x_trend, p(x_trend), 'r--', linewidth=2)

        # Configure axis - xlabel on each subplot, ylabel only on first
        configure_axis(
            ax,
            xlabel='Euclidean Distance to Goal (m)',
            ylabel='L1 Dist. to Goal Repr.' if idx == 0 else '',
            title=axis_name
        )

        # Add correlation as text annotation (no misleading legend symbol)
        ax.text(
            0.95, 0.05, f'Pearson r = {d["correlation"]:.2f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=PLOT_PARAMS['legend_size'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    plt.tight_layout()

    # Save plot
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {args.output_path}")

    # Print summary
    print("\nSummary:")
    for axis_key, axis_name in zip(['X_axis', 'Y_axis', 'Z_axis'], axis_names):
        d = axes_data[axis_key]
        print(f"  {axis_name}: r = {d['correlation']:.4f}, N = {len(d['euclidean_dist'])} points")


if __name__ == '__main__':
    main()
