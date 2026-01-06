#!/usr/bin/env python3
"""
Plot results from IK error analysis with actuator dynamics.

This script reads the saved results and regenerates all plots.
Useful for adjusting plot styling without re-running the experiment.

Usage:
    python plot_results.py
    python plot_results.py --input_dir /path/to/results
"""

import sys
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add thesis root to path for plot_config
sys.path.insert(0, '/home/s185927/thesis')
from plot_config import PLOT_PARAMS, apply_plot_params, configure_axis


@click.command()
@click.option(
    '--input_dir',
    type=str,
    default=None,
    help='Directory containing results. Defaults to script directory.'
)
def main(input_dir):
    """Plot results from IK actuator error analysis."""

    # Default to script directory
    if input_dir is None:
        input_dir = Path(__file__).parent
    else:
        input_dir = Path(input_dir)

    output_path = input_dir

    # Load data
    csv_path = input_dir / "actuator_ik_error_results.csv"
    summary_csv_path = input_dir / "actuator_ik_error_summary.csv"

    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        print("Run the experiment first with run_experiment.sh")
        sys.exit(1)

    print(f"Loading results from: {input_dir}")
    df = pd.read_csv(csv_path)
    summary_df = pd.read_csv(summary_csv_path)

    radii = sorted(df['radius'].unique())
    print(f"Found data for radii: {radii}")

    print(f"\nGenerating plots...")

    # 1. Absolute error vs radius
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        summary_df['radius'],
        summary_df['mean_absolute_error'],
        yerr=summary_df['std_absolute_error'],
        marker='o',
        capsize=PLOT_PARAMS['euclid_capsize'],
        linewidth=PLOT_PARAMS['euclid_linewidth'],
        markersize=PLOT_PARAMS['euclid_markersize'],
        color='black',
        label='Mean ± Std'
    )
    configure_axis(ax, 'Target Radius (m)', 'Absolute Error (m)',
                  'IK Error with Actuator Dynamics vs Target Distance')
    ax.legend(fontsize=PLOT_PARAMS['legend_size'])
    plot1_path = output_path / "actuator_absolute_error_vs_radius.png"
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot1_path}")
    plt.close()

    # 2. Relative error vs radius
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        summary_df['radius'],
        100 * summary_df['mean_relative_error'],
        yerr=100 * summary_df['std_relative_error'],
        marker='o',
        capsize=PLOT_PARAMS['euclid_capsize'],
        linewidth=PLOT_PARAMS['euclid_linewidth'],
        markersize=PLOT_PARAMS['euclid_markersize'],
        color='black',
        label='Mean ± Std'
    )
    configure_axis(ax, 'Target Radius (m)', 'Relative Error (%)',
                  'IK Relative Error with Actuator Dynamics')
    ax.legend(fontsize=PLOT_PARAMS['legend_size'])
    plot2_path = output_path / "actuator_relative_error_vs_radius.png"
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot2_path}")
    plt.close()

    # 3. Combined 2x2 figure: errorbar (left) and boxplot (right)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey='row')
    ax_abs_line, ax_abs_box = axes[0]
    ax_rel_line, ax_rel_box = axes[1]

    # Prepare box plot data
    data_by_radius = [df[df['radius'] == r]['absolute_error'].values for r in radii]
    data_by_radius_rel = [100 * df[df['radius'] == r]['relative_error'].values for r in radii]

    # Calculate shared y-axis limits
    abs_min = min(summary_df['mean_absolute_error'].min() - summary_df['std_absolute_error'].max(),
                  min(d.min() for d in data_by_radius))
    abs_max = max(summary_df['mean_absolute_error'].max() + summary_df['std_absolute_error'].max(),
                  max(d.max() for d in data_by_radius))
    abs_margin = (abs_max - abs_min) * 0.1
    abs_ylim = (max(0, abs_min - abs_margin), abs_max + abs_margin)

    rel_min = min((100 * summary_df['mean_relative_error']).min() - (100 * summary_df['std_relative_error']).max(),
                  min(d.min() for d in data_by_radius_rel))
    rel_max = max((100 * summary_df['mean_relative_error']).max() + (100 * summary_df['std_relative_error']).max(),
                  max(d.max() for d in data_by_radius_rel))
    rel_margin = (rel_max - rel_min) * 0.1
    rel_ylim = (max(0, rel_min - rel_margin), rel_max + rel_margin)

    # Top-left: Absolute error errorbar
    ax_abs_line.errorbar(
        summary_df['radius'],
        summary_df['mean_absolute_error'],
        yerr=summary_df['std_absolute_error'],
        marker='o',
        capsize=PLOT_PARAMS['euclid_capsize'],
        linewidth=PLOT_PARAMS['euclid_linewidth'],
        markersize=PLOT_PARAMS['euclid_markersize'],
        color='black'
    )
    ax_abs_line.set_ylabel('Absolute Error (m)', fontsize=PLOT_PARAMS['label_size'])
    ax_abs_line.set_title('(a) Mean ± Std', fontsize=PLOT_PARAMS['subtitle_size'])
    ax_abs_line.tick_params(axis='both', labelsize=16)
    ax_abs_line.set_ylim(abs_ylim)
    ax_abs_line.grid(True, alpha=PLOT_PARAMS['grid_alpha'])

    # Top-right: Absolute error boxplot
    ax_abs_box.boxplot(data_by_radius, labels=[f"{r:.2f}" for r in radii])
    ax_abs_box.set_title('(b) Distribution', fontsize=PLOT_PARAMS['subtitle_size'])
    ax_abs_box.tick_params(axis='both', labelsize=16)
    ax_abs_box.set_ylim(abs_ylim)
    ax_abs_box.grid(True, alpha=PLOT_PARAMS['grid_alpha'])

    # Bottom-left: Relative error errorbar
    ax_rel_line.errorbar(
        summary_df['radius'],
        100 * summary_df['mean_relative_error'],
        yerr=100 * summary_df['std_relative_error'],
        marker='o',
        capsize=PLOT_PARAMS['euclid_capsize'],
        linewidth=PLOT_PARAMS['euclid_linewidth'],
        markersize=PLOT_PARAMS['euclid_markersize'],
        color='black'
    )
    ax_rel_line.set_xlabel('Target Radius (m)', fontsize=PLOT_PARAMS['label_size'])
    ax_rel_line.set_ylabel('Relative Error (%)', fontsize=PLOT_PARAMS['label_size'])
    ax_rel_line.set_title('(c) Mean ± Std', fontsize=PLOT_PARAMS['subtitle_size'])
    ax_rel_line.tick_params(axis='both', labelsize=16)
    ax_rel_line.set_ylim(rel_ylim)
    ax_rel_line.grid(True, alpha=PLOT_PARAMS['grid_alpha'])

    # Bottom-right: Relative error boxplot
    ax_rel_box.boxplot(data_by_radius_rel, labels=[f"{r:.2f}" for r in radii])
    ax_rel_box.set_xlabel('Target Radius (m)', fontsize=PLOT_PARAMS['label_size'])
    ax_rel_box.set_title('(d) Distribution', fontsize=PLOT_PARAMS['subtitle_size'])
    ax_rel_box.tick_params(axis='both', labelsize=16)
    ax_rel_box.set_ylim(rel_ylim)
    ax_rel_box.grid(True, alpha=PLOT_PARAMS['grid_alpha'])

    plt.tight_layout()
    plot3_path = output_path / "actuator_error_combined.png"
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot3_path}")
    plt.close()

    # 4. Maximum distance during trajectory (overshoot analysis)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        summary_df['radius'],
        summary_df['mean_max_distance'],
        yerr=summary_df['std_max_distance'],
        marker='s',
        capsize=PLOT_PARAMS['euclid_capsize'],
        linewidth=PLOT_PARAMS['euclid_linewidth'],
        markersize=PLOT_PARAMS['euclid_markersize'],
        color='black',
        label='Max distance during trajectory'
    )
    # Add reference line for target radius
    ax.plot(summary_df['radius'], summary_df['radius'], 'k--',
            linewidth=1, alpha=0.5, label='Target radius (reference)')
    configure_axis(ax, 'Target Radius (m)', 'Maximum Distance to Target (m)',
                  'Trajectory Overshoot Analysis')
    ax.legend(fontsize=PLOT_PARAMS['legend_size'])
    plot4_path = output_path / "actuator_max_distance_overshoot.png"
    plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot4_path}")
    plt.close()

    # 5. Scatter plot: target distance vs absolute error
    fig, ax = plt.subplots(figsize=(10, 6))
    for radius in radii:
        radius_data = df[df['radius'] == radius]
        ax.scatter(
            radius_data['target_distance'],
            radius_data['absolute_error'],
            alpha=0.5,
            s=20,
            label=f'R={radius:.2f}m'
        )
    configure_axis(ax, 'Target Distance (m)', 'Absolute Error (m)',
                  'Absolute Error vs Target Distance (Actuator Execution)')
    ax.legend(fontsize=PLOT_PARAMS['legend_size'])
    plot5_path = output_path / "actuator_error_scatter.png"
    plt.savefig(plot5_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot5_path}")
    plt.close()

    print(f"\nAll plots saved to: {output_path}")


if __name__ == '__main__':
    main()
