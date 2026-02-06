#!/usr/bin/env python3
"""
Plot Diagnostic Results

Generates plots from the checkpoint diagnostics results.csv file.

Usage:
    python plot_diagnostics.py \
        --results /home/s185927/thesis/experiments/03_checkpoint_diagnostics/diagnostics/results.csv \
        --output_dir /home/s185927/thesis/experiments/03_checkpoint_diagnostics/plots
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import plot config for consistent styling
import sys
sys.path.insert(0, "/home/s185927/thesis")
from plot_config import PLOT_PARAMS, configure_axis, apply_plot_params


def load_results(results_path):
    """Load results from CSV file."""
    results = []
    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = {}
            for k, v in row.items():
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
            results.append(result)
    return results


def load_raw_data(results_path):
    """Load raw per-probe data for error bars.

    Returns a dict mapping epoch -> metric -> array of per-probe values.
    """
    raw_dir = Path(results_path).parent / 'raw'
    if not raw_dir.exists():
        return None

    raw_data = {}
    for json_file in sorted(raw_dir.glob('epoch_*_per_probe.json')):
        # Extract epoch from filename
        epoch = int(json_file.stem.split('_')[1])

        with open(json_file, 'r') as f:
            data = json.load(f)

        raw_data[epoch] = {
            'sharpness_stdE': np.array(data.get('probe_a', {}).get('stdE', [])),
            'sharpness_rangeE': np.array(data.get('probe_a', {}).get('rangeE', [])),
            'align_cos': np.array(data.get('probe_a', {}).get('align_cos', [])),
            'onpolicy_pred_loss': np.array(data.get('probe_b', {}).get('mean_loss_per_probe', [])),
            'planning_final_dist': np.array(data.get('probe_c', {}).get('final_distances', [])),
            'planning_delta_per_step': np.array(data.get('probe_c', {}).get('delta_per_steps', [])),
        }

    return raw_data


def get_error_bars(raw_data, epochs, metric):
    """Compute standard error for a metric across epochs.

    Returns (means, stderrs) arrays aligned with epochs.
    """
    if raw_data is None:
        return None, None

    means = []
    stderrs = []

    for epoch in epochs:
        epoch_int = int(epoch)
        if epoch_int in raw_data and len(raw_data[epoch_int].get(metric, [])) > 0:
            values = raw_data[epoch_int][metric]
            means.append(np.mean(values))
            stderrs.append(np.std(values) / np.sqrt(len(values)))  # Standard error
        else:
            means.append(np.nan)
            stderrs.append(np.nan)

    return np.array(means), np.array(stderrs)


def plot_sharpness(results, output_dir, raw_data=None):
    """Plot sharpness metrics vs epoch."""
    epochs = [r['epoch'] for r in results]
    stdE = [r.get('sharpness_stdE', -1) for r in results]
    rangeE = [r.get('sharpness_rangeE', -1) for r in results]

    # Filter out invalid values
    valid_idx = [i for i in range(len(epochs)) if stdE[i] >= 0 and rangeE[i] >= 0]
    if not valid_idx:
        print("No valid sharpness data to plot")
        return

    epochs = [epochs[i] for i in valid_idx]
    stdE = [stdE[i] for i in valid_idx]
    rangeE = [rangeE[i] for i in valid_idx]

    # Get error bars
    _, stdE_err = get_error_bars(raw_data, epochs, 'sharpness_stdE')
    _, rangeE_err = get_error_bars(raw_data, epochs, 'sharpness_rangeE')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, stdE, 'b-o', linewidth=2, markersize=6)
    if stdE_err is not None:
        ax1.fill_between(epochs, np.array(stdE) - stdE_err, np.array(stdE) + stdE_err,
                         alpha=0.2, color='b')
    configure_axis(ax1, 'Epoch', r'$\sigma(\mathcal{E})$', r'$\mathcal{E}$ Sharpness (Std Dev)')

    ax2.plot(epochs, rangeE, 'r-o', linewidth=2, markersize=6)
    if rangeE_err is not None:
        ax2.fill_between(epochs, np.array(rangeE) - rangeE_err, np.array(rangeE) + rangeE_err,
                         alpha=0.2, color='r')
    configure_axis(ax2, 'Epoch', r'$\mathcal{E}$ Range', r'$\mathcal{E}$ Sharpness (Range)')

    plt.tight_layout()
    output_path = output_dir / 'sharpness_vs_epoch.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_alignment(results, output_dir, raw_data=None):
    """Plot alignment metric vs epoch."""
    epochs = [r['epoch'] for r in results]
    align_cos = [r.get('align_cos', -999) for r in results]

    valid_idx = [i for i in range(len(epochs)) if align_cos[i] > -999]
    if not valid_idx:
        print("No valid alignment data to plot")
        return

    epochs = [epochs[i] for i in valid_idx]
    align_cos = [align_cos[i] for i in valid_idx]

    # Get error bars
    _, align_err = get_error_bars(raw_data, epochs, 'align_cos')

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, align_cos, 'g-o', linewidth=2, markersize=6)
    if align_err is not None:
        ax.fill_between(epochs, np.array(align_cos) - align_err, np.array(align_cos) + align_err,
                        alpha=0.2, color='g')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero alignment')
    configure_axis(ax, 'Epoch', r'$\cos(-\nabla \mathcal{E}, \mathbf{g})$', 'Descent-Goal Alignment')
    ax.set_ylim(-0.2, 1.1)
    ax.legend(fontsize=PLOT_PARAMS['legend_size'])

    plt.tight_layout()
    output_path = output_dir / 'alignment_vs_epoch.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_prediction_loss(results, output_dir, raw_data=None):
    """Plot on-policy prediction loss vs validation loss."""
    epochs = [r['epoch'] for r in results]
    val_loss = [r.get('val_loss_dataset', -1) for r in results]
    onpolicy = [r.get('onpolicy_pred_loss', -1) for r in results]

    valid_idx = [i for i in range(len(epochs)) if val_loss[i] >= 0 and onpolicy[i] >= 0]
    if not valid_idx:
        print("No valid prediction loss data to plot")
        return

    epochs = [epochs[i] for i in valid_idx]
    val_loss = [val_loss[i] for i in valid_idx]
    onpolicy = [onpolicy[i] for i in valid_idx]

    # Get error bars (only for on-policy, val_loss is a single value)
    _, onpolicy_err = get_error_bars(raw_data, epochs, 'onpolicy_pred_loss')

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, val_loss, 'b-o', linewidth=2, markersize=6, label='Validation (dataset)')
    ax.plot(epochs, onpolicy, 'r-s', linewidth=2, markersize=6, label='On-policy (planning)')
    if onpolicy_err is not None:
        ax.fill_between(epochs, np.array(onpolicy) - onpolicy_err, np.array(onpolicy) + onpolicy_err,
                        alpha=0.2, color='r')
    configure_axis(ax, 'Epoch', 'Prediction Loss (L1)', 'Prediction Loss')
    ax.legend(fontsize=PLOT_PARAMS['legend_size'])

    plt.tight_layout()
    output_path = output_dir / 'prediction_loss_vs_epoch.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_planning(results, output_dir, raw_data=None):
    """Plot planning metrics vs epoch."""
    epochs = [r['epoch'] for r in results]
    final_dist = [r.get('planning_final_dist_mean', -1) for r in results]
    delta = [r.get('planning_delta_per_step', -999) for r in results]

    valid_idx = [i for i in range(len(epochs)) if final_dist[i] >= 0]
    if not valid_idx:
        print("No valid planning data to plot")
        return

    epochs = [epochs[i] for i in valid_idx]
    final_dist = [final_dist[i] for i in valid_idx]
    delta = [delta[i] for i in valid_idx]

    # Get error bars
    _, dist_err = get_error_bars(raw_data, epochs, 'planning_final_dist')
    _, delta_err = get_error_bars(raw_data, epochs, 'planning_delta_per_step')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, final_dist, 'm-o', linewidth=2, markersize=6)
    if dist_err is not None:
        ax1.fill_between(epochs, np.array(final_dist) - dist_err, np.array(final_dist) + dist_err,
                         alpha=0.2, color='m')
    ax1.axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='Success threshold')
    configure_axis(ax1, 'Epoch', 'Final Distance (m)', 'Planning Performance')
    ax1.legend(fontsize=PLOT_PARAMS['legend_size'])

    ax2.plot(epochs, delta, 'c-o', linewidth=2, markersize=6)
    if delta_err is not None:
        ax2.fill_between(epochs, np.array(delta) - delta_err, np.array(delta) + delta_err,
                         alpha=0.2, color='c')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    configure_axis(ax2, 'Epoch', 'Dist. Reduction (m/step)', 'Planning Progress')

    plt.tight_layout()
    output_path = output_dir / 'planning_vs_epoch.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_dashboard(results, output_dir, raw_data=None):
    """Create combined dashboard with all metrics."""
    epochs = [r['epoch'] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Use larger tick labels for dashboard (will be scaled down in report)
    dashboard_tick_size = 14

    # Helper to add error band
    def add_error_band(ax, x, y, err, color):
        if err is not None:
            y_arr = np.array([v if not np.isnan(v) else np.nan for v in y])
            ax.fill_between(x, y_arr - err, y_arr + err, alpha=0.2, color=color)

    # 1. Sharpness stdE
    ax = axes[0, 0]
    stdE = [r.get('sharpness_stdE', np.nan) for r in results]
    valid = [e for e in stdE if e >= 0]
    if valid:
        ax.plot(epochs, [e if e >= 0 else np.nan for e in stdE], 'b-o')
        _, err = get_error_bars(raw_data, epochs, 'sharpness_stdE')
        add_error_band(ax, epochs, [e if e >= 0 else np.nan for e in stdE], err, 'b')
    configure_axis(ax, 'Epoch', r'$\sigma(\mathcal{E})$', r'$\mathcal{E}$ Sharpness (Std Dev)')
    ax.tick_params(axis='both', labelsize=dashboard_tick_size)

    # 2. Sharpness rangeE
    ax = axes[0, 1]
    rangeE = [r.get('sharpness_rangeE', np.nan) for r in results]
    valid = [e for e in rangeE if e >= 0]
    if valid:
        ax.plot(epochs, [e if e >= 0 else np.nan for e in rangeE], 'r-o')
        _, err = get_error_bars(raw_data, epochs, 'sharpness_rangeE')
        add_error_band(ax, epochs, [e if e >= 0 else np.nan for e in rangeE], err, 'r')
    configure_axis(ax, 'Epoch', r'$\mathcal{E}$ Range', r'$\mathcal{E}$ Sharpness (Range)')
    ax.tick_params(axis='both', labelsize=dashboard_tick_size)

    # 3. Alignment
    ax = axes[0, 2]
    align = [r.get('align_cos', np.nan) for r in results]
    valid = [e for e in align if e > -999]
    if valid:
        ax.plot(epochs, [e if e > -999 else np.nan for e in align], 'g-o')
        _, err = get_error_bars(raw_data, epochs, 'align_cos')
        add_error_band(ax, epochs, [e if e > -999 else np.nan for e in align], err, 'g')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    configure_axis(ax, 'Epoch', r'$\cos(-\nabla \mathcal{E}, \mathbf{g})$', 'Descent-Goal Alignment')
    ax.set_ylim(-0.2, 1.1)
    ax.tick_params(axis='both', labelsize=dashboard_tick_size)

    # 4. Validation loss vs on-policy loss
    ax = axes[1, 0]
    val_loss = [r.get('val_loss_dataset', np.nan) for r in results]
    onpolicy = [r.get('onpolicy_pred_loss', np.nan) for r in results]
    if any(v >= 0 for v in val_loss):
        ax.plot(epochs, [v if v >= 0 else np.nan for v in val_loss], 'b-o', label='Validation')
    if any(v >= 0 for v in onpolicy):
        ax.plot(epochs, [v if v >= 0 else np.nan for v in onpolicy], 'r-s', label='On-policy')
        _, err = get_error_bars(raw_data, epochs, 'onpolicy_pred_loss')
        add_error_band(ax, epochs, [v if v >= 0 else np.nan for v in onpolicy], err, 'r')
    configure_axis(ax, 'Epoch', 'Prediction Loss (L1)', 'Prediction Loss')
    ax.legend(fontsize=PLOT_PARAMS['legend_size'])
    ax.tick_params(axis='both', labelsize=dashboard_tick_size)

    # 5. Planning final distance
    ax = axes[1, 1]
    final_dist = [r.get('planning_final_dist_mean', np.nan) for r in results]
    if any(v >= 0 for v in final_dist):
        ax.plot(epochs, [v if v >= 0 else np.nan for v in final_dist], 'm-o')
        _, err = get_error_bars(raw_data, epochs, 'planning_final_dist')
        add_error_band(ax, epochs, [v if v >= 0 else np.nan for v in final_dist], err, 'm')
    ax.axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='Success')
    configure_axis(ax, 'Epoch', 'Final Distance (m)', 'Planning Performance')
    ax.legend(fontsize=PLOT_PARAMS['legend_size'])
    ax.tick_params(axis='both', labelsize=dashboard_tick_size)

    # 6. Delta per step
    ax = axes[1, 2]
    delta = [r.get('planning_delta_per_step', np.nan) for r in results]
    if any(v > -999 for v in delta):
        ax.plot(epochs, [v if v > -999 else np.nan for v in delta], 'c-o')
        _, err = get_error_bars(raw_data, epochs, 'planning_delta_per_step')
        add_error_band(ax, epochs, [v if v > -999 else np.nan for v in delta], err, 'c')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    configure_axis(ax, 'Epoch', 'Dist. Reduction (m/step)', 'Planning Progress')
    ax.tick_params(axis='both', labelsize=dashboard_tick_size)

    plt.tight_layout()
    output_path = output_dir / 'diagnostics_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot Diagnostic Results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for plots')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(args.results)
    print(f"Loaded {len(results)} epochs of results")

    # Load raw per-probe data for error bars
    raw_data = load_raw_data(args.results)
    if raw_data:
        print(f"Loaded raw per-probe data for {len(raw_data)} epochs (error bars enabled)")
    else:
        print("No raw per-probe data found (error bars disabled)")

    if not results:
        print("No results to plot!")
        return

    # Generate all plots
    plot_sharpness(results, output_dir, raw_data)
    plot_alignment(results, output_dir, raw_data)
    plot_prediction_loss(results, output_dir, raw_data)
    plot_planning(results, output_dir, raw_data)
    plot_dashboard(results, output_dir, raw_data)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
