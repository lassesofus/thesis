#!/usr/bin/env python3
"""
Compare VJEPA2 models fine-tuned on different fractions of the x_axis dataset.
Generates three plots:
1. Performance comparison at best validation epoch (loss + convergence speed)
2. Learning curves: training and validation loss over epochs
3. Performance at fixed epochs (15, 25, 35): training and validation losses
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import shared plotting configuration
sys.path.insert(0, '/home/s185927/thesis')
from plot_config import PLOT_PARAMS

# Define model directories and their data fractions
BASE_DIR = Path("/data/s185927/vjepa2/weights/droid")
MODELS = {
    "25%": BASE_DIR / "x_axis_finetune_025pct",
    "50%": BASE_DIR / "x_axis_finetune_050pct",
    "75%": BASE_DIR / "x_axis_finetune_075pct",
    "100%": BASE_DIR / "x_axis_finetune_100pct",
}

def load_validation_data(model_dir):
    """Load validation log for a model."""
    val_log_path = model_dir / "val_log_r0.csv"
    return pd.read_csv(val_log_path)

def get_early_stopping_best_epoch(val_df, min_delta=0.001):
    """
    Find the epoch that would trigger early stopping criterion.
    Returns the last epoch where validation loss improved by more than min_delta.

    This matches the early stopping logic in train_with_early_stopping.py:
        if val_loss < (best_val_loss - min_delta):
            # considered an improvement
    """
    best_val_loss = float('inf')
    best_epoch = None
    best_val_loss_at_epoch = None

    for _, row in val_df.sort_values('epoch').iterrows():
        if row['val_loss'] < (best_val_loss - min_delta):
            best_val_loss = row['val_loss']
            best_epoch = row['epoch']
            best_val_loss_at_epoch = row['val_loss']

    return best_epoch, best_val_loss_at_epoch

def load_training_data(model_dir):
    """Load training log and compute per-epoch average loss."""
    train_log_path = model_dir / "log_r0.csv"
    df = pd.read_csv(train_log_path)
    # Group by epoch and compute mean loss
    epoch_losses = df.groupby('epoch')['loss'].mean().reset_index()
    return epoch_losses

def get_best_epoch_metrics(model_dir):
    """Get training and validation metrics at the epoch with best validation loss."""
    # Load validation data
    val_df = load_validation_data(model_dir)

    # Find epoch with minimum validation loss
    best_idx = val_df['val_loss'].idxmin()
    best_epoch = val_df.loc[best_idx, 'epoch']
    best_val_loss = val_df.loc[best_idx, 'val_loss']
    best_val_jloss = val_df.loc[best_idx, 'val_jloss']
    best_val_sloss = val_df.loc[best_idx, 'val_sloss']

    # Load training data and get loss at best epoch
    train_df = load_training_data(model_dir)
    train_loss_at_best = train_df[train_df['epoch'] == best_epoch]['loss'].values[0]

    return {
        'epoch': best_epoch,
        'train_loss': train_loss_at_best,
        'val_loss': best_val_loss,
        'val_jloss': best_val_jloss,
        'val_sloss': best_val_sloss,
    }

def get_fixed_epoch_metrics(model_dir, epochs):
    """Get training and validation metrics at specific fixed epochs."""
    # Load validation data
    val_df = load_validation_data(model_dir)

    # Load training data
    train_df = load_training_data(model_dir)

    metrics = {}
    for epoch in epochs:
        # Check if epoch exists in data
        val_row = val_df[val_df['epoch'] == epoch]
        train_row = train_df[train_df['epoch'] == epoch]

        if not val_row.empty and not train_row.empty:
            metrics[epoch] = {
                'train_loss': train_row['loss'].values[0],
                'val_loss': val_row['val_loss'].values[0],
            }
        else:
            # Use NaN if epoch doesn't exist
            metrics[epoch] = {
                'train_loss': np.nan,
                'val_loss': np.nan,
            }

    return metrics

def plot_performance_comparison(results, save_path=None):
    """Plot training and validation loss at best epoch vs dataset fraction."""
    fractions = [25, 50, 75, 100]
    train_losses = [results[f"{f}%"]['train_loss'] for f in fractions]
    val_losses = [results[f"{f}%"]['val_loss'] for f in fractions]
    best_epochs = [results[f"{f}%"]['epoch'] for f in fractions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss comparison
    ax1.plot(fractions, train_losses, marker='o', linewidth=PLOT_PARAMS["euclid_linewidth"],
             markersize=PLOT_PARAMS["euclid_markersize"],
             label='Training Loss', color='#2E86AB')
    ax1.plot(fractions, val_losses, marker='s', linewidth=PLOT_PARAMS["euclid_linewidth"],
             markersize=PLOT_PARAMS["euclid_markersize"],
             label='Validation Loss', color='#A23B72')
    ax1.set_xlabel('Dataset Fraction (%)', fontsize=PLOT_PARAMS["label_size"])
    ax1.set_ylabel('Loss (at Best Val Epoch)', fontsize=PLOT_PARAMS["label_size"])
    ax1.set_title('Model Performance vs Dataset Size', fontsize=PLOT_PARAMS["title_size"])
    ax1.legend(fontsize=PLOT_PARAMS["legend_size"])
    ax1.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
    ax1.set_xticks(fractions)
    ax1.tick_params(labelsize=PLOT_PARAMS["tick_label_size"], length=PLOT_PARAMS["tick_length"])

    # Annotate with epoch numbers
    for i, (frac, epoch) in enumerate(zip(fractions, best_epochs)):
        ax1.annotate(f'epoch {epoch}',
                    xy=(frac, val_losses[i]),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center', fontsize=PLOT_PARAMS["tick_label_size"], alpha=0.7)

    # Plot 2: Best epoch
    ax2.bar(fractions, best_epochs, color='#F18F01', alpha=0.7, width=8)
    ax2.set_xlabel('Dataset Fraction (%)', fontsize=PLOT_PARAMS["label_size"])
    ax2.set_ylabel('Epoch with Best Val Loss', fontsize=PLOT_PARAMS["label_size"])
    ax2.set_title('Convergence Speed', fontsize=PLOT_PARAMS["title_size"])
    ax2.grid(True, alpha=PLOT_PARAMS["grid_alpha"], axis='y')
    ax2.set_xticks(fractions)
    ax2.tick_params(labelsize=PLOT_PARAMS["tick_label_size"], length=PLOT_PARAMS["tick_length"])

    # Add value labels on bars
    for i, (frac, epoch) in enumerate(zip(fractions, best_epochs)):
        ax2.text(frac, epoch, str(epoch), ha='center', va='bottom', fontsize=PLOT_PARAMS["tick_label_size"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison to {save_path}")

    return fig

def plot_learning_curves(save_path=None):
    """Plot training and validation loss over epochs for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

    for (label, model_dir), color in zip(MODELS.items(), colors):
        # Load validation data
        val_df = load_validation_data(model_dir)

        # Load training data
        train_df = load_training_data(model_dir)

        # Filter training data to only include epochs that match validation epochs
        train_df_sampled = train_df[train_df['epoch'].isin(val_df['epoch'])]

        # Plot training loss (sampled at validation intervals)
        ax1.plot(train_df_sampled['epoch'], train_df_sampled['loss'],
                marker='o', linewidth=PLOT_PARAMS["euclid_linewidth"],
                markersize=PLOT_PARAMS["euclid_markersize"],
                label=f'{label} data', color=color, alpha=0.8)

        # Plot validation loss
        ax2.plot(val_df['epoch'], val_df['val_loss'],
                marker='o', linewidth=PLOT_PARAMS["euclid_linewidth"],
                markersize=PLOT_PARAMS["euclid_markersize"],
                label=f'{label} data', color=color, alpha=0.8)

        # Mark the early stopping best epoch (last significant improvement)
        best_epoch, best_val_loss = get_early_stopping_best_epoch(val_df, min_delta=0.001)
        ax2.scatter([best_epoch], [best_val_loss], s=150, color=color,
                  edgecolors='black', linewidths=2, zorder=10)

        # Mark the same epoch on training plot
        train_at_best = train_df[train_df['epoch'] == best_epoch]['loss'].values[0]
        ax1.scatter([best_epoch], [train_at_best], s=150, color=color,
                  edgecolors='black', linewidths=2, zorder=10)

    # Add legend entry for early stopping markers
    ax1.scatter([], [], s=150, color='gray', edgecolors='black', linewidths=2,
                label='Early Stop Epoch')
    ax2.scatter([], [], s=150, color='gray', edgecolors='black', linewidths=2,
                label='Early Stop Epoch')

    # Configure training loss plot
    ax1.set_xlabel('Epoch', fontsize=PLOT_PARAMS["label_size"])
    ax1.set_ylabel('Training Loss', fontsize=PLOT_PARAMS["label_size"])
    ax1.set_title('Training Loss over Epochs', fontsize=PLOT_PARAMS["title_size"])
    ax1.legend(fontsize=PLOT_PARAMS["legend_size"], loc='best')
    ax1.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
    ax1.tick_params(labelsize=PLOT_PARAMS["tick_label_size"], length=PLOT_PARAMS["tick_length"])

    # Configure validation loss plot
    ax2.set_xlabel('Epoch', fontsize=PLOT_PARAMS["label_size"])
    ax2.set_ylabel('Validation Loss', fontsize=PLOT_PARAMS["label_size"])
    ax2.set_title('Validation Loss over Epochs', fontsize=PLOT_PARAMS["title_size"])
    ax2.legend(fontsize=PLOT_PARAMS["legend_size"], loc='best')
    ax2.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
    ax2.tick_params(labelsize=PLOT_PARAMS["tick_label_size"], length=PLOT_PARAMS["tick_length"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning curves to {save_path}")

    return fig

def plot_fixed_epoch_comparison(fixed_epoch_results, epochs, save_path=None):
    """Plot training and validation losses at fixed epochs vs dataset fraction."""
    fractions = [25, 50, 75, 100]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot for each fixed epoch
    for i, epoch in enumerate(epochs):
        train_losses = []
        val_losses = []

        for frac in fractions:
            metrics = fixed_epoch_results[f"{frac}%"][epoch]
            train_losses.append(metrics['train_loss'])
            val_losses.append(metrics['val_loss'])

        # Plot 1: Training losses
        ax1.plot(fractions, train_losses, marker=markers[i],
                linewidth=PLOT_PARAMS["euclid_linewidth"],
                markersize=PLOT_PARAMS["euclid_markersize"],
                label=f'Epoch {epoch}', color=colors[i])

        # Plot 2: Validation losses
        ax2.plot(fractions, val_losses, marker=markers[i],
                linewidth=PLOT_PARAMS["euclid_linewidth"],
                markersize=PLOT_PARAMS["euclid_markersize"],
                label=f'Epoch {epoch}', color=colors[i])

    # Configure training loss plot
    ax1.set_xlabel('Dataset Fraction (%)', fontsize=PLOT_PARAMS["label_size"])
    ax1.set_ylabel('Training Loss', fontsize=PLOT_PARAMS["label_size"])
    ax1.set_title('Training Loss at Fixed Epochs', fontsize=PLOT_PARAMS["title_size"])
    ax1.legend(fontsize=PLOT_PARAMS["legend_size"])
    ax1.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
    ax1.set_xticks(fractions)
    ax1.tick_params(labelsize=PLOT_PARAMS["tick_label_size"], length=PLOT_PARAMS["tick_length"])

    # Configure validation loss plot
    ax2.set_xlabel('Dataset Fraction (%)', fontsize=PLOT_PARAMS["label_size"])
    ax2.set_ylabel('Validation Loss', fontsize=PLOT_PARAMS["label_size"])
    ax2.set_title('Validation Loss at Fixed Epochs', fontsize=PLOT_PARAMS["title_size"])
    ax2.legend(fontsize=PLOT_PARAMS["legend_size"])
    ax2.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
    ax2.set_xticks(fractions)
    ax2.tick_params(labelsize=PLOT_PARAMS["tick_label_size"], length=PLOT_PARAMS["tick_length"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved fixed epoch comparison to {save_path}")

    return fig

def print_summary_table(results):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("SUMMARY: Model Performance at Best Validation Epoch")
    print("="*80)
    print(f"{'Fraction':<12} {'Best Epoch':<12} {'Train Loss':<12} {'Val Loss':<12} {'Val JLoss':<12} {'Val SLoss':<12}")
    print("-"*80)

    for fraction in ["25%", "50%", "75%", "100%"]:
        r = results[fraction]
        print(f"{fraction:<12} {r['epoch']:<12} {r['train_loss']:<12.5f} {r['val_loss']:<12.5f} {r['val_jloss']:<12.5f} {r['val_sloss']:<12.5f}")

    print("="*80)
    print("\nNote: Metrics shown are from the epoch with the best (minimum) validation loss.")
    print("Stars (*) on learning curves plot indicate the best epoch for each model.\n")

def main():
    """Main function to generate all plots and summaries."""
    print("Loading data and computing metrics...")

    # Fixed epochs to compare
    fixed_epochs = [15, 25, 35]

    # Collect results for all models
    results = {}
    fixed_epoch_results = {}
    for label, model_dir in MODELS.items():
        print(f"Processing {label} model...")
        results[label] = get_best_epoch_metrics(model_dir)
        fixed_epoch_results[label] = get_fixed_epoch_metrics(model_dir, fixed_epochs)

    # Print summary table
    print_summary_table(results)

    # Generate plots
    print("\nGenerating plots...")
    output_dir = Path(__file__).parent
    plot_performance_comparison(results, save_path=output_dir / "performance_comparison.png")
    plot_learning_curves(save_path=output_dir / "learning_curves.png")
    plot_fixed_epoch_comparison(fixed_epoch_results, fixed_epochs, save_path=output_dir / "fixed_epoch_comparison.png")

    print("\nDone! Plots saved to:")
    print(f"  - {output_dir / 'performance_comparison.png'}")
    print(f"  - {output_dir / 'learning_curves.png'}")
    print(f"  - {output_dir / 'fixed_epoch_comparison.png'}")

    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
