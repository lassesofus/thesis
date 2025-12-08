#!/usr/bin/env python3
"""
Plot comparison of V-JEPA model performance on test samples.

Visualizes mean final distance and success rates across models.

EXAMPLE USAGE:
    python plot_model_comparison.py \
        --eval_dir /data/s185927/vjepa_eval_results \
        --out_path /data/s185927/vjepa_eval_results/model_comparison.png

    # Also plot distance evolution over planning steps
    python plot_model_comparison.py \
        --eval_dir /data/s185927/vjepa_eval_results \
        --out_path /data/s185927/vjepa_eval_results/model_comparison.png \
        --plot_evolution
"""

import os
import json
from pathlib import Path

import click
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use non-interactive backend for headless systems
mpl.use('Agg')


def load_results(eval_dir):
    """Load all model results from evaluation directory."""
    eval_path = Path(eval_dir)
    
    # Load summary
    summary_path = eval_path / "evaluation_summary.json"
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Load detailed results for each model
    detailed_results = {}
    for model_name in summary.keys():
        result_path = eval_path / f"{model_name}_results.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                detailed_results[model_name] = json.load(f)
    
    return summary, detailed_results


def plot_bar_comparison(summary, out_path):
    """Create bar plot comparing mean final distances."""
    
    # Sort models by performance
    model_names = []
    mean_dists = []
    std_dists = []
    success_rates = []
    
    # Sort by mean distance (best first)
    sorted_models = sorted(summary.items(), key=lambda x: x[1]['mean_final_distance'])
    
    for model_name, stats in sorted_models:
        # Format model name for display
        if 'pct' in model_name:
            # Extract percentage from name like '4.8.vitg16-256px-8f_025pct'
            parts = model_name.split('_')
            if len(parts) > 1:
                pct = parts[-1].replace('pct', '')
                display_name = f"{pct}% trained"
            else:
                display_name = model_name
        elif 'baseline' in model_name.lower():
            display_name = "Meta Baseline"
        else:
            display_name = model_name
        
        model_names.append(display_name)
        mean_dists.append(stats['mean_final_distance'])
        std_dists.append(stats['std_final_distance'])
        success_rates.append(stats['success_rate'] * 100)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mean final distance
    x = np.arange(len(model_names))
    bars1 = ax1.bar(x, mean_dists, yerr=std_dists, capsize=5, alpha=0.7,
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(model_names)])
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Final Distance (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Error Distance After 5 Planning Steps', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.05, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Success Threshold (5cm)')
    ax1.legend()
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars1, mean_dists, std_dists)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.002,
                f'{val:.3f}m', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Success rate
    bars2 = ax2.bar(x, success_rates, alpha=0.7,
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(model_names)])
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Success Rate (Distance < 5cm)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, success_rates)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved bar comparison: {out_path}")
    plt.close()


def plot_distance_evolution(detailed_results, out_path):
    """Plot how distance evolves over planning steps for each model."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Sort models by final performance
    model_performance = {}
    for model_name, results in detailed_results.items():
        all_dists = [r['distances_per_step'] for r in results]
        # Average across all samples at each step
        max_len = max(len(d) for d in all_dists)
        avg_per_step = []
        for step in range(max_len):
            step_dists = [d[step] for d in all_dists if len(d) > step]
            avg_per_step.append(np.mean(step_dists))
        model_performance[model_name] = avg_per_step[-1]  # Final distance
    
    sorted_models = sorted(model_performance.items(), key=lambda x: x[1])
    
    for idx, (model_name, _) in enumerate(sorted_models):
        results = detailed_results[model_name]
        
        # Format name
        if 'pct' in model_name:
            parts = model_name.split('_')
            if len(parts) > 1:
                pct = parts[-1].replace('pct', '')
                display_name = f"{pct}% trained"
            else:
                display_name = model_name
        elif 'baseline' in model_name.lower():
            display_name = "Meta Baseline"
        else:
            display_name = model_name
        
        # Compute average distance per step across all samples
        all_distances = [r['distances_per_step'] for r in results]
        max_steps = max(len(d) for d in all_distances)
        
        mean_per_step = []
        std_per_step = []
        for step in range(max_steps):
            step_distances = [d[step] for d in all_distances if len(d) > step]
            mean_per_step.append(np.mean(step_distances))
            std_per_step.append(np.std(step_distances))
        
        steps = np.arange(len(mean_per_step))
        
        # Plot mean with std shading
        ax.plot(steps, mean_per_step, marker='o', linewidth=2, label=display_name,
               color=colors[idx % len(colors)])
        ax.fill_between(steps,
                        np.array(mean_per_step) - np.array(std_per_step),
                        np.array(mean_per_step) + np.array(std_per_step),
                        alpha=0.2, color=colors[idx % len(colors)])
    
    ax.axhline(y=0.05, color='r', linestyle='--', linewidth=2, alpha=0.7,
              label='Success Threshold (5cm)')
    ax.set_xlabel('Planning Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance to Target (m)', fontsize=12, fontweight='bold')
    ax.set_title('Distance Evolution During CEM Planning', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved evolution plot: {out_path}")
    plt.close()


@click.command(help=__doc__)
@click.option('--eval_dir', type=str, required=True,
              help='Directory containing evaluation results')
@click.option('--out_path', type=str, required=True,
              help='Output path for plot (PNG)')
@click.option('--plot_evolution', is_flag=True, default=False,
              help='Also create distance evolution plot')
def main(eval_dir, out_path, plot_evolution):
    """Create comparison plots from evaluation results."""
    
    print(f"Loading results from: {eval_dir}")
    summary, detailed_results = load_results(eval_dir)
    
    print(f"\nFound {len(summary)} models:")
    for model_name in summary.keys():
        print(f"  - {model_name}")
    
    # Create main comparison plot
    print(f"\nCreating bar comparison plot...")
    plot_bar_comparison(summary, out_path)
    
    # Create evolution plot if requested
    if plot_evolution:
        evolution_path = Path(out_path).parent / (Path(out_path).stem + "_evolution.png")
        print(f"\nCreating distance evolution plot...")
        plot_distance_evolution(detailed_results, evolution_path)
    
    print(f"\nDone! Plots saved.")


if __name__ == '__main__':
    main()
