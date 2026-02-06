#!/usr/bin/env python3
"""
Run All Diagnostics

Orchestrates running all three diagnostic probes (A, B, C) across all checkpoints.
Produces a results.csv file with per-epoch metrics.

Usage:
    python run_all_diagnostics.py \
        --checkpoint_dir /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics \
        --output_dir /home/s185927/thesis/experiments/03_checkpoint_diagnostics/diagnostics
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from probe_a_energy_landscape import run_probe_a, generate_or_load_actions, set_seed
from probe_b_onpolicy_loss import run_probe_b
from probe_c_planning import run_probe_c


def discover_checkpoints(checkpoint_dir):
    """Discover all checkpoint files in directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []

    # Find epoch checkpoints (e0.pt, e5.pt, etc.)
    for ckpt_file in checkpoint_dir.glob("e*.pt"):
        match = re.match(r"e(\d+)\.pt", ckpt_file.name)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((str(ckpt_file), epoch))

    # Sort by epoch
    checkpoints = sorted(checkpoints, key=lambda x: x[1])
    return checkpoints


def load_val_loss_from_log(log_dir, epoch):
    """Load validation loss for a specific epoch from training logs."""
    val_log_path = Path(log_dir) / "val_log_r0.csv"
    if not val_log_path.exists():
        return None

    with open(val_log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get('epoch', -1)) == epoch:
                return float(row.get('val_loss', row.get('loss', 0)))
    return None


def main():
    parser = argparse.ArgumentParser(description='Run All Checkpoint Diagnostics')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoints')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--config', type=str,
                        default='/home/s185927/thesis/experiments/03_checkpoint_diagnostics/configs/x_axis_finetune_100pct_diagnostics.yaml',
                        help='Path to training config')
    parser.add_argument('--probe_ids_file', type=str,
                        default='/home/s185927/thesis/experiments/03_checkpoint_diagnostics/diagnostics/probe_ids.txt',
                        help='Path to probe_ids.txt')
    parser.add_argument('--metadata', type=str,
                        default='/data/s185927/droid_sim/axis_aligned/x_axis/trajectory_metadata.json',
                        help='Path to trajectory metadata')
    parser.add_argument('--data_dir', type=str,
                        default='/data/s185927/droid_sim/axis_aligned/x_axis',
                        help='Path to trajectory data')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nsamples', type=int, default=512,
                        help='Number of action samples for Probe A')
    parser.add_argument('--probes', type=str, nargs='+', default=['A', 'B', 'C'],
                        help='Which probes to run (A, B, C)')
    parser.add_argument('--resume_from_epoch', type=int, default=None,
                        help='Resume from specific epoch (skip earlier epochs)')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Load probe IDs
    probe_ids = []
    with open(args.probe_ids_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                probe_ids.append(int(line))
    print(f"Loaded {len(probe_ids)} probe IDs")

    # Generate or load actions for Probe A
    actions_path = output_dir / f"actions_seed_{args.seed}_M{args.nsamples}.npy"
    actions = generate_or_load_actions(str(actions_path), args.nsamples, seed=args.seed)

    # Discover checkpoints
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints")

    if not checkpoints:
        print("No checkpoints found!")
        return

    # Filter if resuming
    if args.resume_from_epoch is not None:
        checkpoints = [(p, e) for p, e in checkpoints if e >= args.resume_from_epoch]
        print(f"Resuming from epoch {args.resume_from_epoch}, {len(checkpoints)} checkpoints remaining")

    # Results storage
    results = []
    results_csv_path = output_dir / "results.csv"

    # Load existing results if resuming
    if args.resume_from_epoch is not None and results_csv_path.exists():
        with open(results_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['epoch']) < args.resume_from_epoch:
                    # Convert back to proper types
                    converted_row = {'epoch': int(row['epoch'])}
                    for k, v in row.items():
                        if k != 'epoch':
                            try:
                                converted_row[k] = float(v)
                            except:
                                converted_row[k] = v
                    results.append(converted_row)

    # Process each checkpoint
    for ckpt_path, epoch in checkpoints:
        print(f"\n{'='*60}")
        print(f"Processing epoch {epoch}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        epoch_result = {'epoch': epoch}

        # Get validation loss from training log
        val_loss = load_val_loss_from_log(args.checkpoint_dir, epoch)
        epoch_result['val_loss_dataset'] = val_loss if val_loss is not None else -1

        raw_data = {}

        # Probe A: Energy Landscape
        if 'A' in args.probes:
            print(f"\n--- Probe A: Energy Landscape ---")
            try:
                aggregated_a, raw_a = run_probe_a(
                    checkpoint_path=ckpt_path,
                    config_path=args.config,
                    probe_ids=probe_ids,
                    data_dir=args.data_dir,
                    actions=actions,
                    device=args.device,
                )
                epoch_result['sharpness_stdE'] = aggregated_a['sharpness_stdE']
                epoch_result['sharpness_rangeE'] = aggregated_a['sharpness_rangeE']
                epoch_result['align_cos'] = aggregated_a['align_cos']
                raw_data['probe_a'] = raw_a
                print(f"  sharpness_stdE: {aggregated_a['sharpness_stdE']:.4f}")
                print(f"  sharpness_rangeE: {aggregated_a['sharpness_rangeE']:.4f}")
                print(f"  align_cos: {aggregated_a['align_cos']:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                epoch_result['sharpness_stdE'] = -1
                epoch_result['sharpness_rangeE'] = -1
                epoch_result['align_cos'] = -1

            # Clear GPU memory
            torch.cuda.empty_cache()

        # Probe B: On-Policy Prediction Loss
        if 'B' in args.probes:
            print(f"\n--- Probe B: On-Policy Prediction Loss ---")
            try:
                aggregated_b, raw_b = run_probe_b(
                    checkpoint_path=ckpt_path,
                    config_path=args.config,
                    probe_ids=probe_ids,
                    metadata_path=args.metadata,
                    device=args.device,
                    k_visit=2,
                )
                epoch_result['onpolicy_pred_loss'] = aggregated_b['onpolicy_pred_loss']
                raw_data['probe_b'] = raw_b
                print(f"  onpolicy_pred_loss: {aggregated_b['onpolicy_pred_loss']:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                epoch_result['onpolicy_pred_loss'] = -1

            torch.cuda.empty_cache()

        # Probe C: Planning
        if 'C' in args.probes:
            print(f"\n--- Probe C: Planning ---")
            try:
                aggregated_c, raw_c = run_probe_c(
                    checkpoint_path=ckpt_path,
                    config_path=args.config,
                    probe_ids=probe_ids,
                    metadata_path=args.metadata,
                    device=args.device,
                    planning_steps=5,
                )
                epoch_result['planning_final_dist_mean'] = aggregated_c['planning_final_dist_mean']
                epoch_result['planning_delta_per_step'] = aggregated_c['planning_delta_per_step']
                raw_data['probe_c'] = raw_c
                print(f"  planning_final_dist_mean: {aggregated_c['planning_final_dist_mean']:.4f}")
                print(f"  planning_delta_per_step: {aggregated_c['planning_delta_per_step']:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                epoch_result['planning_final_dist_mean'] = -1
                epoch_result['planning_delta_per_step'] = -1

            torch.cuda.empty_cache()

        # Save raw data for this epoch
        raw_path = raw_dir / f"epoch_{epoch}_per_probe.json"
        with open(raw_path, 'w') as f:
            json.dump(raw_data, f, indent=2)

        results.append(epoch_result)

        # Update results CSV after each epoch (for checkpointing)
        fieldnames = ['epoch', 'val_loss_dataset', 'sharpness_stdE', 'sharpness_rangeE',
                     'align_cos', 'onpolicy_pred_loss', 'planning_final_dist_mean',
                     'planning_delta_per_step']
        with open(results_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        print(f"\nSaved results to {results_csv_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("DIAGNOSTICS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_csv_path}")
    print(f"Raw data saved to: {raw_dir}")
    print(f"Total epochs processed: {len(results)}")


if __name__ == '__main__':
    main()
