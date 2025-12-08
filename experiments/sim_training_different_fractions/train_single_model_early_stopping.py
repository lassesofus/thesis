#!/usr/bin/env python3
"""
Standalone training script with early stopping.

Usage:
    python scripts/train_single_model_early_stopping.py \
        --fname configs/train/vitg16/ablation/droid-256px-8f_010pct.yaml \
        --devices cuda:0
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, '/home/s185927/thesis/vjepa2')

# Import the early stopping training function
from app.vjepa_droid.train_with_early_stopping import main_with_early_stopping


def main():
    parser = argparse.ArgumentParser(description='Train V-JEPA with early stopping')
    parser.add_argument(
        '--fname',
        type=str,
        required=True,
        help='Path to config YAML file',
    )
    parser.add_argument(
        '--devices',
        type=str,
        default='cuda:0',
        help='Device to use (e.g., cuda:0)',
    )
    parser.add_argument(
        '--debugmode',
        action='store_true',
        help='Debug mode (single process)',
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.fname}")
    with open(args.fname, 'r') as f:
        config = yaml.safe_load(f)

    # Verify early stopping params exist
    meta = config.get('meta', {})
    if 'early_stopping_patience' not in meta:
        print("WARNING: early_stopping_patience not in config. Using default=10")
        meta['early_stopping_patience'] = 10
    if 'val_freq' not in meta:
        print("WARNING: val_freq not in config. Using default=1")
        meta['val_freq'] = 1

    # Verify validation data exists
    if 'val_data' not in config:
        print("ERROR: Config must have 'val_data' section for validation!")
        print("Run: python scripts/add_early_stopping_to_configs.py")
        sys.exit(1)

    print(f"\nEarly Stopping Configuration:")
    print(f"  Patience: {meta.get('early_stopping_patience', 10)} epochs")
    print(f"  Validation frequency: every {meta.get('val_freq', 1)} epoch(s)")
    print(f"  Min delta: {meta.get('early_stopping_min_delta', 0.0001)}")
    print(f"\nOutput folder: {config['folder']}")
    print()

    # Set CUDA device
    device_id = args.devices.split(':')[-1] if ':' in args.devices else '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    # Run training with early stopping
    print("Starting training with early stopping...")
    main_with_early_stopping(args=config, resume_preempt=False)


if __name__ == '__main__':
    main()
