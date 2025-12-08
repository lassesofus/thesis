#!/usr/bin/env python3
"""
Add early stopping parameters to all ablation configs.
"""

import argparse
from pathlib import Path

import yaml


def add_early_stopping(config_path, patience=10, val_freq=1, min_delta=0.0001):
    """Add early stopping parameters to a config file."""

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Add early stopping parameters to meta section
    if 'meta' not in config:
        config['meta'] = {}

    config['meta']['early_stopping_patience'] = patience
    config['meta']['val_freq'] = val_freq
    config['meta']['early_stopping_min_delta'] = min_delta

    # Write back
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_dir',
        type=str,
        default='/home/s185927/thesis/vjepa2/configs/train/vitg16/ablation',
        help='Directory containing config files',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (epochs without improvement)',
    )
    parser.add_argument(
        '--val_freq',
        type=int,
        default=1,
        help='Validation frequency (validate every N epochs)',
    )
    parser.add_argument(
        '--min_delta',
        type=float,
        default=0.0001,
        help='Minimum improvement to count as progress',
    )

    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    configs = list(config_dir.glob('*.yaml'))

    print(f"Adding early stopping parameters to {len(configs)} configs...")
    print(f"  Patience: {args.patience} epochs")
    print(f"  Validation frequency: every {args.val_freq} epoch(s)")
    print(f"  Minimum delta: {args.min_delta}")
    print()

    for config_path in sorted(configs):
        config = add_early_stopping(
            config_path,
            patience=args.patience,
            val_freq=args.val_freq,
            min_delta=args.min_delta,
        )
        print(f"Updated: {config_path.name}")

    print(f"\nAll configs updated!")


if __name__ == '__main__':
    main()
