#!/usr/bin/env python3
"""
Create training config files for each data percentage.

This script generates YAML configs for training with different fractions
of training data, each with validation support.
"""

import argparse
import os
from pathlib import Path

import yaml


def create_configs(
    base_config: str,
    output_dir: str,
    splits_dir: str,
    percentages: list = None,
):
    """
    Create config files for each data percentage.

    Args:
        base_config: Path to base config YAML
        output_dir: Directory to save new configs
        splits_dir: Directory containing data split CSVs
        percentages: List of percentages (default [10, 20, ..., 100])
    """
    if percentages is None:
        percentages = list(range(10, 101, 10))

    # Load base config
    print(f"Loading base config: {base_config}")
    with open(base_config, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits_path = Path(splits_dir)

    for pct in percentages:
        # Create a copy of base config
        cfg = base_cfg.copy()

        # Update datasets to use the split CSVs
        train_csv = str(splits_path / f"train_{pct:03d}pct.csv")
        val_csv = str(splits_path / "val_trajectories.csv")

        # Update data section
        cfg['data']['datasets'] = [train_csv]

        # Add validation dataset (create new section if needed)
        if 'val_data' not in cfg:
            cfg['val_data'] = cfg['data'].copy()
        cfg['val_data']['datasets'] = [val_csv]

        # Update output folder to include percentage
        base_folder = cfg['folder']
        cfg['folder'] = f"{base_folder}_{pct:03d}pct"

        # Adjust training epochs based on data size
        # Smaller datasets might need different training schedules
        # For now, keep same epochs but you can adjust this

        # Save config
        output_file = output_path / f"droid-256px-8f_{pct:03d}pct.yaml"
        with open(output_file, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        print(f"Created config for {pct}%: {output_file}")
        print(f"  Train: {train_csv}")
        print(f"  Val: {val_csv}")
        print(f"  Output: {cfg['folder']}")

    print(f"\nAll configs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create V-JEPA training configs")
    parser.add_argument(
        "--base_config",
        type=str,
        default="/home/s185927/thesis/vjepa2/configs/train/vitg16/droid-256px-8f.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/s185927/thesis/vjepa2/configs/train/vitg16/ablation",
        help="Directory to save new configs",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/data/s185927/droid_sim/y_axis/splits",
        help="Directory containing data split CSVs",
    )
    parser.add_argument(
        "--percentages",
        type=int,
        nargs="+",
        default=None,
        help="Percentages to create configs for (default: 10 20 30 ... 100)",
    )

    args = parser.parse_args()

    create_configs(
        base_config=args.base_config,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        percentages=args.percentages,
    )


if __name__ == "__main__":
    main()
