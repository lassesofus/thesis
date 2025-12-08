#!/usr/bin/env python3
"""
Create training data splits for ablation study.

This script:
1. Reads the full training CSV
2. Creates a train/val split (default 80/20)
3. Generates 10 different training sets with 10%, 20%, ..., 100% of training data
4. Keeps validation set constant across all experiments
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def create_data_splits(
    input_csv: str,
    output_dir: str,
    val_fraction: float = 0.2,
    seed: int = 42,
    percentages: list = None,
):
    """
    Create train/val splits and varying fractions of training data.

    Args:
        input_csv: Path to original train_trajectories.csv
        output_dir: Directory to save split CSV files
        val_fraction: Fraction of data to use for validation (default 0.2 = 20%)
        seed: Random seed for reproducibility
        percentages: List of percentages to create (default [10, 20, ..., 100])
    """
    if percentages is None:
        percentages = list(range(10, 101, 10))

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Read the input CSV (just paths, no header)
    print(f"Reading {input_csv}...")
    with open(input_csv, 'r') as f:
        all_paths = [line.strip() for line in f if line.strip()]

    total_episodes = len(all_paths)
    print(f"Total episodes: {total_episodes}")

    # Shuffle the data
    indices = np.arange(total_episodes)
    np.random.shuffle(indices)
    shuffled_paths = [all_paths[i] for i in indices]

    # Split into train and validation
    val_size = int(total_episodes * val_fraction)
    train_size = total_episodes - val_size

    val_paths = shuffled_paths[:val_size]
    train_paths = shuffled_paths[val_size:]

    print(f"Train episodes: {train_size}")
    print(f"Val episodes: {val_size}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save validation set (constant across all experiments)
    val_csv = output_path / "val_trajectories.csv"
    with open(val_csv, 'w') as f:
        for path in val_paths:
            f.write(f"{path}\n")
    print(f"Saved validation set: {val_csv}")

    # Create training sets with different percentages
    for pct in percentages:
        # Calculate how many episodes for this percentage
        n_episodes = int(train_size * pct / 100.0)
        n_episodes = max(1, n_episodes)  # At least 1 episode

        # Sample episodes (without replacement, preserving order)
        subset_paths = train_paths[:n_episodes]

        # Save to CSV
        train_csv = output_path / f"train_{pct:03d}pct.csv"
        with open(train_csv, 'w') as f:
            for path in subset_paths:
                f.write(f"{path}\n")

        print(f"Created {pct}% split: {n_episodes} episodes -> {train_csv}")

    # Save metadata
    metadata = {
        "total_episodes": total_episodes,
        "train_size": train_size,
        "val_size": val_size,
        "val_fraction": val_fraction,
        "seed": seed,
        "percentages": percentages,
    }

    metadata_file = output_path / "split_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("Data Split Metadata\n")
        f.write("=" * 50 + "\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"\nMetadata saved to: {metadata_file}")
    print(f"\nAll data splits saved to: {output_dir}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Create data splits for V-JEPA training")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/data/s185927/droid_sim/y_axis/train_trajectories.csv",
        help="Path to input training CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/s185927/droid_sim/y_axis/splits",
        help="Directory to save split CSV files",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default 0.2 = 20%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--percentages",
        type=int,
        nargs="+",
        default=None,
        help="Percentages to create (default: 10 20 30 ... 100)",
    )

    args = parser.parse_args()

    create_data_splits(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        percentages=args.percentages,
    )


if __name__ == "__main__":
    main()
