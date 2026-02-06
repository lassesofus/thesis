#!/usr/bin/env python3
"""
Verify that generated DROID-compatible data can be loaded by V-JEPA2's DROIDVideoDataset.

EXAMPLE USAGE:
    python verify_droid_data.py --csv_path /tmp/test_droid_train.csv
"""

import os
import sys
import click

# Add V-JEPA2 to path
_vjepa_root = "/home/s185927/thesis/vjepa2"
if os.path.isdir(_vjepa_root) and _vjepa_root not in sys.path:
    sys.path.insert(0, _vjepa_root)

import torch
import numpy as np
from app.vjepa_droid.droid import DROIDVideoDataset


@click.command()
@click.option('--csv_path', type=str, required=True, help='Path to CSV file with trajectory paths')
@click.option('--num_samples', type=int, default=5, help='Number of samples to test loading')
def main(csv_path, num_samples):
    """Verify DROID-compatible data can be loaded by DROIDVideoDataset."""

    print("=" * 80)
    print("DROID Data Verification")
    print("=" * 80)
    print(f"CSV path: {csv_path}")
    print(f"Testing {num_samples} samples...")
    print()

    # Create dataset
    try:
        dataset = DROIDVideoDataset(
            data_path=csv_path,
            frames_per_clip=8,  # Match V-JEPA2 config
            frameskip=2,  # Tubelet size
            fps=4,  # Target FPS
            camera_views=['left_mp4_path'],
            transform=None,
            camera_frame=False,
        )
        print(f"✓ Dataset created successfully")
        print(f"  Total trajectories: {len(dataset)}")
        print()
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test loading samples
    print("Testing sample loading...")
    for i in range(min(num_samples, len(dataset))):
        try:
            buffer, actions, states, extrinsics, indices = dataset[i]

            print(f"\nSample {i}:")
            print(f"  Video buffer shape: {buffer.shape}")  # [C, T, H, W] or similar
            print(f"  States shape: {states.shape}")  # [T, 7]
            print(f"  Actions shape: {actions.shape}")  # [T-1, 7]
            print(f"  Extrinsics shape: {extrinsics.shape}")  # [T, 6]
            print(f"  Indices: {indices}")

            # Print some statistics
            print(f"  Video range: [{buffer.min():.3f}, {buffer.max():.3f}]")
            print(f"  Position (xyz): {states[0, :3]}")
            print(f"  Gripper: {states[0, 6]:.3f}")

            print(f"  ✓ Sample loaded successfully")

        except Exception as e:
            print(f"\n✗ Failed to load sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
