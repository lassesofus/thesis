#!/usr/bin/env python3
"""
Analyze generated trajectory metadata to verify diversity.

Usage:
    python analyze_trajectories.py /data/s185927/droid_sim/x_axis/trajectory_metadata.json
"""

import json
import sys
import numpy as np
from pathlib import Path


def analyze_metadata(metadata_path):
    """Analyze trajectory metadata and print statistics."""

    print("=" * 80)
    print("Trajectory Metadata Analysis")
    print("=" * 80)
    print(f"File: {metadata_path}\n")

    # Load metadata
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {metadata_path}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON file: {metadata_path}")
        return

    if not metadata:
        print("ERROR: Empty metadata file")
        return

    print(f"Total trajectories: {len(metadata)}\n")

    # Split statistics
    train = [m for m in metadata if m['split'] == 'train']
    test = [m for m in metadata if m['split'] == 'test']
    print(f"Train: {len(train)} ({100*len(train)/len(metadata):.1f}%)")
    print(f"Test:  {len(test)} ({100*len(test)/len(metadata):.1f}%)\n")

    # Extract data
    target_positions = np.array([m['target_position'] for m in metadata])
    target_distances = np.array([m['target_distance'] for m in metadata])
    final_distances = np.array([m['final_distance'] for m in metadata])
    successes = np.array([m['success'] for m in metadata])
    directions = [m['trajectory_direction'] for m in metadata]

    # Target distance statistics
    print("Target Distance Statistics:")
    print(f"  Min:    {np.min(target_distances):.4f}m")
    print(f"  Max:    {np.max(target_distances):.4f}m")
    print(f"  Mean:   {np.mean(target_distances):.4f}m")
    print(f"  Median: {np.median(target_distances):.4f}m")
    print(f"  Std:    {np.std(target_distances):.4f}m")
    print(f"  Unique values: {len(np.unique(target_distances))}/{len(target_distances)}\n")

    # Final distance statistics (IK baseline performance)
    print("Final Distance Statistics (IK Baseline):")
    print(f"  Min:    {np.min(final_distances):.4f}m")
    print(f"  Max:    {np.max(final_distances):.4f}m")
    print(f"  Mean:   {np.mean(final_distances):.4f}m")
    print(f"  Median: {np.median(final_distances):.4f}m")
    print(f"  Std:    {np.std(final_distances):.4f}m\n")

    # Success rate
    success_rate = np.mean(successes) * 100
    print(f"IK Success Rate: {np.sum(successes)}/{len(successes)} ({success_rate:.1f}%)\n")

    # Direction-specific analysis
    unique_dirs = set(directions)
    if len(unique_dirs) == 1:
        direction = list(unique_dirs)[0]
        print(f"Trajectory Direction: {direction.upper()}-axis")

        if direction != 'sphere':
            # Analyze position variation along the specified axis
            dim_idx = {'x': 0, 'y': 1, 'z': 2}[direction.lower()]
            start_pos = target_positions[0] - np.array([0, 0, 0])  # Approximate
            deltas = target_positions[:, dim_idx] - target_positions[0, dim_idx]

            print(f"\nPosition variation along {direction.upper()}-axis:")
            print(f"  Min delta:  {np.min(deltas):.4f}m")
            print(f"  Max delta:  {np.max(deltas):.4f}m")
            print(f"  Mean delta: {np.mean(deltas):.4f}m")
            print(f"  Std delta:  {np.std(deltas):.4f}m")

            # Check for negative deltas (bidirectional)
            num_positive = np.sum(deltas > 0)
            num_negative = np.sum(deltas < 0)
            num_zero = np.sum(np.abs(deltas) < 1e-6)

            print(f"\nDirection distribution:")
            print(f"  Positive {direction}: {num_positive} ({100*num_positive/len(deltas):.1f}%)")
            print(f"  Negative {direction}: {num_negative} ({100*num_negative/len(deltas):.1f}%)")
            if num_zero > 0:
                print(f"  Near-zero: {num_zero} ({100*num_zero/len(deltas):.1f}%)")

    else:
        print(f"Multiple directions found: {unique_dirs}")

    # Sample some trajectories
    print("\n" + "=" * 80)
    print("Sample Trajectories (first 10):")
    print("=" * 80)
    for i in range(min(10, len(metadata))):
        m = metadata[i]
        print(f"Traj {m['trajectory_index']:4d} [{m['split']:5s}]: "
              f"target_dist={m['target_distance']:6.3f}m, "
              f"final_dist={m['final_distance']:6.3f}m, "
              f"success={'✓' if m['success'] else '✗'}, "
              f"pos={m['target_position']}")

    # Distribution histogram (text-based)
    print("\n" + "=" * 80)
    print("Target Distance Distribution (histogram):")
    print("=" * 80)

    # Create 10 bins
    num_bins = 10
    hist, bin_edges = np.histogram(target_distances, bins=num_bins)
    max_count = np.max(hist)

    for i in range(num_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        count = hist[i]
        bar_width = int(50 * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_width
        print(f"  [{left:.3f}, {right:.3f}): {count:4d} {bar}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python analyze_trajectories.py <trajectory_metadata.json>")
        print("\nExample:")
        print("  python analyze_trajectories.py /data/s185927/droid_sim/x_axis/trajectory_metadata.json")
        sys.exit(1)

    metadata_path = sys.argv[1]
    analyze_metadata(metadata_path)
