"""
DROID Action Distribution Analysis

Analyzes translational velocity correlations and off-axis motion patterns
in the DROID dataset to understand learned action biases.

Usage:
    python analyze_action_distribution.py [--n_trajectories 5000] [--save_plots]
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

DROID_DATA_PATH = Path("/data/droid_raw/1.0.1")
OUTPUT_DIR = Path(__file__).parent


def collect_trajectories(base_path: Path, n_trajectories: int) -> list[Path]:
    """Collect trajectory file paths from DROID dataset."""
    trajectory_files = list(base_path.rglob("trajectory.h5"))
    print(f"Found {len(trajectory_files)} total trajectories")

    if len(trajectory_files) > n_trajectories:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(trajectory_files), size=n_trajectories, replace=False)
        trajectory_files = [trajectory_files[i] for i in indices]

    print(f"Using {len(trajectory_files)} trajectories for analysis")
    return trajectory_files


def extract_velocities(trajectory_files: list[Path]) -> np.ndarray:
    """Extract translational velocities (dx, dy, dz) from trajectories."""
    all_velocities = []

    for i, traj_path in enumerate(trajectory_files):
        if (i + 1) % 500 == 0:
            print(f"Processing trajectory {i + 1}/{len(trajectory_files)}")

        try:
            with h5py.File(traj_path, "r") as f:
                # Get Cartesian positions
                positions = f["observation/robot_state/cartesian_position"][:]
                # Compute velocities as position differences
                velocities = np.diff(positions[:, :3], axis=0)  # Only translation (x, y, z)
                all_velocities.append(velocities)
        except Exception as e:
            print(f"Error processing {traj_path}: {e}")
            continue

    return np.vstack(all_velocities)


def compute_correlation_matrix(velocities: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix for velocity components."""
    return np.corrcoef(velocities.T)


def compute_conditional_statistics(velocities: np.ndarray, threshold: float = 0.005):
    """
    Compute statistics when one axis dominates.

    For each axis, select samples where:
    - |v_i| > threshold
    - |v_i| > |v_j| for all j != i
    """
    results = {}
    axis_names = ["x", "y", "z"]

    for i, axis in enumerate(axis_names):
        # Get absolute velocities
        abs_vel = np.abs(velocities)

        # Mask: primary axis exceeds threshold and is larger than other axes
        mask = abs_vel[:, i] > threshold
        for j in range(3):
            if j != i:
                mask &= abs_vel[:, i] > abs_vel[:, j]

        n_samples = mask.sum()
        if n_samples == 0:
            continue

        # Compute mean absolute velocities under this condition
        mean_abs = np.mean(abs_vel[mask], axis=0)

        # Compute off-axis ratio: sum of off-axis / primary
        off_axis_sum = sum(mean_abs[j] for j in range(3) if j != i)
        off_axis_ratio = off_axis_sum / mean_abs[i]

        results[axis] = {
            "n_samples": n_samples,
            "mean_abs_dx": mean_abs[0],
            "mean_abs_dy": mean_abs[1],
            "mean_abs_dz": mean_abs[2],
            "off_axis_ratio": off_axis_ratio,
        }

    return results


def compute_forward_x_statistics(velocities: np.ndarray, threshold: float = 0.005):
    """Analyze coupling patterns for forward x motion."""
    dx, dy, dz = velocities[:, 0], velocities[:, 1], velocities[:, 2]

    # Forward x motion
    mask = dx > threshold
    n_forward = mask.sum()

    if n_forward == 0:
        return None

    # Use half the threshold for classifying dz as "substantial"
    dz_threshold = threshold / 2
    return {
        "n_samples": n_forward,
        "mean_dz": np.mean(dz[mask]),
        "frac_dz_negative": np.mean(dz[mask] < -dz_threshold),
        "frac_dz_positive": np.mean(dz[mask] > dz_threshold),
        "mean_dy": np.mean(dy[mask]),
    }


def print_results(
    n_samples: int,
    correlation_matrix: np.ndarray,
    conditional_stats: dict,
    forward_x_stats: dict | None,
):
    """Print analysis results in a formatted way."""
    print("\n" + "=" * 60)
    print(f"DROID Action Distribution Analysis ({n_samples:,} samples)")
    print("=" * 60)

    print("\n1. Correlation Matrix (dx, dy, dz):")
    print("-" * 40)
    print("      dx       dy       dz")
    for i, axis in enumerate(["dx", "dy", "dz"]):
        row = "  ".join(f"{correlation_matrix[i, j]:+.3f}" for j in range(3))
        print(f"{axis}  {row}")

    print("\n2. Conditional Statistics (dominant axis > 5mm):")
    print("-" * 40)
    print(f"{'Axis':<8} {'n':<10} {'|dx|':<8} {'|dy|':<8} {'|dz|':<8} {'Off-axis ratio':<15}")
    for axis, stats in conditional_stats.items():
        print(
            f"{axis:<8} {stats['n_samples']:<10,} "
            f"{stats['mean_abs_dx']:<8.3f} {stats['mean_abs_dy']:<8.3f} {stats['mean_abs_dz']:<8.3f} "
            f"{stats['off_axis_ratio']:<15.2f}"
        )

    if forward_x_stats:
        print("\n3. Forward X Motion Analysis (dx > 5mm):")
        print("-" * 40)
        print(f"  n samples: {forward_x_stats['n_samples']:,}")
        print(f"  Mean dz: {forward_x_stats['mean_dz']:.3f} m/s")
        print(f"  Fraction dz < -0.05: {forward_x_stats['frac_dz_negative']:.1%}")
        print(f"  Fraction dz > +0.05: {forward_x_stats['frac_dz_positive']:.1%}")
        print(f"  Mean dy: {forward_x_stats['mean_dy']:.3f} m/s")


def save_results(
    velocities: np.ndarray,
    correlation_matrix: np.ndarray,
    conditional_stats: dict,
    forward_x_stats: dict | None,
    output_path: Path,
):
    """Save analysis results to npz file."""
    data = {
        "velocities": velocities,
        "correlation_matrix": correlation_matrix,
        "conditional_stats": conditional_stats,
        "forward_x_stats": forward_x_stats,
    }
    np.savez(output_path, **data)
    print(f"\nResults saved to {output_path}")


def create_plots(
    velocities: np.ndarray,
    correlation_matrix: np.ndarray,
    output_dir: Path,
):
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Velocity histograms
    axis_names = ["dx", "dy", "dz"]
    for i, (ax, name) in enumerate(zip(axes[0], axis_names)):
        ax.hist(velocities[:, i], bins=100, density=True, alpha=0.7)
        ax.set_xlabel(f"{name} (m/s)")
        ax.set_ylabel("Density")
        ax.set_title(f"{name} Distribution")
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)

    # Scatter plots for correlations
    pairs = [(0, 1, "dx", "dy"), (0, 2, "dx", "dz"), (1, 2, "dy", "dz")]
    for ax, (i, j, name_i, name_j) in zip(axes[1], pairs):
        # Subsample for visualization
        idx = np.random.choice(len(velocities), min(10000, len(velocities)), replace=False)
        ax.scatter(velocities[idx, i], velocities[idx, j], alpha=0.1, s=1)
        ax.set_xlabel(f"{name_i} (m/s)")
        ax.set_ylabel(f"{name_j} (m/s)")
        ax.set_title(f"r = {correlation_matrix[i, j]:.3f}")
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "action_distribution_analysis.png", dpi=150)
    plt.close()
    print(f"Plot saved to {output_dir / 'action_distribution_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DROID action distribution")
    parser.add_argument(
        "--n_trajectories", type=int, default=5000, help="Number of trajectories to sample"
    )
    parser.add_argument("--save_plots", action="store_true", help="Generate visualization plots")
    parser.add_argument(
        "--data_path", type=str, default=str(DROID_DATA_PATH), help="Path to DROID data"
    )
    args = parser.parse_args()

    # Collect and process trajectories
    trajectory_files = collect_trajectories(Path(args.data_path), args.n_trajectories)
    velocities = extract_velocities(trajectory_files)
    print(f"Extracted {len(velocities):,} velocity samples")

    # Compute statistics
    correlation_matrix = compute_correlation_matrix(velocities)
    conditional_stats = compute_conditional_statistics(velocities)
    forward_x_stats = compute_forward_x_statistics(velocities)

    # Print results
    print_results(len(velocities), correlation_matrix, conditional_stats, forward_x_stats)

    # Save results
    save_results(
        velocities,
        correlation_matrix,
        conditional_stats,
        forward_x_stats,
        OUTPUT_DIR / "action_statistics.npz",
    )

    # Create plots if requested
    if args.save_plots:
        create_plots(velocities, correlation_matrix, OUTPUT_DIR)


if __name__ == "__main__":
    main()
