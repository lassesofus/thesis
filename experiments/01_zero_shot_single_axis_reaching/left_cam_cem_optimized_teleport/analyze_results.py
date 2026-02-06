#!/usr/bin/env python3
"""
Analyze zero-shot reaching experiment results and print statistics for the results.tex file.
"""

import json
import numpy as np
from pathlib import Path

def load_summary(axis_dir: Path) -> dict:
    """Load the distance summary JSON file for an axis."""
    json_files = list(axis_dir.glob("run_*_distance_summary.json"))
    if not json_files:
        raise FileNotFoundError(f"No summary JSON found in {axis_dir}")
    with open(json_files[0]) as f:
        return json.load(f)

def analyze_axis(data: dict, axis_name: str) -> dict:
    """Analyze data for a single axis."""
    # Cartesian distances (in meters)
    vjepa_distances = np.array(data["phase3_vjepa_distances"])  # shape: (episodes, steps)
    final_distances = np.array(data["phase3_final_distance"])  # shape: (episodes,)

    # Latent L1 distances
    repr_l1_distances = np.array(data["phase3_repr_l1_distances"])  # shape: (episodes, steps)

    n_episodes = len(final_distances)
    success_threshold = 0.05  # 5cm

    # Cartesian statistics
    initial_distance = vjepa_distances[:, 0].mean()
    final_distance_mean = final_distances.mean()
    final_distance_std = final_distances.std()
    final_distance_min = final_distances.min()
    final_distance_max = final_distances.max()

    # Success rate
    successes = (final_distances < success_threshold).sum()

    # Step-by-step Cartesian distances (mean across episodes)
    step_distances_mean = vjepa_distances.mean(axis=0)
    step_distances_std = vjepa_distances.std(axis=0)

    # Latent L1 statistics
    initial_l1 = repr_l1_distances[:, 0].mean()
    final_l1 = repr_l1_distances[:, -1].mean()
    l1_reduction_pct = (1 - final_l1 / initial_l1) * 100

    # Step-by-step latent L1 (mean across episodes)
    step_l1_mean = repr_l1_distances.mean(axis=0)
    step_l1_std = repr_l1_distances.std(axis=0)

    # Check for non-monotonicity in latent space
    l1_increases = []
    for i in range(1, len(step_l1_mean)):
        if step_l1_mean[i] > step_l1_mean[i-1]:
            l1_increases.append(i)

    return {
        "axis": axis_name,
        "n_episodes": n_episodes,
        "initial_distance_cm": initial_distance * 100,
        "final_distance_mean_cm": final_distance_mean * 100,
        "final_distance_std_cm": final_distance_std * 100,
        "final_distance_min_cm": final_distance_min * 100,
        "final_distance_max_cm": final_distance_max * 100,
        "successes": successes,
        "success_rate": successes / n_episodes * 100,
        "step_distances_cm": step_distances_mean * 100,
        "step_distances_std_cm": step_distances_std * 100,
        "initial_l1": initial_l1,
        "final_l1": final_l1,
        "l1_reduction_pct": l1_reduction_pct,
        "step_l1_mean": step_l1_mean,
        "step_l1_std": step_l1_std,
        "l1_increases_at_steps": l1_increases,
    }

def main():
    base_dir = Path(__file__).parent

    axes = {
        "x": base_dir / "reach_along_x",
        "y": base_dir / "reach_along_y",
        "z": base_dir / "reach_along_z",
    }

    results = {}
    for axis_name, axis_dir in axes.items():
        data = load_summary(axis_dir)
        results[axis_name] = analyze_axis(data, axis_name)

    # Print summary
    print("=" * 80)
    print("ZERO-SHOT REACHING EXPERIMENT RESULTS")
    print("=" * 80)

    total_episodes = sum(r["n_episodes"] for r in results.values())
    total_successes = sum(r["successes"] for r in results.values())

    for axis in ["x", "y", "z"]:
        r = results[axis]
        print(f"\n{axis.upper()}-AXIS:")
        print("-" * 40)
        print(f"  Episodes: {r['n_episodes']}")
        print(f"  Initial distance: {r['initial_distance_cm']:.1f} cm")
        print(f"  Final distance (mean ± std): {r['final_distance_mean_cm']:.1f} ± {r['final_distance_std_cm']:.1f} cm")
        print(f"  Final distance range: {r['final_distance_min_cm']:.1f} - {r['final_distance_max_cm']:.1f} cm")
        print(f"  Successes (<5cm): {r['successes']}/{r['n_episodes']} ({r['success_rate']:.0f}%)")
        print(f"  Step-by-step distances (cm): {', '.join(f'{d:.1f}' for d in r['step_distances_cm'])}")
        print(f"  Initial latent L1: {r['initial_l1']:.2f}")
        print(f"  Final latent L1: {r['final_l1']:.2f}")
        print(f"  Latent L1 reduction: {r['l1_reduction_pct']:.0f}%")
        print(f"  Step-by-step L1: {', '.join(f'{d:.2f}' for d in r['step_l1_mean'])}")
        if r['l1_increases_at_steps']:
            print(f"  L1 increases at steps: {r['l1_increases_at_steps']}")

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {total_successes}/{total_episodes} ({total_successes/total_episodes*100:.0f}%)")

    # Identify which axes contribute to successes
    successful_axes = [axis for axis in ["x", "y", "z"] if results[axis]["successes"] > 0]
    print(f"Axes with successes: {', '.join(successful_axes)}")

    print("\n" + "=" * 80)
    print("LATEX-READY TEXT")
    print("=" * 80)

    # Generate LaTeX-ready descriptions
    y = results["y"]
    z = results["z"]
    x = results["x"]

    print(f"""
For the $y$-axis: reduces from {y['initial_distance_cm']:.0f}\\,cm to approximately {y['final_distance_mean_cm']:.0f}\\,cm
  (final errors ranging from {y['final_distance_min_cm']:.1f}\\,cm to {y['final_distance_max_cm']:.1f}\\,cm)
  {y['successes']}/{y['n_episodes']} reach the 5\\,cm success threshold.

For the $z$-axis: reduces from {z['initial_distance_cm']:.0f}\\,cm to approximately {z['final_distance_mean_cm']:.0f}\\,cm
  (final errors ranging from {z['final_distance_min_cm']:.1f}\\,cm to {z['final_distance_max_cm']:.1f}\\,cm)
  {z['successes']}/{z['n_episodes']} reach the 5\\,cm success threshold.

For the $x$-axis: reduces from {x['initial_distance_cm']:.0f}\\,cm to approximately {x['final_distance_mean_cm']:.0f}\\,cm
  (final errors ranging from {x['final_distance_min_cm']:.1f}\\,cm to {x['final_distance_max_cm']:.1f}\\,cm)
  {x['successes']}/{x['n_episodes']} reach the 5\\,cm success threshold.

Total: {total_successes}/{total_episodes} episodes achieve the success threshold.

Latent L1 distances:
  $z$-axis: {z['initial_l1']:.2f} to {z['final_l1']:.2f} (roughly {z['l1_reduction_pct']:.0f}\\% reduction)
  $y$-axis: {y['initial_l1']:.2f} to {y['final_l1']:.2f} (roughly {y['l1_reduction_pct']:.0f}\\% reduction)
  $x$-axis: {x['initial_l1']:.2f} to {x['final_l1']:.2f} (roughly {x['l1_reduction_pct']:.0f}\\% reduction)
""")

if __name__ == "__main__":
    main()
