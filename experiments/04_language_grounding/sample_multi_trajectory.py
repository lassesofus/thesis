"""
Sample frame pairs from multiple DROID trajectories to test filtering generalization.
"""

import json
import random
import argparse
from pathlib import Path
from dataclasses import asdict
import numpy as np

from frame_pair_sampler import FramePairSampler, SamplingConfig


def find_trajectories(droid_root: Path, n_per_lab: int = 5, success_only: bool = False) -> list[Path]:
    """Find trajectory directories from multiple labs.

    Args:
        droid_root: Root directory of DROID dataset
        n_per_lab: Number of trajectories per lab. Use -1 for ALL trajectories.
        success_only: If True, only sample from success/ subdirectory
    """

    trajectories = []
    labs = ["AUTOLab", "CLVR", "GuptaLab", "ILIAD", "IPRL", "IRIS"]

    for lab in labs:
        lab_path = droid_root / lab
        if not lab_path.exists():
            continue

        # Find all trajectory dirs with trajectory.h5
        if success_only:
            search_path = lab_path / "success"
        else:
            search_path = lab_path

        if not search_path.exists():
            search_path = lab_path

        traj_files = list(search_path.rglob("trajectory.h5"))
        traj_dirs = [f.parent for f in traj_files]

        if traj_dirs:
            if n_per_lab == -1:
                # Use ALL trajectories
                sampled = traj_dirs
            else:
                # Sample n_per_lab trajectories
                sampled = random.sample(traj_dirs, min(n_per_lab, len(traj_dirs)))
            trajectories.extend(sampled)
            print(f"{lab}: {len(sampled)} trajectories")

    return trajectories


def sample_from_trajectories(
    trajectories: list[Path],
    output_dir: Path,
    config: SamplingConfig
) -> dict:
    """Sample frame pairs from multiple trajectories and collect statistics."""
    import time

    sampler = FramePairSampler(config)

    all_pairs = []
    stats_by_traj = []
    start_time = time.time()

    for i, traj_dir in enumerate(trajectories):
        elapsed = time.time() - start_time
        if i > 0:
            eta = (elapsed / i) * (len(trajectories) - i)
            eta_str = f"ETA: {eta/60:.1f}min"
        else:
            eta_str = ""
        print(f"[{i+1}/{len(trajectories)}] {traj_dir.name[:40]}... {eta_str}")

        try:
            # Create trajectory-specific output dir
            traj_output = output_dir / f"traj_{i:03d}"

            pairs = sampler.sample_from_trajectory(
                traj_dir,
                traj_output,
                extract_frames=True
            )

            # Update paths to be relative to main output dir
            for pair in pairs:
                if pair.frame_k_path:
                    pair.frame_k_path = str(Path(f"traj_{i:03d}") / "frames" / Path(pair.frame_k_path).name)
                if pair.frame_k_d_path:
                    pair.frame_k_d_path = str(Path(f"traj_{i:03d}") / "frames" / Path(pair.frame_k_d_path).name)

            stats = sampler.get_sampling_stats(pairs)
            stats["trajectory"] = str(traj_dir)
            stats["lab"] = traj_dir.parts[-4] if len(traj_dir.parts) >= 4 else "unknown"

            stats_by_traj.append(stats)
            all_pairs.extend(pairs)

            print(f"  -> {len(pairs)} pairs sampled")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            stats_by_traj.append({
                "trajectory": str(traj_dir),
                "error": str(e),
                "total_pairs": 0
            })

    return {
        "all_pairs": all_pairs,
        "stats_by_traj": stats_by_traj,
        "total_pairs": len(all_pairs),
        "total_trajectories": len(trajectories),
        "successful_trajectories": sum(1 for s in stats_by_traj if s.get("total_pairs", 0) > 0)
    }


def main():
    parser = argparse.ArgumentParser(description="Sample frame pairs from multiple DROID trajectories")
    parser.add_argument("--droid_root", type=str, default="/data/droid_raw/1.0.1",
                        help="Root directory of DROID dataset")
    parser.add_argument("--output_dir", type=str, default="sampled_pairs_multi",
                        help="Output directory")
    parser.add_argument("--n_per_lab", type=int, default=3,
                        help="Number of trajectories to sample per lab (-1 for ALL)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for trajectory selection")
    parser.add_argument("--max_rx_deviation", type=float, default=40.0,
                        help="Maximum Rx deviation in degrees")
    parser.add_argument("--success_only", action="store_true",
                        help="Only sample from success/ trajectories")
    parser.add_argument("--horizons", type=int, nargs="+", default=[30, 50, 100, 200],
                        help="Time horizons to sample")
    parser.add_argument("--stride", type=int, default=30,
                        help="Stride for sampling start frames")
    parser.add_argument("--max_pairs_per_traj", type=int, default=12,
                        help="Maximum pairs per trajectory (for scene diversity)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    droid_root = Path(args.droid_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"DROID root: {droid_root}")
    print(f"Output dir: {output_dir}")
    if args.n_per_lab == -1:
        print("Sampling ALL trajectories")
    else:
        print(f"Sampling {args.n_per_lab} trajectories per lab")
    print(f"Horizons: {args.horizons}, Stride: {args.stride}, Max pairs/traj: {args.max_pairs_per_traj}")
    print("=" * 60)

    # Find trajectories
    trajectories = find_trajectories(droid_root, n_per_lab=args.n_per_lab, success_only=args.success_only)
    print(f"\nTotal trajectories to process: {len(trajectories)}")

    # Configure sampler
    config = SamplingConfig(
        time_horizons=tuple(args.horizons),
        stride=args.stride,
        max_rotation_delta=np.deg2rad(180),  # Disabled
        max_rx_deviation=np.deg2rad(args.max_rx_deviation),
        max_pairs_per_trajectory=args.max_pairs_per_traj,
    )

    # Sample
    results = sample_from_trajectories(trajectories, output_dir, config)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total trajectories processed: {results['total_trajectories']}")
    print(f"Successful trajectories: {results['successful_trajectories']}")
    print(f"Total frame pairs: {results['total_pairs']}")

    # By tier
    tier_a = sum(1 for p in results['all_pairs'] if p.salience_tier == "A")
    tier_b = sum(1 for p in results['all_pairs'] if p.salience_tier == "B")
    total = results['total_pairs']
    if total > 0:
        print(f"\nBy salience tier:")
        print(f"  Tier A (high):     {tier_a:6d} ({100*tier_a/total:.1f}%)")
        print(f"  Tier B (moderate): {tier_b:6d} ({100*tier_b/total:.1f}%)")

    # By lab
    print("\nPairs by lab:")
    by_lab = {}
    for s in results['stats_by_traj']:
        lab = s.get('lab', 'unknown')
        by_lab[lab] = by_lab.get(lab, 0) + s.get('total_pairs', 0)
    for lab, count in sorted(by_lab.items()):
        print(f"  {lab}: {count}")

    # Save results
    pairs_path = output_dir / "frame_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump([asdict(p) for p in results['all_pairs']], f, indent=2)
    print(f"\nSaved pairs to: {pairs_path}")

    stats_path = output_dir / "sampling_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total_pairs": results['total_pairs'],
            "total_trajectories": results['total_trajectories'],
            "successful_trajectories": results['successful_trajectories'],
            "by_tier": {"A": tier_a, "B": tier_b},
            "by_lab": by_lab,
            "by_trajectory": results['stats_by_traj'],
            "config": {
                "max_rx_deviation": args.max_rx_deviation,
                "time_horizons": list(config.time_horizons),
                "stride": config.stride,
                "tier_a_quota": config.tier_a_quota,
                "tier_b_quota": config.tier_b_quota,
                "max_pairs_per_trajectory": config.max_pairs_per_trajectory,
            }
        }, f, indent=2)
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
