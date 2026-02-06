"""
Export sampled frame pairs to a permanent location with organized structure.

Structure:
/data/s185927/droid_raw/sampled_pairs/
    frames/
        {lab}_{traj_short_id}/
            frame_{k:05d}.jpg
            frame_{k_d:05d}.jpg
    frame_pairs.json      # All pairs with metadata
    sampling_config.json  # Configuration used for sampling
"""

import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def export_sampled_pairs(
    input_dir: Path,
    output_dir: Path,
    copy_frames: bool = True
):
    """Export sampled pairs to organized permanent storage."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Load frame pairs
    pairs_file = input_dir / "frame_pairs.json"
    with open(pairs_file) as f:
        pairs = json.load(f)

    # Load stats if available
    stats_file = input_dir / "sampling_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
    else:
        stats = {}

    # Create output structure
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Process each pair
    updated_pairs = []

    for pair in pairs:
        traj_id = pair["trajectory_id"]

        # Create short trajectory identifier: lab_date_time
        parts = traj_id.split("+")
        lab = parts[0] if parts else "unknown"
        traj_short = parts[-1][:20] if len(parts) > 1 else traj_id[:20]
        traj_folder = f"{lab}_{traj_short}".replace(":", "-").replace(" ", "_")

        # Create trajectory folder
        traj_frames_dir = frames_dir / traj_folder
        traj_frames_dir.mkdir(exist_ok=True)

        # Copy frames if requested
        if copy_frames and pair.get("frame_k_path"):
            # Find source frames
            src_k = input_dir / pair["frame_k_path"].split("/")[0] / "frames" / f"frame_{pair['frame_k']:05d}.jpg"
            src_kd = input_dir / pair["frame_k_path"].split("/")[0] / "frames" / f"frame_{pair['frame_k_d']:05d}.jpg"

            dst_k = traj_frames_dir / f"frame_{pair['frame_k']:05d}.jpg"
            dst_kd = traj_frames_dir / f"frame_{pair['frame_k_d']:05d}.jpg"

            if src_k.exists() and not dst_k.exists():
                shutil.copy2(src_k, dst_k)
            if src_kd.exists() and not dst_kd.exists():
                shutil.copy2(src_kd, dst_kd)

        # Update paths in pair
        updated_pair = pair.copy()
        updated_pair["frame_k_path"] = f"frames/{traj_folder}/frame_{pair['frame_k']:05d}.jpg"
        updated_pair["frame_k_d_path"] = f"frames/{traj_folder}/frame_{pair['frame_k_d']:05d}.jpg"
        updated_pairs.append(updated_pair)

    # Save updated pairs
    output_pairs_file = output_dir / "frame_pairs.json"
    with open(output_pairs_file, "w") as f:
        json.dump(updated_pairs, f, indent=2)

    # Save config/metadata
    config = {
        "export_date": datetime.now().isoformat(),
        "source_dir": str(input_dir),
        "total_pairs": len(updated_pairs),
        "sampling_stats": stats.get("config", {}),
        "by_lab": {},
        "by_horizon": {},
    }

    # Compute statistics
    for pair in updated_pairs:
        lab = pair["trajectory_id"].split("+")[0]
        config["by_lab"][lab] = config["by_lab"].get(lab, 0) + 1

        d = pair["d"]
        config["by_horizon"][d] = config["by_horizon"].get(d, 0) + 1

    config_file = output_dir / "sampling_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Exported {len(updated_pairs)} pairs to {output_dir}")
    print(f"\nBy lab:")
    for lab, count in sorted(config["by_lab"].items()):
        print(f"  {lab}: {count}")
    print(f"\nBy horizon:")
    for d, count in sorted(config["by_horizon"].items()):
        print(f"  d={d}: {count}")

    return updated_pairs


def main():
    parser = argparse.ArgumentParser(description="Export sampled frame pairs")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory with sampled pairs")
    parser.add_argument("--output_dir", type=str,
                        default="/data/s185927/droid_raw/sampled_pairs",
                        help="Output directory for exported pairs")
    parser.add_argument("--no-copy", action="store_true",
                        help="Don't copy frames, just update metadata")
    args = parser.parse_args()

    export_sampled_pairs(
        Path(args.input_dir),
        Path(args.output_dir),
        copy_frames=not args.no_copy
    )


if __name__ == "__main__":
    main()
