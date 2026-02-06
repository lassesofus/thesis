"""
Batch Pipeline for Language Grounding Data Generation

Process multiple DROID trajectories to generate (frame_k, frame_k+d, instruction) triplets
for training language-conditioned latent goal predictors.

Usage:
    # Single instruction per pair (default)
    python batch_pipeline.py --input_dir /data/droid_raw/1.0.1 --output_dir ./output --provider openai

    # Diverse instructions (5 per pair) for contrastive training
    python batch_pipeline.py --input_dir /data/droid_raw/1.0.1 --output_dir ./output --provider openai --diverse

    # Process specific lab
    python batch_pipeline.py --input_dir /data/droid_raw/1.0.1/AUTOLab --output_dir ./output --provider gemini

    # Limit for testing
    python batch_pipeline.py --input_dir /data/droid_raw/1.0.1 --output_dir ./output --provider openai --max_trajectories 5
"""

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import time
from datetime import datetime

from frame_pair_sampler import FramePairSampler, SamplingConfig, FramePair
from instruction_generator_v2 import get_generator
from instruction_generator_v3 import get_diverse_generator


def find_trajectories(input_dir: Path, max_trajectories: Optional[int] = None) -> list[Path]:
    """
    Find all valid DROID trajectory directories.

    A valid trajectory has:
    - trajectory.h5 file
    - metadata file with left_mp4_path
    - corresponding left camera MP4 video file
    """
    trajectories = []

    # Walk through directory structure
    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)

        # Check if this is a trajectory directory
        h5_path = root_path / "trajectory.h5"
        if h5_path.exists():
            # Check for metadata with left_mp4_path
            metadata_files = list(root_path.glob("metadata_*.json"))
            if metadata_files:
                with open(metadata_files[0]) as f:
                    metadata = json.load(f)
                left_mp4_path = metadata.get("left_mp4_path", "")
                if left_mp4_path:
                    video_filename = Path(left_mp4_path).name
                    video_path = root_path / "recordings" / "MP4" / video_filename
                    if video_path.exists():
                        trajectories.append(root_path)

                        if max_trajectories and len(trajectories) >= max_trajectories:
                            break

    return sorted(trajectories)


def process_trajectory(
    trajectory_dir: Path,
    output_dir: Path,
    sampler: FramePairSampler,
    generator,
    diverse: bool = False,
    rate_limit_delay: float = 1.0,
    verbose: bool = True
) -> dict:
    """
    Process a single trajectory: sample frame pairs and generate instructions.

    Args:
        diverse: If True, generate multiple diverse instructions per pair (v3).
                 If False, generate single instruction per pair (v2).

    Returns dict with statistics and any errors.
    """
    trajectory_id = trajectory_dir.name
    traj_output_dir = output_dir / trajectory_id

    result = {
        "trajectory_id": trajectory_id,
        "trajectory_path": str(trajectory_dir),
        "status": "success",
        "n_pairs_sampled": 0,
        "n_pairs_with_instructions": 0,
        "n_total_instructions": 0,
        "errors": []
    }

    try:
        # Step 1: Sample frame pairs
        if verbose:
            print(f"\n  Sampling frame pairs...")

        pairs = sampler.sample_from_trajectory(
            trajectory_dir,
            traj_output_dir,
            extract_frames=True
        )

        result["n_pairs_sampled"] = len(pairs)

        if not pairs:
            result["status"] = "no_valid_pairs"
            if verbose:
                print(f"  No valid frame pairs found")
            return result

        if verbose:
            stats = sampler.get_sampling_stats(pairs)
            print(f"  Sampled {len(pairs)} pairs")
            for d, count in sorted(stats["by_horizon"].items()):
                print(f"    d={d}: {count}")

        # Step 2: Generate instructions
        if verbose:
            mode = "diverse (5 per pair)" if diverse else "single"
            print(f"\n  Generating instructions ({mode})...")

        output_path = traj_output_dir / "frame_pairs_with_instructions.json"

        results = []
        for i, pair in enumerate(pairs):
            if not pair.frame_k_path or not pair.frame_k_d_path:
                result["errors"].append(f"Missing frame paths for pair {i}")
                continue

            try:
                if diverse:
                    # V3: Generate multiple diverse instructions
                    vlm_result = generator.generate_instructions(
                        pair.frame_k_path,
                        pair.frame_k_d_path
                    )
                    output = {
                        "trajectory_id": pair.trajectory_id,
                        "frame_k": pair.frame_k,
                        "frame_k_d": pair.frame_k_d,
                        "d": pair.d,
                        "frame_k_path": str(pair.frame_k_path),
                        "frame_k_d_path": str(pair.frame_k_d_path),
                        "position_delta": pair.position_delta,
                        "gripper_delta": pair.gripper_delta,
                        "z_delta": pair.z_delta,
                        "what_changed": vlm_result["what_changed"],
                        "instructions": vlm_result["instructions"],
                        "raw_response": vlm_result["raw_response"]
                    }
                    result["n_total_instructions"] += len(vlm_result["instructions"])
                    if verbose:
                        print(f"    [{i+1}/{len(pairs)}] {len(vlm_result['instructions'])} instructions generated")
                else:
                    # V2: Generate single instruction
                    vlm_result = generator.generate_instruction(
                        pair.frame_k_path,
                        pair.frame_k_d_path
                    )
                    output = {
                        "trajectory_id": pair.trajectory_id,
                        "frame_k": pair.frame_k,
                        "frame_k_d": pair.frame_k_d,
                        "d": pair.d,
                        "frame_k_path": str(pair.frame_k_path),
                        "frame_k_d_path": str(pair.frame_k_d_path),
                        "position_delta": pair.position_delta,
                        "gripper_delta": pair.gripper_delta,
                        "z_delta": pair.z_delta,
                        "what_changed": vlm_result["what_changed"],
                        "instruction": vlm_result["instruction"],
                        "raw_response": vlm_result["raw_response"]
                    }
                    result["n_total_instructions"] += 1
                    if verbose:
                        print(f"    [{i+1}/{len(pairs)}] {vlm_result['instruction'][:60]}...")

                results.append(output)

            except Exception as e:
                result["errors"].append(f"VLM error for pair {i}: {str(e)}")
                if verbose:
                    print(f"    [{i+1}/{len(pairs)}] Error: {e}")

            # Rate limiting
            if rate_limit_delay > 0 and i < len(pairs) - 1:
                time.sleep(rate_limit_delay)

        result["n_pairs_with_instructions"] = len(results)

        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        if verbose:
            print(f"  Saved {len(results)} pairs with instructions")

    except Exception as e:
        result["status"] = "error"
        result["errors"].append(str(e))
        if verbose:
            print(f"  Error: {e}")

    return result


def run_pipeline(
    input_dir: str,
    output_dir: str,
    provider: str = "openai",
    model: str = None,
    max_trajectories: Optional[int] = None,
    diverse: bool = False,
    rate_limit_delay: float = 1.0,
    verbose: bool = True
):
    """
    Run the full pipeline on multiple trajectories.

    Args:
        diverse: If True, generate 5 diverse instructions per pair (for contrastive training).
                 If False, generate single instruction per pair.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find trajectories
    print(f"Searching for trajectories in: {input_dir}")
    trajectories = find_trajectories(input_dir, max_trajectories)
    print(f"Found {len(trajectories)} trajectories")

    if not trajectories:
        print("No trajectories found!")
        return

    # Initialize sampler and generator
    # Note: Video source is determined by left_mp4_path in metadata
    config = SamplingConfig(
        time_horizons=(30, 50, 100, 200),
        stride=20,
        min_position_delta=0.10,
        min_gripper_delta=0.30,
        min_z_delta=0.05,
        min_x_for_visibility=0.50,
        max_z_for_visibility=0.42,
    )
    sampler = FramePairSampler(config)

    # Use appropriate generator based on diverse flag
    if diverse:
        generator = get_diverse_generator(provider, model)
    else:
        generator = get_generator(provider, model)

    print(f"\nPipeline Configuration:")
    print(f"  Provider: {provider}")
    print(f"  Model: {model or 'default'}")
    print(f"  Mode: {'diverse (5 instructions per pair)' if diverse else 'single instruction'}")
    print(f"  Time horizons: {config.time_horizons}")
    print(f"  Rate limit delay: {rate_limit_delay}s")
    print("=" * 60)

    # Process trajectories
    all_results = []
    start_time = datetime.now()

    for i, traj_dir in enumerate(trajectories):
        print(f"\n[{i+1}/{len(trajectories)}] Processing: {traj_dir.name}")

        result = process_trajectory(
            traj_dir,
            output_dir,
            sampler,
            generator,
            diverse=diverse,
            rate_limit_delay=rate_limit_delay,
            verbose=verbose
        )
        all_results.append(result)

        # Save progress
        progress_path = output_dir / "pipeline_progress.json"
        with open(progress_path, "w") as f:
            json.dump({
                "started": start_time.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "completed": i + 1,
                "total": len(trajectories),
                "results": all_results
            }, f, indent=2)

    # Final summary
    elapsed = datetime.now() - start_time

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Time elapsed: {elapsed}")
    print(f"Trajectories processed: {len(all_results)}")

    total_sampled = sum(r["n_pairs_sampled"] for r in all_results)
    total_with_instructions = sum(r["n_pairs_with_instructions"] for r in all_results)
    total_instructions = sum(r.get("n_total_instructions", 0) for r in all_results)

    print(f"Total pairs sampled: {total_sampled}")
    print(f"Total pairs with instructions: {total_with_instructions}")
    if diverse:
        print(f"Total instructions generated: {total_instructions}")
        print(f"Avg instructions per pair: {total_instructions / total_with_instructions:.1f}" if total_with_instructions else "")

    # Count by status
    by_status = {}
    for r in all_results:
        status = r["status"]
        by_status[status] = by_status.get(status, 0) + 1

    print(f"\nBy status:")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")

    # Aggregate all results into single file
    all_pairs = []
    for result in all_results:
        if result["status"] == "success":
            traj_output = output_dir / result["trajectory_id"] / "frame_pairs_with_instructions.json"
            if traj_output.exists():
                with open(traj_output) as f:
                    pairs = json.load(f)
                    all_pairs.extend(pairs)

    aggregated_path = output_dir / "all_frame_pairs.json"
    with open(aggregated_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\nAggregated {len(all_pairs)} pairs saved to: {aggregated_path}")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch pipeline for language grounding data")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root directory containing DROID trajectories")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--provider", type=str, required=True,
                        choices=["openai", "anthropic", "gemini"])
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to use")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Maximum number of trajectories to process")
    parser.add_argument("--diverse", action="store_true",
                        help="Generate 5 diverse instructions per pair (for contrastive training)")
    parser.add_argument("--rate_limit", type=float, default=1.0,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        max_trajectories=args.max_trajectories,
        diverse=args.diverse,
        rate_limit_delay=args.rate_limit,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
