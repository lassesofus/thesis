#!/usr/bin/env python3
"""
Run the scene-state diffing pipeline on DROID trajectories.

Usage:
    python run_pipeline.py --trajectory_dir <path> --provider openai
    python run_pipeline.py --trajectory_dir <path> --provider anthropic
    python run_pipeline.py --keyframes_only  # Just extract keyframes, no VLM
"""

import argparse
import json
import os
import cv2
import h5py
import numpy as np
from pathlib import Path

from scene_state_diff import diff_scene_states, generate_instruction
from vlm_clients import get_client


def segment_trajectory(pos: np.ndarray, gripper: np.ndarray, min_segment_len: int = 20) -> list[int]:
    """
    Find keyframe indices based on gripper state changes.
    Returns list of trajectory indices.
    """
    n = len(pos)
    keyframes = [0]  # Always include start

    gripper_threshold = 0.5
    was_gripping = gripper[0] < gripper_threshold

    for i in range(1, n):
        is_gripping = gripper[i] < gripper_threshold

        if is_gripping != was_gripping and (i - keyframes[-1]) >= min_segment_len:
            keyframes.append(i)
            was_gripping = is_gripping

    if keyframes[-1] != n - 1:
        keyframes.append(n - 1)  # Always include end

    return keyframes


def extract_keyframes(
    trajectory_dir: str,
    output_dir: str,
    camera: str = "ext1"
) -> tuple[list[str], list[int], dict]:
    """
    Extract keyframes from a DROID trajectory.

    Returns:
        - List of keyframe file paths
        - List of trajectory indices
        - Metadata dict
    """
    trajectory_dir = Path(trajectory_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectory
    h5_path = trajectory_dir / "trajectory.h5"
    with h5py.File(h5_path, "r") as f:
        pos = f["observation/robot_state/cartesian_position"][:]
        gripper = f["observation/robot_state/gripper_position"][:]

    # Load metadata
    metadata_files = list(trajectory_dir.glob("metadata_*.json"))
    if metadata_files:
        with open(metadata_files[0]) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Find keyframe indices
    keyframe_indices = segment_trajectory(pos, gripper)

    # Determine video path based on camera selection
    if camera == "ext1":
        cam_serial = metadata.get("ext1_cam_serial", "22008760")
    elif camera == "ext2":
        cam_serial = metadata.get("ext2_cam_serial", "24400334")
    else:  # wrist
        cam_serial = metadata.get("wrist_cam_serial", "18026681")

    video_path = trajectory_dir / "recordings" / "MP4" / f"{cam_serial}.mp4"

    # Extract frames
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_traj = len(pos)

    keyframe_paths = []
    for i, traj_idx in enumerate(keyframe_indices):
        video_frame_idx = int(traj_idx * (total_frames / n_traj))
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()

        if ret:
            frame_path = output_dir / f"keyframe_{i:02d}_traj{traj_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            keyframe_paths.append(str(frame_path))

    cap.release()

    # Save trajectory info with keyframes
    info = {
        "trajectory_dir": str(trajectory_dir),
        "task": metadata.get("current_task", "unknown"),
        "keyframe_indices": keyframe_indices,
        "keyframe_paths": keyframe_paths,
        "gripper_states": [float(gripper[i]) for i in keyframe_indices],
        "positions": [pos[i].tolist() for i in keyframe_indices]
    }

    with open(output_dir / "keyframe_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return keyframe_paths, keyframe_indices, metadata


def run_vlm_pipeline(
    keyframe_paths: list[str],
    provider: str = "openai",
    **vlm_kwargs
) -> list[dict]:
    """
    Run VLM on keyframes and generate instructions.
    """
    client = get_client(provider, **vlm_kwargs)

    # Get scene states for all keyframes
    print(f"Querying {provider} VLM for {len(keyframe_paths)} keyframes...")
    scene_states = []
    for i, kf_path in enumerate(keyframe_paths):
        print(f"  Processing keyframe {i + 1}/{len(keyframe_paths)}: {Path(kf_path).name}")
        try:
            state = client.get_scene_state(kf_path)
            scene_states.append(state)
            print(f"    Objects: {[obj['name'] for obj in state.get('objects', [])]}")
            print(f"    Gripper: {state.get('gripper_state')}")
        except Exception as e:
            print(f"    ERROR: {e}")
            scene_states.append(None)

    # Generate instructions by diffing consecutive states
    results = []
    for i in range(len(scene_states) - 1):
        if scene_states[i] and scene_states[i + 1]:
            changes = diff_scene_states(scene_states[i], scene_states[i + 1])
            instruction = generate_instruction(changes)
        else:
            changes = []
            instruction = "[VLM failed for one or both keyframes]"

        results.append({
            "segment": i + 1,
            "from_keyframe": keyframe_paths[i],
            "to_keyframe": keyframe_paths[i + 1],
            "state_before": scene_states[i],
            "state_after": scene_states[i + 1],
            "changes": changes,
            "instruction": instruction
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Scene-state diffing pipeline for DROID trajectories")
    parser.add_argument("--trajectory_dir", type=str,
                        default="/data/droid_raw/1.0.1/AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023",
                        help="Path to DROID trajectory directory")
    parser.add_argument("--output_dir", type=str,
                        default="/home/s185927/thesis/experiments/04_language_grounding/output",
                        help="Output directory for keyframes and results")
    parser.add_argument("--camera", type=str, default="ext1",
                        choices=["ext1", "ext2", "wrist"],
                        help="Which camera to use")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["openai", "anthropic", "ollama"],
                        help="VLM provider (omit to skip VLM, just extract keyframes)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to use (provider-dependent)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Step 1: Extract keyframes
    print("=" * 60)
    print("Step 1: Extracting keyframes")
    print("=" * 60)

    keyframe_paths, keyframe_indices, metadata = extract_keyframes(
        args.trajectory_dir,
        output_dir / "keyframes",
        camera=args.camera
    )

    print(f"\nTask: {metadata.get('current_task', 'unknown')}")
    print(f"Extracted {len(keyframe_paths)} keyframes at indices: {keyframe_indices}")

    if not args.provider:
        print("\nNo VLM provider specified. Keyframes extracted only.")
        print(f"Keyframes saved to: {output_dir / 'keyframes'}")
        return

    # Step 2: Run VLM pipeline
    print("\n" + "=" * 60)
    print(f"Step 2: Running VLM pipeline ({args.provider})")
    print("=" * 60)

    vlm_kwargs = {}
    if args.model:
        vlm_kwargs["model"] = args.model

    results = run_vlm_pipeline(keyframe_paths, provider=args.provider, **vlm_kwargs)

    # Save results
    results_path = output_dir / "instructions.json"
    with open(results_path, "w") as f:
        json.dump({
            "task": metadata.get("current_task", "unknown"),
            "trajectory_dir": args.trajectory_dir,
            "segments": results
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Generated Instructions")
    print("=" * 60)

    for r in results:
        print(f"\nSegment {r['segment']}:")
        print(f"  {r['instruction']}")

    # Full narrative
    all_instructions = [r["instruction"] for r in results if "[" not in r["instruction"]]
    narrative = ". ".join(all_instructions) + "."
    print("\n" + "-" * 60)
    print("Full narrative:")
    print(f"  '{narrative}'")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
