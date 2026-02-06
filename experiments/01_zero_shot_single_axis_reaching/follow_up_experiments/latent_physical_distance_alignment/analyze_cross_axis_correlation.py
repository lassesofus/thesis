#!/usr/bin/env python3
"""
Analyze cross-axis correlation between latent L1 distance and Euclidean
distance for all 9 (goal_axis, movement_axis) combinations.

For each cell in the 3x3 grid:
  - Goal: the robot at start + 20cm along goal_axis (last frame/position of
    a goal_axis trajectory)
  - Movement: frames from a movement_axis trajectory (robot moving along
    movement_axis)
  - For each movement frame: compute latent distance to goal frame (V-JEPA)
    and L2 physical distance to goal position

Diagonal cells (goal=movement) reproduce the original single-axis analysis.
Off-diagonal cells reveal whether the latent distance is sensitive to
cross-axis movement.

Requires GPU for V-JEPA encoding.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, "/home/s185927/thesis/vjepa2")
sys.path.insert(0, "/home/s185927/thesis")

from app.vjepa_droid.utils import init_video_model
from app.vjepa_droid.transforms import make_transforms

MODEL_CONFIG = {
    "checkpoint": "/home/s185927/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt",
    "config": "/home/s185927/thesis/vjepa2/configs/train/vitg16/x_axis_finetune/x_axis_finetune_025pct.yaml",
}


def load_model(checkpoint_path, config, device="cuda:0"):
    """Load V-JEPA 2 encoder from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    cfgs_data = config.get("data", {})
    cfgs_model = config.get("model", {})
    cfgs_meta = config.get("meta", {})

    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size", 16)
    tubelet_size = cfgs_data.get("tubelet_size", 2)

    model_name = cfgs_model.get("model_name", "vit_giant_xformers")
    pred_depth = cfgs_model.get("pred_depth", 24)
    pred_num_heads = cfgs_model.get("pred_num_heads", 16)
    pred_embed_dim = cfgs_model.get("pred_embed_dim", 1024)
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    uniform_power = cfgs_model.get("uniform_power", True)
    use_rope = cfgs_model.get("use_rope", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_sdpa = cfgs_meta.get("use_sdpa", True)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device)
        torch.cuda.set_device(device)

    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=512,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=7,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder_dict = checkpoint["encoder"]
    encoder_dict = {k.replace("module.", ""): v for k, v in encoder_dict.items()}
    encoder.load_state_dict(encoder_dict, strict=False)

    import copy
    target_encoder = copy.deepcopy(encoder)
    if "target_encoder" in checkpoint:
        target_encoder_dict = checkpoint["target_encoder"]
        target_encoder_dict = {
            k.replace("module.", ""): v for k, v in target_encoder_dict.items()
        }
        target_encoder.load_state_dict(target_encoder_dict, strict=False)

    encoder.eval()
    target_encoder.eval()

    return encoder, target_encoder, device


def create_transform(crop_size=256):
    """Create inference transform (matching robo_samples.py)."""
    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )


def encode_frames_together(
    encoder, current_frame, goal_frame, transform, device, tokens_per_frame, normalize=True
):
    """Encode current and goal frames together as a 2-frame video."""
    combined_rgb = np.stack([current_frame, goal_frame], axis=0)
    clips = transform(combined_rgb).unsqueeze(0).to(device)
    B, C, T, H, W = clips.size()

    with torch.no_grad():
        c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = encoder(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if normalize:
            h = F.layer_norm(h, (h.size(-1),))
        z_current = h[:, :tokens_per_frame].contiguous().squeeze(0)
        z_goal = h[:, -tokens_per_frame:].contiguous().squeeze(0)

    return z_current, z_goal


def load_trajectory_data(episode_path):
    """Load trajectory data (frames + ee positions) from an episode directory."""
    episode_path = Path(episode_path)

    with h5py.File(episode_path / "trajectory.h5", "r") as f:
        ee_pos = f["observation/robot_state/cartesian_position"][:, :3]

    import glob
    import imageio

    video_files = glob.glob(str(episode_path / "recordings" / "MP4" / "*.mp4"))
    if not video_files:
        raise FileNotFoundError(
            f"No video files found in {episode_path / 'recordings' / 'MP4'}"
        )
    reader = imageio.get_reader(video_files[0])
    all_frames = [frame for frame in reader]
    reader.close()

    num_trajectory_steps = len(ee_pos)
    num_video_frames = len(all_frames)

    if num_video_frames > num_trajectory_steps:
        indices = np.linspace(0, num_video_frames - 1, num_trajectory_steps, dtype=int)
        frames = [all_frames[i] for i in indices]
    elif num_video_frames < num_trajectory_steps:
        indices = np.linspace(0, num_trajectory_steps - 1, num_video_frames, dtype=int)
        frames = all_frames
        ee_pos = ee_pos[indices]
    else:
        frames = all_frames

    return {"frames": frames, "ee_pos": ee_pos}


def analyze_cross_axis_pair(
    encoder,
    goal_data,
    movement_data,
    device,
    transform,
    tokens_per_frame,
):
    """
    Analyze correlation for a (goal, movement) episode pair.

    Args:
        goal_data: Trajectory data dict for the goal episode (use last frame/pos)
        movement_data: Trajectory data dict for the movement episode (iterate frames)

    Returns:
        euclidean_dists: L2 distances from each movement frame to goal position
        latent_dists: Latent L1 distances from each movement frame to goal frame
    """
    goal_frame = goal_data["frames"][-1]
    goal_pos = goal_data["ee_pos"][-1]

    movement_frames = movement_data["frames"]
    movement_ee_pos = movement_data["ee_pos"]

    euclidean_dists = []
    latent_dists = []

    # Exclude last frame of movement trajectory (it's the movement's own goal,
    # not the cross-axis goal we're measuring against)
    for i in range(len(movement_frames) - 1):
        frame = movement_frames[i]
        pos = movement_ee_pos[i]

        euclidean_dist = np.linalg.norm(goal_pos - pos)

        z_current, z_goal = encode_frames_together(
            encoder, frame, goal_frame, transform, device, tokens_per_frame
        )
        latent_dist = torch.abs(z_current - z_goal).mean().item()

        euclidean_dists.append(euclidean_dist)
        latent_dists.append(latent_dist)

    return euclidean_dists, latent_dists


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cross-axis latent-physical correlation (3x3 grid)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/s185927/droid_sim/zero_shot_correlation",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/"
        "follow_up_experiments/latent_physical_distance_alignment",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=5,
        help="Number of episodes to process per axis (default: 5)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    with open(MODEL_CONFIG["config"], "r") as f:
        config = yaml.safe_load(f)

    encoder, target_encoder, device = load_model(
        MODEL_CONFIG["checkpoint"], config, args.device
    )
    encoder = target_encoder  # Use target encoder for consistency

    crop_size = config["data"]["crop_size"]
    patch_size = config["data"]["patch_size"]
    transform = create_transform(crop_size=crop_size)
    tokens_per_frame = (crop_size // patch_size) ** 2
    print(f"Tokens per frame: {tokens_per_frame}")

    axes = ["x", "y", "z"]

    # Pre-load all trajectory data to avoid redundant I/O
    print("\nPre-loading trajectory data...")
    trajectory_cache = {}
    for axis in axes:
        csv_path = data_dir / f"{axis}_axis" / "test_trajectories.csv"
        with open(csv_path) as f:
            episodes = [l.strip() for l in f if l.strip()]
        episodes = episodes[: args.max_trajectories]

        trajectory_cache[axis] = []
        for ep_path in tqdm(episodes, desc=f"Loading {axis}-axis"):
            trajectory_cache[axis].append(load_trajectory_data(ep_path))

    n_episodes = args.max_trajectories

    # Analyze all 9 (goal_axis, movement_axis) combinations
    save_dict = {}
    print("\nAnalyzing cross-axis correlations...")

    for goal_axis in axes:
        for movement_axis in axes:
            label = f"goal={goal_axis}, move={movement_axis}"
            is_diagonal = goal_axis == movement_axis
            tag = " [diagonal]" if is_diagonal else ""

            all_euclidean = []
            all_latent = []
            all_episode_idx = []

            for ep_idx in tqdm(
                range(n_episodes), desc=f"{label}{tag}", unit="ep"
            ):
                goal_data = trajectory_cache[goal_axis][ep_idx]
                movement_data = trajectory_cache[movement_axis][ep_idx]

                euclidean_dists, latent_dists = analyze_cross_axis_pair(
                    encoder,
                    goal_data,
                    movement_data,
                    device,
                    transform,
                    tokens_per_frame,
                )

                all_euclidean.extend(euclidean_dists)
                all_latent.extend(latent_dists)
                all_episode_idx.extend([ep_idx] * len(euclidean_dists))

            all_euclidean = np.array(all_euclidean)
            all_latent = np.array(all_latent)
            all_episode_idx = np.array(all_episode_idx)

            correlation = np.corrcoef(all_euclidean, all_latent)[0, 1]
            print(f"  {label}: r={correlation:.4f}, N={len(all_euclidean)}")

            key = f"{goal_axis}_{movement_axis}"
            save_dict[f"{key}_euclidean"] = all_euclidean
            save_dict[f"{key}_latent"] = all_latent
            save_dict[f"{key}_episode_idx"] = all_episode_idx
            save_dict[f"{key}_correlation"] = correlation

    # Clean up GPU
    del encoder, target_encoder
    torch.cuda.empty_cache()

    # Save
    output_path = output_dir / "cross_axis_correlation_data.npz"
    np.savez(output_path, **save_dict)
    print(f"\nSaved to {output_path}")

    # Print summary matrix
    print("\n" + "=" * 60)
    print("CORRELATION MATRIX (Pearson r)")
    print("=" * 60)
    print(f"{'':>14s} {'x-move':>10s} {'y-move':>10s} {'z-move':>10s}")
    print("-" * 46)
    for goal_axis in axes:
        row = f"  {goal_axis}-goal:  "
        for movement_axis in axes:
            r = save_dict[f"{goal_axis}_{movement_axis}_correlation"]
            row += f"  {r:>8.4f}"
        print(row)
    print("=" * 60)


if __name__ == "__main__":
    main()
