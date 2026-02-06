#!/usr/bin/env python3
"""
Test whether encoding identical images produces non-zero L1 distance.

This diagnostic script checks if the encode_frames_together function
introduces differences when encoding the same frame as both current and goal.
"""

import glob
import json
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Add vjepa2 directory to import modules
sys.path.insert(0, "/home/s185927/thesis/vjepa2")

from app.vjepa_droid.utils import init_video_model
from app.vjepa_droid.transforms import make_transforms


# Use pretrained model config
CONFIG_PATH = "/home/s185927/thesis/vjepa2/configs/train/vitg16/x_axis_finetune/x_axis_finetune_025pct.yaml"
CHECKPOINT_PATH = "/home/s185927/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt"
TEST_CSV = "/data/s185927/droid_sim/axis_aligned/splits/val_trajectories.csv"


def load_model(checkpoint_path, config, device='cuda:0'):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    cfgs_data = config.get('data', {})
    cfgs_model = config.get('model', {})
    cfgs_meta = config.get('meta', {})

    crop_size = cfgs_data.get('crop_size', 256)
    patch_size = cfgs_data.get('patch_size', 16)
    tubelet_size = cfgs_data.get('tubelet_size', 2)

    model_name = cfgs_model.get('model_name', 'vit_giant_xformers')
    pred_depth = cfgs_model.get('pred_depth', 24)
    pred_num_heads = cfgs_model.get('pred_num_heads', 16)
    pred_embed_dim = cfgs_model.get('pred_embed_dim', 1024)
    pred_is_frame_causal = cfgs_model.get('pred_is_frame_causal', True)
    uniform_power = cfgs_model.get('uniform_power', True)
    use_rope = cfgs_model.get('use_rope', True)
    use_extrinsics = cfgs_model.get('use_extrinsics', False)
    use_silu = cfgs_model.get('use_silu', False)
    use_pred_silu = cfgs_model.get('use_pred_silu', False)
    wide_silu = cfgs_model.get('wide_silu', True)
    use_sdpa = cfgs_meta.get('use_sdpa', True)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device)
        torch.cuda.set_device(device)

    print("Initializing model...")
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

    encoder_dict = checkpoint['encoder']
    encoder_dict = {k.replace("module.", ""): v for k, v in encoder_dict.items()}
    encoder.load_state_dict(encoder_dict, strict=False)

    encoder.eval()

    return encoder, device


def create_transform(crop_size=256):
    """Create transform for inference."""
    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )


def encode_frames_together(encoder, current_frame, goal_frame, transform, device, tokens_per_frame, normalize=True):
    """
    Encode current and goal frames together as a 2-frame video.
    This is copied from analyze_latent_physical_correlation.py.
    """
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


def main():
    print("=" * 60)
    print("TEST: Encoding identical images")
    print("=" * 60)

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    crop_size = config['data']['crop_size']
    patch_size = config['data']['patch_size']
    tokens_per_frame = (crop_size // patch_size) ** 2

    print(f"Crop size: {crop_size}")
    print(f"Patch size: {patch_size}")
    print(f"Tokens per frame: {tokens_per_frame}")

    # Load model
    encoder, device = load_model(CHECKPOINT_PATH, config)
    transform = create_transform(crop_size=crop_size)

    # Create a random test image
    print("\nCreating random test image...")
    test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Encode the SAME frame as both current and goal
    print("Encoding identical frames...")
    z_current, z_goal = encode_frames_together(
        encoder, test_frame, test_frame, transform, device, tokens_per_frame
    )

    print(f"\nz_current shape: {z_current.shape}")
    print(f"z_goal shape: {z_goal.shape}")

    # Compute L1 distance (same as in analyze script)
    l1_per_token = torch.abs(z_current - z_goal).sum(dim=-1)
    l1_distance = l1_per_token.mean().item()

    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"{'=' * 60}")
    print(f"L1 distance (mean over tokens): {l1_distance:.6f}")
    print(f"L1 per token - min: {l1_per_token.min().item():.6f}")
    print(f"L1 per token - max: {l1_per_token.max().item():.6f}")
    print(f"L1 per token - std: {l1_per_token.std().item():.6f}")

    # Check if representations are exactly equal
    are_equal = torch.allclose(z_current, z_goal, atol=1e-6)
    print(f"\nRepresentations exactly equal (atol=1e-6): {are_equal}")

    # Check element-wise differences
    abs_diff = torch.abs(z_current - z_goal)
    print(f"\nElement-wise absolute differences:")
    print(f"  Max: {abs_diff.max().item():.6f}")
    print(f"  Mean: {abs_diff.mean().item():.6f}")
    print(f"  Non-zero elements: {(abs_diff > 1e-6).sum().item()} / {abs_diff.numel()}")

    if l1_distance > 1.0:
        print(f"\n⚠️  WARNING: L1 distance is {l1_distance:.2f} for IDENTICAL images!")
        print("This explains the 400+ baseline in the correlation plot.")
        print("The encoding function treats the two frame positions differently.")
    elif l1_distance > 0.01:
        print(f"\n⚠️  Small but non-zero L1 distance detected: {l1_distance:.6f}")
    else:
        print(f"\n✓ L1 distance is essentially zero as expected.")


def load_trajectory_data(episode_path):
    """Load trajectory data from an episode directory."""
    episode_path = Path(episode_path)

    # Load trajectory
    with h5py.File(episode_path / "trajectory.h5", 'r') as f:
        ee_pos_full = f['observation/robot_state/cartesian_position'][:]
        ee_pos = ee_pos_full[:, :3]

    # Load video frames
    video_files = glob.glob(str(episode_path / "recordings" / "MP4" / "*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video files found in {episode_path / 'recordings' / 'MP4'}")
    video_path = video_files[0]

    reader = imageio.get_reader(video_path)
    all_frames = []
    for frame in reader:
        all_frames.append(frame)
    reader.close()

    # Subsample to match trajectory
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

    return {'frames': frames, 'ee_pos': ee_pos}


def test_real_trajectory():
    """Test L1 distances on a real trajectory to understand the baseline."""
    print("\n" + "=" * 60)
    print("TEST: Real trajectory frames")
    print("=" * 60)

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    crop_size = config['data']['crop_size']
    patch_size = config['data']['patch_size']
    tokens_per_frame = (crop_size // patch_size) ** 2

    # Load model
    encoder, device = load_model(CHECKPOINT_PATH, config)
    transform = create_transform(crop_size=crop_size)

    # Load first test trajectory
    with open(TEST_CSV, 'r') as f:
        episodes = [line.strip() for line in f if line.strip()]

    traj_data = load_trajectory_data(episodes[0])
    frames = traj_data['frames']
    ee_pos = traj_data['ee_pos']

    print(f"Trajectory: {episodes[0]}")
    print(f"Number of frames: {len(frames)}")

    goal_frame = frames[-1]
    goal_pos = ee_pos[-1]

    # Test a few frames at different distances from goal
    test_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -2]

    print(f"\nComparing frames to goal (last frame):")
    print(f"{'Frame':<10} {'Eucl Dist (m)':<15} {'L1 Dist':<15}")
    print("-" * 40)

    for idx in test_indices:
        frame = frames[idx]
        pos = ee_pos[idx]

        eucl_dist = np.linalg.norm(goal_pos - pos)

        z_current, z_goal = encode_frames_together(
            encoder, frame, goal_frame, transform, device, tokens_per_frame
        )

        l1_dist = torch.abs(z_current - z_goal).sum(dim=-1).mean().item()

        print(f"{idx:<10} {eucl_dist:<15.4f} {l1_dist:<15.2f}")

    # Now test: what's the L1 distance between the second-to-last frame and the goal?
    print(f"\n--- Key test: second-to-last frame vs goal ---")
    second_last_frame = frames[-2]
    second_last_pos = ee_pos[-2]

    eucl_dist = np.linalg.norm(goal_pos - second_last_pos)

    z_current, z_goal = encode_frames_together(
        encoder, second_last_frame, goal_frame, transform, device, tokens_per_frame
    )
    l1_dist = torch.abs(z_current - z_goal).sum(dim=-1).mean().item()

    print(f"Euclidean distance: {eucl_dist:.4f} m")
    print(f"L1 distance: {l1_dist:.2f}")

    # Pixel difference
    pixel_diff = np.abs(second_last_frame.astype(float) - goal_frame.astype(float)).mean()
    print(f"Mean pixel difference: {pixel_diff:.2f}")

    # Save the two frames for visual inspection
    from PIL import Image
    output_dir = Path("/home/s185927/thesis/experiments/sim_training_different_fractions_x_axis_finetune")
    Image.fromarray(second_last_frame).save(output_dir / "debug_second_last_frame.png")
    Image.fromarray(goal_frame).save(output_dir / "debug_goal_frame.png")
    print(f"\nSaved frames to {output_dir} for visual inspection")


if __name__ == '__main__':
    main()
    test_real_trajectory()
