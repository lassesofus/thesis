#!/usr/bin/env python3
"""
Analyze correlation between Euclidean distance and latent L1 distance
using the pretrained V-JEPA2 backbone on RAW DROID data (in-distribution).

This tests whether the pretrained encoder performs better on real-world data
compared to simulated data.

NOTE: This script uses the same encoding approach as the V-JEPA2 training code
(app/vjepa_droid/train.py) to ensure consistency with how the model was trained.
"""

import argparse
import glob
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add vjepa2 directory to import modules
sys.path.insert(0, "/home/s185927/thesis/vjepa2")
# Add thesis directory for plot config
sys.path.insert(0, "/home/s185927/thesis")

from app.vjepa_droid.transforms import make_transforms
from plot_config import PLOT_PARAMS, configure_axis

# Default crop size for pretrained vitg-384 checkpoint
DEFAULT_CROP_SIZE = 384
DEFAULT_PATCH_SIZE = 16


def load_pretrained_encoder(checkpoint_path, device='cuda:0', crop_size=DEFAULT_CROP_SIZE):
    """Load pretrained V-JEPA2 encoder from checkpoint."""
    from src.models.vision_transformer import vit_giant_xformers

    print(f"Loading pretrained checkpoint: {checkpoint_path}")

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device)
        torch.cuda.set_device(device)

    print("Initializing encoder...")
    encoder = vit_giant_xformers(
        img_size=crop_size,
        patch_size=DEFAULT_PATCH_SIZE,
        num_frames=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
        use_sdpa=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder_state = checkpoint.get('target_encoder', checkpoint.get('encoder', {}))

    new_state = {}
    for k, v in encoder_state.items():
        if k.startswith('module.backbone.'):
            new_key = k.replace('module.backbone.', '')
            new_state[new_key] = v
        elif k.startswith('module.'):
            new_key = k.replace('module.', '')
            new_state[new_key] = v
        else:
            new_state[k] = v

    missing, unexpected = encoder.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    encoder = encoder.to(device).eval()
    print("  Encoder loaded successfully")
    return encoder, device


def create_transform(crop_size=DEFAULT_CROP_SIZE):
    """Create transform matching training/inference setup."""
    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )


def encode_frames(encoder, frames, transform, device, crop_size=DEFAULT_CROP_SIZE,
                  patch_size=DEFAULT_PATCH_SIZE, normalize=True):
    """
    Encode multiple frames using the same approach as training code.

    This matches the forward_target function in app/vjepa_droid/train.py:
    - Permute: [B, C, T, H, W] -> [B, T, C, H, W]
    - Flatten batch+time: [B*T, C, H, W]
    - Duplicate frame: [B*T, C, 2, H, W]
    - Encode and reshape back to [B, T*tokens, D]
    - Apply layer norm

    Args:
        encoder: V-JEPA2 encoder
        frames: numpy array of frames [T, H, W, C] or list of frames
        transform: preprocessing transform
        device: torch device
        crop_size: image crop size (default 256)
        patch_size: patch size (default 16)
        normalize: whether to apply layer norm

    Returns:
        Tensor of shape [T, tokens, D] where tokens = (crop_size/patch_size)^2
    """
    # Convert frames to numpy array if list
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)  # [T, H, W, C]

    T = len(frames)
    tokens_per_frame = (crop_size // patch_size) ** 2

    # Apply transform (expects [T, H, W, C], returns [C, T, H, W])
    clips = transform(frames)  # [C, T, H, W]
    clips = clips.unsqueeze(0).to(device)  # [1, C, T, H, W]

    with torch.no_grad():
        # Match forward_target from train.py exactly
        B, C, T_dim, H, W = clips.size()
        # Permute to [B, T, C, H, W] then flatten to [B*T, C, H, W]
        c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [T, C, H, W]
        # Add temporal dim and duplicate: [T, C, 2, H, W]
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        # Encode
        h = encoder(c)  # [T, tokens, D]
        # Reshape to [B, T, tokens, D] then flatten to [B, T*tokens, D]
        h = h.view(B, T_dim, -1, h.size(-1)).flatten(1, 2)  # [1, T*tokens, D]
        if normalize:
            h = F.layer_norm(h, (h.size(-1),))

    # Return [T, tokens, D] by reshaping
    h = h.view(T_dim, tokens_per_frame, -1)  # [T, tokens, D]
    return h


def encode_single_frame(encoder, frame, transform, device, crop_size=DEFAULT_CROP_SIZE,
                        patch_size=DEFAULT_PATCH_SIZE, normalize=True):
    """
    Encode a single frame, returning full token representation.

    Returns:
        Tensor of shape [tokens, D] where tokens = (crop_size/patch_size)^2
    """
    frames = np.expand_dims(frame, axis=0)  # [1, H, W, C]
    h = encode_frames(encoder, frames, transform, device, crop_size, patch_size, normalize)
    return h.squeeze(0)  # [tokens, D]


def load_droid_trajectory_data(episode_path, droid_base_path):
    """Load trajectory data from a DROID episode directory."""
    episode_path = Path(episode_path)
    droid_base_path = Path(droid_base_path)

    # Load trajectory
    with h5py.File(episode_path / "trajectory.h5", 'r') as f:
        ee_pos_full = f['observation/robot_state/cartesian_position'][:]
        ee_pos = ee_pos_full[:, :3]

    # Find metadata file and get left_mp4_path
    metadata_files = glob.glob(str(episode_path / "metadata_*.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata file found in {episode_path}")

    import json
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)

    # Use left_mp4_path from metadata (relative to lab directory)
    if 'left_mp4_path' not in metadata:
        raise KeyError(f"No left_mp4_path in metadata for {episode_path}")
    if 'lab' not in metadata:
        raise KeyError(f"No lab field in metadata for {episode_path}")

    # Path is: {droid_base}/{lab}/{left_mp4_path}
    video_path = droid_base_path / metadata['lab'] / metadata['left_mp4_path']
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    import imageio
    reader = imageio.get_reader(str(video_path))
    all_frames = []
    for frame in reader:
        all_frames.append(frame)
    reader.close()

    # Subsample to match trajectory length
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

    return {
        'frames': frames,
        'ee_pos': ee_pos,
    }


def analyze_trajectory_correlation(encoder, trajectory_data, transform, device,
                                    crop_size=DEFAULT_CROP_SIZE, patch_size=DEFAULT_PATCH_SIZE):
    """
    Analyze correlation for a single trajectory.

    Uses the same encoding approach as training:
    - Full token representations (no mean pooling)
    - L1 distance computed over all tokens then averaged
    """
    frames = trajectory_data['frames']
    ee_pos = trajectory_data['ee_pos']

    # Encode all frames at once for efficiency
    all_reprs = encode_frames(encoder, frames, transform, device, crop_size, patch_size)
    # all_reprs shape: [T, tokens, D]

    goal_repr = all_reprs[-1]  # [tokens, D]
    goal_pos = ee_pos[-1]

    euclidean_dist_list = []
    latent_dist_list = []

    for i in range(len(frames) - 1):
        pos = ee_pos[i]
        frame_repr = all_reprs[i]  # [tokens, D]

        # Euclidean distance in physical space
        euclidean_dist = np.linalg.norm(goal_pos - pos)

        # L1 distance in latent space (matching loss_fn from train.py)
        # Compute mean absolute difference across all tokens and dimensions
        latent_dist = torch.abs(frame_repr - goal_repr).mean().item()

        euclidean_dist_list.append(euclidean_dist)
        latent_dist_list.append(latent_dist)

    return euclidean_dist_list, latent_dist_list


def find_droid_episodes(base_path, max_episodes=None):
    """Find DROID episode directories."""
    trajectory_files = glob.glob(f"{base_path}/**/success/**/trajectory.h5", recursive=True)
    episodes = [str(Path(f).parent) for f in trajectory_files]

    if max_episodes:
        episodes = episodes[:max_episodes]

    return episodes


def main():
    parser = argparse.ArgumentParser(
        description='Analyze latent-physical correlation on raw DROID data'
    )
    parser.add_argument(
        '--droid_path',
        type=str,
        default='/data/droid_raw/1.0.1',
        help='Path to DROID raw data',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/data/s185927/vjepa2/weights/vitg-384.pt',
        help='Path to pretrained checkpoint',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/s185927/thesis/experiments/sim_training_different_fractions_x_axis_finetune/test_results',
        help='Output directory',
    )
    parser.add_argument(
        '--max_trajectories',
        type=int,
        default=10,
        help='Maximum number of trajectories to process',
    )
    parser.add_argument(
        '--crop_size',
        type=int,
        default=DEFAULT_CROP_SIZE,
        help=f'Crop size for preprocessing (default: {DEFAULT_CROP_SIZE}, matching checkpoint)',
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f'Patch size (default: {DEFAULT_PATCH_SIZE})',
    )

    args = parser.parse_args()

    # Find episodes
    print(f"Finding DROID episodes in {args.droid_path}...")
    episodes = find_droid_episodes(args.droid_path, args.max_trajectories)
    print(f"Found {len(episodes)} episodes")

    # Load encoder from checkpoint
    encoder, device = load_pretrained_encoder(args.checkpoint, args.device, args.crop_size)

    # Create transform (same as training/notebook)
    transform = create_transform(args.crop_size)
    print(f"Using crop_size={args.crop_size}, patch_size={args.patch_size}")
    tokens_per_frame = (args.crop_size // args.patch_size) ** 2
    print(f"Tokens per frame: {tokens_per_frame}")

    # Collect data
    all_euclidean_dist = []
    all_latent_dist = []

    for episode_path in tqdm(episodes, desc="Processing DROID trajectories", unit="traj"):
        try:
            trajectory_data = load_droid_trajectory_data(episode_path, args.droid_path)
            euclidean_dist_list, latent_dist_list = analyze_trajectory_correlation(
                encoder, trajectory_data, transform, device, args.crop_size, args.patch_size
            )
            all_euclidean_dist.extend(euclidean_dist_list)
            all_latent_dist.extend(latent_dist_list)
        except Exception as e:
            print(f"  Warning: Failed to process {episode_path}: {e}")
            continue

    print(f"\nCollected {len(all_euclidean_dist)} data points")

    # Compute correlation
    correlation = np.corrcoef(all_euclidean_dist, all_latent_dist)[0, 1]
    print(f"Correlation coefficient: {correlation:.4f}")

    # Create plot
    print("Creating plot...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=PLOT_PARAMS["figsize_plots_only"])

    ax.scatter(all_euclidean_dist, all_latent_dist, alpha=0.3, s=20, color='purple')

    # Trend line
    z = np.polyfit(all_euclidean_dist, all_latent_dist, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(all_euclidean_dist), max(all_euclidean_dist), 100)
    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2)

    configure_axis(
        ax,
        xlabel='Euclidean Distance to Goal (m)',
        ylabel='Latent L1 Distance to Goal (mean)',
        title=f'Raw DROID Data (Pretrained Backbone, {args.crop_size}px)\nr = {correlation:.4f}'
    )

    plt.tight_layout()

    plot_path = output_dir / f'latent_physical_correlation_droid_raw_{args.crop_size}px.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # Save data
    data_path = output_dir / f'latent_physical_correlation_droid_raw_{args.crop_size}px.npz'
    np.savez(
        data_path,
        euclidean_dist=all_euclidean_dist,
        latent_dist=all_latent_dist,
        correlation=correlation,
        crop_size=args.crop_size,
        patch_size=args.patch_size,
    )
    print(f"Data saved to: {data_path}")

    # Compute variance at low Euclidean distance
    near_goal_mask = np.array(all_euclidean_dist) < 0.02
    if near_goal_mask.sum() > 0:
        near_goal_latent = np.array(all_latent_dist)[near_goal_mask]
        print(f"\nNear-goal analysis (Euclidean < 0.02m):")
        print(f"  N points: {len(near_goal_latent)}")
        print(f"  Latent distance range: {near_goal_latent.min():.2f} - {near_goal_latent.max():.2f}")
        print(f"  Latent distance std: {near_goal_latent.std():.2f}")
        print(f"  Latent distance mean: {near_goal_latent.mean():.2f}")

    # Save a sample frame for visual comparison
    if episodes:
        try:
            sample_data = load_droid_trajectory_data(episodes[0], args.droid_path)
            sample_frame = Image.fromarray(sample_data['frames'][0])
            sample_frame.save(output_dir / 'sample_frame_droid_raw.png')
            print(f"Sample frame saved to: {output_dir / 'sample_frame_droid_raw.png'}")
        except Exception as e:
            print(f"Could not save sample frame: {e}")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
