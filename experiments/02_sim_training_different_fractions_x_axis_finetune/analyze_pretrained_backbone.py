#!/usr/bin/env python3
"""
Analyze correlation between Euclidean distance and latent L1 distance
using the pretrained V-JEPA2 backbone (not action-conditioned).

This tests whether the variance in latent representations for similar physical
states is inherent to the V-JEPA2 encoder or introduced during action-conditioned training.
"""

import argparse
import json
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

from src.models.vision_transformer import vit_giant_xformers
from plot_config import PLOT_PARAMS, configure_axis


def load_pretrained_encoder(checkpoint_path, device='cuda:0', crop_size=384):
    """Load pretrained V-JEPA2 encoder (not action-conditioned)."""
    print(f"Loading pretrained checkpoint: {checkpoint_path}")

    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device)
        torch.cuda.set_device(device)

    # Initialize encoder with 384px crop size
    # num_frames > 1 is required to use 3D patch embedding (video mode)
    print("Initializing encoder...")
    encoder = vit_giant_xformers(
        img_size=crop_size,
        patch_size=16,
        num_frames=16,  # Use video mode with 3D patch embedding
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
        use_sdpa=True,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle the module.backbone. prefix in state dict
    encoder_state = checkpoint.get('target_encoder', checkpoint.get('encoder', {}))

    # Remove 'module.backbone.' prefix if present
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

    # Load state dict
    missing, unexpected = encoder.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    encoder = encoder.to(device)
    encoder.eval()

    print(f"  Encoder loaded successfully")
    return encoder, device


def preprocess_frame(frame, crop_size=384):
    """Preprocess a single frame for model input."""
    img = Image.fromarray(frame)

    # Resize to crop_size (the pretrained model uses 384px)
    img = img.resize((crop_size, crop_size), Image.BILINEAR)

    # Convert to tensor and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    return img_array


def encode_frame(encoder, frame, device, crop_size=384, normalize=True):
    """Encode a single frame using the encoder."""
    # Preprocess
    img = preprocess_frame(frame, crop_size)

    # Convert to tensor: (C, H, W)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

    # Add batch and temporal dimensions: (B, C, T, H, W)
    # For single frame, we repeat it twice as required by the model
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        # Encode
        h = encoder(img_tensor)

        # Normalize if requested
        if normalize:
            h = F.layer_norm(h, (h.size(-1),))

        # Average pool over patches to get single vector per frame
        h = h.mean(dim=1)  # (1, embed_dim)

    return h.squeeze(0)  # (embed_dim,)


def load_trajectory_data(episode_path):
    """Load trajectory data from an episode directory."""
    episode_path = Path(episode_path)

    # Load metadata
    with open(episode_path / "metadata_sim.json", 'r') as f:
        metadata = json.load(f)

    # Load trajectory
    with h5py.File(episode_path / "trajectory.h5", 'r') as f:
        ee_pos_full = f['observation/robot_state/cartesian_position'][:]
        ee_pos = ee_pos_full[:, :3]

    # Load video frames
    import glob
    video_files = glob.glob(str(episode_path / "recordings" / "MP4" / "*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video files found in {episode_path / 'recordings' / 'MP4'}")
    video_path = video_files[0]

    import imageio
    reader = imageio.get_reader(video_path)
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
        'metadata': metadata,
    }


def analyze_trajectory_correlation(encoder, trajectory_data, device, crop_size=384):
    """Analyze correlation for a single trajectory."""
    frames = trajectory_data['frames']
    ee_pos = trajectory_data['ee_pos']

    goal_frame = frames[-1]
    goal_pos = ee_pos[-1]
    goal_repr = encode_frame(encoder, goal_frame, device, crop_size)

    euclidean_dist_list = []
    latent_dist_list = []

    for i in range(len(frames) - 1):
        frame = frames[i]
        pos = ee_pos[i]

        euclidean_dist = np.linalg.norm(goal_pos - pos)
        frame_repr = encode_frame(encoder, frame, device, crop_size)
        latent_dist = torch.abs(frame_repr - goal_repr).sum().item()

        euclidean_dist_list.append(euclidean_dist)
        latent_dist_list.append(latent_dist)

    return euclidean_dist_list, latent_dist_list


def main():
    parser = argparse.ArgumentParser(
        description='Analyze latent-physical correlation for pretrained V-JEPA2 backbone'
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        default='/data/s185927/droid_sim/axis_aligned/x/test_trajectories.csv',
        help='Path to test trajectories CSV',
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
        default=None,
        help='Maximum number of trajectories to process',
    )
    parser.add_argument(
        '--crop_size',
        type=int,
        default=384,
        help='Crop size for preprocessing (pretrained uses 384)',
    )

    args = parser.parse_args()

    # Load test trajectories
    print("Loading test trajectories...")
    with open(args.test_csv, 'r') as f:
        test_episodes = [line.strip() for line in f if line.strip()]
    print(f"Found {len(test_episodes)} test trajectories")

    if args.max_trajectories:
        test_episodes = test_episodes[:args.max_trajectories]

    # Load encoder
    encoder, device = load_pretrained_encoder(args.checkpoint, args.device, args.crop_size)

    # Collect data
    all_euclidean_dist = []
    all_latent_dist = []

    for episode_path in tqdm(test_episodes, desc="Processing trajectories", unit="traj"):
        try:
            trajectory_data = load_trajectory_data(episode_path)
            euclidean_dist_list, latent_dist_list = analyze_trajectory_correlation(
                encoder, trajectory_data, device, args.crop_size
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

    ax.scatter(all_euclidean_dist, all_latent_dist, alpha=0.3, s=20, color='green')

    # Trend line
    z = np.polyfit(all_euclidean_dist, all_latent_dist, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(all_euclidean_dist), max(all_euclidean_dist), 100)
    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2)

    configure_axis(
        ax,
        xlabel='Euclidean Distance to Goal (m)',
        ylabel='Latent L1 Distance to Goal',
        title=f'Pretrained Backbone (384px)\nr = {correlation:.4f}'
    )

    plt.tight_layout()

    plot_path = output_dir / 'latent_physical_correlation_pretrained_backbone.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # Save data
    data_path = output_dir / 'latent_physical_correlation_pretrained_backbone.npz'
    np.savez(
        data_path,
        euclidean_dist=all_euclidean_dist,
        latent_dist=all_latent_dist,
        correlation=correlation,
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

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
