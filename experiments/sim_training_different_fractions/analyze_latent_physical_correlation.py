#!/usr/bin/env python3
"""
Analyze correlation between Euclidean distance and latent L1 distance.

This script creates a scatter plot to investigate whether the latent distance
correlates with physical distance (3D Euclidean distance) to the goal. If the
scatter plot is a cloud rather than a tight increasing curve, CEM is optimizing
the wrong metric.
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
import yaml
from PIL import Image
from tqdm import tqdm

# Add vjepa2 directory to import modules
sys.path.insert(0, "/home/s185927/thesis/vjepa2")
# Add thesis directory for plot config
sys.path.insert(0, "/home/s185927/thesis")

from app.vjepa_droid.utils import init_video_model
from plot_config import PLOT_PARAMS, configure_axis


def load_model(checkpoint_path, config, device='cuda:0'):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Extract config parameters
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

    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device)
        torch.cuda.set_device(device)

    # Initialize model
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'], strict=False)

    # Load target encoder if available
    import copy
    target_encoder = copy.deepcopy(encoder)
    if 'target_encoder' in checkpoint:
        target_encoder.load_state_dict(checkpoint['target_encoder'], strict=False)

    encoder.eval()
    target_encoder.eval()

    return encoder, target_encoder, device


def preprocess_frame(frame, crop_size=256):
    """Preprocess a single frame for model input."""
    # Resize to crop_size maintaining aspect ratio
    img = Image.fromarray(frame)

    # Center crop
    width, height = img.size
    new_width, new_height = crop_size, crop_size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to tensor and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    return img_array


def encode_frame(encoder, frame, device, crop_size=256, patch_size=16, normalize=True):
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

        # h shape: (B, num_patches, embed_dim)
        # For our single frame: (1, num_patches, embed_dim)

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
        # Load end-effector positions (cartesian position)
        # Note: This is (T, 6) where first 3 are xyz position
        ee_pos_full = f['observation/robot_state/cartesian_position'][:]
        ee_pos = ee_pos_full[:, :3]  # Take only xyz position, Shape: (T, 3)

    # Load video frames - find the first MP4 file
    import glob
    video_files = glob.glob(str(episode_path / "recordings" / "MP4" / "*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video files found in {episode_path / 'recordings' / 'MP4'}")
    video_path = video_files[0]

    # Use imageio to read frames
    import imageio
    reader = imageio.get_reader(video_path)
    all_frames = []
    for frame in reader:
        all_frames.append(frame)
    reader.close()

    # Subsample video frames to match trajectory length
    # The video might be recorded at higher FPS than trajectory sampling
    num_trajectory_steps = len(ee_pos)
    num_video_frames = len(all_frames)

    if num_video_frames > num_trajectory_steps:
        # Subsample video frames to match trajectory
        indices = np.linspace(0, num_video_frames - 1, num_trajectory_steps, dtype=int)
        frames = [all_frames[i] for i in indices]
    elif num_video_frames < num_trajectory_steps:
        # Subsample trajectory to match video
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


def analyze_trajectory_correlation(
    encoder,
    trajectory_data,
    device,
    crop_size=256,
    patch_size=16,
):
    """
    Analyze correlation between Euclidean distance and latent distance for a single trajectory.

    Returns:
        euclidean_dist_list: List of Euclidean distances (3D physical distance to goal)
        latent_dist_list: List of latent L1 distances to goal representation
    """
    frames = trajectory_data['frames']
    ee_pos = trajectory_data['ee_pos']

    # Goal is the last frame
    goal_frame = frames[-1]
    goal_pos = ee_pos[-1]

    # Encode goal
    goal_repr = encode_frame(encoder, goal_frame, device, crop_size, patch_size)

    euclidean_dist_list = []
    latent_dist_list = []

    # Analyze each frame along the trajectory
    for i in range(len(frames) - 1):  # Exclude goal frame itself
        frame = frames[i]
        pos = ee_pos[i]

        # Compute Euclidean distance (3D distance in xyz space)
        euclidean_dist = np.linalg.norm(goal_pos - pos)

        # Encode frame
        frame_repr = encode_frame(encoder, frame, device, crop_size, patch_size)

        # Compute latent L1 distance
        latent_dist = torch.abs(frame_repr - goal_repr).sum().item()

        euclidean_dist_list.append(euclidean_dist)
        latent_dist_list.append(latent_dist)

    return euclidean_dist_list, latent_dist_list


def main():
    parser = argparse.ArgumentParser(
        description='Analyze correlation between Euclidean distance and latent distance'
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        default='/data/s185927/droid_sim/y_axis/test_trajectories.csv',
        help='Path to test trajectories CSV',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_100pct/best.pt',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='/home/s185927/thesis/vjepa2/configs/train/vitg16/ablation/droid-256px-8f_100pct.yaml',
        help='Path to training config',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use',
    )
    parser.add_argument(
        '--output_plot',
        type=str,
        default='/home/s185927/thesis/experiments/sim_training_different_fractions/latent_physical_correlation.png',
        help='Output plot path',
    )
    parser.add_argument(
        '--max_trajectories',
        type=int,
        default=None,
        help='Maximum number of trajectories to process (for testing)',
    )

    args = parser.parse_args()

    # Load config
    print("Loading config...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    encoder, target_encoder, device = load_model(args.checkpoint, config, args.device)

    # Use target encoder for consistency with training
    encoder = target_encoder

    # Load test trajectories
    print("Loading test trajectories...")
    with open(args.test_csv, 'r') as f:
        test_episodes = [line.strip() for line in f if line.strip()]

    if args.max_trajectories:
        test_episodes = test_episodes[:args.max_trajectories]

    print(f"Processing {len(test_episodes)} test trajectories...")

    # Collect all data points
    all_euclidean_dist = []
    all_latent_dist = []

    for idx, episode_path in tqdm(enumerate(test_episodes), total=len(test_episodes), desc="Processing trajectories", unit="trajectory"):
        # print(f"Processing trajectory {idx + 1}/{len(test_episodes)}: {episode_path}")

        try:
            # Load trajectory
            trajectory_data = load_trajectory_data(episode_path)

            # Analyze
            euclidean_dist_list, latent_dist_list = analyze_trajectory_correlation(
                encoder,
                trajectory_data,
                device,
                crop_size=config['data']['crop_size'],
                patch_size=config['data']['patch_size'],
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

    # Create scatter plot
    print("Creating scatter plot...")
    fig, ax = plt.subplots(figsize=PLOT_PARAMS["figsize_plots_only"])

    # Scatter plot with transparency
    ax.scatter(all_euclidean_dist, all_latent_dist, alpha=0.3, s=20)

    # Configure axis with consistent styling
    configure_axis(
        ax,
        xlabel='Euclidean Distance to Goal (m)',
        ylabel='Latent L1 Distance to Goal',
        title=f'Correlation between Physical Distance and Latent Distance\n'
              f'(Correlation: {correlation:.4f}, N={len(all_euclidean_dist)} points)'
    )

    # Add trend line
    z = np.polyfit(all_euclidean_dist, all_latent_dist, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(all_euclidean_dist), max(all_euclidean_dist), 100)
    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')

    ax.legend(fontsize=PLOT_PARAMS["legend_size"])

    plt.tight_layout()

    # Save plot
    output_path = Path(args.output_plot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Save data
    data_path = output_path.with_suffix('.npz')
    np.savez(
        data_path,
        euclidean_dist=all_euclidean_dist,
        latent_dist=all_latent_dist,
        correlation=correlation,
    )
    print(f"Data saved to: {data_path}")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
