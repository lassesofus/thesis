#!/usr/bin/env python3
"""
Analyze correlation between Euclidean distance and latent L1 distance
for zero-shot axis-aligned reaching trajectories.

This script analyzes the pretrained V-JEPA 2 encoder's latent space
to investigate how well latent distances correlate with physical distances
along x, y, and z axes separately.
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
import pdb

# Add vjepa2 directory to import modules
sys.path.insert(0, "/home/s185927/thesis/vjepa2")
# Add thesis directory for plot config
sys.path.insert(0, "/home/s185927/thesis")

from app.vjepa_droid.utils import init_video_model
from app.vjepa_droid.transforms import make_transforms
from plot_config import PLOT_PARAMS, configure_axis


# Model configuration (pretrained encoder only for zero-shot analysis)
MODEL_CONFIG = {
    "checkpoint": "/home/s185927/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt",
    "config": "/home/s185927/thesis/vjepa2/configs/train/vitg16/x_axis_finetune/x_axis_finetune_025pct.yaml",
}


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

    # Strip 'module.' prefix if present (Meta checkpoints have this prefix)
    encoder_dict = checkpoint['encoder']
    encoder_dict = {k.replace("module.", ""): v for k, v in encoder_dict.items()}
    encoder.load_state_dict(encoder_dict, strict=False)

    # Load target encoder if available
    import copy
    target_encoder = copy.deepcopy(encoder)
    if 'target_encoder' in checkpoint:
        target_encoder_dict = checkpoint['target_encoder']
        target_encoder_dict = {k.replace("module.", ""): v for k, v in target_encoder_dict.items()}
        target_encoder.load_state_dict(target_encoder_dict, strict=False)

    encoder.eval()
    target_encoder.eval()

    return encoder, target_encoder, device


def create_transform(crop_size=256):
    """Create the same transform used in robo_samples.py for inference."""
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
    This matches the approach used in robo_samples.py.

    Args:
        encoder: The encoder model
        current_frame: Current frame as numpy array (H, W, C)
        goal_frame: Goal frame as numpy array (H, W, C)
        transform: Transform to apply to frames
        device: Device to use
        tokens_per_frame: Number of tokens per frame
        normalize: Whether to apply layer normalization

    Returns:
        z_current: Current frame representation (tokens_per_frame, D)
        z_goal: Goal frame representation (tokens_per_frame, D)
    """
    # Stack frames as [2, H, W, C] - this is what the transform expects
    combined_rgb = np.stack([current_frame, goal_frame], axis=0)

    # Apply transform: outputs [C, 2, H, W], then unsqueeze to [1, C, 2, H, W]
    clips = transform(combined_rgb).unsqueeze(0).to(device)

    B, C, T, H, W = clips.size()

    with torch.no_grad():
        # Permute to [B*T, C, H, W] = [2, C, H, W]
        c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # Model expects tubelet_size=2, so repeat to get [2, C, 2, H, W]
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)

        # Encode: output shape [2, num_tokens, D]
        h = encoder(c)

        # Reshape back to [B, T, num_tokens, D] then flatten to [B, T*num_tokens, D]
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)  # [1, 2*num_tokens, D]

        if normalize:
            h = F.layer_norm(h, (h.size(-1),))

        # Split into current and goal representations
        z_current = h[:, :tokens_per_frame].contiguous().squeeze(0)  # [num_tokens, D]
        z_goal = h[:, -tokens_per_frame:].contiguous().squeeze(0)    # [num_tokens, D]

    return z_current, z_goal


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
    transform,
    tokens_per_frame,
    crop_size=256,
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

    euclidean_dist_list = []
    latent_dist_list = []

    # Analyze each frame along the trajectory
    for i in range(len(frames) - 1):  # Exclude goal frame itself
        frame = frames[i]
        pos = ee_pos[i]

        # Compute Euclidean distance (3D distance in xyz space)
        euclidean_dist = np.linalg.norm(goal_pos - pos)

        # Encode current frame and goal frame together (proper approach from robo_samples.py)
        z_current, z_goal = encode_frames_together(
            encoder, frame, goal_frame, transform, device, tokens_per_frame
        )

        # Compute latent L1 distance (mean over all elements, matching CEM objective)
        latent_dist = torch.abs(z_current - z_goal).mean().item()

        euclidean_dist_list.append(euclidean_dist)
        latent_dist_list.append(latent_dist)

    return euclidean_dist_list, latent_dist_list


def analyze_axis(axis_name, test_csv_path, encoder, device, transform, tokens_per_frame, crop_size, max_trajectories=None):
    """Analyze a single axis and return correlation data."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {axis_name}-axis trajectories")
    print(f"{'='*60}")

    # Load test trajectories
    with open(test_csv_path, 'r') as f:
        test_episodes = [line.strip() for line in f if line.strip()]
    print(f"  Found {len(test_episodes)} test trajectories")

    # Collect all data points
    all_euclidean_dist = []
    all_latent_dist = []
    all_episode_idx = []

    episodes_to_process = test_episodes[:max_trajectories] if max_trajectories else test_episodes

    for ep_idx, episode_path in enumerate(tqdm(episodes_to_process, desc=f"Processing {axis_name}-axis", unit="traj")):
        try:
            trajectory_data = load_trajectory_data(episode_path)
            euclidean_dist_list, latent_dist_list = analyze_trajectory_correlation(
                encoder,
                trajectory_data,
                device,
                transform=transform,
                tokens_per_frame=tokens_per_frame,
                crop_size=crop_size,
            )
            all_euclidean_dist.extend(euclidean_dist_list)
            all_latent_dist.extend(latent_dist_list)
            all_episode_idx.extend([ep_idx] * len(euclidean_dist_list))
        except Exception as e:
            print(f"  Warning: Failed to process {episode_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute correlation
    correlation = np.corrcoef(all_euclidean_dist, all_latent_dist)[0, 1]
    print(f"  Correlation: {correlation:.4f}, N={len(all_euclidean_dist)} points")

    return {
        'euclidean_dist': np.array(all_euclidean_dist),
        'latent_dist': np.array(all_latent_dist),
        'episode_idx': np.array(all_episode_idx),
        'correlation': correlation,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze latent-physical correlation for zero-shot axis-aligned reaching'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/data/s185927/droid_sim/zero_shot_correlation',
        help='Base directory containing {x,y,z}_axis subdirectories',
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
        default='/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching',
        help='Output directory for plots and data',
    )
    parser.add_argument(
        '--max_trajectories',
        type=int,
        default=None,
        help='Maximum number of trajectories to process per axis (for testing)',
    )
    parser.add_argument(
        '--axes',
        type=str,
        nargs='+',
        default=['x', 'y', 'z'],
        help='Axes to analyze',
    )

    args = parser.parse_args()

    # Load config
    with open(MODEL_CONFIG['config'], 'r') as f:
        config = yaml.safe_load(f)

    # Load model once (same encoder for all axes)
    encoder, target_encoder, device = load_model(MODEL_CONFIG['checkpoint'], config, args.device)
    encoder = target_encoder  # Use target encoder for consistency

    # Verify checkpoint was loaded correctly (debug logging)
    param_sum = sum(p.sum().item() for p in encoder.parameters())
    print(f"Checkpoint: {MODEL_CONFIG['checkpoint']}")
    print(f"Encoder param checksum: {param_sum:.2f}")

    # Create transform
    crop_size = config['data']['crop_size']
    patch_size = config['data']['patch_size']
    transform = create_transform(crop_size=crop_size)

    # Calculate tokens per frame
    tokens_per_frame = (crop_size // patch_size) ** 2
    print(f"Tokens per frame: {tokens_per_frame}")

    # Analyze each axis
    results = {}
    for axis in args.axes:
        test_csv = Path(args.data_dir) / f"{axis}_axis" / "test_trajectories.csv"
        if not test_csv.exists():
            print(f"Warning: {test_csv} not found, skipping {axis}-axis")
            continue
        results[axis] = analyze_axis(
            axis,
            test_csv,
            encoder,
            device,
            transform,
            tokens_per_frame,
            crop_size,
            args.max_trajectories,
        )

    # Clean up GPU memory
    del encoder, target_encoder
    torch.cuda.empty_cache()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison plot (1 row, 3 columns for x, y, z)
    print("\nCreating comparison plot...")
    n_axes = len(results)
    fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 5))
    if n_axes == 1:
        axes = [axes]

    # Match colors from analyze_axis_correlation.py: Blue, Green, Orange for X, Y, Z
    colors = {'x': '#1f77b4', 'y': '#2ca02c', 'z': '#ff7f0e'}

    for idx, (axis_name, data) in enumerate(results.items()):
        ax = axes[idx]

        # Scatter plot with transparency
        ax.scatter(
            data['euclidean_dist'],
            data['latent_dist'],
            alpha=0.3,
            s=20,
            color=colors.get(axis_name, 'tab:gray'),
        )

        # Add trend line
        z = np.polyfit(data['euclidean_dist'], data['latent_dist'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(data['euclidean_dist']), max(data['euclidean_dist']), 100)
        ax.plot(x_trend, p(x_trend), 'k--', linewidth=2)

        # Add Pearson correlation as text annotation
        ax.text(
            0.95, 0.05, f"Pearson r = {data['correlation']:.2f}",
            transform=ax.transAxes,
            fontsize=PLOT_PARAMS["legend_size"] + 2,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Configure axis
        configure_axis(
            ax,
            xlabel='Euclidean Distance to Goal (m)',
            ylabel='Mean L1 Distance to Goal',
            title=f'{axis_name.upper()}-Axis Movement'
        )

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / 'latent_physical_correlation_by_axis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")

    # Create combined plot (all axes on single figure)
    fig, ax = plt.subplots(figsize=(8, 6))

    for axis_name, data in results.items():
        ax.scatter(
            data['euclidean_dist'],
            data['latent_dist'],
            alpha=0.3,
            s=20,
            color=colors.get(axis_name, 'tab:gray'),
            label=f"{axis_name.upper()}-axis (r={data['correlation']:.2f})",
        )

    ax.legend(loc='lower right', fontsize=PLOT_PARAMS["legend_size"])
    configure_axis(
        ax,
        xlabel='Euclidean Distance to Goal (m)',
        ylabel='Mean L1 Distance to Goal',
        title='Latent-Physical Correlation by Axis (Zero-Shot)'
    )

    plt.tight_layout()
    combined_plot_path = output_dir / 'latent_physical_correlation_combined.png'
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {combined_plot_path}")

    # Save data
    data_path = output_dir / 'latent_physical_correlation_data.npz'
    save_dict = {}
    for axis_name, data in results.items():
        save_dict[f'{axis_name}_euclidean'] = data['euclidean_dist']
        save_dict[f'{axis_name}_latent'] = data['latent_dist']
        save_dict[f'{axis_name}_episode_idx'] = data['episode_idx']
        save_dict[f'{axis_name}_correlation'] = data['correlation']
    np.savez(data_path, **save_dict)
    print(f"Data saved to: {data_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Latent-Physical Distance Correlations (Zero-Shot)")
    print("=" * 60)
    for axis_name, data in results.items():
        print(f"  {axis_name.upper()}-axis: r = {data['correlation']:.4f}, N = {len(data['euclidean_dist'])}")
    print("=" * 60)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
