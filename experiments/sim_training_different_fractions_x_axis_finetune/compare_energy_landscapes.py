#!/usr/bin/env python3
"""
Part 3: Energy Landscape Comparison

Compare energy landscapes across all 5 models (Meta baseline + 4 finetuned)
for the same start/goal pair to visualize if finetuning improves the landscape.

This script:
1. Loads a test trajectory's start and goal frames
2. Computes energy landscapes for all 5 models
3. Creates a side-by-side comparison plot
"""

import argparse
import copy
import glob
import json
import sys
from pathlib import Path

import h5py
import imageio
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
from app.vjepa_droid.transforms import make_transforms
from plot_config import PLOT_PARAMS, configure_axis


# Model configurations - same as analyze_latent_physical_correlation.py
MODELS = {
    "Meta Baseline": {
        "checkpoint": "/home/s185927/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt",
        "config": "/home/s185927/thesis/vjepa2/configs/train/vitg16/x_axis_finetune/x_axis_finetune_025pct.yaml",
    },
    "25% Finetuned": {
        "checkpoint": "/data/s185927/vjepa2/weights/droid/fine_tuned/x_axis_finetune_025pct/best.pt",
        "config": "/home/s185927/thesis/vjepa2/configs/train/vitg16/x_axis_finetune/x_axis_finetune_025pct.yaml",
    },
    "50% Finetuned": {
        "checkpoint": "/data/s185927/vjepa2/weights/droid/fine_tuned/x_axis_finetune_050pct/best.pt",
        "config": "/home/s185927/thesis/vjepa2/configs/train/vitg16/x_axis_finetune/x_axis_finetune_050pct.yaml",
    },
    "75% Finetuned": {
        "checkpoint": "/data/s185927/vjepa2/weights/droid/fine_tuned/x_axis_finetune_075pct/best.pt",
        "config": "/home/s185927/thesis/vjepa2/configs/train/vitg16/x_axis_finetune/x_axis_finetune_075pct.yaml",
    },
    "100% Finetuned": {
        "checkpoint": "/data/s185927/vjepa2/weights/droid/fine_tuned/x_axis_finetune_100pct/best.pt",
        "config": "/home/s185927/thesis/vjepa2/configs/train/vitg16/x_axis_finetune/x_axis_finetune_100pct.yaml",
    },
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
    encoder.load_state_dict(checkpoint['encoder'], strict=False)

    # Load predictor if available
    if 'predictor' in checkpoint:
        predictor.load_state_dict(checkpoint['predictor'], strict=False)

    # Load target encoder if available
    target_encoder = copy.deepcopy(encoder)
    if 'target_encoder' in checkpoint:
        target_encoder.load_state_dict(checkpoint['target_encoder'], strict=False)

    encoder.eval()
    predictor.eval()
    target_encoder.eval()

    return encoder, predictor, target_encoder, device


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
    Encode current and goal frames together in a single forward pass.

    This matches the approach used in robo_samples.py:
    - Stack frames as a 2-frame video [2, H, W, C]
    - Apply transform to get [C, 2, H, W]
    - Forward through encoder
    - Extract z_n (first frame tokens) and z_goal (last frame tokens)

    Args:
        encoder: V-JEPA encoder model
        current_frame: Current observation frame (H, W, C) numpy array
        goal_frame: Goal frame (H, W, C) numpy array
        transform: Transform function from make_transforms
        device: Torch device
        tokens_per_frame: Number of tokens per frame
        normalize: Whether to apply layer normalization

    Returns:
        z_n: Current frame representation [1, tokens_per_frame, D]
        z_goal: Goal frame representation [1, tokens_per_frame, D]
    """
    # Stack frames as [2, H, W, C] - this is the format expected by the transform
    combined_rgb = np.stack([current_frame, goal_frame], axis=0)

    # Apply transform: output is [C, T, H, W] where T=2
    clips = transform(combined_rgb).unsqueeze(0).to(device)  # [1, C, 2, H, W]

    # Forward through encoder using the same approach as robo_samples.py
    # The forward_target function in robo_samples.py does:
    # 1. Permute to [B, T, C, H, W]
    # 2. Flatten B and T: [B*T, C, H, W]
    # 3. Unsqueeze and repeat to make each frame a 2-frame clip: [B*T, C, 2, H, W]
    # 4. Encode to get [B*T, num_tokens, D]
    # 5. View back to [B, T, num_tokens, D] and flatten to [B, T*num_tokens, D]

    B, C, T, H, W = clips.size()

    with torch.no_grad():
        # Permute to [B, T, C, H, W], flatten to [B*T, C, H, W]
        c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [2, C, H, W]
        # Unsqueeze and repeat to make 2-frame clips: [B*T, C, 2, H, W]
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [2, C, 2, H, W]
        # Encode
        h = encoder(c)  # [2, num_tokens, D]
        # View back to [B, T, num_tokens, D] and flatten to [B, T*num_tokens, D]
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)  # [1, 2*num_tokens, D]

        if normalize:
            h = F.layer_norm(h, (h.size(-1),))

        # Extract z_n (first frame) and z_goal (last frame)
        z_n = h[:, :tokens_per_frame].contiguous()  # [1, tokens_per_frame, D]
        z_goal = h[:, -tokens_per_frame:].contiguous()  # [1, tokens_per_frame, D]

    return z_n, z_goal


def compute_energy_landscape(
    encoder, predictor, tokens_per_frame,
    current_rep, current_state, goal_rep,
    nsamples=9, grid_size=0.075, device='cuda'
):
    """
    Compute energy landscape using 3D action grid (like official V-JEPA 2 notebook).

    Samples a 3D grid of actions (dx, dy, dz) and projects onto X-Y plane.

    Returns:
        heatmap: 2D numpy array of energy values (summed across dz)
        x_edges, y_edges: Bin edges for axes
        optimal_action: The action with minimum energy
    """
    # Create 3D action grid (like official notebook)
    action_samples = []
    for dx in np.linspace(-grid_size, grid_size, nsamples):
        for dy in np.linspace(-grid_size, grid_size, nsamples):
            for dz in np.linspace(-grid_size, grid_size, nsamples):
                action = [dx, dy, dz, 0.0, 0.0, 0.0, 0.0]
                action_samples.append(torch.tensor(action, device=device, dtype=current_rep.dtype))

    action_samples = torch.stack(action_samples, dim=0).unsqueeze(1)  # [N^3, 1, 7]
    num_samples = nsamples ** 3

    # Expand representations for batch processing
    z_batch = current_rep[:, :tokens_per_frame].repeat(num_samples, 1, 1)
    s_batch = current_state.repeat(num_samples, 1, 1)

    # Predict next representations
    with torch.no_grad():
        z_pred = predictor(z_batch, action_samples, s_batch)[:, -tokens_per_frame:]
        z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))

    # Compute L1 energy (distance to goal)
    goal_expanded = goal_rep[:, :tokens_per_frame].repeat(num_samples, 1, 1)
    energy = torch.mean(torch.abs(z_pred - goal_expanded), dim=[1, 2])
    energy = energy.cpu().numpy()

    # Find optimal action
    min_idx = np.argmin(energy)
    optimal_action = action_samples[min_idx, 0, :3].cpu().numpy()  # dx, dy, dz

    # Extract action deltas for histogram (project onto X-Y plane)
    actions_np = action_samples[:, 0, :].cpu().numpy()
    x_vals = actions_np[:, 0]  # dx
    y_vals = actions_np[:, 1]  # dy

    # Create 2D histogram (summing energies across dz values)
    heatmap, x_edges, y_edges = np.histogram2d(
        x_vals, y_vals, weights=energy, bins=nsamples
    )

    return heatmap, x_edges, y_edges, optimal_action, energy.min()


def load_test_trajectory(episode_dir):
    """Load a test trajectory."""
    episode_path = Path(episode_dir)

    # Load trajectory
    with h5py.File(episode_path / "trajectory.h5", 'r') as f:
        ee_pos = f['observation/robot_state/cartesian_position'][:][:, :3]

    # Load video frames
    video_files = glob.glob(str(episode_path / "recordings" / "MP4" / "*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video files found in {episode_path / 'recordings' / 'MP4'}")

    reader = imageio.get_reader(video_files[0])
    all_frames = [frame for frame in reader]
    reader.close()

    # Subsample if needed
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

    return frames, ee_pos


def plot_comparison(
    landscapes, model_names, ground_truth_action, start_pos, goal_pos, output_path
):
    """Create side-by-side comparison plot of energy landscapes."""
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, landscape_data) in enumerate(zip(model_names, landscapes)):
        ax = axes[idx]
        heatmap, x_edges, y_edges, optimal_action, min_energy = landscape_data

        im = ax.imshow(
            heatmap.T,
            origin='lower',
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap='viridis',
            aspect='equal'
        )

        # Mark the ground truth action direction
        ax.scatter([ground_truth_action[0]], [ground_truth_action[1]],
                   c='red', marker='x', s=100, linewidths=2, label='Ground Truth')

        # Mark the optimal action (minimum energy)
        ax.scatter([optimal_action[0]], [optimal_action[1]],
                   c='white', marker='o', s=100, linewidths=2, edgecolors='black',
                   label='Predicted Optimal')

        # Mark origin (no action)
        ax.scatter([0], [0], c='gray', marker='+', s=80, linewidths=2)

        ax.set_xlabel('Delta X (m)', fontsize=PLOT_PARAMS["label_size"] - 6)
        ax.set_ylabel('Delta Y (m)', fontsize=PLOT_PARAMS["label_size"] - 6)
        ax.set_title(f'{model_name}\nMin E={min_energy:.4f}', fontsize=PLOT_PARAMS["title_size"] - 10)

        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add overall title
    distance_to_goal = np.linalg.norm(goal_pos - start_pos)
    fig.suptitle(f'Energy Landscape Comparison\nDistance to Goal: {distance_to_goal:.3f}m',
                 fontsize=PLOT_PARAMS["title_size"] - 4, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare energy landscapes across x-axis finetuned models'
    )
    parser.add_argument(
        '--episode_dir',
        type=str,
        default='/data/s185927/droid_sim/axis_aligned/episode_0002',
        help='Path to test episode directory',
    )
    parser.add_argument(
        '--frame_idx',
        type=int,
        default=10,
        help='Frame index to use as start (goal is always the last frame)',
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
        default='/home/s185927/thesis/experiments/sim_training_different_fractions_x_axis_finetune',
        help='Output directory for plots',
    )
    parser.add_argument(
        '--nsamples',
        type=int,
        default=15,
        help='Grid resolution for energy landscape',
    )
    parser.add_argument(
        '--grid_size',
        type=float,
        default=0.075,
        help='Range of action sampling in meters',
    )

    args = parser.parse_args()

    print("Part 3: Energy Landscape Comparison")
    print("=" * 60)

    # Load test trajectory
    print(f"\nLoading test trajectory from: {args.episode_dir}")
    frames, ee_pos = load_test_trajectory(args.episode_dir)
    print(f"Loaded {len(frames)} frames")

    # Get start and goal
    start_frame = frames[args.frame_idx]
    goal_frame = frames[-1]
    start_pos = ee_pos[args.frame_idx]
    goal_pos = ee_pos[-1]

    # Compute ground truth action direction (normalized to grid_size)
    delta = goal_pos - start_pos
    distance = np.linalg.norm(delta)
    if distance > 0:
        # Scale to be visible in the grid
        scale = min(args.grid_size * 0.8, distance)
        ground_truth_action = (delta / distance) * scale
    else:
        ground_truth_action = np.array([0.0, 0.0, 0.0])

    print(f"\nStart position: {start_pos}")
    print(f"Goal position: {goal_pos}")
    print(f"Distance to goal: {distance:.4f}m")
    print(f"Ground truth direction (scaled): {ground_truth_action[:2]}")

    # Compute energy landscapes for each model
    landscapes = []
    model_names = list(MODELS.keys())

    for model_name, model_info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Computing energy landscape for: {model_name}")
        print(f"{'='*60}")

        # Load config
        with open(model_info['config'], 'r') as f:
            config = yaml.safe_load(f)

        crop_size = config['data']['crop_size']
        patch_size = config['data']['patch_size']
        tokens_per_frame = (crop_size // patch_size) ** 2

        # Load model
        encoder, predictor, target_encoder, device = load_model(
            model_info['checkpoint'], config, args.device
        )

        # Create transform (same as robo_samples.py)
        transform = create_transform(crop_size)

        # Encode frames together (same approach as robo_samples.py)
        print("Encoding start and goal frames together...")
        current_rep, goal_rep = encode_frames_together(
            encoder, start_frame, goal_frame, transform, device, tokens_per_frame
        )

        # Create current state tensor
        current_state = torch.tensor(
            [[list(start_pos) + [0.0, 0.0, 0.0, 0.0]]],  # [x,y,z,roll,pitch,yaw,grip]
            device=device, dtype=current_rep.dtype
        )

        # Compute energy landscape
        print("Computing energy landscape...")
        landscape_data = compute_energy_landscape(
            encoder, predictor, tokens_per_frame,
            current_rep, current_state, goal_rep,
            nsamples=args.nsamples, grid_size=args.grid_size, device=device
        )
        landscapes.append(landscape_data)

        print(f"  Optimal action: dx={landscape_data[3][0]:.4f}, dy={landscape_data[3][1]:.4f}")
        print(f"  Min energy: {landscape_data[4]:.4f}")

        # Clean up GPU memory
        del encoder, predictor, target_encoder
        torch.cuda.empty_cache()

    # Create comparison plot
    print("\n" + "=" * 60)
    print("Creating comparison plot...")
    output_path = Path(args.output_dir) / 'energy_landscape_comparison.png'
    plot_comparison(
        landscapes, model_names, ground_truth_action, start_pos, goal_pos, output_path
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Energy Landscape Analysis")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Optimal dX':<12} {'Optimal dY':<12} {'Min Energy':<12}")
    print("-" * 60)
    for model_name, landscape_data in zip(model_names, landscapes):
        opt = landscape_data[3]
        min_e = landscape_data[4]
        print(f"{model_name:<20} {opt[0]:<12.4f} {opt[1]:<12.4f} {min_e:<12.4f}")
    print(f"\nGround Truth: dX={ground_truth_action[0]:.4f}, dY={ground_truth_action[1]:.4f}")
    print("=" * 60)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
