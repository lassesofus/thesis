#!/usr/bin/env python3
"""
Probe A Extended: Energy Landscape with Action-Outcome Correlation

Extends the original Probe A to save:
- Full energy array E(a) for all sampled actions
- Start position s_k and goal position s_goal
- True distances D(a) = ||s_k + a[:3] - s_goal|| for each action
- Spearman and Pearson correlation between E(a) and D(a)

This allows direct validation of whether the energy ranking matches true action quality.
"""

import argparse
import copy
import glob
import json
import os
import random
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# Add vjepa2 directory to import modules
sys.path.insert(0, "/home/s185927/thesis/vjepa2")

from app.vjepa_droid.utils import init_video_model
from app.vjepa_droid.transforms import make_transforms


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

    def strip_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    encoder.load_state_dict(strip_module_prefix(checkpoint['encoder']), strict=False)

    if 'predictor' in checkpoint:
        predictor.load_state_dict(strip_module_prefix(checkpoint['predictor']), strict=False)

    target_encoder = copy.deepcopy(encoder)
    if 'target_encoder' in checkpoint:
        target_encoder.load_state_dict(strip_module_prefix(checkpoint['target_encoder']), strict=False)

    encoder.eval()
    predictor.eval()
    target_encoder.eval()

    return encoder, predictor, target_encoder, device


def create_transform(crop_size=256):
    """Create the same transform used for inference."""
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
    """Encode current and goal frames together in a single forward pass."""
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

        z_n = h[:, :tokens_per_frame].contiguous()
        z_goal = h[:, -tokens_per_frame:].contiguous()

    return z_n, z_goal


def load_trajectory(episode_dir):
    """Load a trajectory's frames and positions."""
    episode_path = Path(episode_dir)

    with h5py.File(episode_path / "trajectory.h5", 'r') as f:
        ee_pos = f['observation/robot_state/cartesian_position'][:][:, :3]

    video_files = glob.glob(str(episode_path / "recordings" / "MP4" / "*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video files found in {episode_path / 'recordings' / 'MP4'}")

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

    return frames, ee_pos


def compute_energy_for_actions(
    predictor, tokens_per_frame,
    current_rep, current_state, goal_rep,
    actions, device, batch_size=64
):
    """Compute energy (L1 distance to goal) for a set of actions."""
    num_samples = len(actions)
    action_tensor = torch.tensor(actions, device=device, dtype=current_rep.dtype)
    action_tensor = action_tensor.unsqueeze(1)

    all_energies = []
    goal_single = goal_rep[:, :tokens_per_frame]
    z_single = current_rep[:, :tokens_per_frame]

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_actions = action_tensor[i:batch_end]
            actual_batch_size = batch_actions.shape[0]

            z_batch = z_single.repeat(actual_batch_size, 1, 1)
            s_batch = current_state.repeat(actual_batch_size, 1, 1)

            z_pred = predictor(z_batch, batch_actions, s_batch)[:, -tokens_per_frame:]
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))

            goal_expanded = goal_single.repeat(actual_batch_size, 1, 1)
            energy = torch.mean(torch.abs(z_pred - goal_expanded), dim=[1, 2])
            all_energies.append(energy.cpu())

    return torch.cat(all_energies, dim=0).numpy()


def compute_true_distances(actions, start_pos, goal_pos):
    """
    Compute true distance to goal after applying each action.
    D(a) = ||s_k + a[:3] - s_goal||

    Note: Actions are in DROID frame (dx, dy, dz), positions are in RoboHive frame.
    The action transform swaps x/y and negates, but for distance computation
    we just need the magnitude of movement, so we use actions directly.
    """
    true_distances = []
    for a in actions:
        # Action displacement (first 3 components are dx, dy, dz)
        # Note: This assumes actions are small deltas in the same frame as positions
        # If there's a frame transform, we may need to apply it here
        s_next = start_pos + a[:3]
        d = np.linalg.norm(s_next - goal_pos)
        true_distances.append(d)
    return np.array(true_distances)


def generate_or_load_actions(actions_path, nsamples=512, grid_size=0.075, seed=42):
    """Generate or load fixed action samples."""
    if os.path.exists(actions_path):
        print(f"Loading existing actions from {actions_path}")
        return np.load(actions_path)

    print(f"Generating {nsamples} action samples with seed={seed}")
    np.random.seed(seed)

    actions = []
    for _ in range(nsamples):
        dx = np.random.uniform(-grid_size, grid_size)
        dy = np.random.uniform(-grid_size, grid_size)
        action = [dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0]
        actions.append(action)

    actions = np.array(actions)
    np.save(actions_path, actions)
    print(f"Saved actions to {actions_path}")
    return actions


def run_probe_a_extended(
    checkpoint_path,
    config_path,
    probe_ids,
    data_dir,
    actions,
    device='cuda:0',
    frame_idx=10,
):
    """Run extended Probe A with full energy and correlation data."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    crop_size = config['data']['crop_size']
    patch_size = config['data']['patch_size']
    tokens_per_frame = (crop_size // patch_size) ** 2

    encoder, predictor, target_encoder, device = load_model(checkpoint_path, config, device)
    transform = create_transform(crop_size)

    # Results per probe - now including full arrays
    results = {
        # Original metrics
        'stdE': [],
        'rangeE': [],
        'min_energy': [],
        # New: full data for correlation analysis
        'energies': [],           # Full energy array per probe
        'true_distances': [],     # True distance D(a) per action
        'start_pos': [],          # s_k
        'goal_pos': [],           # s_goal
        # Correlation metrics
        'spearman_corr': [],
        'spearman_pval': [],
        'pearson_corr': [],
        'pearson_pval': [],
    }

    for probe_id in tqdm(probe_ids, desc="Processing probes"):
        episode_dir = os.path.join(data_dir, f"episode_{probe_id:04d}")

        try:
            frames, ee_pos = load_trajectory(episode_dir)
        except Exception as e:
            print(f"Warning: Failed to load trajectory {probe_id}: {e}")
            continue

        start_frame = frames[frame_idx]
        goal_frame = frames[-1]
        start_pos = ee_pos[frame_idx]
        goal_pos = ee_pos[-1]

        # Encode frames
        current_rep, goal_rep = encode_frames_together(
            encoder, start_frame, goal_frame, transform, device, tokens_per_frame
        )

        current_state = torch.tensor(
            [[list(start_pos) + [0.0, 0.0, 0.0, 0.0]]],
            device=device, dtype=current_rep.dtype
        )

        # Compute energy for all actions
        energies = compute_energy_for_actions(
            predictor, tokens_per_frame,
            current_rep, current_state, goal_rep,
            actions, device
        )

        # Compute true distances D(a) = ||s_k + a[:3] - s_goal||
        true_distances = compute_true_distances(actions, start_pos, goal_pos)

        # Compute correlations
        spearman_result = spearmanr(energies, true_distances)
        pearson_result = pearsonr(energies, true_distances)

        # Original metrics
        stdE = np.std(energies)
        rangeE = np.max(energies) - np.min(energies)
        min_energy = np.min(energies)

        # Store all results
        results['stdE'].append(float(stdE))
        results['rangeE'].append(float(rangeE))
        results['min_energy'].append(float(min_energy))
        results['energies'].append(energies.tolist())
        results['true_distances'].append(true_distances.tolist())
        results['start_pos'].append(start_pos.tolist())
        results['goal_pos'].append(goal_pos.tolist())
        results['spearman_corr'].append(float(spearman_result.correlation))
        results['spearman_pval'].append(float(spearman_result.pvalue))
        results['pearson_corr'].append(float(pearson_result[0]))
        results['pearson_pval'].append(float(pearson_result[1]))

    # Aggregate metrics
    aggregated = {
        'sharpness_stdE': float(np.mean(results['stdE'])),
        'sharpness_rangeE': float(np.mean(results['rangeE'])),
        'min_energy_mean': float(np.mean(results['min_energy'])),
        # Correlation aggregates
        'spearman_corr_mean': float(np.mean(results['spearman_corr'])),
        'spearman_corr_std': float(np.std(results['spearman_corr'])),
        'pearson_corr_mean': float(np.mean(results['pearson_corr'])),
        'pearson_corr_std': float(np.std(results['pearson_corr'])),
    }

    # Cleanup
    del encoder, predictor, target_encoder
    torch.cuda.empty_cache()

    return aggregated, results


def main():
    parser = argparse.ArgumentParser(description='Probe A Extended: Energy-Action Correlation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--probe_ids_file', type=str, required=True, help='Path to probe_ids.txt')
    parser.add_argument('--data_dir', type=str, default='/data/s185927/droid_sim/axis_aligned/x_axis',
                        help='Path to trajectory data')
    parser.add_argument('--actions_file', type=str, required=True, help='Path to save/load actions')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--nsamples', type=int, default=512, help='Number of action samples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--frame_idx', type=int, default=10, help='Frame index for start observation')

    args = parser.parse_args()

    set_seed(args.seed)

    # Load probe IDs
    probe_ids = []
    with open(args.probe_ids_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                probe_ids.append(int(line))
    print(f"Loaded {len(probe_ids)} probe IDs")

    # Generate or load actions
    actions = generate_or_load_actions(args.actions_file, args.nsamples, seed=args.seed)

    # Run extended probe
    aggregated, raw_results = run_probe_a_extended(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        probe_ids=probe_ids,
        data_dir=args.data_dir,
        actions=actions,
        device=args.device,
        frame_idx=args.frame_idx,
    )

    # Save results
    output = {
        'aggregated': aggregated,
        'per_probe': raw_results,
        'probe_ids': probe_ids,
        'checkpoint': args.checkpoint,
        'nsamples': args.nsamples,
        'seed': args.seed,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"\nAggregated metrics:")
    for k, v in aggregated.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
