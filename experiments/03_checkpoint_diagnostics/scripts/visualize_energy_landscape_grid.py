#!/usr/bin/env python3
"""
Visualize Energy Landscape Grid

Creates a large grid showing energy landscape heatmaps:
- Rows: different training checkpoints (epochs)
- Columns: planning steps (using predictor rollout)

This shows how the landscape flattens both over training and across planning steps.

The planning steps use predictor-based rollout (imagined planning), which shows
the energy landscape as the planner perceives it - the relevant quantity for
understanding why CEM optimization fails as the landscape flattens.
"""
import argparse
import copy
import glob
import os
import random
import sys
from pathlib import Path

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.colors import Normalize

sys.path.insert(0, "/home/s185927/thesis/vjepa2")
sys.path.insert(0, "/home/s185927/thesis")
from app.vjepa_droid.utils import init_video_model
from app.vjepa_droid.transforms import make_transforms
from plot_config import PLOT_PARAMS, configure_axis


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(ckpt_path, config, device='cuda:0'):
    d, m, meta = config.get('data', {}), config.get('model', {}), config.get('meta', {})
    crop, patch = d.get('crop_size', 256), d.get('patch_size', 16)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device)
        torch.cuda.set_device(device)

    encoder, predictor = init_video_model(
        uniform_power=m.get('uniform_power', True), device=device, patch_size=patch,
        max_num_frames=512, tubelet_size=d.get('tubelet_size', 2),
        model_name=m.get('model_name', 'vit_giant_xformers'), crop_size=crop,
        pred_depth=m.get('pred_depth', 24), pred_num_heads=m.get('pred_num_heads', 16),
        pred_embed_dim=m.get('pred_embed_dim', 1024), action_embed_dim=7,
        pred_is_frame_causal=m.get('pred_is_frame_causal', True),
        use_extrinsics=m.get('use_extrinsics', False), use_sdpa=meta.get('use_sdpa', True),
        use_silu=m.get('use_silu', False), use_pred_silu=m.get('use_pred_silu', False),
        wide_silu=m.get('wide_silu', True), use_rope=m.get('use_rope', True),
        use_activation_checkpointing=False)

    ckpt = torch.load(ckpt_path, map_location=device)
    strip = lambda sd: {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}

    encoder.load_state_dict(strip(ckpt['encoder']), strict=False)
    if 'predictor' in ckpt:
        predictor.load_state_dict(strip(ckpt['predictor']), strict=False)
    encoder.eval()
    predictor.eval()
    return encoder, predictor, device


def create_transform(crop_size=256):
    return make_transforms(
        random_horizontal_flip=False, random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.), reprob=0., auto_augment=False,
        motion_shift=False, crop_size=crop_size)


def encode_frames(encoder, curr, goal, transform, device, tpf):
    clips = transform(np.stack([curr, goal])).unsqueeze(0).to(device)
    B, C, T, H, W = clips.size()
    with torch.no_grad():
        c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = encoder(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        h = F.layer_norm(h, (h.size(-1),))
    return h[:, :tpf].contiguous(), h[:, -tpf:].contiguous()


def load_traj(ep_dir, frame_idx=10):
    """Load only the specific frames we need."""
    ep = Path(ep_dir)
    with h5py.File(ep / "trajectory.h5", 'r') as f:
        pos = f['observation/robot_state/cartesian_position'][:][:, :3]
    vids = glob.glob(str(ep / "recordings" / "MP4" / "*.mp4"))
    reader = imageio.get_reader(vids[0])
    n_video_frames = reader.count_frames()
    n_traj = len(pos)

    if n_video_frames > n_traj:
        indices = np.linspace(0, n_video_frames - 1, n_traj, dtype=int)
    else:
        indices = np.arange(n_video_frames)
        pos = pos[np.linspace(0, n_traj - 1, n_video_frames, dtype=int)]

    start_vid_idx = indices[frame_idx]
    goal_vid_idx = indices[-1]
    start_frame = reader.get_data(start_vid_idx)
    goal_frame = reader.get_data(goal_vid_idx)
    reader.close()
    return start_frame, goal_frame, pos[frame_idx], pos[-1]


def compute_grid(pred, tpf, z, state, g, res, size, dev, bs=64):
    """Compute energy grid and return best action."""
    x = np.linspace(-size, size, res)
    xx, yy = np.meshgrid(x, x)
    acts = np.array([[xx[i, j], yy[i, j], 0, 0, 0, 0, 0]
                     for i in range(res) for j in range(res)])
    at = torch.tensor(acts, device=dev, dtype=z.dtype).unsqueeze(1)
    Es = []
    with torch.no_grad():
        for i in range(0, len(acts), bs):
            ba = at[i:i + bs]
            n = ba.shape[0]
            zp = pred(z.repeat(n, 1, 1), ba, state.repeat(n, 1, 1))[:, -tpf:]
            zp = F.layer_norm(zp, (zp.size(-1),))
            Es.append(torch.mean(torch.abs(zp - g[:, :tpf].repeat(n, 1, 1)), dim=[1, 2]).cpu())

    energies = torch.cat(Es).numpy()
    grid = energies.reshape(res, res)

    # Find best action (lowest energy)
    best_idx = np.argmin(energies)
    best_action = acts[best_idx]

    return grid, x, best_action


def predict_next_state(pred, tpf, z, state, action, dev):
    """Use predictor to get next state representation."""
    action_t = torch.tensor(action, device=dev, dtype=z.dtype).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        z_next = pred(z, action_t, state)[:, -tpf:]
        z_next = F.layer_norm(z_next, (z_next.size(-1),))
    return z_next


def compute_grids_for_planning_steps(pred, tpf, z_init, state_init, g, res, size, dev, goal_pos, n_steps=5):
    """Compute energy grids for each planning step.

    Also tracks goal direction and distance at each step (from current position to goal).
    """
    grids = []
    goal_dirs = []
    goal_dists = []
    x = None
    z = z_init
    state = state_init

    for step in range(n_steps):
        # Compute goal direction and distance from current position
        current_pos = state[0, 0, :3].cpu().numpy()
        delta = goal_pos - current_pos
        gdir = delta[:2] / (np.linalg.norm(delta[:2]) + 1e-8)
        gdist = np.linalg.norm(delta)  # 3D Euclidean distance
        goal_dirs.append(gdir)
        goal_dists.append(gdist)

        grid, x, best_action = compute_grid(pred, tpf, z, state, g, res, size, dev)
        grids.append(grid)

        # Roll forward using predictor
        z = predict_next_state(pred, tpf, z, state, best_action, dev)

        # Update state (simulate position change)
        new_pos = current_pos + best_action[:3]
        state = torch.tensor([[list(new_pos) + [0, 0, 0, 0]]], device=dev, dtype=z.dtype)

    return grids, x, goal_dirs, goal_dists


def plot_grid(all_grids, epochs, n_steps, x, out, all_goal_dirs=None, all_goal_dists=None):
    """Plot large grid: rows=epochs, columns=planning steps.

    Args:
        all_goal_dirs: List of lists, [n_epochs][n_steps] goal directions per panel
        all_goal_dists: List of lists, [n_epochs][n_steps] distances to goal per panel
    """
    n_epochs = len(epochs)

    fig, axes = plt.subplots(n_epochs, n_steps, figsize=(3.2 * n_steps + 1, 2.8 * n_epochs))

    # Global colorscale across all panels
    all_vals = np.concatenate([g.flatten() for grids in all_grids for g in grids])
    vmin, vmax = np.percentile(all_vals, [2, 98])

    for row, (epoch, grids) in enumerate(zip(epochs, all_grids)):
        for col, grid in enumerate(grids):
            ax = axes[row, col] if n_epochs > 1 else axes[col]
            im = ax.imshow(grid, extent=[x[0], x[-1], x[0], x[-1]], origin='lower',
                           cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
            ax.contour(x, x, grid, levels=8, colors='white', alpha=0.25, linewidths=0.4)
            ax.plot(0, 0, 'w+', ms=8, mew=1.5)

            # Goal direction arrow (per-step direction)
            if all_goal_dirs is not None:
                gdir = all_goal_dirs[row][col]
                ax.arrow(0, 0, gdir[0] * 0.04, gdir[1] * 0.04,
                         head_width=0.008, head_length=0.005, fc='red', ec='red', lw=1.5)

            # Axis labels only on edges
            if row == n_epochs - 1:
                ax.set_xlabel(r'$\Delta x$ (m)', fontsize=13)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(f'Epoch {epoch}\n' + r'$\Delta y$ (m)', fontsize=13)
            else:
                ax.set_yticklabels([])

            # Column titles on top row
            if row == 0:
                ax.set_title(f'Planning Step {col + 1}', fontsize=14, fontweight='bold')

            ax.set_xticks([-0.05, 0, 0.05])
            ax.set_yticks([-0.05, 0, 0.05])
            ax.tick_params(labelsize=11)

            # Sharpness annotation
            ax.text(0.03, 0.97, f'$\\sigma(\\mathcal{{E}})$={np.std(grid):.4f}',
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.75, pad=0.15))

            # Distance to goal annotation
            if all_goal_dists is not None:
                dist = all_goal_dists[row][col]
                ax.text(0.97, 0.97, f'd={dist:.3f}m',
                        transform=ax.transAxes, fontsize=10, va='top', ha='right',
                        bbox=dict(boxstyle='round', fc='white', alpha=0.75, pad=0.15))

    # Colorbar on the right
    fig.subplots_adjust(right=0.9, hspace=0.12, wspace=0.08)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'Energy $\mathcal{E}$ (latent distance)', fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def save_cache(cache_path, all_grids, epochs, x, all_goal_dirs, all_goal_dists, n_steps):
    """Save computed grids to cache file."""
    np.savez(
        cache_path,
        all_grids=np.array(all_grids),
        epochs=np.array(epochs),
        x=x,
        all_goal_dirs=np.array(all_goal_dirs),
        all_goal_dists=np.array(all_goal_dists),
        n_steps=n_steps
    )
    print(f"Saved cache to: {cache_path}")


def load_cache(cache_path):
    """Load grids from cache file."""
    data = np.load(cache_path)
    all_grids = [list(grids) for grids in data['all_grids']]
    epochs = list(data['epochs'])
    x = data['x']
    all_goal_dirs = [list(gds) for gds in data['all_goal_dirs']]
    # Handle backwards compatibility for caches without distances
    if 'all_goal_dists' in data:
        all_goal_dists = [list(dists) for dists in data['all_goal_dists']]
    else:
        all_goal_dists = None
    n_steps = int(data['n_steps'])
    print(f"Loaded cache from: {cache_path}")
    return all_grids, epochs, x, all_goal_dirs, all_goal_dists, n_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', help='Paths to checkpoints')
    p.add_argument('--epochs', type=int, nargs='+', help='Epoch numbers')
    p.add_argument('--config', help='Path to config YAML')
    p.add_argument('--data_dir', default='/data/s185927/droid_sim/axis_aligned/x_axis')
    p.add_argument('--probe_id', type=int, default=0)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--grid_resolution', type=int, default=40)
    p.add_argument('--grid_size', type=float, default=0.075)
    p.add_argument('--frame_idx', type=int, default=10)
    p.add_argument('--n_steps', type=int, default=5)
    # Caching options
    p.add_argument('--save_cache', type=str, help='Save computed grids to this file')
    p.add_argument('--load_cache', type=str, help='Load grids from cache (skip computation)')
    p.add_argument('--plot_only', action='store_true', help='Only plot from cache, no computation')
    a = p.parse_args()

    set_seed(42)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # If loading from cache, skip computation
    if a.load_cache or a.plot_only:
        cache_path = a.load_cache or (out / 'energy_grid_cache.npz')
        all_grids, epochs, x, all_goal_dirs, all_goal_dists, n_steps = load_cache(cache_path)
    else:
        # Need checkpoints for computation
        assert a.checkpoints and a.epochs and a.config, \
            "Must provide --checkpoints, --epochs, --config (or use --load_cache)"
        assert len(a.checkpoints) == len(a.epochs)

        with open(a.config) as f:
            cfg = yaml.safe_load(f)
        tpf = (cfg['data']['crop_size'] // cfg['data']['patch_size']) ** 2

        # Load trajectory
        sf, gf, sp, gp = load_traj(os.path.join(a.data_dir, f"episode_{a.probe_id:04d}"), a.frame_idx)

        all_grids = []
        all_goal_dirs = []
        all_goal_dists = []
        x = None
        epochs = a.epochs
        n_steps = a.n_steps

        for ckpt, ep in zip(a.checkpoints, epochs):
            print(f"\nEpoch {ep}: {ckpt}")
            enc, pred, dev = load_model(ckpt, cfg, a.device)
            tr = create_transform(cfg['data']['crop_size'])

            # Encode initial frames
            z_init, g = encode_frames(enc, sf, gf, tr, dev, tpf)
            state_init = torch.tensor([[list(sp) + [0, 0, 0, 0]]], device=dev, dtype=z_init.dtype)

            # Compute grids for all planning steps (with per-step goal directions and distances)
            grids, x, goal_dirs, goal_dists = compute_grids_for_planning_steps(
                pred, tpf, z_init, state_init, g,
                a.grid_resolution, a.grid_size, dev, gp, n_steps
            )
            all_grids.append(grids)
            all_goal_dirs.append(goal_dirs)
            all_goal_dists.append(goal_dists)

            # Print sharpness and distance for each step
            for step, (grid, dist) in enumerate(zip(grids, goal_dists)):
                print(f"  Step {step + 1}: std={np.std(grid):.4f}, range={np.ptp(grid):.4f}, dist={dist:.3f}m")

            del enc, pred
            torch.cuda.empty_cache()

        # Save cache
        cache_path = a.save_cache or (out / 'energy_grid_cache.npz')
        save_cache(cache_path, all_grids, epochs, x, all_goal_dirs, all_goal_dists, n_steps)

    # Plot the full grid
    plot_grid(all_grids, epochs, n_steps, x, out / 'energy_landscape_grid.png', all_goal_dirs, all_goal_dists)
    print(f"\nDone! Output in {out}")


if __name__ == '__main__':
    main()
