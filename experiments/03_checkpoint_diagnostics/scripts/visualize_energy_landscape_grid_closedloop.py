#!/usr/bin/env python3
"""
Visualize Energy Landscape Grid - Closed-Loop Version

Creates a grid showing energy landscape heatmaps with actual MuJoCo execution:
- Rows: different training checkpoints (epochs)
- Columns: planning steps (with real observations after each action)

Unlike the predictor-based rollout version, this script:
1. Plans an action using CEM
2. Executes the action in MuJoCo simulation
3. Captures the real resulting frame
4. Encodes the real frame to get the true next latent state
5. Computes the energy landscape from that actual observation

This shows the "ground truth" of what happens during closed-loop execution.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

# Set EGL rendering before importing MuJoCo
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Add paths for imports
sys.path.insert(0, "/home/s185927/thesis/vjepa2")
sys.path.insert(0, "/home/s185927/thesis/robohive/robohive/robohive")
sys.path.insert(0, "/home/s185927/thesis")

from app.vjepa_droid.utils import init_video_model
from app.vjepa_droid.transforms import make_transforms
from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import generate_joint_space_min_jerk
from robohive.utils.xml_utils import reassign_parent
from plot_config import PLOT_PARAMS

# Constants from robo_samples.py
ARM_nJnt = 7
EE_SITE = 'end_effector'
ARM_JNT0 = np.array([
    -0.0321842, -0.394346, 0.00932319, -2.77917,
    -0.011826, 0.713889, 0.74663
])
FRANKA_MODEL = '/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_v0.xml'
ROBOTIQ_MODEL = '/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_robotiq_v0.xml'


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def robohive_to_droid_pos(pos):
    """Convert RoboHive position to DROID frame.

    Inverse of droid_to_robohive:
        DROID_x = RoboHive_y
        DROID_y = -RoboHive_x
    """
    return np.array([pos[1], -pos[0], pos[2]])


def droid_to_robohive_pos(pos):
    """Convert DROID position/action to RoboHive frame.

    Based on robo_samples.py swap_xy_negate_x transform:
        RoboHive_x = -DROID_y
        RoboHive_y = DROID_x
    """
    return np.array([-pos[1], pos[0], pos[2]])


def load_trajectory(ep_dir, frame_idx=10):
    """Load trajectory data from droid_sim episode directory.

    Returns:
        start_frame: RGB frame at frame_idx
        goal_frame: RGB frame at end of trajectory
        start_pos: EE position at frame_idx (DROID coordinates)
        goal_pos: EE position at end (DROID coordinates)
    """
    ep = Path(ep_dir)
    with h5py.File(ep / "trajectory.h5", 'r') as f:
        pos = f['observation/robot_state/cartesian_position'][:][:, :3]

    vids = glob.glob(str(ep / "recordings" / "MP4" / "*.mp4"))
    reader = imageio.get_reader(vids[0])

    n_video_frames = reader.count_frames()
    n_traj = len(pos)

    # Compute frame indices mapping
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


def load_model(ckpt_path, config, device='cuda:0'):
    """Load trained model from checkpoint."""
    d, m, meta = config.get('data', {}), config.get('model', {}), config.get('meta', {})
    crop_size = d.get('crop_size', 256)
    patch_size = d.get('patch_size', 16)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device)
        torch.cuda.set_device(device)

    encoder, predictor = init_video_model(
        uniform_power=m.get('uniform_power', True), device=device, patch_size=patch_size,
        max_num_frames=512, tubelet_size=d.get('tubelet_size', 2),
        model_name=m.get('model_name', 'vit_giant_xformers'), crop_size=crop_size,
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


def encode_frame(encoder, frame, transform, device, tpf):
    """Encode a single frame."""
    # Transform expects [T, H, W, C], output is [C, T, H, W]
    clip = transform(frame[np.newaxis, ...]).unsqueeze(0).to(device)  # [1, C, 1, H, W]
    B, C, T, H, W = clip.size()
    with torch.no_grad():
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [1, C, H, W]
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [1, C, 2, H, W]
        h = encoder(c)  # [1, tokens, D]
        h = F.layer_norm(h, (h.size(-1),))
    return h


def encode_frames_together(encoder, curr, goal, transform, device, tpf):
    """Encode current and goal frames together."""
    combined = np.stack([curr, goal], axis=0)  # [2, H, W, C]
    clips = transform(combined).unsqueeze(0).to(device)  # [1, C, 2, H, W]
    B, C, T, H, W = clips.size()
    with torch.no_grad():
        c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = encoder(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        h = F.layer_norm(h, (h.size(-1),))
    z_curr = h[:, :tpf].contiguous()
    z_goal = h[:, -tpf:].contiguous()
    return z_curr, z_goal


def compute_energy_grid(predictor, tpf, z, state, g, res, size, dev, bs=64):
    """Compute energy landscape grid."""
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
            zp = predictor(z.repeat(n, 1, 1), ba, state.repeat(n, 1, 1))[:, -tpf:]
            zp = F.layer_norm(zp, (zp.size(-1),))
            Es.append(torch.mean(torch.abs(zp - g[:, :tpf].repeat(n, 1, 1)), dim=[1, 2]).cpu())

    energies = torch.cat(Es).numpy()
    grid = energies.reshape(res, res)
    best_idx = np.argmin(energies)
    best_action = acts[best_idx]
    return grid, x, best_action


def cem_plan(predictor, tpf, z, state, g, dev,
             samples=512, cem_steps=5, topk=10, maxnorm=0.075):
    """Simple CEM planning to get best action."""
    mean = np.zeros(7)
    std = np.ones(7) * maxnorm / 2

    for _ in range(cem_steps):
        # Sample actions
        actions = np.random.randn(samples, 7) * std + mean
        actions[:, :3] = np.clip(actions[:, :3], -maxnorm, maxnorm)
        actions[:, 3:] = 0  # Zero out rotation and gripper

        # Evaluate
        at = torch.tensor(actions, device=dev, dtype=z.dtype).unsqueeze(1)
        with torch.no_grad():
            zp = predictor(z.repeat(samples, 1, 1), at, state.repeat(samples, 1, 1))[:, -tpf:]
            zp = F.layer_norm(zp, (zp.size(-1),))
            energies = torch.mean(torch.abs(zp - g[:, :tpf].repeat(samples, 1, 1)), dim=[1, 2]).cpu().numpy()

        # Select top-k
        top_idx = np.argsort(energies)[:topk]
        elite = actions[top_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-6

    return mean


def render_frame(sim, camera_name='left_cam', width=256, height=256, device_id=0):
    """Render a frame from the simulation."""
    frame = sim.renderer.render_offscreen(
        width=width, height=height, camera_id=camera_name, device_id=device_id
    )
    return frame


def execute_action(sim, action_droid, ee_sid, current_joint_pos, horizon=0.5):
    """Execute an action in the simulation (position delta in DROID frame).

    Uses IK + min-jerk trajectory like robo_samples.py.
    Returns new joint positions for chaining.
    """
    # Get current EE position
    current_ee = sim.data.site_xpos[ee_sid].copy()

    # Convert DROID action to RoboHive frame
    action_robohive = droid_to_robohive_pos(action_droid[:3])

    # Target position
    target_pos = current_ee + action_robohive

    # Use IK to find joint positions
    ik_result = qpos_from_site_pose(
        physics=sim,
        site_name=EE_SITE,
        target_pos=target_pos,
        target_quat=None,
        inplace=False,
        regularization_strength=1.0,
        max_steps=2000,
        tol=1e-4
    )

    # Generate smooth trajectory
    step_dt = sim.model.opt.timestep
    waypoints = generate_joint_space_min_jerk(
        start=current_joint_pos,
        goal=ik_result.qpos[:ARM_nJnt],
        time_to_go=horizon,
        dt=step_dt
    )

    # Execute trajectory
    for wp in waypoints:
        sim.data.ctrl[:ARM_nJnt] = wp['position']
        sim.advance(render=False)

    # Hold briefly
    hold_steps = max(5, int(0.1 / step_dt))
    for _ in range(hold_steps):
        sim.data.ctrl[:ARM_nJnt] = waypoints[-1]['position']
        sim.advance(render=False)

    return ik_result.qpos[:ARM_nJnt].copy()


def run_closed_loop_episode(encoder, predictor, tpf, transform, sim, ee_sid, target_sid,
                            target_pos_robohive, goal_frame, device, n_steps=5,
                            grid_res=40, grid_size=0.075, camera='left_cam', crop_size=256):
    """Run a closed-loop episode and collect energy landscapes."""
    grids = []
    goal_dirs = []
    goal_dists = []
    frames = []

    # Encode goal frame once
    z_goal = encode_frame(encoder, goal_frame, transform, device, tpf)
    goal_pos_droid = robohive_to_droid_pos(target_pos_robohive)

    # Track joint positions for smooth trajectory execution
    current_joint_pos = sim.data.qpos[:ARM_nJnt].copy()

    for step in range(n_steps):
        # Render current frame
        current_frame = render_frame(sim, camera, width=crop_size, height=crop_size)
        frames.append(current_frame)

        # Get current EE position
        current_ee = sim.data.site_xpos[ee_sid].copy()
        current_ee_droid = robohive_to_droid_pos(current_ee)

        # Compute goal direction and distance
        delta = goal_pos_droid - current_ee_droid
        gdir = delta[:2] / (np.linalg.norm(delta[:2]) + 1e-8)
        gdist = np.linalg.norm(delta)
        goal_dirs.append(gdir)
        goal_dists.append(gdist)

        # Encode current frame
        z_curr = encode_frame(encoder, current_frame, transform, device, tpf)
        state = torch.tensor([[list(current_ee_droid) + [0, 0, 0, 0]]],
                             device=device, dtype=z_curr.dtype)

        # Compute energy grid
        grid, x, _ = compute_energy_grid(predictor, tpf, z_curr, state, z_goal,
                                          grid_res, grid_size, device)
        grids.append(grid)

        # Plan action using CEM
        best_action = cem_plan(predictor, tpf, z_curr, state, z_goal, device)

        # Execute action in simulation
        current_joint_pos = execute_action(sim, best_action, ee_sid, current_joint_pos)

    return grids, x, goal_dirs, goal_dists, frames


def reset_sim(sim, ee_sid, target_pos_robohive):
    """Reset simulation to initial state and set target."""
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
    sim.data.ctrl[:ARM_nJnt] = ARM_JNT0.copy()
    sim.forward()

    # Set target position
    target_sid = sim.model.site_name2id("target")
    sim.model.site_pos[target_sid] = target_pos_robohive
    sim.forward()

    return sim.data.site_xpos[ee_sid].copy()


def plot_grid(all_grids, epochs, n_steps, x, out, all_goal_dirs=None, all_goal_dists=None):
    """Plot large grid: rows=epochs, columns=planning steps."""
    n_epochs = len(epochs)

    fig, axes = plt.subplots(n_epochs, n_steps, figsize=(3.2 * n_steps + 1, 2.8 * n_epochs))

    # Global colorscale
    all_vals = np.concatenate([g.flatten() for grids in all_grids for g in grids])
    vmin, vmax = np.percentile(all_vals, [2, 98])

    for row, (epoch, grids) in enumerate(zip(epochs, all_grids)):
        for col, grid in enumerate(grids):
            ax = axes[row, col] if n_epochs > 1 else axes[col]
            im = ax.imshow(grid, extent=[x[0], x[-1], x[0], x[-1]], origin='lower',
                           cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
            ax.contour(x, x, grid, levels=8, colors='white', alpha=0.25, linewidths=0.4)
            # Current position cross with black outline for visibility
            ax.plot(0, 0, '+', color='black', ms=14, mew=4, zorder=9)
            ax.plot(0, 0, '+', color='white', ms=14, mew=2, zorder=10)

            # Goal direction arrow
            if all_goal_dirs is not None:
                gdir = all_goal_dirs[row][col]
                ax.arrow(0, 0, gdir[0] * 0.04, gdir[1] * 0.04,
                         head_width=0.008, head_length=0.005, fc='red', ec='red', lw=1.5)

            # Axis labels
            if row == n_epochs - 1:
                ax.set_xlabel(r'$\Delta x$ (m)', fontsize=13)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(f'Epoch {epoch}\n' + r'$\Delta y$ (m)', fontsize=13)
            else:
                ax.set_yticklabels([])

            if row == 0:
                ax.set_title(f'Planning Step {col + 1}', fontsize=14, fontweight='bold')

            ax.set_xticks([-0.05, 0, 0.05])
            ax.set_yticks([-0.05, 0, 0.05])
            ax.tick_params(labelsize=11)

            # Annotations
            ax.text(0.03, 0.97, f'$\\sigma(\\mathcal{{E}})$={np.std(grid):.4f}',
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.75, pad=0.15))

            if all_goal_dists is not None:
                dist = all_goal_dists[row][col]
                ax.text(0.97, 0.97, f'd={dist:.3f}m',
                        transform=ax.transAxes, fontsize=10, va='top', ha='right',
                        bbox=dict(boxstyle='round', fc='white', alpha=0.75, pad=0.15))

    # Colorbar
    fig.subplots_adjust(right=0.9, hspace=0.12, wspace=0.08)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'Energy $\mathcal{E}$ (latent distance)', fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    # Legend - positioned in bottom right, below the colorbar
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as path_effects
    # Create cross marker with stroke effect to match subplot style
    cross_marker = Line2D([0], [0], marker='+', color='white',
                          markersize=14, markeredgewidth=2, linestyle='None',
                          label='Current position',
                          path_effects=[path_effects.Stroke(linewidth=4, foreground='black'),
                                        path_effects.Normal()])
    legend_elements = [
        cross_marker,
        Line2D([0], [0], marker=r'$\rightarrow$', color='red',
               markersize=15, linestyle='None', label='Goal direction'),
    ]
    fig.legend(handles=legend_elements, loc='lower right',
               fontsize=11, bbox_to_anchor=(1, 0.075), frameon=True,
               facecolor='white', edgecolor='gray', framealpha=0.9)

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
    all_goal_dists = [list(dists) for dists in data['all_goal_dists']]
    n_steps = int(data['n_steps'])
    print(f"Loaded cache from: {cache_path}")
    return all_grids, epochs, x, all_goal_dirs, all_goal_dists, n_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', help='Paths to checkpoints')
    p.add_argument('--epochs', type=int, nargs='+', help='Epoch numbers')
    p.add_argument('--config', help='Path to config YAML')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--grid_resolution', type=int, default=40)
    p.add_argument('--grid_size', type=float, default=0.075)
    p.add_argument('--n_steps', type=int, default=5)
    p.add_argument('--camera', default='left_cam')
    # Trajectory options (use test trajectories from droid_sim)
    p.add_argument('--data_dir', type=str, default='/data/s185927/droid_sim/axis_aligned/x_axis',
                   help='Path to droid_sim trajectory directory')
    p.add_argument('--probe_id', type=int, default=2,
                   help='Trajectory ID to use (default: 2, first test trajectory)')
    p.add_argument('--frame_idx', type=int, default=10,
                   help='Starting frame index in trajectory')
    # Legacy option (ignored when using trajectory)
    p.add_argument('--target_offset', type=float, default=0.2,
                   help='[DEPRECATED] Target offset - now using trajectory goal instead')
    # Gripper option
    p.add_argument('--gripper', choices=['franka', 'robotiq'], default='robotiq',
                   help='End-effector gripper type (default: robotiq)')
    # Caching options
    p.add_argument('--save_cache', type=str, help='Save computed grids to this file')
    p.add_argument('--load_cache', type=str, help='Load grids from cache')
    p.add_argument('--plot_only', action='store_true', help='Only plot from cache')
    a = p.parse_args()

    set_seed(42)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # If loading from cache, skip computation
    if a.load_cache or a.plot_only:
        cache_path = a.load_cache or (out / 'energy_grid_closedloop_cache.npz')
        all_grids, epochs, x, all_goal_dirs, all_goal_dists, n_steps = load_cache(cache_path)
    else:
        assert a.checkpoints and a.epochs and a.config, \
            "Must provide --checkpoints, --epochs, --config (or use --load_cache)"
        assert len(a.checkpoints) == len(a.epochs)

        with open(a.config) as f:
            cfg = yaml.safe_load(f)
        crop_size = cfg['data']['crop_size']
        tpf = (crop_size // cfg['data']['patch_size']) ** 2

        # Initialize MuJoCo simulation
        print("Loading MuJoCo simulation...")
        sim_path = ROBOTIQ_MODEL if a.gripper == 'robotiq' else FRANKA_MODEL
        sim = SimScene.get_sim(model_handle=sim_path)

        # For RobotiQ model, reparent ee_mount to panda0_link7
        if a.gripper == 'robotiq':
            raw_xml = sim.model.get_xml()
            processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="ee_mount")
            processed_path = os.path.join(os.path.dirname(os.path.abspath(sim_path)), '_robotiq_processed.xml')
            with open(processed_path, 'w') as f:
                f.write(processed_xml)
            sim = SimScene.get_sim(model_handle=processed_path)
            print(f"Using RobotiQ gripper model: {sim_path}")

        ee_sid = sim.model.site_name2id(EE_SITE)
        target_sid = sim.model.site_name2id("target")

        # Load goal offset from test trajectory
        traj_dir = os.path.join(a.data_dir, f"episode_{a.probe_id:04d}")
        print(f"Loading trajectory from: {traj_dir}")
        with h5py.File(os.path.join(traj_dir, "trajectory.h5"), 'r') as f:
            pos = f['observation/robot_state/cartesian_position'][:][:, :3]
        start_pos_droid = pos[a.frame_idx]
        goal_pos_droid = pos[-1]
        goal_offset_droid = goal_pos_droid - start_pos_droid
        print(f"Trajectory goal offset (DROID): {goal_offset_droid}")

        # Set simulation to default starting position
        sim.data.qpos[:ARM_nJnt] = ARM_JNT0
        sim.forward()
        ee_start = sim.data.site_xpos[ee_sid].copy()

        # Apply trajectory's goal offset (converted to RoboHive frame)
        goal_offset_robohive = droid_to_robohive_pos(goal_offset_droid)
        target_pos_robohive = ee_start + goal_offset_robohive
        print(f"Goal offset (RoboHive): {goal_offset_robohive}")
        print(f"Target position (RoboHive): {target_pos_robohive}")

        # Set target marker in simulation
        sim.model.site_pos[target_sid] = target_pos_robohive

        # Render goal frame by moving robot to target via IK
        ik_result = qpos_from_site_pose(
            physics=sim,
            site_name=EE_SITE,
            target_pos=target_pos_robohive,
            target_quat=None,
            inplace=False,
            regularization_strength=1.0,
            max_steps=2000,
            tol=1e-4
        )
        sim.data.qpos[:ARM_nJnt] = ik_result.qpos[:ARM_nJnt]
        sim.forward()
        goal_frame = render_frame(sim, a.camera, width=crop_size, height=crop_size)
        print(f"Rendered goal frame at EE position: {sim.data.site_xpos[ee_sid]}")

        # Save goal frame
        plt.imsave(out / 'goal_frame.png', goal_frame)
        print(f"Saved goal frame to {out / 'goal_frame.png'}")

        all_grids = []
        all_goal_dirs = []
        all_goal_dists = []
        x = None
        epochs = a.epochs
        n_steps = a.n_steps

        for ckpt, ep in zip(a.checkpoints, epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {ep}: {ckpt}")
            print(f"{'='*60}")

            # Load model
            enc, pred, dev = load_model(ckpt, cfg, a.device)
            tr = create_transform(crop_size)

            # Reset simulation
            reset_sim(sim, ee_sid, target_pos_robohive)

            # Run closed-loop episode
            grids, x, goal_dirs, goal_dists, frames = run_closed_loop_episode(
                enc, pred, tpf, tr, sim, ee_sid, target_sid,
                target_pos_robohive, goal_frame, dev,
                n_steps=n_steps, grid_res=a.grid_resolution, grid_size=a.grid_size,
                camera=a.camera, crop_size=crop_size
            )

            all_grids.append(grids)
            all_goal_dirs.append(goal_dirs)
            all_goal_dists.append(goal_dists)

            # Save frames
            frames_dir = out / 'frames' / f'epoch_{ep}'
            frames_dir.mkdir(parents=True, exist_ok=True)
            for step, frame in enumerate(frames):
                frame_path = frames_dir / f'step_{step:02d}.png'
                plt.imsave(frame_path, frame)
            print(f"  Saved {len(frames)} frames to {frames_dir}")

            # Print stats
            for step, (grid, dist) in enumerate(zip(grids, goal_dists)):
                print(f"  Step {step + 1}: std={np.std(grid):.4f}, dist={dist:.3f}m")

            del enc, pred
            torch.cuda.empty_cache()

        # Save cache
        cache_path = a.save_cache or (out / 'energy_grid_closedloop_cache.npz')
        save_cache(cache_path, all_grids, epochs, x, all_goal_dirs, all_goal_dists, n_steps)

    # Plot
    plot_grid(all_grids, epochs, n_steps, x,
              out / 'energy_landscape_grid_closedloop.png',
              all_goal_dirs, all_goal_dists)
    print(f"\nDone! Output in {out}")


if __name__ == '__main__':
    main()
