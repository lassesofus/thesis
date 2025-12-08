#!/usr/bin/env python3
"""
Evaluate V-JEPA planning performance on test samples.

Compares multiple trained models against the Meta baseline on reaching tasks.
For each test sample, performs:
1. Move to target position to capture goal RGB
2. Return to start position
3. Run CEM planning for N steps
4. Measure final distance to target

EXAMPLE USAGE:
    # Evaluate all models in directory
    python eval_vjepa_planning.py \
        --metadata /data/s185927/droid_sim/y_axis/trajectory_metadata.json \
        --model_dir /data/s185927/vjepa2/weights/droid \
        --planning_steps 5 \
        --out_dir /data/s185927/vjepa_eval_results

    # Test on limited samples
    python eval_vjepa_planning.py \
        --metadata /data/s185927/droid_sim/y_axis/trajectory_metadata.json \
        --model_dir /data/s185927/vjepa2/weights/droid \
        --planning_steps 5 \
        --out_dir /data/s185927/vjepa_eval_results \
        --max_samples 10
"""

import warnings

# Silence pydantic warnings
try:
    from pydantic import PydanticUserWarning
    warnings.filterwarnings("ignore", category=PydanticUserWarning)
except Exception:
    warnings.filterwarnings("ignore", message=r".*The 'repr' attribute.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*The 'frozen' attribute.*", category=UserWarning)

warnings.filterwarnings("ignore", message=r".*Importing from timm.models.layers.*", category=FutureWarning)

import os
import sys
import json
import time
from pathlib import Path

import click
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import time

import click
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

# Add V-JEPA to path
_vjepa_root = "/home/s185927/thesis/vjepa2"
if os.path.isdir(_vjepa_root) and _vjepa_root not in sys.path:
    sys.path.insert(0, _vjepa_root)

from notebooks.utils.world_model_wrapper import WorldModel
from app.vjepa_droid.transforms import make_transforms
from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import generate_joint_space_min_jerk


# Configuration
ARM_nJnt = 7
EE_SITE = "end_effector"
ARM_JNT0 = np.array([
    -0.0321842,  -0.394346,   0.00932319,
    -2.77917,    -0.011826,   0.713889,    1.53183
])


class ModelConfig:
    """Configuration for a V-JEPA model to evaluate."""
    def __init__(self, name, checkpoint_path=None, use_hub=False):
        self.name = name
        self.checkpoint_path = checkpoint_path
        self.use_hub = use_hub


def load_model(config, device):
    """
    Load V-JEPA model from checkpoint or PyTorch Hub.

    Args:
        config: ModelConfig instance
        device: torch device

    Returns:
        encoder, predictor, transform, world_model, tokens_per_frame
    """
    print(f"\nLoading model: {config.name}")

    if config.use_hub:
        print("  Loading from PyTorch Hub (Meta baseline)...")
        encoder, predictor = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_ac_vit_giant"
        )
    else:
        print(f"  Loading from checkpoint: {config.checkpoint_path}")
        # Load checkpoint
        checkpoint = torch.load(config.checkpoint_path, map_location=device)

        # First load the architecture from hub
        encoder, predictor = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_ac_vit_giant"
        )

        # Then load the trained weights
        if 'encoder' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder'])
        elif 'target_encoder' in checkpoint:
            encoder.load_state_dict(checkpoint['target_encoder'])
        else:
            raise ValueError(f"Cannot find encoder weights in checkpoint")

        if 'predictor' in checkpoint:
            predictor.load_state_dict(checkpoint['predictor'])
        else:
            raise ValueError(f"Cannot find predictor weights in checkpoint")

    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    # Create transform
    crop_size = 256
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    # Create WorldModel wrapper
    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args={
            "rollout": 1,
            "samples": 800,
            "topk": 10,
            "cem_steps": 10,
            "momentum_mean": 0.15,
            "momentum_mean_gripper": 0.15,
            "momentum_std": 0.75,
            "momentum_std_gripper": 0.15,
            "maxnorm": 0.075,
            "verbose": False
        },
        normalize_reps=True,
        device=str(device)
    )

    print(f"  Model loaded successfully")
    return encoder, predictor, transform, world_model, tokens_per_frame


def compute_new_pose(current_pos, delta_pos, delta_rpy):
    """Compute new end-effector pose from deltas."""
    new_pos = current_pos + delta_pos
    new_rpy = delta_rpy
    return new_pos, new_rpy


def transform_action(action, transform_type='none'):
    """Transform actions from camera frame to robot frame."""
    transformed = action.copy()

    if transform_type == 'swap_xy':
        transformed[0], transformed[1] = action[1], action[0]
        transformed[3], transformed[4] = action[4], action[3]
    elif transform_type == 'negate_x':
        transformed[0] = -action[0]
    elif transform_type == 'negate_y':
        transformed[1] = -action[1]
    elif transform_type == 'swap_xy_negate_x':
        transformed[0], transformed[1] = -action[1], action[0]
        transformed[3], transformed[4] = action[4], action[3]
    elif transform_type == 'swap_xy_negate_y':
        transformed[0], transformed[1] = action[1], -action[0]
        transformed[3], transformed[4] = action[4], action[3]

    return transformed


def execute_waypoints(waypoints, sim, horizon, step_dt):
    """Execute joint-space waypoints."""
    t = 0.0
    idx = 0
    num_waypoints = len(waypoints)

    while t <= horizon and idx < num_waypoints:
        sim.data.ctrl[:ARM_nJnt] = waypoints[idx]['position']
        sim.advance(render=False)
        t += step_dt
        idx += 1

    # Hold final position
    if waypoints:
        final_ctrl = waypoints[-1]['position'].copy()
        hold_steps = max(5, int(0.2 / sim.model.opt.timestep))
        for _ in range(hold_steps):
            sim.data.ctrl[:ARM_nJnt] = final_ctrl
            sim.advance(render=False)

    return final_ctrl if waypoints else sim.data.qpos[:ARM_nJnt].copy()


@click.command(help=__doc__)
@click.option(
    '--test_csv',
    type=str,
    required=True,
    help='CSV file containing test trajectory paths'
)
@click.option(
    '--metadata',
    type=str,
    required=True,
    help='JSON file containing trajectory metadata'
)
@click.option(
    '--sim_path',
    type=str,
    default='/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_v0.xml',
    help='Path to MuJoCo XML model'
)
@click.option(
    '--out_dir',
    type=str,
    required=True,
    help='Output directory for evaluation results'
)
@click.option(
    '--planning_steps',
    type=int,
    default=10,
    help='Number of CEM planning steps'
)
@click.option(
    '--action_transform',
    type=str,
    default='swap_xy_negate_x',
    help='Action transformation: none, swap_xy, negate_x, negate_y, swap_xy_negate_x, swap_xy_negate_y'
)
@click.option(
    '--success_threshold',
    type=float,
    default=0.05,
    help='Distance threshold (m) to consider target reached'
)
@click.option(
    '--horizon',
    type=float,
    default=3.0,
    help='Time per planning step (seconds)'
)
@click.option(
    '--width',
    type=int,
    default=640,
    help='Render width'
)
@click.option(
    '--height',
    type=int,
    default=480,
    help='Render height'
)
@click.option(
    '--device_id',
    type=int,
    default=0,
    help='Rendering device ID'
)
@click.option(
    '--camera_name',
    type=str,
    default='left_cam',
    help='MuJoCo camera name'
)
def main(
    test_csv,
    metadata,
    sim_path,
    out_dir,
    planning_steps,
    action_transform,
    success_threshold,
    horizon,
    width,
    height,
    device_id,
    camera_name
):
    """Evaluate V-JEPA planning on test trajectories."""

    print("=" * 80)
    print("V-JEPA Planning Evaluation")
    print("=" * 80)
    print(f"Test CSV: {test_csv}")
    print(f"Metadata: {metadata}")
    print(f"Output directory: {out_dir}")
    print(f"Planning steps: {planning_steps}")
    print(f"Action transform: {action_transform}")
    print(f"Success threshold: {success_threshold}m")
    print("=" * 80)

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load test trajectories
    print(f"\nLoading test trajectories from {test_csv}...")
    with open(test_csv, 'r') as f:
        test_paths = [line.strip() for line in f if line.strip()]

    print(f"Found {len(test_paths)} test trajectories")

    # Load metadata
    print(f"Loading metadata from {metadata}...")
    with open(metadata, 'r') as f:
        all_metadata = json.load(f)

    # Filter to test set only
    test_metadata = {m['trajectory_path']: m for m in all_metadata if m['split'] == 'test'}

    # Load simulation
    print(f"Loading simulation: {sim_path}")
    sim = SimScene.get_sim(model_handle=sim_path)
    ee_sid = sim.model.site_name2id(EE_SITE)
    step_dt = sim.model.opt.timestep

    # Starting position
    ARM_JNT0 = np.array([
        -0.0321842, -0.394346, 0.00932319, -2.77917,
        -0.011826, 0.713889, 1.53183
    ])

    # Reset and get start position
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
    sim.data.ctrl[:ARM_nJnt] = ARM_JNT0.copy()
    sim.forward()
    ee_start = sim.data.site_xpos[ee_sid].copy()

    # Load V-JEPA models
    print(f"\nLoading V-JEPA models...")
    encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    crop_size = 256
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args={
            "rollout": 1,
            "samples": 800,
            "topk": 10,
            "cem_steps": 10,
            "momentum_mean": 0.15,
            "momentum_mean_gripper": 0.15,
            "momentum_std": 0.75,
            "momentum_std_gripper": 0.15,
            "maxnorm": 0.075,
            "verbose": False
        },
        normalize_reps=True,
        device=str(device)
    )

    print("V-JEPA models loaded successfully")

    # Evaluation results
    eval_results = []

    # Evaluate each test trajectory
    print(f"\n{'=' * 80}")
    print(f"Evaluating {len(test_paths)} test trajectories")
    print("=" * 80)

    for test_idx, traj_path in enumerate(test_paths):
        print(f"\n--- Test {test_idx + 1}/{len(test_paths)} ---")
        print(f"Trajectory: {traj_path}")

        # Get metadata for this trajectory
        traj_meta = test_metadata.get(traj_path)
        if traj_meta is None:
            print(f"WARNING: No metadata found for {traj_path}, skipping")
            continue

        target_pos = np.array(traj_meta['target_position'])
        target_distance = traj_meta['target_distance']
        traj_direction = traj_meta['trajectory_direction']

        print(f"Target position: {target_pos}")
        print(f"Target distance: {target_distance:.4f}m along {traj_direction}")

        # Reset to start
        sim.data.qpos[:ARM_nJnt] = ARM_JNT0
        sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
        sim.data.ctrl[:ARM_nJnt] = ARM_JNT0.copy()
        sim.forward()

        # Load goal image from trajectory
        goal_img_path = Path(traj_path) / "recordings" / "MP4"
        # Get the last frame as goal (we'll use the saved images or re-render)
        # For simplicity, render the target position

        # First, move to target using IK to get goal image
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

        sim.data.qpos[:ARM_nJnt] = ik_result.qpos[:ARM_nJnt]
        sim.data.ctrl[:ARM_nJnt] = ik_result.qpos[:ARM_nJnt]
        sim.forward()

        goal_rgb = sim.renderer.render_offscreen(
            width=width, height=height, camera_id=camera_name, device_id=device_id
        )

        # Reset to start for planning
        sim.data.qpos[:ARM_nJnt] = ARM_JNT0
        sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
        sim.data.ctrl[:ARM_nJnt] = ARM_JNT0.copy()
        sim.forward()

        # Track planning metrics
        distances = []
        repr_distances = []
        actions_taken = []

        current_joint_pos = ARM_JNT0.copy()

        # Planning loop
        with torch.no_grad():
            for step_idx in range(planning_steps):
                sim.forward()
                current_ee_pos = sim.data.site_xpos[ee_sid].copy()
                distance = np.linalg.norm(current_ee_pos - target_pos)
                distances.append(distance)

                print(f"  Step {step_idx + 1}/{planning_steps}: Distance = {distance:.4f}m")

                # Capture current observation
                current_rgb = sim.renderer.render_offscreen(
                    width=width, height=height, camera_id=camera_name, device_id=device_id
                )

                # Stack and transform
                combined_rgb = np.stack([current_rgb, goal_rgb], axis=0)
                clips = transform(combined_rgb).unsqueeze(0).to(device)

                # Get representations
                B, C, T, H, W = clips.size()
                c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
                h = encoder(c)
                h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
                h = F.layer_norm(h, (h.size(-1),))

                z_n = h[:, :tokens_per_frame].contiguous().clone()
                z_goal = h[:, -tokens_per_frame:].contiguous().clone()

                repr_l1_distance = torch.mean(torch.abs(z_n - z_goal)).item()
                repr_distances.append(repr_l1_distance)

                # Build state
                pos = current_ee_pos
                rpy = np.zeros(3)
                gripper = 0.0
                current_state = np.concatenate([pos, rpy, [gripper]])
                states = torch.tensor(current_state, device=device).unsqueeze(0).unsqueeze(0)
                s_n = states[:, :1].to(dtype=z_n.dtype)

                # Plan action
                start_time = time.time()
                actions = world_model.infer_next_action(z_n, s_n, z_goal).cpu().numpy()
                planning_time = time.time() - start_time

                # Transform action
                transformed_action = transform_action(actions[0], action_transform)
                actions_taken.append(transformed_action.tolist())

                print(f"    Raw action: [{actions[0, 0]:.3f}, {actions[0, 1]:.3f}, {actions[0, 2]:.3f}]")
                print(f"    Transformed: [{transformed_action[0]:.3f}, {transformed_action[1]:.3f}, {transformed_action[2]:.3f}]")
                print(f"    Planning time: {planning_time:.3f}s, Repr dist: {repr_l1_distance:.6f}")

                # Execute action
                try:
                    planned_delta = transformed_action[:7]
                    new_pos, new_rpy = compute_new_pose(pos, planned_delta[:3], planned_delta[3:6])

                    ik_res = qpos_from_site_pose(
                        physics=sim,
                        site_name=EE_SITE,
                        target_pos=new_pos,
                        target_quat=None,
                        inplace=False,
                        regularization_strength=1.0,
                        max_steps=2000,
                        tol=1e-4
                    )

                    waypoints = generate_joint_space_min_jerk(
                        start=current_joint_pos,
                        goal=ik_res.qpos[:ARM_nJnt],
                        time_to_go=horizon,
                        dt=step_dt
                    )

                    current_joint_pos = execute_waypoints(waypoints, sim, horizon, step_dt)

                except Exception as e:
                    print(f"    WARNING: Planning failed: {e}")
                    # Hold current position
                    current_joint_pos = sim.data.qpos[:ARM_nJnt].copy()

        # Final measurement
        sim.forward()
        final_ee_pos = sim.data.site_xpos[ee_sid].copy()
        final_distance = np.linalg.norm(final_ee_pos - target_pos)
        success = final_distance <= success_threshold

        print(f"\n  Final distance: {final_distance:.4f}m [{'SUCCESS' if success else 'FAIL'}]")

        # Store results
        result = {
            'trajectory_index': traj_meta['trajectory_index'],
            'trajectory_path': traj_path,
            'target_position': target_pos.tolist(),
            'target_distance': float(target_distance),
            'trajectory_direction': traj_direction,
            'final_distance': float(final_distance),
            'success': bool(success),
            'distances_per_step': distances,
            'repr_distances_per_step': repr_distances,
            'actions_taken': actions_taken,
            'planning_steps': planning_steps
        }
        eval_results.append(result)

        sim.reset()

    # Save evaluation results
    results_path = out_path / "eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print("=" * 80)

    successes = [r['success'] for r in eval_results]
    final_distances = [r['final_distance'] for r in eval_results]
    target_distances = [r['target_distance'] for r in eval_results]

    print(f"\nTotal trajectories evaluated: {len(eval_results)}")
    print(f"Success rate: {sum(successes)}/{len(successes)} ({100*sum(successes)/len(successes):.1f}%)")
    print(f"Success threshold: {success_threshold}m")
    print(f"\nTarget distances: mean={np.mean(target_distances):.4f}m, "
          f"std={np.std(target_distances):.4f}m")
    print(f"Final distances:  mean={np.mean(final_distances):.4f}m, "
          f"std={np.std(final_distances):.4f}m")

    # Per-direction statistics if available
    directions = set(r['trajectory_direction'] for r in eval_results)
    if len(directions) > 1:
        print(f"\nPer-direction statistics:")
        for direction in sorted(directions):
            dir_results = [r for r in eval_results if r['trajectory_direction'] == direction]
            dir_successes = [r['success'] for r in dir_results]
            dir_final_dists = [r['final_distance'] for r in dir_results]

            print(f"  {direction.upper()}-axis: "
                  f"{sum(dir_successes)}/{len(dir_successes)} success "
                  f"({100*sum(dir_successes)/len(dir_successes):.1f}%), "
                  f"mean final dist: {np.mean(dir_final_dists):.4f}m")

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {results_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
