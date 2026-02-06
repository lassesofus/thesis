#!/usr/bin/env python3
"""
Evaluate multiple V-JEPA models on test samples and compare performance.

Tests trained checkpoints against Meta baseline using CEM planning.

EXAMPLE USAGE:
    # Evaluate all models (training data collected WITHOUT coordinate transformation)
    python eval_vjepa_models.py \
        --metadata /data/s185927/droid_sim/y_axis/trajectory_metadata.json \
        --model_dir /data/s185927/vjepa2/weights/droid \
        --planning_steps 5 \
        --out_dir /data/s185927/vjepa_eval_results \
        --action_transform none

    # Evaluate models trained on DROID-frame data (WITH coordinate transformation)
    python eval_vjepa_models.py \
        --metadata /data/s185927/droid_sim/y_axis/trajectory_metadata.json \
        --model_dir /data/s185927/vjepa2/weights/droid \
        --planning_steps 5 \
        --out_dir /data/s185927/vjepa_eval_results \
        --action_transform swap_xy_negate_x

    # Test run with limited samples
    python eval_vjepa_models.py \
        --metadata /data/s185927/droid_sim/y_axis/trajectory_metadata.json \
        --model_dir /data/s185927/vjepa2/weights/droid \
        --planning_steps 5 \
        --out_dir /tmp/test_eval \
        --max_samples 5 \
        --action_transform none
"""

import warnings
try:
    from pydantic import PydanticUserWarning
    warnings.filterwarnings("ignore", category=PydanticUserWarning)
except:
    pass
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import json
import time
from pathlib import Path

import click
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
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
from robohive.utils.xml_utils import reassign_parent

# Configuration
ARM_nJnt = 7
EE_SITE = "end_effector"
# Default ARM_JNT0 for Franka gripper
ARM_JNT0_FRANKA = np.array([
    -0.0321842, -0.394346, 0.00932319,
    -2.77917, -0.011826, 0.713889, 1.53183
])
# ARM_JNT0 for RobotiQ gripper (Joint 7 rotated -45 degrees to match real robot orientation)
ARM_JNT0_ROBOTIQ = np.array([
    -0.0321842,  # Joint 1
    -0.394346,   # Joint 2
    0.00932319,  # Joint 3
    -2.77917,    # Joint 4
    -0.011826,   # Joint 5
    0.713889,    # Joint 6
    0.74663      # Joint 7 (original 1.53183, -π/4 ≈ 0.785 for angled gripper)
])


def get_ee_orientation(sim, ee_sid):
    """
    Get end-effector orientation as euler angles (roll, pitch, yaw).
    """
    xmat = sim.data.site_xmat[ee_sid].reshape(3, 3)
    rpy = Rotation.from_matrix(xmat).as_euler('xyz', degrees=False)
    return rpy


class ModelConfig:
    """Configuration for a V-JEPA model to evaluate."""
    def __init__(self, name, checkpoint_path=None, use_hub=False, action_transform='none'):
        self.name = name
        self.checkpoint_path = checkpoint_path
        self.use_hub = use_hub
        self.action_transform = action_transform


def load_model(config, device):
    """Load V-JEPA model from checkpoint or PyTorch Hub."""
    print(f"\nLoading model: {config.name}")

    if config.use_hub:
        print("  Loading from PyTorch Hub (Meta baseline)...")
        encoder, predictor = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_ac_vit_giant"
        )
    else:
        print(f"  Loading checkpoint: {config.checkpoint_path}")
        # Load checkpoint to CPU first to save GPU memory
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')

        # Load architecture from hub
        encoder, predictor = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_ac_vit_giant"
        )

        # Load trained weights
        if 'encoder' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder'])
        elif 'target_encoder' in checkpoint:
            encoder.load_state_dict(checkpoint['target_encoder'])
        else:
            raise ValueError("Cannot find encoder weights in checkpoint")

        if 'predictor' in checkpoint:
            predictor.load_state_dict(checkpoint['predictor'])
        else:
            raise ValueError("Cannot find predictor weights in checkpoint")

        # Free checkpoint memory
        del checkpoint
        torch.cuda.empty_cache()

    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    crop_size = 256
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1.777, 1.777),  # Match training config
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

    print(f"  Model loaded")
    return encoder, predictor, transform, world_model, tokens_per_frame


def execute_waypoints(waypoints, sim, horizon, step_dt):
    """Execute joint waypoints in simulation."""
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
        return final_ctrl
    return sim.data.qpos[:ARM_nJnt].copy()


def compute_new_pose(current_pos, delta_pos, delta_rpy):
    """Compute new EE pose from deltas."""
    return current_pos + delta_pos, delta_rpy


def transform_action(action, transform_type='none'):
    """
    Transform actions between coordinate frames.

    Args:
        action: Action array [dx, dy, dz, droll, dpitch, dyaw, gripper, ...]
        transform_type: Type of transformation
            - 'none': No transformation (training data in RoboHive frame)
            - 'swap_xy_negate_x': DROID → RoboHive (for data generated with generate_droid_sim_data.py)
            - 'swap_xy': Simple x-y swap (legacy)

    Returns:
        Transformed action array
    """
    transformed = action.copy()

    if transform_type == 'none':
        # No transformation - model trained on RoboHive frame directly
        pass
    elif transform_type == 'swap_xy_negate_x':
        # DROID → RoboHive: x'=-y, y'=x, z'=z
        transformed[0], transformed[1] = -action[1], action[0]
        transformed[3], transformed[4] = action[4], action[3]
    elif transform_type == 'swap_xy':
        # Simple swap: x'=y, y'=x, z'=z
        transformed[0], transformed[1] = action[1], action[0]
        transformed[3], transformed[4] = action[4], action[3]
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    return transformed


def transform_pose_to_droid_frame(pose):
    """
    Transform pose from RoboHive coordinate frame to DROID coordinate frame.

    This is the same transformation used in generate_droid_sim_data.py when saving
    training data. It must be applied to the current EE pose before passing to the
    model during evaluation, since the model was trained on DROID-frame data.

    Transformation (RoboHive → DROID):
        DROID_x = RoboHive_y
        DROID_y = -RoboHive_x
        DROID_z = RoboHive_z

    Args:
        pose: [x, y, z, roll, pitch, yaw] in RoboHive frame

    Returns:
        transformed_pose: [x, y, z, roll, pitch, yaw] in DROID frame
    """
    transformed = pose.copy()

    # Transform position: DROID_x = RoboHive_y, DROID_y = -RoboHive_x
    transformed[0] = pose[1]    # DROID_x = RoboHive_y
    transformed[1] = -pose[0]   # DROID_y = -RoboHive_x
    transformed[2] = pose[2]    # DROID_z = RoboHive_z

    # Transform orientation: swap roll/pitch and negate new pitch
    transformed[3] = pose[4]    # new_roll = old_pitch
    transformed[4] = -pose[3]   # new_pitch = -old_roll
    transformed[5] = pose[5]    # new_yaw = old_yaw

    return transformed


def evaluate_sample(sample, sim, ee_sid, encoder, predictor, transform,
                   world_model, tokens_per_frame, planning_steps,
                   horizon, step_dt, device, action_transform='none',
                   camera_name='left_cam', width=640, height=480, device_id=0,
                   save_images=False, image_dir=None):
    """Evaluate planning on a single test sample.

    If save_images=True and image_dir is provided, saves:
    - goal.png: The goal image
    - step_N.png: Current observation at each planning step
    - final.png: Final observation after planning
    """
    target_pos = np.array(sample['target_position'])
    saved_images = {}  # Store images if requested

    # Reset to start
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
    sim.data.ctrl[:ARM_nJnt] = ARM_JNT0.copy()
    sim.forward()

    # Phase 1: Move to target to capture goal RGB
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

    waypoints = generate_joint_space_min_jerk(
        start=ARM_JNT0,
        goal=ik_result.qpos[:ARM_nJnt],
        time_to_go=horizon,
        dt=step_dt
    )
    current_joint_pos = execute_waypoints(waypoints, sim, horizon, step_dt)

    # Capture goal RGB
    sim.forward()
    goal_rgb = sim.renderer.render_offscreen(
        width=width, height=height, camera_id=camera_name, device_id=device_id
    )

    # Save goal image if requested
    if save_images and image_dir:
        os.makedirs(image_dir, exist_ok=True)
        goal_img = Image.fromarray(goal_rgb)
        goal_img.save(os.path.join(image_dir, 'goal.png'))

    # Phase 2: Return to start
    return_waypoints = generate_joint_space_min_jerk(
        start=current_joint_pos,
        goal=ARM_JNT0,
        time_to_go=horizon,
        dt=step_dt
    )
    current_joint_pos = execute_waypoints(return_waypoints, sim, horizon, step_dt)

    # Phase 3: CEM Planning
    distances = []
    repr_distances = []

    with torch.no_grad():
        for step_idx in range(planning_steps):
            sim.forward()
            current_ee_pos = sim.data.site_xpos[ee_sid].copy()
            distance = np.linalg.norm(current_ee_pos - target_pos)
            distances.append(distance)
            print(f"      Step {step_idx + 1}/{planning_steps}: dist={distance:.4f}m")

            # Capture current RGB
            current_rgb = sim.renderer.render_offscreen(
                width=width, height=height, camera_id=camera_name, device_id=device_id
            )

            # Save step image if requested
            if save_images and image_dir:
                step_img = Image.fromarray(current_rgb)
                step_img.save(os.path.join(image_dir, f'step_{step_idx}.png'))

            # Prepare input
            combined_rgb = np.stack([current_rgb, goal_rgb], axis=0)
            clips = transform(combined_rgb).unsqueeze(0).to(device)

            # Forward pass
            B, C, T, H, W = clips.size()
            c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            h = encoder(c)
            h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
            h = F.layer_norm(h, (h.size(-1),))

            z_n = h[:, :tokens_per_frame].contiguous().clone()
            z_goal = h[:, -tokens_per_frame:].contiguous().clone()

            repr_l1_distance = torch.mean(torch.abs(z_n - z_goal)).item()
            repr_distances.append(repr_l1_distance)

            # Build state in DROID frame (model was trained on DROID-frame data)
            robohive_rpy = get_ee_orientation(sim, ee_sid)
            robohive_pose = np.concatenate([current_ee_pos, robohive_rpy])
            droid_pose = transform_pose_to_droid_frame(robohive_pose)
            gripper = 1.0  # Match training data (DROID closed convention)
            current_state = np.concatenate([droid_pose, [gripper]])
            states = torch.tensor(current_state, device=device).unsqueeze(0).unsqueeze(0)
            s_n = states[:, :1].to(dtype=z_n.dtype)

            # Plan action (model outputs DROID frame, transform to RoboHive for execution)
            actions = world_model.infer_next_action(z_n, s_n, z_goal).cpu().numpy()
            transformed_action = transform_action(actions[0], action_transform)

            # Execute action (use RoboHive frame positions for IK)
            try:
                planned_delta = transformed_action[:7]
                new_pos, new_rpy = compute_new_pose(current_ee_pos, planned_delta[:3], planned_delta[3:6])

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

                planned_waypoints = generate_joint_space_min_jerk(
                    start=current_joint_pos,
                    goal=ik_res.qpos[:ARM_nJnt],
                    time_to_go=horizon,
                    dt=step_dt
                )
                current_joint_pos = execute_waypoints(planned_waypoints, sim, horizon, step_dt)
            except Exception as e:
                # Fallback: hold position
                pass

        # Final measurement
        sim.forward()
        final_ee_pos = sim.data.site_xpos[ee_sid].copy()
        final_distance = np.linalg.norm(final_ee_pos - target_pos)
        distances.append(final_distance)

        # Save final image if requested
        if save_images and image_dir:
            final_rgb = sim.renderer.render_offscreen(
                width=width, height=height, camera_id=camera_name, device_id=device_id
            )
            final_img = Image.fromarray(final_rgb)
            final_img.save(os.path.join(image_dir, 'final.png'))

    return {
        'final_distance': float(final_distance),
        'distances_per_step': [float(d) for d in distances],
        'repr_distances_per_step': [float(d) for d in repr_distances],
        'success': bool(final_distance < 0.05),
        'target_distance': float(sample['target_distance'])
    }


@click.command(help=__doc__)
@click.option('--metadata', type=str, required=True,
              help='Path to trajectory_metadata.json')
@click.option('--model_dir', type=str, required=True,
              help='Directory containing model subdirectories')
@click.option('--planning_steps', type=int, default=5,
              help='Number of CEM planning steps')
@click.option('--out_dir', type=str, required=True,
              help='Output directory for results')
@click.option('--horizon', type=float, default=3.0,
              help='Time per action execution (seconds)')
@click.option('--sim_path', type=str,
              default='/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_v0.xml',
              help='Path to MuJoCo XML')
@click.option('--max_samples', type=int, default=None,
              help='Max test samples to evaluate')
@click.option('--checkpoint_name', type=str, default='best.pt',
              help='Checkpoint filename (best.pt, latest.pt)')
@click.option('--action_transform', type=click.Choice(['none', 'swap_xy_negate_x', 'swap_xy']),
              default='none',
              help='Action transformation: none (training data in RoboHive frame), swap_xy_negate_x (DROID→RoboHive), swap_xy (legacy)')
@click.option('--save_images', is_flag=True, default=False,
              help='Save visualization images (goal, step observations, final) for each sample')
@click.option('--split', type=click.Choice(['train', 'test']), default='test',
              help='Which data split to evaluate on (train or test)')
@click.option('--gripper', type=click.Choice(['franka', 'robotiq']), default='franka',
              help='End-effector gripper type: franka (default parallel-jaw) or robotiq (2F-85)')
def main(metadata, model_dir, planning_steps, out_dir, horizon, sim_path,
         max_samples, checkpoint_name, action_transform, save_images, split, gripper):
    """Evaluate V-JEPA planning on samples from the specified split."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Action transformation: {action_transform}\n")

    # Load metadata
    print(f"Loading metadata: {metadata}")
    with open(metadata, 'r') as f:
        all_metadata = json.load(f)

    samples = [m for m in all_metadata if m['split'] == split]
    print(f"Found {len(samples)} {split} samples")

    if max_samples:
        samples = samples[:max_samples]
        print(f"Limiting to {max_samples} samples")

    os.makedirs(out_dir, exist_ok=True)

    # Configure models
    model_dir_path = Path(model_dir)
    model_configs = []

    # Add trained models (use custom action_transform)
    for subdir in sorted(model_dir_path.iterdir()):
        if subdir.is_dir():
            checkpoint_path = subdir / checkpoint_name
            if checkpoint_path.exists():
                model_configs.append(ModelConfig(
                    name=subdir.name,
                    checkpoint_path=str(checkpoint_path),
                    use_hub=False,
                    action_transform=action_transform  # Use CLI-specified transform
                ))

    # Add Meta baseline (always uses swap_xy_negate_x since trained on DROID data)
    model_configs.append(ModelConfig(
        name="meta_baseline",
        use_hub=True,
        action_transform='swap_xy_negate_x'  # Meta baseline trained on DROID frame
    ))

    print(f"\nEvaluating {len(model_configs)} models:")
    for cfg in model_configs:
        print(f"  - {cfg.name}")

    # Gripper-aware model path selection
    FRANKA_MODEL = '/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_v0.xml'
    ROBOTIQ_MODEL = '/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_robotiq_v0.xml'

    # Override sim_path if using default Franka model and robotiq gripper is specified
    if sim_path == FRANKA_MODEL and gripper == 'robotiq':
        sim_path = ROBOTIQ_MODEL
        print(f"Using RobotiQ gripper model: {sim_path}")

    # Load simulation
    print(f"\nLoading simulation: {sim_path}")
    sim = SimScene.get_sim(model_handle=sim_path)

    # For RobotiQ model, reparent ee_mount to panda0_link7
    if gripper == 'robotiq':
        raw_xml = sim.model.get_xml()
        processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="ee_mount")
        # Keep processed file in same directory to preserve relative mesh paths
        processed_path = os.path.join(os.path.dirname(os.path.abspath(sim_path)), '_robotiq_processed.xml')
        with open(processed_path, 'w') as f:
            f.write(processed_xml)
        sim = SimScene.get_sim(model_handle=processed_path)
        os.remove(processed_path)  # Clean up temp file
        print("RobotiQ gripper attached to Franka arm (panda0_link7)")

    # Select ARM_JNT0 based on gripper type
    global ARM_JNT0
    if gripper == 'robotiq':
        ARM_JNT0 = ARM_JNT0_ROBOTIQ
        print(f"Using RobotiQ ARM_JNT0 (Joint 7 = {ARM_JNT0[6]:.5f})")
    else:
        ARM_JNT0 = ARM_JNT0_FRANKA
        print(f"Using Franka ARM_JNT0 (Joint 7 = {ARM_JNT0[6]:.5f})")

    ee_sid = sim.model.site_name2id(EE_SITE)
    step_dt = sim.model.opt.timestep

    # Initialize results storage
    import time
    all_results = {config.name: [] for config in model_configs}
    model_stats = {config.name: {'total_distance': 0.0, 'successes': 0} for config in model_configs}

    # Process samples in batches of 25
    batch_size = 25
    total_samples = len(samples)
    num_batches = (total_samples + batch_size - 1) // batch_size

    overall_start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_samples)
        batch_samples = samples[batch_start:batch_end]

        print(f"\n{'#'*80}")
        print(f"BATCH {batch_idx+1}/{num_batches}: Samples {batch_start+1}-{batch_end}/{total_samples}")
        print(f"{'#'*80}")

        # Evaluate each model on this batch (load -> evaluate -> offload)
        for model_idx, model_config in enumerate(model_configs):
            print(f"\n{'─'*80}")
            print(f"Model {model_idx+1}/{len(model_configs)}: {model_config.name}")
            print(f"{'─'*80}")

            # Load model
            load_start = time.time()
            encoder, predictor, transform, world_model, tokens_per_frame = load_model(model_config, device)
            print(f"  Loaded in {time.time() - load_start:.1f}s")

            # Evaluate on batch samples
            for local_idx, sample in enumerate(batch_samples):
                sample_idx = batch_start + local_idx
                sample_start_time = time.time()

                print(f"\n  Sample {sample_idx+1}/{total_samples} (ep {sample['trajectory_index']})")
                print(f"    Target: {sample['target_position']}, Distance: {sample['target_distance']:.4f}m")

                try:
                    # Set up image directory if saving images
                    image_dir = None
                    if save_images:
                        image_dir = Path(out_dir) / model_config.name / f"sample_{sample_idx}"

                    result = evaluate_sample(
                        sample, sim, ee_sid, encoder, predictor, transform,
                        world_model, tokens_per_frame, planning_steps,
                        horizon, step_dt, device,
                        action_transform=model_config.action_transform,
                        save_images=save_images,
                        image_dir=str(image_dir) if image_dir else None
                    )
                    result['sample_idx'] = sample_idx
                    result['trajectory_index'] = sample['trajectory_index']
                    all_results[model_config.name].append(result)

                    # Update statistics
                    model_stats[model_config.name]['total_distance'] += result['final_distance']
                    if result['success']:
                        model_stats[model_config.name]['successes'] += 1

                    elapsed = time.time() - sample_start_time
                    print(f"    Final: {result['final_distance']:.4f}m [{'✓ SUCCESS' if result['success'] else '✗ FAIL'}] ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

            # Offload model to free memory
            del encoder, predictor, transform, world_model
            torch.cuda.empty_cache()
            print(f"  Model offloaded")

        # Save checkpoint for all models after this batch
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx+1} COMPLETED - Saving checkpoints")
        print(f"{'='*80}")

        for model_name in all_results.keys():
            n_samples = len(all_results[model_name])
            if n_samples > 0:
                checkpoint_path = Path(out_dir) / f"{model_name}_results_n{n_samples}.json"
                with open(checkpoint_path, 'w') as f:
                    json.dump(all_results[model_name], f, indent=2)

                mean_dist = model_stats[model_name]['total_distance'] / n_samples
                success_rate = model_stats[model_name]['successes'] / n_samples * 100
                print(f"  {model_name}: {n_samples} samples, Mean={mean_dist:.4f}m, Success={success_rate:.1f}%")

        elapsed_total = time.time() - overall_start_time
        print(f"\nElapsed time: {elapsed_total:.1f}s")

    # Print final summary
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")

    for model_name in all_results.keys():
        n_samples = len(all_results[model_name])
        if n_samples > 0:
            mean_dist = model_stats[model_name]['total_distance'] / n_samples
            success_rate = model_stats[model_name]['successes'] / n_samples * 100
            print(f"\n{model_name}:")
            print(f"  Total samples: {n_samples}")
            print(f"  Mean distance: {mean_dist:.4f}m")
            print(f"  Success rate: {success_rate:.1f}%")

            # Save final results
            result_path = Path(out_dir) / f"{model_name}_results.json"
            with open(result_path, 'w') as f:
                json.dump(all_results[model_name], f, indent=2)
            print(f"  Saved: {result_path}")

    # Save combined summary
    summary = {}
    for model_name, results in all_results.items():
        final_dists = [r['final_distance'] for r in results]
        successes = [r['success'] for r in results]
        summary[model_name] = {
            'mean_final_distance': float(np.mean(final_dists)),
            'std_final_distance': float(np.std(final_dists)),
            'median_final_distance': float(np.median(final_dists)),
            'success_rate': float(sum(successes) / len(successes)),
            'num_samples': len(results)
        }

    summary_path = Path(out_dir) / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")
    for model_name in sorted(summary.keys()):
        stats = summary[model_name]
        print(f"{model_name:30s}: {stats['mean_final_distance']:.4f}m ± {stats['std_final_distance']:.4f}m "
              f"(success: {stats['success_rate']*100:.1f}%)")

    print(f"\nSaved summary: {summary_path}")
    print(f"Done! Results in: {out_dir}")


if __name__ == '__main__':
    main()
