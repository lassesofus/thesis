#!/usr/bin/env python3
"""
Probe B: On-Policy Prediction Error

Measures prediction loss on states visited during planning execution.
This tests whether the predictor is accurate on actually visited states
(not just the training distribution).

Based on eval_vjepa_models.py from robohive/utils
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

# Set MuJoCo rendering backend before any mujoco imports
os.environ["MUJOCO_GL"] = "egl"

# Enable CEM planning optimizations (GPU pose computation, cudnn.benchmark, TF32)
os.environ["OPT_AMP"] = "0"
os.environ["VJEPA_OPTIMIZE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Add paths
sys.path.insert(0, "/home/s185927/thesis/vjepa2")
sys.path.insert(0, "/home/s185927/thesis/robohive/robohive")

from app.vjepa_droid.utils import init_video_model
from app.vjepa_droid.transforms import make_transforms
from notebooks.utils.world_model_wrapper import WorldModel
from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import generate_joint_space_min_jerk
from robohive.utils.xml_utils import reassign_parent

# Configuration
ARM_nJnt = 7
EE_SITE = "end_effector"
ARM_JNT0_ROBOTIQ = np.array([
    -0.0321842, -0.394346, 0.00932319, -2.77917, -0.011826, 0.713889, 0.74663
])


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

    encoder, predictor = init_video_model(
        uniform_power=cfgs_model.get('uniform_power', True),
        device=torch.device(device),
        patch_size=patch_size,
        max_num_frames=512,
        tubelet_size=cfgs_data.get('tubelet_size', 2),
        model_name=cfgs_model.get('model_name', 'vit_giant_xformers'),
        crop_size=crop_size,
        pred_depth=cfgs_model.get('pred_depth', 24),
        pred_num_heads=cfgs_model.get('pred_num_heads', 16),
        pred_embed_dim=cfgs_model.get('pred_embed_dim', 1024),
        action_embed_dim=7,
        pred_is_frame_causal=cfgs_model.get('pred_is_frame_causal', True),
        use_extrinsics=cfgs_model.get('use_extrinsics', False),
        use_sdpa=cfgs_meta.get('use_sdpa', True),
        use_silu=cfgs_model.get('use_silu', False),
        use_pred_silu=cfgs_model.get('use_pred_silu', False),
        wide_silu=cfgs_model.get('wide_silu', True),
        use_rope=cfgs_model.get('use_rope', True),
        use_activation_checkpointing=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    def strip_module_prefix(state_dict):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}

    encoder.load_state_dict(strip_module_prefix(checkpoint['encoder']), strict=False)
    if 'predictor' in checkpoint:
        predictor.load_state_dict(strip_module_prefix(checkpoint['predictor']), strict=False)

    target_encoder = copy.deepcopy(encoder)
    if 'target_encoder' in checkpoint:
        target_encoder.load_state_dict(strip_module_prefix(checkpoint['target_encoder']), strict=False)

    encoder.eval()
    predictor.eval()
    target_encoder.eval()

    tokens_per_frame = (crop_size // patch_size) ** 2

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1.777, 1.777),
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

    return encoder, predictor, target_encoder, transform, world_model, tokens_per_frame, device


def get_ee_orientation(sim, ee_sid):
    """Get end-effector orientation as euler angles."""
    xmat = sim.data.site_xmat[ee_sid].reshape(3, 3)
    return Rotation.from_matrix(xmat).as_euler('xyz', degrees=False)


def transform_pose_to_droid_frame(pose):
    """Transform RoboHive pose to DROID frame."""
    transformed = pose.copy()
    transformed[0] = pose[1]
    transformed[1] = -pose[0]
    transformed[2] = pose[2]
    transformed[3] = pose[4]
    transformed[4] = -pose[3]
    transformed[5] = pose[5]
    return transformed


def transform_action(action, transform_type='swap_xy_negate_x'):
    """Transform action from DROID to RoboHive frame."""
    transformed = action.copy()
    if transform_type == 'swap_xy_negate_x':
        transformed[0], transformed[1] = -action[1], action[0]
        transformed[3], transformed[4] = action[4], action[3]
    return transformed


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

    if waypoints:
        final_ctrl = waypoints[-1]['position'].copy()
        hold_steps = max(5, int(0.2 / sim.model.opt.timestep))
        for _ in range(hold_steps):
            sim.data.ctrl[:ARM_nJnt] = final_ctrl
            sim.advance(render=False)
        return final_ctrl
    return sim.data.qpos[:ARM_nJnt].copy()


def load_sim(gripper='robotiq'):
    """Load simulation with specified gripper."""
    sim_path = '/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_robotiq_v0.xml'
    sim = SimScene.get_sim(model_handle=sim_path)

    # Reparent ee_mount for robotiq
    raw_xml = sim.model.get_xml()
    processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="ee_mount")
    processed_path = os.path.join(os.path.dirname(os.path.abspath(sim_path)), '_robotiq_processed.xml')
    with open(processed_path, 'w') as f:
        f.write(processed_xml)
    sim = SimScene.get_sim(model_handle=processed_path)
    os.remove(processed_path)

    return sim


def encode_frame(encoder, frame, transform, device, tokens_per_frame, normalize=True):
    """Encode a single frame."""
    # Stack frame with itself (model expects 2-frame input)
    combined = np.stack([frame, frame], axis=0)
    clips = transform(combined).unsqueeze(0).to(device)

    B, C, T, H, W = clips.size()
    with torch.no_grad():
        c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = encoder(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if normalize:
            h = F.layer_norm(h, (h.size(-1),))
        z = h[:, :tokens_per_frame].contiguous()
    return z


def run_probe_b_for_trajectory(
    probe_id, target_pos, sim, ee_sid, encoder, predictor, target_encoder,
    transform, world_model, tokens_per_frame, device,
    horizon=3.0, step_dt=None, k_visit=2,
    camera_name='left_cam', width=640, height=480, device_id=0
):
    """Run planning for k_visit steps and compute prediction loss."""
    if step_dt is None:
        step_dt = sim.model.opt.timestep

    # Reset to start
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0_ROBOTIQ
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
    sim.data.ctrl[:ARM_nJnt] = ARM_JNT0_ROBOTIQ.copy()
    sim.forward()

    # Phase 1: Move to target to capture goal RGB
    ik_result = qpos_from_site_pose(
        physics=sim, site_name=EE_SITE, target_pos=target_pos,
        target_quat=None, inplace=False, regularization_strength=1.0,
        max_steps=2000, tol=1e-4
    )
    waypoints = generate_joint_space_min_jerk(
        start=ARM_JNT0_ROBOTIQ, goal=ik_result.qpos[:ARM_nJnt],
        time_to_go=horizon, dt=step_dt
    )
    current_joint_pos = execute_waypoints(waypoints, sim, horizon, step_dt)

    # Capture goal RGB
    sim.forward()
    goal_rgb = sim.renderer.render_offscreen(
        width=width, height=height, camera_id=camera_name, device_id=device_id
    )

    # Phase 2: Return to start
    return_waypoints = generate_joint_space_min_jerk(
        start=current_joint_pos, goal=ARM_JNT0_ROBOTIQ,
        time_to_go=horizon, dt=step_dt
    )
    current_joint_pos = execute_waypoints(return_waypoints, sim, horizon, step_dt)

    # Phase 3: Planning with prediction loss tracking
    prediction_losses = []
    physical_distances = []

    with torch.no_grad():
        # Encode goal once
        z_goal = encode_frame(encoder, goal_rgb, transform, device, tokens_per_frame)

        for step_idx in range(k_visit):
            sim.forward()
            current_ee_pos = sim.data.site_xpos[ee_sid].copy()
            distance = np.linalg.norm(current_ee_pos - target_pos)
            physical_distances.append(float(distance))

            # Capture current RGB
            current_rgb = sim.renderer.render_offscreen(
                width=width, height=height, camera_id=camera_name, device_id=device_id
            )

            # Encode current frame
            z_current = encode_frame(encoder, current_rgb, transform, device, tokens_per_frame)

            # Build state
            robohive_rpy = get_ee_orientation(sim, ee_sid)
            robohive_pose = np.concatenate([current_ee_pos, robohive_rpy])
            droid_pose = transform_pose_to_droid_frame(robohive_pose)
            current_state = np.concatenate([droid_pose, [1.0]])  # gripper=1.0
            states = torch.tensor(current_state, device=device).unsqueeze(0).unsqueeze(0).to(dtype=z_current.dtype)

            # Plan action
            actions = world_model.infer_next_action(z_current, states, z_goal).cpu().numpy()
            action = actions[0]

            # Get predicted next representation BEFORE executing
            action_tensor = torch.tensor(action, device=device, dtype=z_current.dtype).unsqueeze(0).unsqueeze(0)
            z_predicted = predictor(z_current, action_tensor, states)[:, -tokens_per_frame:]
            z_predicted = F.layer_norm(z_predicted, (z_predicted.size(-1),))

            # Execute action in simulation
            transformed_action = transform_action(action, 'swap_xy_negate_x')
            try:
                new_pos = current_ee_pos + transformed_action[:3]
                ik_res = qpos_from_site_pose(
                    physics=sim, site_name=EE_SITE, target_pos=new_pos,
                    target_quat=None, inplace=False, regularization_strength=1.0,
                    max_steps=2000, tol=1e-4
                )
                planned_waypoints = generate_joint_space_min_jerk(
                    start=current_joint_pos, goal=ik_res.qpos[:ARM_nJnt],
                    time_to_go=horizon, dt=step_dt
                )
                current_joint_pos = execute_waypoints(planned_waypoints, sim, horizon, step_dt)
            except Exception as e:
                print(f"    Warning: IK failed at step {step_idx}: {e}")
                continue

            # Capture actual next state
            sim.forward()
            next_rgb = sim.renderer.render_offscreen(
                width=width, height=height, camera_id=camera_name, device_id=device_id
            )

            # Encode actual next state using target encoder
            z_actual = encode_frame(target_encoder, next_rgb, transform, device, tokens_per_frame)

            # Compute prediction loss
            pred_loss = torch.mean(torch.abs(z_predicted - z_actual)).item()
            prediction_losses.append(pred_loss)

    return {
        'prediction_losses': prediction_losses,
        'physical_distances': physical_distances,
        'mean_prediction_loss': float(np.mean(prediction_losses)) if prediction_losses else 0.0,
    }


def run_probe_b(
    checkpoint_path,
    config_path,
    probe_ids,
    metadata_path,
    device='cuda:0',
    k_visit=2,
):
    """Run Probe B on all probe trajectories."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load metadata to get target positions
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    # Create lookup by trajectory_index
    metadata_lookup = {m['trajectory_index']: m for m in all_metadata}

    # Load model
    encoder, predictor, target_encoder, transform, world_model, tokens_per_frame, device = load_model(
        checkpoint_path, config, device
    )

    # Load simulation
    sim = load_sim(gripper='robotiq')
    ee_sid = sim.model.site_name2id(EE_SITE)
    step_dt = sim.model.opt.timestep

    results = {
        'prediction_losses': [],
        'mean_loss_per_probe': [],
    }

    for probe_id in tqdm(probe_ids, desc="Processing probes"):
        if probe_id not in metadata_lookup:
            print(f"Warning: Probe {probe_id} not in metadata")
            continue

        meta = metadata_lookup[probe_id]
        target_pos = np.array(meta['target_position'])

        try:
            probe_result = run_probe_b_for_trajectory(
                probe_id, target_pos, sim, ee_sid,
                encoder, predictor, target_encoder, transform,
                world_model, tokens_per_frame, device,
                horizon=3.0, step_dt=step_dt, k_visit=k_visit
            )

            results['prediction_losses'].extend(probe_result['prediction_losses'])
            results['mean_loss_per_probe'].append(probe_result['mean_prediction_loss'])

        except Exception as e:
            print(f"Warning: Failed on probe {probe_id}: {e}")
            continue

    # Aggregate
    aggregated = {
        'onpolicy_pred_loss': float(np.mean(results['prediction_losses'])) if results['prediction_losses'] else 0.0,
        'onpolicy_pred_loss_std': float(np.std(results['prediction_losses'])) if results['prediction_losses'] else 0.0,
        'n_samples': len(results['prediction_losses']),
    }

    # Cleanup
    del encoder, predictor, target_encoder, world_model
    torch.cuda.empty_cache()

    return aggregated, results


def main():
    parser = argparse.ArgumentParser(description='Probe B: On-Policy Prediction Loss')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--probe_ids_file', type=str, required=True)
    parser.add_argument('--metadata', type=str, default='/data/s185927/droid_sim/axis_aligned/x_axis/trajectory_metadata.json')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--k_visit', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

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

    # Run probe
    aggregated, raw_results = run_probe_b(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        probe_ids=probe_ids,
        metadata_path=args.metadata,
        device=args.device,
        k_visit=args.k_visit,
    )

    # Save results
    output = {
        'aggregated': aggregated,
        'per_probe': raw_results,
        'probe_ids': probe_ids,
        'checkpoint': args.checkpoint,
        'k_visit': args.k_visit,
        'seed': args.seed,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"\nAggregated metrics:")
    for k, v in aggregated.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
