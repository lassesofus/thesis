# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/robohive
# License :: Apache 2.0
# ================================================= """

DESC = """
TUTORIAL: Calculate min-jerk trajectory using IK (Franka Reach, sphere sampling)\n
EXAMPLE:\n
    python tutorials/ik_minjerk_reach_sphere.py --sim_path envs/arms/franka/assets/franka_reach_v0.xml --horizon 2 --radius 0.10
"""

import warnings

# try to silence pydantic-specific user warnings (v2)
try:
    from pydantic import PydanticUserWarning
    warnings.filterwarnings("ignore", category=PydanticUserWarning)
except Exception:
    warnings.filterwarnings(
        "ignore",
        message=r".*The 'repr' attribute with value False was provided to the `Field\(\)` function.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The 'frozen' attribute with value True was provided to the `Field\(\)` function.*",
        category=UserWarning,
    )

# suppress the timm deprecation message about importing layers
warnings.filterwarnings(
    "ignore",
    message=r".*Importing from timm.models.layers is deprecated.*",
    category=FutureWarning,
)

import os, sys, math, pdb, time

_vjepa_root = "/home/s185927/thesis/vjepa2"
if os.path.isdir(_vjepa_root) and _vjepa_root not in sys.path:
    sys.path.insert(0, _vjepa_root)

from notebooks.utils.world_model_wrapper import WorldModel
from app.vjepa_droid.transforms import make_transforms
from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import generate_joint_space_min_jerk
from robohive.utils.quat_math import euler2quat

import click
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

try:
    import skvideo.io
except Exception:
    skvideo = None


# Config
ARM_nJnt = 7
EE_SITE = "end_effector"  # from the Franka chain include


# Set device and load VJEPA model + transform (used for planning)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def compute_new_pose(current_pos, delta_pos, delta_rpy):
    """
    Compute new end-effector pose given current position and deltas.

    Args:
        current_pos: Current position [x, y, z]
        delta_pos: Position delta [dx, dy, dz]
        delta_rpy: Orientation delta [droll, dpitch, dyaw]

    Returns:
        new_pos: New position [x, y, z]
        new_rpy: New orientation [roll, pitch, yaw]
    """
    new_pos = current_pos + delta_pos
    # For simplicity, just return the delta as the new orientation
    # In practice, you might want to integrate this properly with current orientation
    new_rpy = delta_rpy
    return new_pos, new_rpy


def capture_rgb_and_append(frames_list, sim, render, steps_per_frame,
                           frame_counter, width, height, camera_name, device_id):
    """Render offscreen and append to frames_list according to downsample factor."""
    if render != 'offscreen':
        return None

    frm = sim.renderer.render_offscreen(
        width=width,
        height=height,
        camera_id=camera_name,
        device_id=device_id
    )
    if frm is None:
        return None

    if (frame_counter[0] % steps_per_frame) == 0:
        frames_list.append(frm)

    frame_counter[0] += 1
    return frm


def execute_waypoints_and_record(waypoints, frames_list, horizon, step_dt,
                                 sim, render, steps_per_frame, frame_counter,
                                 width, height, camera_name, device_id):
    t = 0.0
    idx = 0
    num_waypoints = len(waypoints)

    while t <= horizon:
        # clamp index to last waypoint
        idx = min(idx, num_waypoints - 1)

        # directly set actuator target like the tutorial does
        sim.data.ctrl[:ARM_nJnt] = waypoints[idx]['position']

        # capture frame before advancing
        capture_rgb_and_append(
            frames_list, sim, render, steps_per_frame, frame_counter,
            width, height, camera_name, device_id
        )

        # advance simulation
        sim.advance(render=(render == 'onscreen'))
        t += step_dt
        idx += 1

    # hold final waypoint briefly with the planned position (not actual qpos)
    final_ctrl = waypoints[-1]['position'].copy()
    hold_steps = max(5, int(0.2 / sim.model.opt.timestep))
    for _ in range(hold_steps):
        sim.data.ctrl[:ARM_nJnt] = final_ctrl
        capture_rgb_and_append(
            frames_list, sim, render, steps_per_frame, frame_counter,
            width, height, camera_name, device_id
        )
        sim.advance(render=(render == 'onscreen'))

    # Return the final planned position for smooth chaining
    return final_ctrl


def forward_target(c, normalize_reps=True):
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def sample_point_in_sphere(center, R):
    """Uniform-in-volume sample inside a sphere centered at center with radius R."""
    u = np.random.rand()
    r = R * (u ** (1 / 3))
    phi = 2 * np.pi * np.random.rand()
    cost = np.random.uniform(-1, 1)
    sint = math.sqrt(1 - cost * cost)
    dir_vec = np.array([
        sint * math.cos(phi),
        sint * math.sin(phi),
        cost
    ])
    return center + r * dir_vec


def _free_cuda_cache():
    import gc, torch
    for name in [
        'z_hat', 's_hat', 'a_hat', 'loss',
        'plot_data', 'delta_x', 'delta_z', 'energy',
        'heatmap', 'xedges', 'yedges'
    ]:
        if name in globals():
            try:
                del globals()[name]
            except:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@click.command(help=DESC)
@click.option(
    '-s', '--sim_path',
    type=str,
    required=True,
    default='/home/s185927/src/robohive/robohive/envs/arms/franka/assets/franka_reach_v0.xml',
    help='environment XML to load'
)
@click.option('-h', '--horizon', type=float, default=3.0, help='time (s) to simulate per phase')
@click.option('-r', '--radius', type=float, default=0.2, help='sampling sphere radius (meters)')
@click.option(
    '--render',
    type=click.Choice(['onscreen', 'offscreen', 'none']),
    default='onscreen',
    help='visualize onscreen or offscreen'
)
@click.option('-c', '--camera_name', type=str, default='left_cam', help='Camera name for offscreen render')
@click.option('--out_dir', type=str, default='/data/s185927/robohive',
              help='Directory to save rendered frames/videos (used for offscreen)')
@click.option('--out_name', type=str, default='run_', help='Filename prefix for saved videos (offscreen)')
@click.option('--width', type=int, default=640, help='offscreen frame width')
@click.option('--height', type=int, default=480, help='offscreen frame height')
@click.option('--device_id', type=int, default=0, help='renderer device id (if used)')
@click.option('--fps', type=int, default=30, help='frames per second for output video')
@click.option('--max_episodes', type=int, default=1, help='number of episodes to run (only used when render=offscreen)')
@click.option('--fixed_target', is_flag=True, default=False,
              help='If set, use a fixed target offset relative to start instead of random sampling.')
@click.option(
    '--fixed_target_offset',
    type=float, nargs=3,
    default=(0.0, 0.0, 0.2),
    help='dx dy dz (m) offset from start when --fixed_target is used (default: 0 0 0.2).'
)
@click.option(
    '--experiment_type',
    type=click.Choice(['x', 'y', 'z']), default='z',
    help='Experiment type: which dimension to apply 0.2m offset (x, y, or z). Overrides --fixed_target_offset.'
)
@click.option('--num_targets', type=int, default=1,
              help='Number of consecutive targets to reach before returning home.')
@click.option('--planning_steps', type=int, default=10,
              help='Number of CEM planning steps to reach target in Phase 3.')
@click.option('--enable_vjepa_planning', is_flag=True, default=False,
              help='Enable Phase 3: V-JEPA based CEM planning with distance tracking.')
@click.option('--save_distance_data', is_flag=True, default=False,
              help='Save detailed distance data for all phases.')
@click.option(
    '--action_transform',
    type=str,
    default='none',
    help='Coordinate transformation for actions: none, swap_xy, negate_x, negate_y, swap_xy_negate_x'
)
@click.option('--save_planning_images', is_flag=True, default=False,
              help='Save RGB images used during V-JEPA planning for inspection')
@click.option('--visualize_planning', is_flag=True, default=False,
              help='Create side-by-side visualization of current/goal images during planning')
              
def main(sim_path, horizon, radius, render, camera_name, out_dir, out_name,
         width, height, device_id, fps, max_episodes, fixed_target,
         fixed_target_offset, experiment_type, num_targets, planning_steps,
         enable_vjepa_planning, save_distance_data, action_transform,
         save_planning_images, visualize_planning):

    sim = SimScene.get_sim(model_handle=sim_path)
    target_sid = sim.model.site_name2id("target")
    ee_sid = sim.model.site_name2id(EE_SITE)

    # Use the keyframe starting position from XML
    ARM_JNT0 = np.array([
        -0.0321842,  # Joint 1
        -0.394346,   # Joint 2
        0.00932319,  # Joint 3
        -2.77917,    # Joint 4
        -0.011826,   # Joint 5
        0.713889,    # Joint 6
        1.53183      # Joint 7
    ])

    # Seed arm and get start EE
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt, dtype=float)
    sim.data.ctrl[:ARM_nJnt] = sim.data.qpos[:ARM_nJnt].copy()
    sim.forward()
    EE_START = sim.data.site_xpos[ee_sid].copy()
    print(f"End effector starting position: {EE_START}")  # Debug print

    os.makedirs(out_dir, exist_ok=True)
    episode = 0

    # Set up experiment-specific offset
    if experiment_type == 'x':
        experiment_offset = np.array([0.2, 0.0, 0.0])
        print(f"Experiment type: X-axis (offset: {experiment_offset})")
    elif experiment_type == 'y':
        experiment_offset = np.array([0.0, 0.2, 0.0])
        print(f"Experiment type: Y-axis (offset: {experiment_offset})")
    else:  # 'z'
        experiment_offset = np.array([0.0, 0.0, 0.2])
        print(f"Experiment type: Z-axis (offset: {experiment_offset})")

    # Create experiment-specific output directory
    experiment_out_dir = os.path.join(out_dir, f"reach_along_{experiment_type}")
    os.makedirs(experiment_out_dir, exist_ok=True)
    print(f"Output directory: {experiment_out_dir}")

    step_dt = sim.model.opt.timestep
    sim_fps = max(1, int(round(1.0 / step_dt)))
    steps_per_frame = max(1, int(round(sim_fps / max(1, fps))))
    actual_fps = int(round(sim_fps / steps_per_frame))

    # capture frame counter shared across phases
    frame_counter = [0]

    # Storage for all distance data across episodes
    all_episode_data = {
        'phase1_ik_distances': [],   # Distance trajectory for IK-based approach
        'phase3_vjepa_distances': [],  # Distance trajectory for V-JEPA planning
        'phase1_final_distance': [],  # Final distance achieved by IK
        'phase3_final_distance': [],  # Final distance achieved by V-JEPA
        'target_positions': [],       # Target positions for each episode
        'episode_ids': []
    }

    # Load V-JEPA models if planning is enabled
    if enable_vjepa_planning:
        print("Loading V-JEPA models for CEM planning...")
        encoder, predictor = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_ac_vit_giant"
        )
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
        print("V-JEPA models loaded successfully.")
    else:
        world_model = None
        transform = None

    def transform_action(action, transform_type='none'):
        """
        Transform actions from camera frame to robot frame.

        Args:
            action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            transform_type: Type of transformation to apply
        Returns:
            Transformed action in robot frame
        """
        transformed = action.copy()

        if transform_type == 'swap_xy':
            # Swap x and y axes
            transformed[0], transformed[1] = action[1], action[0]
            transformed[3], transformed[4] = action[4], action[3]  # Also swap roll/pitch
        elif transform_type == 'negate_x':
            # Negate x axis
            transformed[0] = -action[0]
        elif transform_type == 'negate_y':
            # Negate y axis
            transformed[1] = -action[1]
        elif transform_type == 'swap_xy_negate_x':
            # Swap xy and negate new x
            transformed[0], transformed[1] = -action[1], action[0]
            transformed[3], transformed[4] = action[4], action[3]
        elif transform_type == 'swap_xy_negate_y':
            # Swap xy and negate new y
            transformed[0], transformed[1] = action[1], -action[0]
            transformed[3], transformed[4] = action[4], action[3]
        elif transform_type == 'rotate_90_cw':
            # Rotate 90 degrees clockwise in xy plane
            transformed[0], transformed[1] = action[1], -action[0]
            transformed[3], transformed[4] = action[4], action[3]
        elif transform_type == 'rotate_90_ccw':
            # Rotate 90 degrees counter-clockwise in xy plane
            transformed[0], transformed[1] = -action[1], action[0]
            transformed[3], transformed[4] = action[4], action[3]

        return transformed

    while True:
        # Episode start: reset to home position
        sim.data.qpos[:ARM_nJnt] = ARM_JNT0
        sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt, dtype=float)
        sim.data.ctrl[:ARM_nJnt] = sim.data.qpos[:ARM_nJnt].copy()
        sim.forward()
        frame_counter[0] = 0
        frames = []
        current_joint_pos = ARM_JNT0.copy()
        targets_reached = 0

        print(f"\n=== Episode {episode}: {num_targets} consecutive targets (reactive planning) ===")

        # Store the final target position for Phase 3
        final_target_pos = None

        # Track distances during Phase 1 (IK approach)
        phase1_distances = []

        # Phase 1: Reactively visit N consecutive targets
        while targets_reached < num_targets:
            # Sample new target only when we need it
            if fixed_target and targets_reached == 0:
                print(
                    f"\nTarget {targets_reached + 1}/{num_targets}: "
                    f"Using fixed target offset ({experiment_type}-axis):",
                    experiment_offset
                )
                target_pos = EE_START + experiment_offset
            else:
                print(
                    f"\nTarget {targets_reached + 1}/{num_targets}: "
                    "Sampling new target (uniform sphere)"
                )
                target_pos = sample_point_in_sphere(EE_START, radius)

            # Save the last target for Phase 3
            if targets_reached == num_targets - 1:
                final_target_pos = target_pos.copy()

            delta_pos = target_pos - EE_START
            print(
                f" Target position (x,y,z) = "
                f"({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})"
            )
            print(
                f" Target delta (x,y,z) = "
                f"({delta_pos[0]:.3f},{delta_pos[1]:.3f},{delta_pos[2]:.3f})"
            )

            # Update visual target marker
            sim.model.site_pos[target_sid][:] = target_pos

            # Plan trajectory from current position to new target
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
            if not ik_result.success:
                print(f" WARNING: IK failed (err: {ik_result.err_norm:.4f})")

            approach_waypoints = generate_joint_space_min_jerk(
                start=current_joint_pos,
                goal=ik_result.qpos[:ARM_nJnt],
                time_to_go=horizon,
                dt=step_dt
            )

            print(
                f" Planned trajectory: {len(approach_waypoints)} steps "
                f"({len(approach_waypoints) * step_dt:.2f}s)"
            )

            # Execute trajectory to reach this target
            current_joint_pos = execute_waypoints_and_record(
                approach_waypoints,
                frames,
                horizon,
                step_dt,
                sim,
                render,
                steps_per_frame,
                frame_counter,
                width,
                height,
                camera_name,
                device_id
            )

            targets_reached += 1
            print(f" Reached target {targets_reached}/{num_targets}")

            # Measure distance after reaching target (for Phase 1)
            if targets_reached == num_targets and save_distance_data:
                sim.forward()
                final_ee_pos = sim.data.site_xpos[ee_sid].copy()
                phase1_final_dist = np.linalg.norm(final_ee_pos - final_target_pos)
                phase1_distances.append(phase1_final_dist)
                print(f" Phase 1 final distance to target: {phase1_final_dist:.4f}m")

            # Save goal snapshot only for the last target
            if targets_reached == num_targets:
                goal_rgb = None
                if render == 'offscreen':
                    goal_rgb = sim.renderer.render_offscreen(
                        width=width,
                        height=height,
                        camera_id=camera_name,
                        device_id=device_id
                    )
                try:
                    goal_path = os.path.join(
                        experiment_out_dir,
                        f"{out_name}{episode}_goal.png"
                    )
                    # Resize to 256x256 for individual image saves
                    goal_rgb_resized = np.array(
                        Image.fromarray(goal_rgb).resize((256, 256), Image.LANCZOS)
                    )
                    Image.fromarray(goal_rgb_resized).save(goal_path)
                    print(" Saved goal snapshot:", goal_path)
                except Exception as e:
                    print(" warning: could not save goal snapshot:", e)

                if (frame_counter[0] % steps_per_frame) == 0:
                    frames.append(goal_rgb)
                frame_counter[0] += 1

        # Phase 2: return to start (record)
        print(f"\nReturning to home position...")
        return_waypoints = generate_joint_space_min_jerk(
            start=current_joint_pos,
            goal=ARM_JNT0,
            time_to_go=horizon,
            dt=step_dt
        )
        print(
            f" Planned return: {len(return_waypoints)} steps "
            f"({len(return_waypoints) * step_dt:.2f}s)"
        )

        final_return_pos = execute_waypoints_and_record(
            return_waypoints,
            frames,
            horizon,
            step_dt,
            sim,
            render,
            steps_per_frame,
            frame_counter,
            width,
            height,
            camera_name,
            device_id
        )

        # At home: save start snapshot
        start_rgb = None
        if render == 'offscreen':
            start_rgb = sim.renderer.render_offscreen(
                width=width,
                height=height,
                camera_id=camera_name,
                device_id=device_id
            )
        try:
            start_path = os.path.join(
                experiment_out_dir,
                f"{out_name}{episode}_start.png"
            )
            # Resize to 256x256 for individual image saves
            start_rgb_resized = np.array(
                Image.fromarray(start_rgb).resize((256, 256), Image.LANCZOS)
            )
            Image.fromarray(start_rgb_resized).save(start_path)
            print("Saved start snapshot:", start_path)
        except Exception as e:
            print("warning: could not save start snapshot:", e)

        if (frame_counter[0] % steps_per_frame) == 0:
            frames.append(start_rgb)
        frame_counter[0] += 1

        # Phase 3: VJEPA-based CEM planning with distance tracking
        phase3_distances = []

        if (
            enable_vjepa_planning and
            render == 'offscreen' and
            (start_rgb is not None) and
            ('goal_rgb' in locals()) and
            (final_target_pos is not None)
        ):
            print(f"\n=== Phase 3: V-JEPA CEM Planning ({planning_steps} steps) ===")
            print(f"Using action transformation: {action_transform}")

            # Create directory for planning images
            if save_planning_images or visualize_planning:
                planning_img_dir = os.path.join(
                    experiment_out_dir,
                    f"planning_images_ep{episode}"
                )
                os.makedirs(planning_img_dir, exist_ok=True)
                print(f"Saving planning images to: {planning_img_dir}")

                # Save goal image for reference
                try:
                    goal_img_path = os.path.join(planning_img_dir, "goal.png")
                    # Resize to 256x256 for individual image saves
                    goal_rgb_resized = np.array(
                        Image.fromarray(goal_rgb).resize((256, 256), Image.LANCZOS)
                    )
                    Image.fromarray(goal_rgb_resized).save(goal_img_path)
                    print(f"Saved goal image: {goal_img_path}")
                except Exception as e:
                    print(f"Warning: Could not save goal image: {e}")

            with torch.no_grad():
                # Iterative planning loop
                current_joint_pos = final_return_pos.copy()
                global_transformed_goal = None

                for step_idx in range(planning_steps):
                    # Get current EE pose
                    sim.forward()
                    current_ee_pos = sim.data.site_xpos[ee_sid].copy()

                    # Calculate distance to target
                    distance = np.linalg.norm(current_ee_pos - final_target_pos)
                    phase3_distances.append(distance)
                    print(
                        f"\nStep {step_idx + 1}/{planning_steps}: "
                        f"Distance to target = {distance:.4f}m"
                    )

                    # Capture current observation RGB image
                    current_rgb = sim.renderer.render_offscreen(
                        width=width,
                        height=height,
                        camera_id=camera_name,
                        device_id=device_id
                    )
                    if current_rgb is None:
                        print(" WARNING: Failed to capture current RGB, skipping step")
                        continue

                    # Stack current observation with goal image [2, H, W, 3]
                    combined_rgb = np.stack([current_rgb, goal_rgb], axis=0)
                    clips = transform(combined_rgb).unsqueeze(0).to(device)  # [1, C, T, H, W]

                    # Save images if requested (save AFTER transform to see what model sees)
                    if save_planning_images or visualize_planning:
                        try:
                            # Save raw current observation - resize to 256x256
                            raw_current_path = os.path.join(
                                planning_img_dir,
                                f"step{step_idx:02d}_raw_current.png"
                            )
                            current_rgb_resized = np.array(
                                Image.fromarray(current_rgb).resize((256, 256), Image.LANCZOS)
                            )
                            Image.fromarray(current_rgb_resized).save(raw_current_path)

                            # Convert transformed tensor back to image for visualization
                            # clips is [1, C, T, H, W], we want the current frame (t=0)
                            transformed_current = clips[0, :, 0, :, :].cpu()  # [C, H, W]

                            transformed_current = transformed_current.permute(1, 2, 0)  # [H, W, C]

                            # Assuming ImageNet normalization was used
                            mean = torch.tensor([0.485, 0.456, 0.406])
                            std = torch.tensor([0.229, 0.224, 0.225])
                            transformed_current = transformed_current * std + mean
                            transformed_current = torch.clamp(
                                transformed_current * 255, 0, 255
                            ).byte().numpy()

                            transformed_current_path = os.path.join(
                                planning_img_dir,
                                f"step{step_idx:02d}_transformed_current.png"
                            )
                            Image.fromarray(transformed_current).save(transformed_current_path)

                            # Similarly for goal image (t=1)
                            if step_idx == 0:
                                # Save transformed goal once
                                transformed_goal = clips[0, :, 1, :, :].cpu().permute(1, 2, 0)
                                transformed_goal = transformed_goal * std + mean
                                transformed_goal = torch.clamp(
                                    transformed_goal * 255, 0, 255
                                ).byte().numpy()
                                transformed_goal_path = os.path.join(
                                    planning_img_dir,
                                    "transformed_goal.png"
                                )
                                Image.fromarray(transformed_goal).save(transformed_goal_path)
                                global_transformed_goal = transformed_goal
                        except Exception as e:
                            print(f" Warning: Could not save transformed images: {e}")

                    # Create side-by-side visualization if requested
                    if visualize_planning:
                        try:
                            # Create visualization with both raw and transformed images
                            # Layout: [raw_current | transformed_current | transformed_goal]
                            h_raw, w_raw = current_rgb.shape[:2]
                            h_trans, w_trans = transformed_current.shape[:2]

                            # Create canvas (use max height, sum widths)
                            max_h = max(h_raw, h_trans)
                            combined = np.zeros(
                                (max_h, w_raw + w_trans * 2 + 20, 3),
                                dtype=np.uint8
                            )

                            # Place raw current
                            combined[:h_raw, :w_raw, :] = current_rgb

                            # Place transformed current (with separator)
                            offset = w_raw + 10
                            combined[:h_trans, offset:offset + w_trans, :] = transformed_current

                            # Place transformed goal (with separator)
                            offset = w_raw + w_trans + 20
                            if global_transformed_goal is not None:
                                combined[:h_trans, offset:offset + w_trans, :] = global_transformed_goal

                            # Add text annotations
                            from PIL import ImageDraw, ImageFont
                            combined_pil = Image.fromarray(combined)
                            draw = ImageDraw.Draw(combined_pil)
                            try:
                                font = ImageFont.truetype(
                                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                    16
                                )
                            except Exception:
                                font = ImageFont.load_default()

                            # Add labels
                            draw.text(
                                (10, 10),
                                f"Raw Current\n(Step {step_idx})",
                                fill=(255, 255, 0),
                                font=font
                            )
                            draw.text(
                                (w_raw + 20, 10),
                                f"Transformed\nCurrent\n({w_trans}x{h_trans})",
                                fill=(0, 255, 255),
                                font=font
                            )
                            draw.text(
                                (w_raw + w_trans + 30, 10),
                                f"Transformed\nGoal\n({w_trans}x{h_trans})",
                                fill=(0, 255, 0),
                                font=font
                            )
                            draw.text(
                                (10, max_h - 25),
                                f"Distance: {distance:.4f}m",
                                fill=(255, 255, 255),
                                font=font
                            )

                            viz_path = os.path.join(
                                planning_img_dir,
                                f"step{step_idx:02d}_full_visualization.png"
                            )
                            combined_pil.save(viz_path)
                        except Exception as e:
                            print(f" Warning: Could not create full visualization: {e}")

                    # Forward pass to get representations
                    def forward_target_local(c, normalize_reps=True):
                        B, C, T, H, W = c.size()
                        c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
                        h = encoder(c)
                        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
                        if normalize_reps:
                            h = F.layer_norm(h, (h.size(-1),))
                        return h

                    h = forward_target_local(clips)
                    z_n = h[:, :tokens_per_frame].contiguous().clone()
                    z_goal = h[:, -tokens_per_frame:].contiguous().clone()
                    del h

                    # Build state tensor
                    pos = current_ee_pos
                    rpy = np.zeros(3, dtype=float)
                    gripper_width = 0.0
                    current_state = np.concatenate([pos, rpy, [gripper_width]], axis=0)
                    states = torch.tensor(
                        current_state,
                        device=device
                    ).unsqueeze(0).unsqueeze(0)  # [1,1,7]
                    s_n = states[:, :1].to(dtype=z_n.dtype)

                    # Plan next action using CEM
                    start_time = time.time()
                    actions = world_model.infer_next_action(z_n, s_n, z_goal).cpu().numpy()
                    end_time = time.time()
                    print(f" Planning time: {end_time - start_time:.3f}s")
                    print(
                        f" Raw planned action (x,y,z): "
                        f"({actions[0, 0]:.3f}, {actions[0, 1]:.3f}, {actions[0, 2]:.3f})"
                    )

                    # Transform action from camera frame to robot frame
                    transformed_action = transform_action(actions[0], action_transform)
                    print(
                        f" Transformed action (x,y,z): "
                        f"({transformed_action[0]:.3f}, "
                        f"{transformed_action[1]:.3f}, "
                        f"{transformed_action[2]:.3f})"
                    )

                    # Convert transformed action to joint-space waypoints
                    planned_delta = transformed_action[:7]  # dx,dy,dz,dp,dy,dr,grip
                    try:
                        new_pos, new_rpy = compute_new_pose(
                            pos,
                            planned_delta[:3],
                            planned_delta[3:6]
                        )
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
                    except Exception as e:
                        print(f" WARNING: IK/planning failed: {e}")
                        # Fallback: hold current position
                        planned_waypoints = [
                            {'position': current_joint_pos.copy()}
                            for _ in range(int(round(horizon / step_dt)))
                        ]

                    # Execute planned waypoints
                    current_joint_pos = execute_waypoints_and_record(
                        planned_waypoints,
                        frames,
                        horizon,
                        step_dt,
                        sim,
                        render,
                        steps_per_frame,
                        frame_counter,
                        width,
                        height,
                        camera_name,
                        device_id
                    )

                # Final distance measurement
                sim.forward()
                final_ee_pos = sim.data.site_xpos[ee_sid].copy()
                final_distance = np.linalg.norm(final_ee_pos - final_target_pos)
                phase3_distances.append(final_distance)
                print(f"\nFinal distance to target: {final_distance:.4f}m")

                # Save final state image if requested
                if save_planning_images or visualize_planning:
                    try:
                        final_rgb = sim.renderer.render_offscreen(
                            width=width,
                            height=height,
                            camera_id=camera_name,
                            device_id=device_id
                        )
                        if final_rgb is not None:
                            final_img_path = os.path.join(
                                planning_img_dir,
                                f"step{planning_steps:02d}_final.png"
                            )
                            # Resize to 256x256 for individual image saves
                            final_rgb_resized = np.array(
                                Image.fromarray(final_rgb).resize((256, 256), Image.LANCZOS)
                            )
                            Image.fromarray(final_rgb_resized).save(final_img_path)
                            print(f"Saved final image: {final_img_path}")
                    except Exception as e:
                        print(f"Warning: Could not save final image: {e}")

        # Store data for this episode
        if save_distance_data:
            all_episode_data['phase1_ik_distances'].append(
                phase1_distances if phase1_distances else [np.nan]
            )
            all_episode_data['phase3_vjepa_distances'].append(
                phase3_distances if phase3_distances else [np.nan]
            )
            all_episode_data['phase1_final_distance'].append(
                phase1_distances[-1] if phase1_distances else np.nan
            )
            all_episode_data['phase3_final_distance'].append(
                phase3_distances[-1] if phase3_distances else np.nan
            )
            all_episode_data['target_positions'].append(
                final_target_pos.tolist() if final_target_pos is not None else [np.nan, np.nan, np.nan]
            )
            all_episode_data['episode_ids'].append(episode)

        # Save a single MP4 containing all captured frames
        if render == 'offscreen':
            if skvideo is None:
                print("skvideo not available: saving individual PNGs instead.")
                for i, f in enumerate(frames):
                    try:
                        Image.fromarray(f).save(
                            os.path.join(
                                experiment_out_dir,
                                f"{out_name}{experiment_type}_{episode}_{i:04d}.png"
                            )
                        )
                    except Exception:
                        pass
            else:
                # No resizing needed - frames are already correct size
                save_frames = np.asarray(frames, dtype=np.uint8)
                file_name = os.path.join(
                    experiment_out_dir,
                    out_name + experiment_type + "_" + str(episode) + ".mp4"
                )
                outputdict = {"-pix_fmt": "yuv420p", "-r": str(actual_fps)}
                skvideo.io.vwrite(file_name, save_frames, outputdict=outputdict)
                print(
                    f"Saved video: {file_name} "
                    f"(size: {save_frames.shape[1]}x{save_frames.shape[2]})"
                )

        episode += 1
        if episode >= max_episodes:
            print("Reached max_episodes, exiting.")
            break

        sim.reset()

    # Save all distance data after all episodes
    if save_distance_data:
        import json
        # Save as JSON for easy loading
        summary_path = os.path.join(
            experiment_out_dir,
            f"{out_name}{experiment_type}_distance_summary.json"
        )
        with open(summary_path, 'w') as f:
            json.dump(all_episode_data, f, indent=2)
        print(f"\nSaved distance summary: {summary_path}")

        # Also save as numpy for easy plotting (use consistent singular keys)
        np_summary_path = os.path.join(
            experiment_out_dir,
            f"{out_name}{experiment_type}_distance_summary.npz"
        )
        np.savez(
            np_summary_path,
            phase1_final_distance=np.array(all_episode_data['phase1_final_distance']),
            phase3_distances_per_episode=np.array(
                [np.array(d) for d in all_episode_data['phase3_vjepa_distances']],
                dtype=object
            ),
            phase3_final_distance=np.array(all_episode_data['phase3_final_distance']),
            target_positions=np.array(all_episode_data['target_positions']),
            episode_ids=np.array(all_episode_data['episode_ids'])
        )
        print(f"Saved numpy summary: {np_summary_path}")

        # Print summary statistics
        print("\n=== Summary Statistics ===")
        phase1_finals = [
            d for d in all_episode_data['phase1_final_distance']
            if not np.isnan(d)
        ]
        phase3_finals = [
            d for d in all_episode_data['phase3_final_distance']
            if not np.isnan(d)
        ]

        if phase1_finals:
            print("Phase 1 (IK baseline):")
            print(
                f" Mean final distance: {np.mean(phase1_finals):.4f}m "
                f"± {np.std(phase1_finals):.4f}m"
            )
            print(
                f" Min: {np.min(phase1_finals):.4f}m, "
                f"Max: {np.max(phase1_finals):.4f}m"
            )

        if phase3_finals:
            print("Phase 3 (V-JEPA planning):")
            print(
                f" Mean final distance: {np.mean(phase3_finals):.4f}m "
                f"± {np.std(phase3_finals):.4f}m"
            )
            print(
                f" Min: {np.min(phase3_finals):.4f}m, "
                f"Max: {np.max(phase3_finals):.4f}m"
            )


if __name__ == '__main__':
    main()
