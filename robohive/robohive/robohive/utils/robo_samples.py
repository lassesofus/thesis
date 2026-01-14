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

# Latent alignment module for sim-to-DROID alignment
_latent_align_path = "/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/inference_time_latent_alignment"
if os.path.isdir(_latent_align_path) and _latent_align_path not in sys.path:
    sys.path.insert(0, _latent_align_path)
try:
    from latent_alignment import LatentAligner
except ImportError:
    LatentAligner = None
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import generate_joint_space_min_jerk
from robohive.utils.quat_math import euler2quat
from robohive.utils.xml_utils import reassign_parent

import click
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
import pdb

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

# Apply VJEPA optimizations if enabled via environment variable
import os
if os.environ.get('VJEPA_OPTIMIZE', '0') == '1' or os.environ.get('OPT_CUDNN', '0') == '1':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("VJEPA optimizations enabled: cudnn.benchmark + TF32")


def robohive_to_droid_pos(pos_robohive):
    """
    Convert RoboHive position to DROID frame for display.

    Transformation: DROID_x = RoboHive_y, DROID_y = -RoboHive_x, DROID_z = RoboHive_z

    Args:
        pos_robohive: Position [x, y, z] in RoboHive frame

    Returns:
        Position [x, y, z] in DROID frame
    """
    return np.array([pos_robohive[1], -pos_robohive[0], pos_robohive[2]])


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


def get_ee_orientation(sim, ee_sid):
    """
    Get end-effector orientation as euler angles (roll, pitch, yaw).

    Args:
        sim: MuJoCo simulation object
        ee_sid: Site ID for end-effector

    Returns:
        rpy: numpy array [roll, pitch, yaw] in radians
    """
    # Get rotation matrix (3x3, stored as flat 9-element array in MuJoCo)
    xmat = sim.data.site_xmat[ee_sid].reshape(3, 3)

    # Convert to euler angles (xyz convention to match DROID/PyBullet)
    rpy = Rotation.from_matrix(xmat).as_euler('xyz', degrees=False)

    return rpy


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
                                 width, height, camera_name, device_id,
                                 track_distances=False, target_pos=None,
                                 ee_sid=None, distance_samples=20):
    t = 0.0
    idx = 0
    num_waypoints = len(waypoints)
    distances = []  # optionally filled if track_distances is True

    # precompute stride for distance sampling
    stride = 1
    if track_distances and distance_samples > 0 and num_waypoints > 0:
        stride = max(1, num_waypoints // distance_samples)

    wp_idx = 0
    while t <= horizon:
        # clamp index to last waypoint
        idx = min(idx, num_waypoints - 1)

        # directly set actuator target like the tutorial does
        sim.data.ctrl[:ARM_nJnt] = waypoints[idx]['position']

        # optionally measure distance for a subset of waypoints
        if track_distances and target_pos is not None and ee_sid is not None:
            if (wp_idx % stride) == 0:
                sim.forward()  # ensure site_xpos is updated before measuring
                ee_pos = sim.data.site_xpos[ee_sid].copy()
                dist = np.linalg.norm(ee_pos - target_pos)
                distances.append(dist)

        # capture frame before advancing
        capture_rgb_and_append(
            frames_list, sim, render, steps_per_frame, frame_counter,
            width, height, camera_name, device_id
        )

        # advance simulation
        sim.advance(render=(render == 'onscreen'))
        t += step_dt
        idx += 1
        wp_idx += 1

    # hold final waypoint briefly with the planned position (not actual qpos)
    final_ctrl = waypoints[-1]['position'].copy()
    hold_steps = max(5, int(0.2 / sim.model.opt.timestep))
    for _ in range(hold_steps):
        sim.data.ctrl[:ARM_nJnt] = final_ctrl
        if track_distances and target_pos is not None and ee_sid is not None:
            sim.forward()
            ee_pos = sim.data.site_xpos[ee_sid].copy()
            dist = np.linalg.norm(ee_pos - target_pos)
            distances.append(dist)
        capture_rgb_and_append(
            frames_list, sim, render, steps_per_frame, frame_counter,
            width, height, camera_name, device_id
        )
        sim.advance(render=(render == 'onscreen'))

    # Return the final planned position for smooth chaining
    if track_distances:
        return final_ctrl, distances
    return final_ctrl


def forward_target(c, normalize_reps=True):
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def compute_energy_landscape(
    encoder, predictor, tokens_per_frame,
    current_rep, current_state, goal_rep,
    experiment_type='z', nsamples=9, grid_size=0.075,
    normalize_reps=True, device='cuda'
):
    """
    Compute energy landscape for visualization.

    Args:
        encoder: V-JEPA encoder model
        predictor: V-JEPA predictor model
        tokens_per_frame: Number of tokens per frame
        current_rep: Current frame representation [1, tokens, D]
        current_state: Current robot state [1, 1, 7]
        goal_rep: Goal frame representation [1, tokens, D]
        experiment_type: 'x', 'y', or 'z' - determines which axis to fix
        nsamples: Grid resolution per axis
        grid_size: Range of action sampling in meters
        normalize_reps: Whether to layer-normalize representations
        device: Torch device

    Returns:
        heatmap: 2D numpy array of energy values
        axis1_edges: Bin edges for axis 1
        axis2_edges: Bin edges for axis 2
        axis1_label: Label for axis 1 (e.g., "Delta Y")
        axis2_label: Label for axis 2 (e.g., "Delta Z")
    """
    # Determine which axes to vary based on experiment_type
    # We include the planning axis in the visualization
    if experiment_type == 'x':
        # Planning along X: show X-Y plane (fix Z)
        vary_axes = (0, 1)  # (X, Y)
        axis1_label, axis2_label = "Delta X (m)", "Delta Y (m)"
    elif experiment_type == 'y':
        # Planning along Y: show X-Y plane (fix Z)
        vary_axes = (0, 1)  # (X, Y)
        axis1_label, axis2_label = "Delta X (m)", "Delta Y (m)"
    else:  # 'z'
        # Planning along Z: show X-Z plane (fix Y)
        vary_axes = (0, 2)  # (X, Z)
        axis1_label, axis2_label = "Delta X (m)", "Delta Z (m)"

    # Create 2D action grid (fix third axis at 0)
    action_samples = []
    for d1 in np.linspace(-grid_size, grid_size, nsamples):
        for d2 in np.linspace(-grid_size, grid_size, nsamples):
            action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [dx,dy,dz,droll,dpitch,dyaw,grip]
            action[vary_axes[0]] = d1
            action[vary_axes[1]] = d2
            action_samples.append(torch.tensor(action, device=device, dtype=current_rep.dtype))

    action_samples = torch.stack(action_samples, dim=0).unsqueeze(1)  # [N^2, 1, 7]
    num_samples = nsamples ** 2

    # Expand representations for batch processing
    z_batch = current_rep[:, :tokens_per_frame].repeat(num_samples, 1, 1)  # [N^2, tokens, D]
    s_batch = current_state.repeat(num_samples, 1, 1)  # [N^2, 1, 7]

    # Predict next representations
    with torch.no_grad():
        z_pred = predictor(z_batch, action_samples, s_batch)[:, -tokens_per_frame:]
        if normalize_reps:
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))

    # Compute L1 energy (distance to goal)
    goal_expanded = goal_rep.repeat(num_samples, 1, 1)  # [N^2, tokens, D]
    energy = torch.mean(torch.abs(z_pred - goal_expanded), dim=[1, 2])  # [N^2]
    energy = energy.cpu().numpy()

    # Reshape energy into 2D grid directly (samples are on a regular grid)
    # The nested loop iterates d1 (axis1) in outer loop, d2 (axis2) in inner loop
    # So energy is ordered as: [d1_0,d2_0], [d1_0,d2_1], ..., [d1_1,d2_0], ...
    heatmap = energy.reshape(nsamples, nsamples)  # [nsamples_axis1, nsamples_axis2]

    # Create axis edges for plotting (cell centers become edges)
    axis_values = np.linspace(-grid_size, grid_size, nsamples)
    half_step = (axis_values[1] - axis_values[0]) / 2 if nsamples > 1 else grid_size
    axis1_edges = np.concatenate([[axis_values[0] - half_step],
                                   axis_values[:-1] + half_step,
                                   [axis_values[-1] + half_step]])
    axis2_edges = axis1_edges.copy()

    return heatmap, axis1_edges, axis2_edges, axis1_label, axis2_label


def plot_energy_landscape(
    heatmap, axis1_edges, axis2_edges,
    axis1_label, axis2_label,
    save_path, step_idx, distance_to_goal,
    optimal_action=None
):
    """
    Create and save energy landscape visualization.

    Args:
        optimal_action: Optional tuple (axis1_val, axis2_val) for the optimal action
                        direction to goal, shown as a red marker.
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        heatmap.T,
        origin='lower',
        extent=[axis1_edges[0], axis1_edges[-1], axis2_edges[0], axis2_edges[-1]],
        cmap='viridis',
        aspect='auto'
    )

    # Plot optimal action marker if provided
    if optimal_action is not None:
        ax.plot(optimal_action[0], optimal_action[1], 'r*', markersize=15,
                markeredgecolor='white', markeredgewidth=1.5, label='Optimal direction')
        ax.legend(loc='upper right', fontsize=8)

    ax.set_xlabel(axis1_label)
    ax.set_ylabel(axis2_label)
    ax.set_title(f'Energy Landscape (Step {step_idx})\nDist to goal: {distance_to_goal:.4f}m')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('L1 Energy')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_energy_landscape_3d(
    heatmap, axis1_edges, axis2_edges,
    axis1_label, axis2_label,
    save_path, step_idx, distance_to_goal
):
    """
    Create and save 3D surface plot of energy landscape (like V-JEPA 2 paper Figure 9).
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    # Create meshgrid for surface plot
    # Use bin centers instead of edges
    axis1_centers = (axis1_edges[:-1] + axis1_edges[1:]) / 2
    axis2_centers = (axis2_edges[:-1] + axis2_edges[1:]) / 2
    X, Y = np.meshgrid(axis1_centers, axis2_centers)

    # Energy values (transpose to match meshgrid orientation)
    Z = heatmap.T

    # Create figure with 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot with colormap
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.coolwarm,
        linewidth=0.5,
        antialiased=True,
        alpha=0.9,
        edgecolor='gray'
    )

    # Labels and title
    ax.set_xlabel(axis1_label, fontsize=10, labelpad=10)
    ax.set_ylabel(axis2_label, fontsize=10, labelpad=10)
    ax.set_zlabel('Energy', fontsize=10, labelpad=10)
    ax.set_title(f'Energy Landscape (Step {step_idx})\nDist to goal: {distance_to_goal:.4f}m', fontsize=12)

    # Adjust view angle for better visualization
    ax.view_init(elev=25, azim=-60)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('L1 Energy', fontsize=9)

    # Scale axes to be more readable (convert to cm)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}'))

    # Update axis labels to show cm
    ax.set_xlabel(axis1_label.replace('(m)', '(cm)'), fontsize=10, labelpad=10)
    ax.set_ylabel(axis2_label.replace('(m)', '(cm)'), fontsize=10, labelpad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def set_seed(seed):
    """Set random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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

@click.command(help=DESC)
@click.option(
    '-s', '--sim_path',
    type=str,
    required=True,
    default='/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_v0.xml',
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
@click.option('--vjepa_checkpoint', type=str, default=None,
              help='Path to trained V-JEPA checkpoint (.pt file). If not specified, loads Meta baseline from Hub.')
@click.option('--visualize_energy_landscape', is_flag=True, default=False,
              help='Generate energy landscape heatmaps during V-JEPA planning steps')
@click.option('--energy_landscape_3d', is_flag=True, default=False,
              help='Use 3D surface plots instead of 2D heatmaps for energy landscape')
@click.option(
    '--gripper',
    type=click.Choice(['franka', 'robotiq']),
    default='franka',
    help='End-effector gripper type: franka (default parallel-jaw) or robotiq (2F-85)'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Base random seed for reproducibility. Each episode uses seed + episode_id.'
)
@click.option(
    '--latent_alignment',
    type=click.Choice(['none', 'mean', 'coral']),
    default='none',
    help='Latent space alignment method: none (baseline), mean (mean-only), coral (whitening-coloring)'
)
@click.option(
    '--latent_alignment_stats_path',
    type=str,
    default='/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/inference_time_latent_alignment/stats',
    help='Path to directory containing alignment statistics (mu_sim.npy, mu_droid.npy, coral_matrix.npy)'
)

def main(sim_path, horizon, radius, render, camera_name, out_dir, out_name,
         width, height, device_id, fps, max_episodes, fixed_target,
         fixed_target_offset, experiment_type, num_targets, planning_steps,
         enable_vjepa_planning, save_distance_data, action_transform,
         save_planning_images, visualize_planning, vjepa_checkpoint,
         visualize_energy_landscape, energy_landscape_3d, gripper, seed,
         latent_alignment, latent_alignment_stats_path):

    # Gripper-aware model path selection
    FRANKA_MODEL = '/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_v0.xml'
    ROBOTIQ_MODEL = '/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_robotiq_v0.xml'

    # Override sim_path if using default Franka model and robotiq gripper is specified
    if sim_path == FRANKA_MODEL and gripper == 'robotiq':
        sim_path = ROBOTIQ_MODEL
        print(f"Using RobotiQ gripper model: {sim_path}")

    # Load the simulation
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

    target_sid = sim.model.site_name2id("target")
    ee_sid = sim.model.site_name2id(EE_SITE)

    # Use the keyframe starting position from XML
    # Joint 7 rotated -45 degrees to match real robot gripper orientation
    ARM_JNT0 = np.array([
        -0.0321842,  # Joint 1
        -0.394346,   # Joint 2
        0.00932319,  # Joint 3
        -2.77917,    # Joint 4
        -0.011826,   # Joint 5
        0.713889,    # Joint 6
        0.74663      # Joint 7 (original 1.53183, -π/4 ≈ 0.785 for angled gripper)
    ])

    # Seed arm and get start EE
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt, dtype=float)
    sim.data.ctrl[:ARM_nJnt] = sim.data.qpos[:ARM_nJnt].copy()
    sim.forward()
    EE_START = sim.data.site_xpos[ee_sid].copy()
    EE_START_DROID = robohive_to_droid_pos(EE_START)
    print(f"End effector starting position (RoboHive): {EE_START}")
    print(f"End effector starting position (DROID): {EE_START_DROID}")

    os.makedirs(out_dir, exist_ok=True)
    episode = 0

    # Set up experiment-specific offset
    # NOTE: Offsets are specified to match DROID coordinate frame conventions:
    #   DROID_x = RoboHive_y, DROID_y = RoboHive_x, DROID_z = RoboHive_z
    # So to move along DROID x-axis, we move along RoboHive y-axis, etc.
    if experiment_type == 'x':
        # DROID X-axis = RoboHive Y-axis
        experiment_offset = np.array([0.0, 0.2, 0.0])
        print(f"Experiment type: X-axis (DROID frame)")
        print(f"  RoboHive offset: {experiment_offset}")
        print(f"  DROID offset: {robohive_to_droid_pos(experiment_offset)}")
    elif experiment_type == 'y':
        # DROID Y-axis = RoboHive X-axis (negative RoboHive X for positive DROID Y)
        experiment_offset = np.array([-0.2, 0.0, 0.0])
        print(f"Experiment type: Y-axis (DROID frame)")
        print(f"  RoboHive offset: {experiment_offset}")
        print(f"  DROID offset: {robohive_to_droid_pos(experiment_offset)}")
    else:  # 'z'
        # DROID Z-axis = RoboHive Z-axis (no change)
        experiment_offset = np.array([0.0, 0.0, 0.2])
        print(f"Experiment type: Z-axis (DROID frame)")
        print(f"  RoboHive offset: {experiment_offset}")
        print(f"  DROID offset: {robohive_to_droid_pos(experiment_offset)}")

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
        'phase1_ik_distances': [],
        'phase1_final_distance': [],
        'phase3_vjepa_distances': [],
        'phase3_repr_l1_distances': [],
        'phase3_final_distance': [],
        'phase3_actions_raw': [],
        'phase3_actions_transformed': [],
        'target_positions': [],
        'initial_ee_positions': [],
        'ik_success': [],
        'episode_ids': [],
        'base_seed': seed,
        'episode_seeds': []
    }

    # Load V-JEPA models if planning is enabled
    if enable_vjepa_planning:
        if vjepa_checkpoint:
            print(f"Loading V-JEPA models from checkpoint: {vjepa_checkpoint}")
            # First load architecture from Hub
            encoder, predictor = torch.hub.load(
                "facebookresearch/vjepa2", "vjepa2_ac_vit_giant"
            )
            # Load checkpoint to CPU first to save GPU memory
            checkpoint = torch.load(vjepa_checkpoint, map_location='cpu')

            # Load encoder weights (try both possible keys)
            if 'encoder' in checkpoint:
                encoder.load_state_dict(checkpoint['encoder'])
                print("  Loaded encoder weights")
            elif 'target_encoder' in checkpoint:
                encoder.load_state_dict(checkpoint['target_encoder'])
                print("  Loaded target_encoder weights")
            else:
                raise ValueError(f"Cannot find encoder weights in checkpoint. Available keys: {list(checkpoint.keys())}")

            # Load predictor weights
            if 'predictor' in checkpoint:
                predictor.load_state_dict(checkpoint['predictor'])
                print("  Loaded predictor weights")
            else:
                raise ValueError(f"Cannot find predictor weights in checkpoint. Available keys: {list(checkpoint.keys())}")

            # Free checkpoint memory
            del checkpoint
            torch.cuda.empty_cache()
            print("  Cleared checkpoint from memory")
        else:
            print("Loading V-JEPA models from PyTorch Hub (Meta baseline)...")
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

        # Initialize latent aligner if requested
        latent_aligner = None
        if latent_alignment != 'none':
            if LatentAligner is None:
                print(f"WARNING: latent_alignment='{latent_alignment}' requested but LatentAligner not available.")
            else:
                try:
                    latent_aligner = LatentAligner.from_path(
                        latent_alignment_stats_path,
                        method=latent_alignment,
                        device=str(device)
                    )
                    print(f"Latent aligner initialized: {latent_aligner}")
                except Exception as e:
                    print(f"WARNING: Failed to initialize latent aligner: {e}")
                    latent_aligner = None
        else:
            print("Latent alignment: none (baseline)")
    else:
        world_model = None
        transform = None
        latent_aligner = None

    def transform_action(action, transform_type='none'):
        """
        Transform actions from DROID frame to RoboHive frame.

        DROID → RoboHive transformation is needed because V-JEPA was trained on DROID data
        and outputs actions in DROID coordinate frame, but RoboHive uses different axes:
            DROID_x = RoboHive_y
            DROID_y = -RoboHive_x
            DROID_z = RoboHive_z

            RoboHive_x = -DROID_y
            Robohive_y = DROID_x
            RoboHive_z = DROID_z

        Args:
            action: [dx, dy, dz, droll, dpitch, dyaw, gripper] in DROID frame
            transform_type: Type of transformation to apply
        Returns:
            Transformed action in RoboHive frame
        """
        transformed = action.copy()

        if transform_type == 'swap_xy': 
            # Swap x and y axes: DROID → RoboHive
            # RoboHive_x = DROID_y, RoboHive_y = DROID_x
            transformed[0], transformed[1] = action[1], action[0]
            transformed[3], transformed[4] = action[4], action[3]  # Also swap roll/pitch
        elif transform_type == 'negate_x':
            # Negate x axis
            transformed[0] = -action[0]
        elif transform_type == 'negate_y':
            # Negate y axis
            transformed[1] = -action[1]
        elif transform_type == 'swap_xy_negate_x': # THIS IS THE RIGHT TRANSFORMATION FOR DROID <-> ROBOHIVE
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
        # Set episode-specific seed for reproducibility with variation between episodes
        episode_seed = seed + episode
        set_seed(episode_seed)
        print(f"\n[Seed: {episode_seed} (base={seed}, episode={episode})]")

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
        phase1_distances_all_targets = []  # Track across all targets (NEW)

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
            target_pos_droid = robohive_to_droid_pos(target_pos)
            delta_pos_droid = robohive_to_droid_pos(delta_pos)
            print(
                f" Target position (RoboHive x,y,z) = "
                f"({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})"
            )
            print(
                f" Target position (DROID x,y,z) = "
                f"({target_pos_droid[0]:.3f},{target_pos_droid[1]:.3f},{target_pos_droid[2]:.3f})"
            )
            print(
                f" Target delta (RoboHive x,y,z) = "
                f"({delta_pos[0]:.3f},{delta_pos[1]:.3f},{delta_pos[2]:.3f})"
            )
            print(
                f" Target delta (DROID x,y,z) = "
                f"({delta_pos_droid[0]:.3f},{delta_pos_droid[1]:.3f},{delta_pos_droid[2]:.3f})"
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

            # Execute trajectory to reach this target with distance tracking (NEW)
            target_distances_this_approach = []  # Track distance per step
            current_joint_pos, target_distances_this_approach = execute_waypoints_and_record(
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
                device_id,
                track_distances=True,
                target_pos=target_pos,
                ee_sid=ee_sid,
                distance_samples=20,
            )

            phase1_distances_all_targets.append(target_distances_this_approach)

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

        # for Phase 2 we don't need distances, so use default signature
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
        phase3_repr_l1_distances = []
        phase3_actions_raw = []  # Store raw actions (NEW)
        phase3_actions_transformed = []  # Store transformed actions (NEW)

        if (
            enable_vjepa_planning and
            render == 'offscreen' and
            (start_rgb is not None) and
            ('goal_rgb' in locals()) and
            (final_target_pos is not None)
        ):
            print(f"\n=== Phase 3: V-JEPA CEM Planning ({planning_steps} steps) ===")
            print(f"Using action transformation: {action_transform}")

            # Create directory for planning images (only for first episode)
            save_images_this_episode = (save_planning_images or visualize_planning) and episode == 0
            if save_images_this_episode:
                planning_img_dir = os.path.join(
                    experiment_out_dir,
                    f"planning_images_ep{episode}"
                )
                os.makedirs(planning_img_dir, exist_ok=True)
                print(f"Saving planning images to: {planning_img_dir}")

            # Save goal image for reference (only for first episode)
            if save_images_this_episode:
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
                    current_ee_pos_droid = robohive_to_droid_pos(current_ee_pos)

                    # Calculate distance to target
                    distance = np.linalg.norm(current_ee_pos - final_target_pos)
                    phase3_distances.append(distance)
                    print(
                        f"\nStep {step_idx + 1}/{planning_steps}: "
                        f"Distance to target = {distance:.4f}m"
                    )
                    print(
                        f" Current EE (RoboHive): ({current_ee_pos[0]:.3f}, "
                        f"{current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f})"
                    )
                    print(
                        f" Current EE (DROID): ({current_ee_pos_droid[0]:.3f}, "
                        f"{current_ee_pos_droid[1]:.3f}, {current_ee_pos_droid[2]:.3f})"
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
                    if save_images_this_episode:
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
                    if save_images_this_episode and visualize_planning:
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

                    # Apply latent alignment if enabled (MODE 1: align both z_n and z_goal)
                    # This ensures predictor and cost operate in the same aligned coordinate system
                    if latent_aligner is not None:
                        z_n = latent_aligner(z_n)
                        z_goal = latent_aligner(z_goal)

                    # Calculate L1 distance between current and goal representations
                    # (in aligned space if alignment is enabled)
                    repr_l1_distance = torch.mean(torch.abs(z_n - z_goal)).item()
                    phase3_repr_l1_distances.append(repr_l1_distance)
                    print(f" Representation L1 distance: {repr_l1_distance:.6f}" +
                          (f" (aligned: {latent_alignment})" if latent_alignment != 'none' else ""))

                    # Build state tensor in DROID frame (model was trained on DROID-frame data)
                    robohive_rpy = get_ee_orientation(sim, ee_sid)
                    robohive_pose = np.concatenate([current_ee_pos, robohive_rpy])
                    droid_pose = transform_pose_to_droid_frame(robohive_pose)
                    gripper_width = 0.0
                    current_state = np.concatenate([droid_pose, [gripper_width]], axis=0)
                    states = torch.tensor(
                        current_state,
                        device=device
                    ).unsqueeze(0).unsqueeze(0)  # [1,1,7]
                    s_n = states[:, :1].to(dtype=z_n.dtype)

                    # Generate energy landscape if requested
                    if visualize_energy_landscape and save_images_this_episode:
                        try:
                            heatmap, a1_edges, a2_edges, a1_label, a2_label = compute_energy_landscape(
                                encoder=encoder,
                                predictor=predictor,
                                tokens_per_frame=tokens_per_frame,
                                current_rep=z_n,
                                current_state=s_n,
                                goal_rep=z_goal,
                                experiment_type=experiment_type,
                                nsamples=9,
                                grid_size=0.075,
                                normalize_reps=True,
                                device=device
                            )

                            # Compute optimal action direction (straight line to goal)
                            # Direction in RoboHive frame, then convert to DROID frame
                            direction_robohive = final_target_pos - current_ee_pos
                            direction_droid = robohive_to_droid_pos(direction_robohive)

                            # Extract the two axes shown in the landscape
                            grid_size = 0.075
                            if experiment_type in ('x', 'y'):
                                # X-Y plane: axis1=X, axis2=Y
                                opt_a1, opt_a2 = direction_droid[0], direction_droid[1]
                            else:  # 'z'
                                # X-Z plane: axis1=X, axis2=Z
                                opt_a1, opt_a2 = direction_droid[0], direction_droid[2]

                            # Scale to fit within grid while preserving direction
                            max_component = max(abs(opt_a1), abs(opt_a2), 1e-8)
                            if max_component > grid_size:
                                scale = grid_size / max_component
                                opt_a1 *= scale
                                opt_a2 *= scale

                            optimal_action = (opt_a1, opt_a2)

                            # Choose between 2D heatmap and 3D surface plot
                            if energy_landscape_3d:
                                energy_path = os.path.join(
                                    planning_img_dir,
                                    f"step{step_idx:02d}_energy_landscape_3d.png"
                                )
                                plot_energy_landscape_3d(
                                    heatmap, a1_edges, a2_edges,
                                    a1_label, a2_label,
                                    energy_path, step_idx, distance
                                )
                            else:
                                energy_path = os.path.join(
                                    planning_img_dir,
                                    f"step{step_idx:02d}_energy_landscape.png"
                                )
                                plot_energy_landscape(
                                    heatmap, a1_edges, a2_edges,
                                    a1_label, a2_label,
                                    energy_path, step_idx, distance,
                                    optimal_action=optimal_action
                                )

                                # Save energy landscape data for later composite plotting
                                energy_data_path = os.path.join(
                                    planning_img_dir,
                                    f"step{step_idx:02d}_energy_data.npz"
                                )
                                np.savez(
                                    energy_data_path,
                                    heatmap=heatmap,
                                    axis1_edges=a1_edges,
                                    axis2_edges=a2_edges,
                                    axis1_label=a1_label,
                                    axis2_label=a2_label,
                                    distance_to_goal=distance,
                                    optimal_action=np.array(optimal_action),
                                )
                            print(f" Saved energy landscape: {energy_path}")
                        except Exception as e:
                            print(f" Warning: Could not generate energy landscape: {e}")

                    # Plan next action using CEM (model outputs DROID frame)
                    start_time = time.time()
                    actions = world_model.infer_next_action(z_n, s_n, z_goal).cpu().numpy()
                    end_time = time.time()
                    print(f" Planning time: {end_time - start_time:.3f}s")
                    print(
                        f" Raw planned action (DROID frame x,y,z): "
                        f"({actions[0, 0]:.3f}, {actions[0, 1]:.3f}, {actions[0, 2]:.3f})"
                    )

                    # Store raw action (NEW)
                    phase3_actions_raw.append(actions[0].copy())

                    # Transform action from DROID frame to RoboHive frame
                    transformed_action = transform_action(actions[0], action_transform)
                    print(
                        f" Transformed action (RoboHive frame x,y,z): "
                        f"({transformed_action[0]:.3f}, "
                        f"{transformed_action[1]:.3f}, "
                        f"{transformed_action[2]:.3f})"
                    )

                    # Store transformed action (NEW)
                    phase3_actions_transformed.append(transformed_action.copy())

                    # Convert transformed action to joint-space waypoints
                    # Use RoboHive frame positions for IK
                    planned_delta = transformed_action[:7]  # dx,dy,dz,dp,dy,dr,grip
                    try:
                        new_pos, new_rpy = compute_new_pose(
                            current_ee_pos,  # Use RoboHive frame for IK
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
                
                # Calculate final representation distance
                with torch.no_grad():
                    final_rgb_for_repr = sim.renderer.render_offscreen(
                        width=width, height=height, camera_id=camera_name, device_id=device_id
                    )
                    if final_rgb_for_repr is not None:
                        combined_rgb_final = np.stack([final_rgb_for_repr, goal_rgb], axis=0)
                        clips_final = transform(combined_rgb_final).unsqueeze(0).to(device)
                        h_final = forward_target_local(clips_final)
                        z_n_final = h_final[:, :tokens_per_frame].contiguous().clone()
                        z_goal_final = h_final[:, -tokens_per_frame:].contiguous().clone()
                        # Apply alignment for consistent final distance computation
                        if latent_aligner is not None:
                            z_n_final = latent_aligner(z_n_final)
                            z_goal_final = latent_aligner(z_goal_final)
                        final_repr_l1_distance = torch.mean(torch.abs(z_n_final - z_goal_final)).item()
                        phase3_repr_l1_distances.append(final_repr_l1_distance)
                        del h_final, z_n_final, z_goal_final

                if phase3_repr_l1_distances:
                    final_repr_distance = phase3_repr_l1_distances[-1]
                    print(f"\nFinal representation L1 distance: {final_repr_distance:.6f}")
                
                print(f"\nFinal distance to target: {final_distance:.4f}m")

                # Save final state image if requested
                if save_images_this_episode:
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
                phase1_distances_all_targets if phase1_distances_all_targets else [[np.nan]]
            )
            all_episode_data['phase3_vjepa_distances'].append(
                phase3_distances if phase3_distances else [np.nan]
            )
            all_episode_data['phase3_repr_l1_distances'].append(
                phase3_repr_l1_distances if phase3_repr_l1_distances else [np.nan]
            )
            all_episode_data['phase1_final_distance'].append(
                phase1_distances[-1] if phase1_distances else np.nan
            )
            all_episode_data['phase3_final_distance'].append(
                phase3_distances[-1] if phase3_distances else np.nan
            )
            all_episode_data['phase3_actions_raw'].append(
                phase3_actions_raw if phase3_actions_raw else []
            )
            all_episode_data['phase3_actions_transformed'].append(
                phase3_actions_transformed if phase3_actions_transformed else []
            )
            all_episode_data['target_positions'].append(
                final_target_pos.tolist() if final_target_pos is not None else [np.nan, np.nan, np.nan]
            )
            all_episode_data['initial_ee_positions'].append(
                EE_START.tolist()
            )
            all_episode_data['episode_ids'].append(episode)
            all_episode_data['episode_seeds'].append(episode_seed)

        # Save video if offscreen rendering
        if render == 'offscreen' and frames:
            save_frames = np.array(frames, dtype=np.uint8)
            if skvideo is None:
                print("Warning: skvideo not available, skipping video save")
            else:
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

        def to_jsonable(obj):
            """Recursively convert numpy types in obj to Python scalars/lists for JSON."""
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, (_np.generic,)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_jsonable(v) for v in obj]
            return obj

        # Save as JSON for easy loading
        summary_path = os.path.join(
            experiment_out_dir,
            f"{out_name}{experiment_type}_distance_summary.json"
        )
        json_safe_data = to_jsonable(all_episode_data)
        with open(summary_path, 'w') as f:
            json.dump(json_safe_data, f, indent=2)
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
            phase3_repr_l1_distances_per_episode=np.array(
                [np.array(d) for d in all_episode_data['phase3_repr_l1_distances']],
                dtype=object
            ),
            phase3_final_distance=np.array(all_episode_data['phase3_final_distance']),
            phase3_actions_raw_per_episode=np.array(
                [np.array(a) for a in all_episode_data['phase3_actions_raw']],
                dtype=object
            ),
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
