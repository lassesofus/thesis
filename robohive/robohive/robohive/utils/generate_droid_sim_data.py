#!/usr/bin/env python3
"""
Generate DROID-compatible simulation data for V-JEPA2 training.

This script creates synthetic robot trajectories in the exact format expected by
V-JEPA2's DROIDVideoDataset class. Each trajectory contains:
- trajectory.h5 with robot state and camera extrinsics
- metadata JSON with video paths
- MP4 video recordings synchronized with state data

IMPORTANT: Coordinate Frame Alignment
---------------------------------------
This script applies a coordinate transformation (swap_xy_negate_y) to all saved poses and
camera extrinsics to ensure consistency with the DROID dataset coordinate conventions.
This transformation converts from RoboHive's coordinate frame to DROID's:
    DROID_x = RoboHive_y
    DROID_y = -RoboHive_x
    DROID_z = RoboHive_z

The data flow is:
    1. [Training] RoboHive → DROID frame (swap_xy_negate_y) → Save to HDF5
    2. [Training] V-JEPA2 learns from DROID-frame data
    3. [Deployment] V-JEPA2 outputs actions in DROID frame
    4. [Deployment] DROID → RoboHive frame (swap_xy_negate_x) → Execute in simulation

When deploying the trained model in robo_samples.py, use the INVERSE transformation:
    python robo_samples.py --action_transform swap_xy_negate_x

EXAMPLE USAGE:
    # Generate 100 reaching trajectories
    python generate_droid_sim_data.py \
        --out_dir /data/sim_droid \
        --num_trajectories 100 \
        --task reaching \
        --camera_name 99999999 \
        --reach_horizon 4.5

    # Generate with dimension-specific sampling
    python generate_droid_sim_data.py \
        --out_dir /data/sim_droid \
        --num_trajectories 1000 \
        --task reaching \
        --traj_dir x \
        --min_reach_distance 0.05 \
        --max_reach_distance 0.3 \
        --bidirectional \
        --train_test_split 0.8 \
        --save_split_info \
        --seed 42 \
        --reach_horizon 4.5
"""

import os
import sys

# Enable headless rendering BEFORE importing MuJoCo/OpenGL libraries
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import json
import click
import numpy as np
import h5py
from pathlib import Path
from scipy.spatial.transform import Rotation

# Import RoboHive
from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import generate_joint_space_min_jerk
from robohive.utils.xml_utils import reassign_parent

try:
    import skvideo.io
except ImportError:
    skvideo = None
    print("Warning: skvideo not available. Install with: pip install scikit-video")

try:
    from decord import VideoReader, cpu as decord_cpu
except ImportError:
    VideoReader = None
    print("Warning: decord not available. Frame count verification will be skipped.")

# Robot configuration
ARM_nJnt = 7
EE_SITE = "end_effector"


def get_camera_extrinsics(sim, camera_name):
    """
    Extract camera extrinsics from MuJoCo simulation in DROID coordinate frame.

    Args:
        sim: MuJoCo simulation object
        camera_name: Name of the camera in the model

    Returns:
        extrinsics: [x, y, z, roll, pitch, yaw] in DROID frame
    """
    # Get camera ID
    cam_id = sim.model.camera_name2id(camera_name)

    # Get camera position (in RoboHive world frame)
    cam_pos = sim.data.cam_xpos[cam_id].copy()

    # Get camera orientation matrix (3x3 rotation matrix)
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()

    # Convert rotation matrix to xyz Euler angles
    rot = Rotation.from_matrix(cam_mat)
    euler_xyz = rot.as_euler('xyz', degrees=False)  # [roll, pitch, yaw] in radians

    # Combine position and orientation
    extrinsics_robohive = np.concatenate([cam_pos, euler_xyz])

    # Transform to DROID coordinate frame
    extrinsics_droid = transform_pose_to_droid_frame(extrinsics_robohive)

    return extrinsics_droid


def transform_pose_to_droid_frame(pose):
    """
    Transform pose from RoboHive coordinate frame to DROID coordinate frame.

    DROID and RoboHive use different coordinate conventions. This function
    applies the coordinate transformation to match DROID's conventions.

    Transformation: swap_xy_negate_y (RoboHive → DROID)
    - DROID_x = RoboHive_y
    - DROID_y = -RoboHive_x
    - DROID_z = RoboHive_z

    Inverse transformation in robo_samples.py: swap_xy_negate_x (DROID → RoboHive)
    - RoboHive_x = -DROID_y
    - RoboHive_y = DROID_x
    - RoboHive_z = DROID_z

    Args:
        pose: [x, y, z, roll, pitch, yaw] in RoboHive frame

    Returns:
        transformed_pose: [x, y, z, roll, pitch, yaw] in DROID frame
    """
    transformed_pose = pose.copy()

    # Transform position: swap x and y, then negate new y
    transformed_pose[0] = pose[1]   # DROID_x = RoboHive_y
    transformed_pose[1] = -pose[0]  # DROID_y = -RoboHive_x
    transformed_pose[2] = pose[2]   # DROID_z = RoboHive_z

    # Transform orientation: swap roll/pitch and negate new pitch
    transformed_pose[3] = pose[4]   # new_roll = old_pitch
    transformed_pose[4] = -pose[3]  # new_pitch = -old_roll
    transformed_pose[5] = pose[5]   # new_yaw = old_yaw

    return transformed_pose


def get_ee_cartesian_pose(sim, ee_site_id):
    """
    Get end-effector Cartesian pose in DROID coordinate frame.

    Args:
        sim: MuJoCo simulation object
        ee_site_id: End-effector site ID

    Returns:
        pose: [x, y, z, roll, pitch, yaw] in DROID frame
    """
    # Get EE position in RoboHive frame
    ee_pos = sim.data.site_xpos[ee_site_id].copy()

    # Get EE orientation matrix
    ee_mat = sim.data.site_xmat[ee_site_id].reshape(3, 3).copy()

    # Convert to xyz Euler angles
    rot = Rotation.from_matrix(ee_mat)
    euler_xyz = rot.as_euler('xyz', degrees=False)

    # Combine into pose [x, y, z, roll, pitch, yaw]
    pose_robohive = np.concatenate([ee_pos, euler_xyz])

    # Transform to DROID coordinate frame
    pose_droid = transform_pose_to_droid_frame(pose_robohive)

    return pose_droid


def get_gripper_position(sim):
    """
    Get gripper position/state in DROID convention.

    IMPORTANT: Gripper Convention Conversion
    ----------------------------------------
    RoboHive convention: 0.0 = fully closed, 1.0 = fully open
    DROID convention: 1.0 = fully closed, 0.0 = fully open

    This function converts from RoboHive to DROID convention using:
        droid_gripper = 1.0 - robohive_gripper

    For Franka Reach environment without an actual gripper, we use a constant
    placeholder (0.0 in RoboHive = closed → 1.0 in DROID = closed).

    In a real gripper scenario, you would:
    1. Read the gripper joint positions
    2. Normalize to [0, 1] range (RoboHive convention)
    3. Apply the conversion: 1.0 - normalized_value

    Args:
        sim: MuJoCo simulation object

    Returns:
        gripper_pos: Scalar gripper state in DROID convention (1.0=closed, 0.0=open)
    """
    # For Franka Reach environment, there's no actual gripper
    # Use constant placeholder: 0.0 in RoboHive convention (closed)
    robohive_gripper = 0.0

    # Convert to DROID convention: invert the value
    droid_gripper = 1.0 - robohive_gripper

    return droid_gripper


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


def sample_sphere_target(center, radius):
    """Sample a point uniformly inside a sphere."""
    u = np.random.rand()
    r = radius * (u ** (1.0 / 3.0))
    phi = 2 * np.pi * np.random.rand()
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    direction = np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])
    return center + r * direction


def sample_target_along_dimension(center, dimension, distance):
    """
    Sample a target position along a specific dimension.

    Args:
        center: Starting position [x, y, z]
        dimension: 'x', 'y', or 'z' - which axis to move along
        distance: Distance to move (can be positive or negative)

    Returns:
        target_pos: Target position [x, y, z]
    """
    offset = np.zeros(3)
    dim_idx = {'x': 0, 'y': 1, 'z': 2}[dimension.lower()]
    offset[dim_idx] = distance
    return center + offset


def generate_reaching_trajectory(sim, ee_sid, start_qpos, target_pos, horizon, step_dt):
    """
    Generate a reaching trajectory using IK and min-jerk.

    Args:
        sim: MuJoCo simulation
        ee_sid: End-effector site ID
        start_qpos: Starting joint positions [7]
        target_pos: Target Cartesian position [3]
        horizon: Time to reach target (seconds)
        step_dt: Simulation timestep

    Returns:
        waypoints: List of {'position': qpos} dictionaries
    """
    # Solve IK for target
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
        print(f"  WARNING: IK failed (err: {ik_result.err_norm:.4f})")

    # Generate smooth trajectory
    waypoints = generate_joint_space_min_jerk(
        start=start_qpos,
        goal=ik_result.qpos[:ARM_nJnt],
        time_to_go=horizon,
        dt=step_dt
    )

    return waypoints


def generate_random_trajectory(sim, start_qpos, num_steps, step_dt):
    """
    Generate a random exploration trajectory with smooth noise.

    Args:
        sim: MuJoCo simulation
        start_qpos: Starting joint positions [7]
        num_steps: Number of steps
        step_dt: Simulation timestep

    Returns:
        waypoints: List of {'position': qpos} dictionaries
    """
    waypoints = []
    current_qpos = start_qpos.copy()

    # Generate random targets with some smoothness
    for i in range(num_steps):
        # Add small random perturbations
        if i % 50 == 0:  # Change direction every 50 steps
            noise = np.random.randn(ARM_nJnt) * 0.1
            target_qpos = current_qpos + noise
            # Clip to reasonable joint limits
            target_qpos = np.clip(target_qpos, -2.8, 2.8)
        else:
            target_qpos = current_qpos

        waypoints.append({'position': target_qpos.copy()})
        current_qpos = target_qpos

    return waypoints


def execute_trajectory_and_record(
    sim,
    waypoints,
    camera_name,
    ee_sid,
    step_dt,
    width=640,
    height=480,
    device_id=0,
    fps=30
):
    """
    Execute trajectory and record all required data with downsampling.

    Args:
        sim: MuJoCo simulation
        waypoints: List of waypoint dictionaries
        camera_name: Camera name for rendering
        ee_sid: End-effector site ID
        step_dt: Simulation timestep
        width: Video width
        height: Video height
        device_id: Rendering device ID
        fps: Target video framerate for downsampling

    Returns:
        data: Dictionary with:
            - frames: List of RGB frames [T, H, W, 3] (downsampled)
            - cartesian_position: [T, 6] array (downsampled)
            - gripper_position: [T] array (downsampled)
            - camera_extrinsics: [T, 6] array (downsampled)
    """
    num_steps = len(waypoints)

    # Calculate downsampling rate
    sim_fps = int(round(1.0 / step_dt))
    steps_per_frame = max(1, int(round(sim_fps / max(1, fps))))

    # Preallocate arrays for downsampled data
    all_cartesian_positions = []
    all_gripper_positions = []
    all_camera_extrinsics = []
    all_frames = []

    # Execute and record (with downsampling)
    for idx, wp in enumerate(waypoints):
        # Set control
        sim.data.ctrl[:ARM_nJnt] = wp['position']

        # Step simulation
        sim.advance(render=False)

        # Update forward kinematics
        sim.forward()

        # Record only at downsampled rate
        if (idx % steps_per_frame) == 0:
            # Record robot state
            all_cartesian_positions.append(get_ee_cartesian_pose(sim, ee_sid))
            all_gripper_positions.append(get_gripper_position(sim))
            all_camera_extrinsics.append(get_camera_extrinsics(sim, camera_name))

            # Render and record frame
            frame = sim.renderer.render_offscreen(
                width=width,
                height=height,
                camera_id=camera_name,
                device_id=device_id
            )
            all_frames.append(frame)

    # Convert lists to arrays
    cartesian_positions = np.array(all_cartesian_positions, dtype=np.float64)
    gripper_positions = np.array(all_gripper_positions, dtype=np.float64)
    camera_extrinsics = np.array(all_camera_extrinsics, dtype=np.float64)

    return {
        'frames': all_frames,
        'cartesian_position': cartesian_positions,
        'gripper_position': gripper_positions,
        'camera_extrinsics': camera_extrinsics
    }


def save_trajectory(traj_dir, data, camera_name, fps=30):
    """
    Save trajectory data in DROID format.

    Args:
        traj_dir: Directory to save trajectory
        data: Dictionary with frames and state data
        camera_name: Camera identifier (e.g., "99999999")
        fps: Video framerate
    """
    traj_path = Path(traj_dir)
    traj_path.mkdir(parents=True, exist_ok=True)

    # Create recordings directory
    mp4_dir = traj_path / "recordings" / "MP4"
    mp4_dir.mkdir(parents=True, exist_ok=True)

    T = len(data['frames'])

    # 1. Save video
    mp4_path = mp4_dir / f"{camera_name}.mp4"
    if skvideo is None:
        print(f"  ERROR: skvideo not available, cannot save video")
        return False

    frames_array = np.array(data['frames'], dtype=np.uint8)
    # Use encoding options that prevent frame count mismatch:
    # inputdict -r: Specify input frame rate (CRITICAL for correct frame count)
    # -g 1: GOP size 1 (keyframe every frame, no B-frame dependencies)
    # -bf 0: Disable B-frames explicitly
    inputdict = {"-r": str(fps)}
    outputdict = {
        "-pix_fmt": "yuv420p",
        "-r": str(fps),
        "-g": "1",
        "-bf": "0"
    }
    skvideo.io.vwrite(str(mp4_path), frames_array, inputdict=inputdict, outputdict=outputdict)
    print(f"  Saved video: {mp4_path} ({frames_array.shape})")

    # Verify frame count matches expected
    if VideoReader is not None:
        vr = VideoReader(str(mp4_path), ctx=decord_cpu(0))
        actual_frames = len(vr)
        expected_frames = len(frames_array)
        if actual_frames != expected_frames:
            print(f"  WARNING: Video frame count mismatch! Expected {expected_frames}, got {actual_frames}")
            print(f"           This may cause data misalignment during training.")
        else:
            print(f"  Verified video frame count: {actual_frames}")

    # 2. Save trajectory.h5
    h5_path = traj_path / "trajectory.h5"
    with h5py.File(h5_path, 'w') as f:
        # Create group structure
        obs_group = f.create_group('observation')
        robot_state_group = obs_group.create_group('robot_state')
        camera_extrinsics_group = obs_group.create_group('camera_extrinsics')

        # Write required datasets
        robot_state_group.create_dataset(
            'cartesian_position',
            data=data['cartesian_position'],
            dtype='float64'
        )
        robot_state_group.create_dataset(
            'gripper_position',
            data=data['gripper_position'],
            dtype='float64'
        )
        camera_extrinsics_group.create_dataset(
            f'{camera_name}_left',
            data=data['camera_extrinsics'],
            dtype='float64'
        )

    print(f"  Saved HDF5: {h5_path}")

    # 3. Save metadata JSON
    metadata = {
        "trajectory_length": T,
        "left_mp4_path": f"recordings/MP4/{camera_name}.mp4",
        "camera_name": camera_name,
        "fps": fps,
        "resolution": [frames_array.shape[2], frames_array.shape[1]],  # [width, height]
        "source": "robohive_simulation"
    }

    json_path = traj_path / "metadata_sim.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {json_path}")

    return True


@click.command(help=__doc__)
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
    help='Output directory for generated trajectories'
)
@click.option(
    '--num_trajectories',
    type=int,
    default=10,
    help='Number of trajectories to generate'
)
@click.option(
    '--task',
    type=click.Choice(['reaching', 'random_exploration', 'multi_target']),
    default='reaching',
    help='Type of task to generate'
)
@click.option(
    '--video_duration',
    type=float,
    default=None,
    help='Target video duration in seconds (default: None = match reach_horizon). Overrides trajectory_length if set.'
)
@click.option(
    '--trajectory_length',
    type=int,
    default=None,
    help='DEPRECATED: Use --reach_horizon to control execution time instead.'
)
@click.option(
    '--camera_name',
    type=str,
    default='99999999',
    help='Camera identifier for video and extrinsics'
)
@click.option(
    '--mujoco_camera',
    type=str,
    default='left_cam',
    help='MuJoCo camera name to render from'
)
@click.option(
    '--width',
    type=int,
    default=640,
    help='Video width'
)
@click.option(
    '--height',
    type=int,
    default=480,
    help='Video height'
)
@click.option(
    '--fps',
    type=int,
    default=30,
    help='Video framerate'
)
@click.option(
    '--device_id',
    type=int,
    default=0,
    help='Rendering device ID'
)
@click.option(
    '--reach_radius',
    type=float,
    default=0.2,
    help='Radius for reach target sampling (meters)'
)
@click.option(
    '--reach_horizon',
    type=float,
    default=4.5,
    help='Time to reach each target (seconds)'
)
@click.option(
    '--csv_output',
    type=str,
    default=None,
    help='Path to output CSV file listing all trajectories (for V-JEPA2 training)'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Random seed for reproducibility (default: None = random)'
)
@click.option(
    '--traj_dir',
    type=click.Choice(['x', 'y', 'z', 'sphere']),
    default='sphere',
    help='Direction to generate trajectories in DROID frame: x, y, z axis, or sphere (random). Note: DROID_x=RoboHive_y, DROID_y=-RoboHive_x'
)
@click.option(
    '--max_reach_distance',
    type=float,
    default=0.3,
    help='Maximum reach distance along specified dimension (meters)'
)
@click.option(
    '--min_reach_distance',
    type=float,
    default=0.05,
    help='Minimum reach distance along specified dimension (meters, default: 0.05m)'
)
@click.option(
    '--bidirectional',
    is_flag=True,
    default=False,
    help='Allow both positive and negative movement along specified axis (e.g., x in [-max, -min] U [min, max])'
)
@click.option(
    '--train_test_split',
    type=float,
    default=0.8,
    help='Fraction of trajectories to use for training (default: 0.8 = 80%% train, 20%% test)'
)
@click.option(
    '--success_threshold',
    type=float,
    default=0.05,
    help='Distance threshold (meters) to consider target reached successfully (default: 0.05m)'
)
@click.option(
    '--save_split_info',
    is_flag=True,
    default=False,
    help='Save train/test split information to separate CSV files'
)
@click.option(
    '--gripper',
    type=click.Choice(['franka', 'robotiq']),
    default='franka',
    help='End-effector gripper type: franka (default parallel-jaw) or robotiq (2F-85)'
)
def main(
    sim_path,
    out_dir,
    num_trajectories,
    task,
    video_duration,
    trajectory_length,
    camera_name,
    mujoco_camera,
    width,
    height,
    fps,
    device_id,
    reach_radius,
    reach_horizon,
    csv_output,
    seed,
    traj_dir,
    max_reach_distance,
    min_reach_distance,
    bidirectional,
    train_test_split,
    success_threshold,
    save_split_info,
    gripper
):
    """Generate DROID-compatible simulation data."""

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        print(f"Random seed set to: {seed}")

    print("=" * 80)
    print("DROID-Compatible Simulation Data Generator")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print(f"Number of trajectories: {num_trajectories}")
    print(f"Task: {task}")
    print(f"Trajectory direction (DROID frame): {traj_dir}")
    if traj_dir != 'sphere':
        # Show the RoboHive axis being used
        droid_to_robohive = {'x': 'y', 'y': 'x', 'z': 'z'}
        robohive_axis = droid_to_robohive.get(traj_dir, traj_dir)
        negation_note = " (negated)" if traj_dir == 'y' else ""
        print(f"  → RoboHive axis: {robohive_axis}{negation_note} (DROID_{traj_dir} = {'−' if traj_dir == 'y' else ''}RoboHive_{robohive_axis})")
        if bidirectional:
            print(f"Reach distance range: [-{max_reach_distance}, -{min_reach_distance}] U [{min_reach_distance}, {max_reach_distance}]m along DROID {traj_dir}-axis")
        else:
            print(f"Reach distance range: [{min_reach_distance}, {max_reach_distance}]m along DROID {traj_dir}-axis")
    else:
        print(f"Sphere radius: {reach_radius}m")
    print(f"Train/test split: {train_test_split:.1%} train / {1-train_test_split:.1%} test")
    print(f"Success threshold: {success_threshold}m")
    print(f"Camera: {camera_name} (MuJoCo: {mujoco_camera})")
    print(f"Resolution: {width}x{height} @ {fps} fps")
    print("=" * 80)

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

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

    # Attach gripper to arm if using robotiq
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

    # Get site IDs
    ee_sid = sim.model.site_name2id(EE_SITE)

    # Get starting position
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

    # Reset simulation
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
    sim.data.ctrl[:ARM_nJnt] = ARM_JNT0.copy()
    sim.forward()

    # Get EE starting position for target sampling
    ee_start = sim.data.site_xpos[ee_sid].copy()
    ee_start_droid = robohive_to_droid_pos(ee_start)
    print(f"End-effector start position (RoboHive): {ee_start}")
    print(f"End-effector start position (DROID): {ee_start_droid}")

    # Get simulation timestep
    step_dt = sim.model.opt.timestep
    print(f"Simulation timestep: {step_dt:.4f}s")

    # Calculate video parameters
    sim_fps = int(round(1.0 / step_dt))
    steps_per_frame = max(1, int(round(sim_fps / max(1, fps))))
    actual_fps = int(round(sim_fps / steps_per_frame))

    # Determine target video duration
    if trajectory_length is not None:
        print(f"WARNING: trajectory_length is deprecated. Using reach_horizon instead.")

    # Calculate actual trajectory length needed
    if task == 'reaching':
        # For reaching: use reach_horizon to determine execution time
        execution_time = reach_horizon
    else:
        # For other tasks: match video duration
        execution_time = video_duration if video_duration is not None else reach_horizon

    computed_trajectory_length = int(execution_time / step_dt)

    # Calculate expected number of frames and video duration
    expected_frames = computed_trajectory_length // steps_per_frame
    expected_video_duration = expected_frames / actual_fps

    print(f"Reach horizon: {reach_horizon}s")
    print(f"Computed trajectory length: {computed_trajectory_length} steps ({execution_time:.2f}s execution)")
    print(f"Downsampling: 1 frame every {steps_per_frame} steps (sim_fps={sim_fps}, video_fps={actual_fps})")
    print(f"Expected video: {expected_frames} frames @ {actual_fps} fps = {expected_video_duration:.2f}s")

    # Validate distance parameters
    if min_reach_distance < 0:
        raise ValueError(f"min_reach_distance must be >= 0, got {min_reach_distance}")
    if max_reach_distance <= min_reach_distance:
        raise ValueError(f"max_reach_distance ({max_reach_distance}) must be > min_reach_distance ({min_reach_distance})")

    # Pre-generate target positions for all trajectories
    print(f"\nPre-generating target positions...")
    target_positions = []
    target_distances = []

    for traj_idx in range(num_trajectories):
        if traj_dir == 'sphere':
            # Random sampling in sphere
            target_pos = sample_sphere_target(ee_start, reach_radius)
            distance = np.linalg.norm(target_pos - ee_start)
        else:
            # Sample distance uniformly along specified dimension
            if bidirectional:
                # Sample from [-max, -min] U [min, max]
                # Randomly choose positive or negative direction
                if np.random.rand() < 0.5:
                    # Positive direction
                    distance = np.random.uniform(min_reach_distance, max_reach_distance)
                else:
                    # Negative direction
                    distance = -np.random.uniform(min_reach_distance, max_reach_distance)
            else:
                # Only positive direction: [min, max]
                distance = np.random.uniform(min_reach_distance, max_reach_distance)

            # Map DROID direction to RoboHive direction
            # traj_dir specifies direction in DROID frame, but we need to sample in RoboHive frame
            # DROID_x = RoboHive_y, DROID_y = -RoboHive_x, DROID_z = RoboHive_z
            # Note: negation is handled by transform_pose_to_droid_frame, here we only map axes
            if traj_dir == 'x':
                # DROID x-axis corresponds to RoboHive y-axis
                robohive_dir = 'y'
            elif traj_dir == 'y':
                # DROID y-axis corresponds to RoboHive x-axis (with negation)
                robohive_dir = 'x'
            else:  # 'z'
                # DROID z-axis = RoboHive z-axis (no change)
                robohive_dir = 'z'

            target_pos = sample_target_along_dimension(ee_start, robohive_dir, distance)

        target_positions.append(target_pos)
        target_distances.append(distance)

    # Split into train/test sets
    num_train = int(num_trajectories * train_test_split)
    num_test = num_trajectories - num_train

    # Create indices and shuffle them
    indices = np.arange(num_trajectories)
    np.random.shuffle(indices)

    train_indices = set(indices[:num_train])
    test_indices = set(indices[num_train:])

    print(f"Train set: {num_train} trajectories")
    print(f"Test set: {num_test} trajectories")

    # Track generated trajectories
    trajectory_paths = []
    trajectory_metadata = []
    train_paths = []
    test_paths = []

    # Generate trajectories
    print(f"\nGenerating {num_trajectories} trajectories...")

    for traj_idx in range(num_trajectories):
        is_train = traj_idx in train_indices
        split_name = "TRAIN" if is_train else "TEST"

        print(f"\n--- Trajectory {traj_idx + 1}/{num_trajectories} [{split_name}] ---")

        # Reset to start
        sim.data.qpos[:ARM_nJnt] = ARM_JNT0
        sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
        sim.data.ctrl[:ARM_nJnt] = ARM_JNT0.copy()
        sim.forward()

        current_qpos = ARM_JNT0.copy()

        # Get pre-generated target
        target_pos = target_positions[traj_idx]
        target_distance = target_distances[traj_idx]

        # Generate waypoints based on task
        if task == 'reaching':
            # Single reach target
            target_pos_droid = robohive_to_droid_pos(target_pos)
            print(f"  Target (RoboHive): [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            print(f"  Target (DROID): [{target_pos_droid[0]:.3f}, {target_pos_droid[1]:.3f}, {target_pos_droid[2]:.3f}]")
            print(f"  Distance: {target_distance:.4f}m along DROID {traj_dir}-axis")
            waypoints = generate_reaching_trajectory(
                sim, ee_sid, current_qpos, target_pos, reach_horizon, step_dt
            )

        elif task == 'multi_target':
            # Multiple consecutive reaches
            num_reaches = np.random.randint(2, 5)
            waypoints = []
            for reach_idx in range(num_reaches):
                target_pos = sample_sphere_target(ee_start, reach_radius)
                sub_waypoints = generate_reaching_trajectory(
                    sim, ee_sid, current_qpos, target_pos, reach_horizon, step_dt
                )
                waypoints.extend(sub_waypoints)
                current_qpos = sub_waypoints[-1]['position']
            print(f"  Generated {num_reaches} reaches, {len(waypoints)} total steps")

        elif task == 'random_exploration':
            waypoints = generate_random_trajectory(
                sim, current_qpos, computed_trajectory_length, step_dt
            )
            print(f"  Generated random exploration: {len(waypoints)} steps")

        # Note: We don't truncate waypoints anymore. The robot executes the full
        # trajectory, and downsampling in execute_trajectory_and_record handles
        # the video frame rate.

        print(f"  Executing trajectory: {len(waypoints)} steps ({len(waypoints)*step_dt:.2f}s)")

        # Execute and record
        data = execute_trajectory_and_record(
            sim=sim,
            waypoints=waypoints,
            camera_name=mujoco_camera,
            ee_sid=ee_sid,
            step_dt=step_dt,
            width=width,
            height=height,
            device_id=device_id,
            fps=fps
        )

        # Measure final distance to target
        # Note: final_ee_pos is in DROID frame, need to transform target_pos to DROID frame
        final_ee_pos = data['cartesian_position'][-1, :3]
        target_pos_droid = robohive_to_droid_pos(target_pos)
        final_distance = np.linalg.norm(final_ee_pos - target_pos_droid)
        success_reached = final_distance <= success_threshold

        print(f"  Final distance: {final_distance:.4f}m [{'SUCCESS' if success_reached else 'FAIL'}]")

        # Save trajectory
        traj_dir_path = out_path / f"episode_{traj_idx:04d}"
        save_success = save_trajectory(traj_dir_path, data, camera_name, fps)

        if save_success:
            traj_abs_path = str(traj_dir_path.absolute())
            trajectory_paths.append(traj_abs_path)

            # Store metadata
            metadata = {
                'trajectory_index': traj_idx,
                'trajectory_path': traj_abs_path,
                'target_position': target_pos.tolist(),
                'target_distance': float(target_distance),
                'final_distance': float(final_distance),
                'success': bool(success_reached),
                'split': 'train' if is_train else 'test',
                'trajectory_direction': traj_dir,
                'task': task
            }
            trajectory_metadata.append(metadata)

            # Add to appropriate split
            if is_train:
                train_paths.append(traj_abs_path)
            else:
                test_paths.append(traj_abs_path)

        # Reset simulation
        sim.reset()

    # Save trajectory metadata as JSON
    metadata_path = out_path / "trajectory_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(trajectory_metadata, f, indent=2)
    print(f"\nSaved trajectory metadata: {metadata_path}")

    # Save CSV file for V-JEPA2 training (all trajectories)
    if csv_output and trajectory_paths:
        csv_path = Path(csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, 'w') as f:
            for path in trajectory_paths:
                f.write(f"{path}\n")

        print(f"Saved all trajectories list: {csv_path}")

    # Save separate train/test CSV files if requested
    if save_split_info:
        train_csv_path = out_path / "train_trajectories.csv"
        test_csv_path = out_path / "test_trajectories.csv"

        with open(train_csv_path, 'w') as f:
            for path in train_paths:
                f.write(f"{path}\n")

        with open(test_csv_path, 'w') as f:
            for path in test_paths:
                f.write(f"{path}\n")

        print(f"Saved train trajectories: {train_csv_path} ({len(train_paths)} trajectories)")
        print(f"Saved test trajectories: {test_csv_path} ({len(test_paths)} trajectories)")

    # Print summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Overall statistics
    train_metadata = [m for m in trajectory_metadata if m['split'] == 'train']
    test_metadata = [m for m in trajectory_metadata if m['split'] == 'test']

    def print_split_stats(split_data, split_name):
        if not split_data:
            return

        final_distances = [m['final_distance'] for m in split_data]
        target_distances = [m['target_distance'] for m in split_data]
        successes = [m['success'] for m in split_data]

        print(f"\n{split_name.upper()} SET ({len(split_data)} trajectories):")
        print(f"  Target distances: mean={np.mean(target_distances):.4f}m, "
              f"std={np.std(target_distances):.4f}m, "
              f"range=[{np.min(target_distances):.4f}, {np.max(target_distances):.4f}]m")
        print(f"  Final distances:  mean={np.mean(final_distances):.4f}m, "
              f"std={np.std(final_distances):.4f}m, "
              f"range=[{np.min(final_distances):.4f}, {np.max(final_distances):.4f}]m")
        print(f"  Success rate: {sum(successes)}/{len(successes)} "
              f"({100*sum(successes)/len(successes):.1f}%) "
              f"[threshold: {success_threshold}m]")

    print_split_stats(train_metadata, "train")
    print_split_stats(test_metadata, "test")

    print(f"\n{'=' * 80}")
    if save_split_info and train_paths and test_paths:
        print(f"\nTo train V-JEPA2 on the training set, update your config:")
        print(f"  data:")
        print(f"    dataset_type: VideoDataset")
        print(f"    datasets: ['DROIDVideoDataset']")
        print(f"    droid_train_paths: '{train_csv_path.absolute()}'")
        print(f"\nTo evaluate on the test set, use the paths in: {test_csv_path}")
    print("=" * 80)

    print(f"\nDone! Generated {len(trajectory_paths)} trajectories in {out_dir}")


if __name__ == '__main__':
    main()
