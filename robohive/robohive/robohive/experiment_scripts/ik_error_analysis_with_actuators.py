#!/usr/bin/env python3
"""
Inverse Kinematics Error Analysis with Actuator Dynamics

This script investigates the IK solver error as a function of target distance
when trajectories are executed via actuator control (matching real robot behavior).

Unlike the original ik_error_analysis.py which directly sets qpos (instantaneous
teleportation), this script executes min-jerk trajectories through the actuators,
capturing the realistic tracking error from position control dynamics.

The experiment records:
- Absolute L2 error (meters) after actuator execution
- Relative error (as fraction of target radius)
- Maximum distance during trajectory (transient overshoot)
- IK convergence statistics

USAGE:
    python ik_error_analysis_with_actuators.py --output_dir /path/to/results
    python ik_error_analysis_with_actuators.py --radii 0.05 0.1 0.15 0.2 0.25 0.3 --samples_per_radius 100
"""

import os
import sys
import json
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Enable headless rendering
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Add thesis root to path for plot_config
sys.path.insert(0, '/home/s185927/thesis')
from plot_config import PLOT_PARAMS, apply_plot_params, configure_axis

from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import generate_joint_space_min_jerk

try:
    import skvideo.io
    SKVIDEO_AVAILABLE = True
except ImportError:
    SKVIDEO_AVAILABLE = False
    print("Warning: skvideo not available. Video recording disabled. Install with: pip install scikit-video")


# Robot configuration
ARM_nJnt = 7
EE_SITE = "end_effector"

# Default starting configuration (from Franka XML)
ARM_JNT0 = np.array([
    -0.0321842,
    -0.394346,
    0.00932319,
    -2.77917,
    -0.011826,
    0.713889,
    1.53183
])


def sample_sphere_target(center, radius, z_min=None, max_attempts=100):
    """
    Sample a point uniformly inside a sphere, with optional z-floor constraint.

    Args:
        center: Center position [x, y, z]
        radius: Sphere radius
        z_min: Minimum z coordinate (e.g., table surface height). If None, no constraint.
        max_attempts: Maximum resampling attempts before giving up

    Returns:
        target_pos: Sampled position [x, y, z], or None if no valid sample found
    """
    for _ in range(max_attempts):
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
        target = center + r * direction

        # Check z-floor constraint
        if z_min is None or target[2] >= z_min:
            return target

    # Failed to find valid sample
    return None


def sample_sphere_surface(center, radius, z_min=None, max_attempts=100):
    """
    Sample a point uniformly on the surface of a sphere, with optional z-floor constraint.

    Args:
        center: Center position [x, y, z]
        radius: Sphere radius
        z_min: Minimum z coordinate (e.g., table surface height). If None, no constraint.
        max_attempts: Maximum resampling attempts before giving up

    Returns:
        target_pos: Sampled position [x, y, z], or None if no valid sample found
    """
    for _ in range(max_attempts):
        phi = 2 * np.pi * np.random.rand()
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta * cos_theta)
        direction = np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ])
        target = center + radius * direction

        # Check z-floor constraint
        if z_min is None or target[2] >= z_min:
            return target

    # Failed to find valid sample
    return None


def run_actuator_ik_test(sim, ee_sid, start_qpos, target_pos, ik_params,
                          horizon=3.0, hold_time=0.5):
    """
    Test IK solver with actuator-based trajectory execution.

    This function executes the IK solution via actuator control (sim.data.ctrl),
    matching the behavior of real robot experiments and robo_samples.py.

    Args:
        sim: MuJoCo simulation object
        ee_sid: End-effector site ID
        start_qpos: Starting joint configuration
        target_pos: Target Cartesian position
        ik_params: Dictionary of IK solver parameters
        horizon: Trajectory duration in seconds
        hold_time: Time to hold at final position for settling

    Returns:
        result_dict: Dictionary with test results
    """
    step_dt = sim.model.opt.timestep

    # Reset to starting configuration (direct qpos for clean start)
    sim.data.qpos[:ARM_nJnt] = start_qpos
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
    sim.data.ctrl[:ARM_nJnt] = start_qpos.copy()
    sim.forward()

    # Get starting EE position
    start_ee_pos = sim.data.site_xpos[ee_sid].copy()

    # Calculate target distance
    target_distance = np.linalg.norm(target_pos - start_ee_pos)
    initial_distance = target_distance  # Should be same at start

    # Run IK to get target joint configuration
    ik_result = qpos_from_site_pose(
        physics=sim,
        site_name=EE_SITE,
        target_pos=target_pos,
        target_quat=None,
        inplace=False,
        **ik_params
    )

    # Generate min-jerk trajectory (same as robo_samples.py)
    waypoints = generate_joint_space_min_jerk(
        start=start_qpos,
        goal=ik_result.qpos[:ARM_nJnt],
        time_to_go=horizon,
        dt=step_dt
    )

    # Execute trajectory via actuators and track distances
    distances_during_trajectory = []
    t = 0.0
    idx = 0
    num_waypoints = len(waypoints)

    while t <= horizon:
        idx = min(idx, num_waypoints - 1)

        # Set actuator target (same as robo_samples.py)
        sim.data.ctrl[:ARM_nJnt] = waypoints[idx]['position']

        # Advance simulation
        sim.advance(render=False)
        sim.forward()

        # Track distance to target
        current_ee_pos = sim.data.site_xpos[ee_sid].copy()
        dist = np.linalg.norm(current_ee_pos - target_pos)
        distances_during_trajectory.append(dist)

        t += step_dt
        idx += 1

    # Hold at final position for settling (same as robo_samples.py)
    final_ctrl = waypoints[-1]['position'].copy()
    hold_steps = max(5, int(hold_time / step_dt))

    for _ in range(hold_steps):
        sim.data.ctrl[:ARM_nJnt] = final_ctrl
        sim.advance(render=False)
        sim.forward()

        # Track distance during hold
        current_ee_pos = sim.data.site_xpos[ee_sid].copy()
        dist = np.linalg.norm(current_ee_pos - target_pos)
        distances_during_trajectory.append(dist)

    # Final measurements
    final_ee_pos = sim.data.site_xpos[ee_sid].copy()
    absolute_error = np.linalg.norm(final_ee_pos - target_pos)
    relative_error = absolute_error / target_distance if target_distance > 0 else 0.0

    # Calculate trajectory statistics
    max_distance = np.max(distances_during_trajectory)
    min_distance = np.min(distances_during_trajectory)

    # Joint displacement
    joint_displacement = np.linalg.norm(ik_result.qpos[:ARM_nJnt] - start_qpos)

    return {
        'target_pos': target_pos.tolist(),
        'target_distance': target_distance,
        'initial_distance': initial_distance,
        'start_ee_pos': start_ee_pos.tolist(),
        'final_ee_pos': final_ee_pos.tolist(),
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'max_distance_during_trajectory': max_distance,
        'min_distance_during_trajectory': min_distance,
        'ik_err_norm': ik_result.err_norm,
        'ik_steps': ik_result.steps,
        'ik_success': ik_result.success,
        'joint_displacement': joint_displacement,
        'num_trajectory_steps': len(distances_during_trajectory)
    }


def record_actuator_video(sim, ee_sid, target_sid, start_qpos, target_pos, ik_params,
                           output_path, horizon=3.0, hold_time=0.5,
                           width=640, height=480, fps=30, camera_name='left_cam'):
    """
    Record a video of the robot executing an IK solution via actuators.

    Args:
        sim: MuJoCo simulation object
        ee_sid: End-effector site ID
        target_sid: Target site ID (for visualization)
        start_qpos: Starting joint configuration
        target_pos: Target Cartesian position
        ik_params: Dictionary of IK solver parameters
        output_path: Path to save MP4 file
        horizon: Trajectory duration in seconds
        hold_time: Hold time at end
        width: Video width
        height: Video height
        fps: Video framerate
        camera_name: MuJoCo camera name for rendering

    Returns:
        result_dict: Dictionary with test results
    """
    if not SKVIDEO_AVAILABLE:
        print("Warning: skvideo not available, cannot record video")
        return None

    frames = []
    step_dt = sim.model.opt.timestep
    sim_fps = int(1.0 / step_dt)
    steps_per_frame = max(1, sim_fps // fps)

    # Reset to starting configuration
    sim.data.qpos[:ARM_nJnt] = start_qpos
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
    sim.data.ctrl[:ARM_nJnt] = start_qpos.copy()
    sim.forward()

    # Update target marker position
    sim.model.site_pos[target_sid][:] = target_pos

    # Capture starting frames (hold for 0.5 seconds)
    for _ in range(int(fps * 0.5)):
        frame = sim.renderer.render_offscreen(
            width=width, height=height, camera_id=camera_name, device_id=0
        )
        frames.append(frame)

    start_ee_pos = sim.data.site_xpos[ee_sid].copy()
    target_distance = np.linalg.norm(target_pos - start_ee_pos)

    # Run IK
    ik_result = qpos_from_site_pose(
        physics=sim,
        site_name=EE_SITE,
        target_pos=target_pos,
        target_quat=None,
        inplace=False,
        **ik_params
    )

    # Generate min-jerk trajectory
    waypoints = generate_joint_space_min_jerk(
        start=start_qpos,
        goal=ik_result.qpos[:ARM_nJnt],
        time_to_go=horizon,
        dt=step_dt
    )

    # Execute trajectory via actuators and record
    distances_during_trajectory = []
    frame_idx = 0

    for idx, wp in enumerate(waypoints):
        sim.data.ctrl[:ARM_nJnt] = wp['position']
        sim.advance(render=False)
        sim.forward()

        # Track distance
        current_ee_pos = sim.data.site_xpos[ee_sid].copy()
        dist = np.linalg.norm(current_ee_pos - target_pos)
        distances_during_trajectory.append(dist)

        # Capture frame at specified rate
        if (frame_idx % steps_per_frame) == 0:
            frame = sim.renderer.render_offscreen(
                width=width, height=height, camera_id=camera_name, device_id=0
            )
            frames.append(frame)
        frame_idx += 1

    # Hold at final position
    final_ctrl = waypoints[-1]['position'].copy()
    hold_steps = max(5, int(hold_time / step_dt))

    for _ in range(hold_steps):
        sim.data.ctrl[:ARM_nJnt] = final_ctrl
        sim.advance(render=False)
        sim.forward()

        current_ee_pos = sim.data.site_xpos[ee_sid].copy()
        dist = np.linalg.norm(current_ee_pos - target_pos)
        distances_during_trajectory.append(dist)

        if (frame_idx % steps_per_frame) == 0:
            frame = sim.renderer.render_offscreen(
                width=width, height=height, camera_id=camera_name, device_id=0
            )
            frames.append(frame)
        frame_idx += 1

    # Hold at final position for video (additional 0.5 seconds)
    for _ in range(int(fps * 0.5)):
        frame = sim.renderer.render_offscreen(
            width=width, height=height, camera_id=camera_name, device_id=0
        )
        frames.append(frame)

    # Measure final error
    final_ee_pos = sim.data.site_xpos[ee_sid].copy()
    absolute_error = np.linalg.norm(final_ee_pos - target_pos)
    relative_error = absolute_error / target_distance if target_distance > 0 else 0.0
    joint_displacement = np.linalg.norm(ik_result.qpos[:ARM_nJnt] - start_qpos)
    max_distance = np.max(distances_during_trajectory)

    # Save video
    frames_array = np.array(frames, dtype=np.uint8)
    outputdict = {"-pix_fmt": "yuv420p", "-r": str(fps)}
    skvideo.io.vwrite(str(output_path), frames_array, outputdict=outputdict)

    return {
        'target_pos': target_pos.tolist(),
        'target_distance': target_distance,
        'start_ee_pos': start_ee_pos.tolist(),
        'final_ee_pos': final_ee_pos.tolist(),
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'max_distance_during_trajectory': max_distance,
        'ik_err_norm': ik_result.err_norm,
        'ik_steps': ik_result.steps,
        'ik_success': ik_result.success,
        'joint_displacement': joint_displacement,
        'video_path': str(output_path)
    }


@click.command(help=__doc__)
@click.option(
    '--sim_path',
    type=str,
    default='/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_v0.xml',
    help='Path to MuJoCo XML model'
)
@click.option(
    '--output_dir',
    type=str,
    default='/home/s185927/thesis/robohive/robohive/robohive/experiment_scripts/ik_actuator_error_results',
    help='Output directory for results'
)
@click.option(
    '--radii',
    type=float,
    multiple=True,
    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    help='Sphere radii to test (meters). Can be specified multiple times.'
)
@click.option(
    '--samples_per_radius',
    type=int,
    default=100,
    help='Number of target samples per radius'
)
@click.option(
    '--sampling_mode',
    type=click.Choice(['volume', 'surface']),
    default='volume',
    help='Sample inside sphere volume or on surface only'
)
@click.option(
    '--z_min',
    type=float,
    default=0.75,
    help='Minimum z coordinate for targets (table surface constraint). Set to 0 to disable.'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducibility'
)
@click.option(
    '--ik_tolerance',
    type=float,
    default=1e-4,
    help='IK convergence tolerance'
)
@click.option(
    '--ik_max_steps',
    type=int,
    default=2000,
    help='Maximum IK iterations'
)
@click.option(
    '--ik_regularization',
    type=float,
    default=1.0,
    help='IK regularization strength'
)
@click.option(
    '--horizon',
    type=float,
    default=3.0,
    help='Trajectory duration in seconds (default: 3.0, matching robo_samples.py)'
)
@click.option(
    '--hold_time',
    type=float,
    default=0.5,
    help='Hold time at final position for settling (default: 0.5s)'
)
@click.option(
    '--plot/--no-plot',
    default=True,
    help='Generate plots'
)
@click.option(
    '--save_videos',
    is_flag=True,
    default=False,
    help='Save one MP4 video per radius for visual inspection'
)
@click.option(
    '--video_width',
    type=int,
    default=640,
    help='Video width in pixels (default: 640)'
)
@click.option(
    '--video_height',
    type=int,
    default=480,
    help='Video height in pixels (default: 480)'
)
@click.option(
    '--video_fps',
    type=int,
    default=30,
    help='Video framerate (default: 30)'
)
@click.option(
    '--video_camera',
    type=str,
    default='left_cam',
    help='MuJoCo camera name for video rendering (default: left_cam)'
)
def main(sim_path, output_dir, radii, samples_per_radius, sampling_mode,
         z_min, seed, ik_tolerance, ik_max_steps, ik_regularization,
         horizon, hold_time, plot, save_videos,
         video_width, video_height, video_fps, video_camera):
    """Run IK error analysis with actuator-based trajectory execution."""

    # Set random seed
    np.random.seed(seed)

    # Handle z_min: 0 means no constraint
    z_min_effective = z_min if z_min > 0 else None

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INVERSE KINEMATICS ERROR ANALYSIS WITH ACTUATOR DYNAMICS")
    print("=" * 80)
    print(f"Model: {sim_path}")
    print(f"Output directory: {output_dir}")
    print(f"Radii to test: {radii}")
    print(f"Samples per radius: {samples_per_radius}")
    print(f"Sampling mode: {sampling_mode}")
    print(f"Z-min constraint: {z_min_effective if z_min_effective else 'None (disabled)'}")
    print(f"Random seed: {seed}")
    print(f"\nTrajectory Parameters:")
    print(f"  Horizon: {horizon}s")
    print(f"  Hold time: {hold_time}s")
    print(f"\nIK Parameters:")
    print(f"  Tolerance: {ik_tolerance}")
    print(f"  Max steps: {ik_max_steps}")
    print(f"  Regularization: {ik_regularization}")
    print("=" * 80)

    # Load simulation
    print(f"\nLoading simulation...")
    sim = SimScene.get_sim(model_handle=sim_path)

    # Get site IDs
    ee_sid = sim.model.site_name2id(EE_SITE)
    target_sid = sim.model.site_name2id("target")

    # Set starting configuration and get EE position
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
    sim.forward()
    ee_start = sim.data.site_xpos[ee_sid].copy()

    print(f"End-effector starting position: {ee_start}")
    print(f"Starting joint configuration: {ARM_JNT0}")
    print(f"Simulation timestep: {sim.model.opt.timestep}s")

    # IK parameters
    ik_params = {
        'tol': ik_tolerance,
        'max_steps': ik_max_steps,
        'regularization_strength': ik_regularization
    }

    # Storage for results
    all_results = []

    # Run experiments for each radius
    for radius in radii:
        print(f"\n{'-' * 80}")
        print(f"Testing radius: {radius:.3f}m ({samples_per_radius} samples)")
        if save_videos and SKVIDEO_AVAILABLE:
            print(f"  → Will record video for first sample")
        print(f"{'-' * 80}")

        radius_results = []
        video_recorded_for_radius = False
        skipped_samples = 0

        for sample_idx in tqdm(range(samples_per_radius), desc=f"R={radius:.3f}m"):
            # Sample target position with z-floor constraint
            if sampling_mode == 'volume':
                target_pos = sample_sphere_target(ee_start, radius, z_min=z_min_effective)
            else:  # surface
                target_pos = sample_sphere_surface(ee_start, radius, z_min=z_min_effective)

            # Skip if no valid sample found (should be rare)
            if target_pos is None:
                skipped_samples += 1
                continue

            # Record video for first sample of each radius (if enabled)
            if save_videos and not video_recorded_for_radius and SKVIDEO_AVAILABLE:
                video_path = output_path / f"actuator_ik_video_radius_{radius:.3f}m.mp4"
                print(f"\n  Recording video: {video_path}")
                result = record_actuator_video(
                    sim, ee_sid, target_sid, ARM_JNT0, target_pos, ik_params,
                    video_path, horizon=horizon, hold_time=hold_time,
                    width=video_width, height=video_height,
                    fps=video_fps, camera_name=video_camera
                )
                video_recorded_for_radius = True
                if result:
                    print(f"  Video saved: {video_path}")
            else:
                # Run standard actuator-based IK test
                result = run_actuator_ik_test(
                    sim, ee_sid, ARM_JNT0, target_pos, ik_params,
                    horizon=horizon, hold_time=hold_time
                )

            result['radius'] = radius
            result['sample_idx'] = sample_idx
            radius_results.append(result)
            all_results.append(result)

        # Print statistics for this radius
        abs_errors = [r['absolute_error'] for r in radius_results]
        rel_errors = [r['relative_error'] for r in radius_results]
        max_dists = [r['max_distance_during_trajectory'] for r in radius_results]
        ik_successes = [r['ik_success'] for r in radius_results]

        print(f"\nResults for radius {radius:.3f}m (with actuator dynamics):")
        if skipped_samples > 0:
            print(f"  WARNING: {skipped_samples} samples skipped (below z_min={z_min_effective})")
        print(f"  Valid samples: {len(radius_results)}")
        print(f"  Final absolute error: {np.mean(abs_errors):.6f} ± {np.std(abs_errors):.6f}m")
        print(f"                        [min: {np.min(abs_errors):.6f}, max: {np.max(abs_errors):.6f}]")
        print(f"  Final relative error: {100 * np.mean(rel_errors):.3f} ± {100 * np.std(rel_errors):.3f}%")
        print(f"  Max dist during traj: {np.mean(max_dists):.6f} ± {np.std(max_dists):.6f}m")
        print(f"  IK success rate:      {100 * np.mean(ik_successes):.1f}%")

    # Save results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print("=" * 80)

    # Save raw results as JSON
    json_path = output_path / "actuator_ik_error_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'experiment_config': {
                'sim_path': sim_path,
                'radii': list(radii),
                'samples_per_radius': samples_per_radius,
                'sampling_mode': sampling_mode,
                'z_min': z_min_effective,
                'seed': seed,
                'ik_params': ik_params,
                'horizon': horizon,
                'hold_time': hold_time,
                'starting_position': ee_start.tolist(),
                'starting_joints': ARM_JNT0.tolist(),
                'execution_method': 'actuator_control'
            },
            'results': all_results
        }, f, indent=2)
    print(f"Saved raw results: {json_path}")

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_results)
    csv_path = output_path / "actuator_ik_error_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Compute summary statistics
    summary_stats = []
    for radius in radii:
        radius_data = df[df['radius'] == radius]
        summary_stats.append({
            'radius': radius,
            'n_samples': len(radius_data),
            'mean_absolute_error': radius_data['absolute_error'].mean(),
            'std_absolute_error': radius_data['absolute_error'].std(),
            'min_absolute_error': radius_data['absolute_error'].min(),
            'max_absolute_error': radius_data['absolute_error'].max(),
            'median_absolute_error': radius_data['absolute_error'].median(),
            'mean_relative_error': radius_data['relative_error'].mean(),
            'std_relative_error': radius_data['relative_error'].std(),
            'min_relative_error': radius_data['relative_error'].min(),
            'max_relative_error': radius_data['relative_error'].max(),
            'median_relative_error': radius_data['relative_error'].median(),
            'mean_max_distance': radius_data['max_distance_during_trajectory'].mean(),
            'std_max_distance': radius_data['max_distance_during_trajectory'].std(),
            'success_rate': radius_data['ik_success'].mean()
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_csv_path = output_path / "actuator_ik_error_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary statistics: {summary_csv_path}")

    # Generate plots
    if plot:
        print(f"\n{'=' * 80}")
        print("GENERATING PLOTS")
        print("=" * 80)

        # 1. Absolute error vs radius
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(
            summary_df['radius'],
            summary_df['mean_absolute_error'],
            yerr=summary_df['std_absolute_error'],
            marker='o',
            capsize=PLOT_PARAMS['euclid_capsize'],
            linewidth=PLOT_PARAMS['euclid_linewidth'],
            markersize=PLOT_PARAMS['euclid_markersize'],
            color='black',
            label='Mean ± Std'
        )
        configure_axis(ax, 'Target Radius (m)', 'Absolute Error (m)',
                      'IK Error with Actuator Dynamics vs Target Distance')
        ax.legend(fontsize=PLOT_PARAMS['legend_size'])
        plot1_path = output_path / "actuator_absolute_error_vs_radius.png"
        plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot1_path}")
        plt.close()

        # 2. Relative error vs radius
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(
            summary_df['radius'],
            100 * summary_df['mean_relative_error'],
            yerr=100 * summary_df['std_relative_error'],
            marker='o',
            capsize=PLOT_PARAMS['euclid_capsize'],
            linewidth=PLOT_PARAMS['euclid_linewidth'],
            markersize=PLOT_PARAMS['euclid_markersize'],
            color='black',
            label='Mean ± Std'
        )
        configure_axis(ax, 'Target Radius (m)', 'Relative Error (%)',
                      'IK Relative Error with Actuator Dynamics')
        ax.legend(fontsize=PLOT_PARAMS['legend_size'])
        plot2_path = output_path / "actuator_relative_error_vs_radius.png"
        plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot2_path}")
        plt.close()

        # 3. Box plot of errors by radius (stacked vertically)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Absolute error box plot
        data_by_radius = [df[df['radius'] == r]['absolute_error'].values for r in radii]
        bp1 = ax1.boxplot(data_by_radius, labels=[f"{r:.2f}" for r in radii])
        configure_axis(ax1, 'Target Radius (m)', 'Absolute Error (m)',
                      'Absolute Error Distribution (Actuator Execution)')
        ax1.tick_params(axis='both', labelsize=16)

        # Relative error box plot
        data_by_radius_rel = [100 * df[df['radius'] == r]['relative_error'].values for r in radii]
        bp2 = ax2.boxplot(data_by_radius_rel, labels=[f"{r:.2f}" for r in radii])
        configure_axis(ax2, 'Target Radius (m)', 'Relative Error (%)',
                      'Relative Error Distribution (Actuator Execution)')
        ax2.tick_params(axis='both', labelsize=16)

        plt.tight_layout()
        plot3_path = output_path / "actuator_error_distributions.png"
        plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot3_path}")
        plt.close()

        # 4. Maximum distance during trajectory (overshoot analysis)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(
            summary_df['radius'],
            summary_df['mean_max_distance'],
            yerr=summary_df['std_max_distance'],
            marker='s',
            capsize=PLOT_PARAMS['euclid_capsize'],
            linewidth=PLOT_PARAMS['euclid_linewidth'],
            markersize=PLOT_PARAMS['euclid_markersize'],
            color='black',
            label='Max distance during trajectory'
        )
        # Add reference line for target radius
        ax.plot(summary_df['radius'], summary_df['radius'], 'k--',
                linewidth=1, alpha=0.5, label='Target radius (reference)')
        configure_axis(ax, 'Target Radius (m)', 'Maximum Distance to Target (m)',
                      'Trajectory Overshoot Analysis')
        ax.legend(fontsize=PLOT_PARAMS['legend_size'])
        plot4_path = output_path / "actuator_max_distance_overshoot.png"
        plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot4_path}")
        plt.close()

        # 5. Scatter plot: target distance vs absolute error
        fig, ax = plt.subplots(figsize=(10, 6))
        for radius in radii:
            radius_data = df[df['radius'] == radius]
            ax.scatter(
                radius_data['target_distance'],
                radius_data['absolute_error'],
                alpha=0.5,
                s=20,
                label=f'R={radius:.2f}m'
            )
        configure_axis(ax, 'Target Distance (m)', 'Absolute Error (m)',
                      'Absolute Error vs Target Distance (Actuator Execution)')
        ax.legend(fontsize=PLOT_PARAMS['legend_size'])
        plot5_path = output_path / "actuator_error_scatter.png"
        plt.savefig(plot5_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot5_path}")
        plt.close()

    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {output_dir}")
    print("\nKey findings (with actuator dynamics):")
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
