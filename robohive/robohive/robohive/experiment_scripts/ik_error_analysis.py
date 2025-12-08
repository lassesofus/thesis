#!/usr/bin/env python3
"""
Inverse Kinematics Error Analysis Experiment

This script investigates the IK solver error as a function of target distance.
For various radii M, we sample N targets uniformly on a sphere centered at the
starting end-effector position and measure the L2 error after IK convergence.

The experiment records:
- Absolute L2 error (meters)
- Relative error (as fraction of target radius)
- IK convergence statistics

USAGE:
    python ik_error_analysis.py --output_dir /path/to/results
    python ik_error_analysis.py --radii 0.05 0.1 0.15 0.2 0.25 0.3 --samples_per_radius 100
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


def sample_sphere_target(center, radius):
    """
    Sample a point uniformly inside a sphere.

    Args:
        center: Center position [x, y, z]
        radius: Sphere radius

    Returns:
        target_pos: Sampled position [x, y, z]
    """
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


def sample_sphere_surface(center, radius):
    """
    Sample a point uniformly on the surface of a sphere.

    Args:
        center: Center position [x, y, z]
        radius: Sphere radius

    Returns:
        target_pos: Sampled position [x, y, z]
    """
    phi = 2 * np.pi * np.random.rand()
    cos_theta = np.random.uniform(-1, 1)
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    direction = np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])
    return center + radius * direction


def record_ik_video(sim, ee_sid, target_sid, start_qpos, target_pos, ik_params,
                    output_path, width=640, height=480, fps=30, camera_name='left_cam'):
    """
    Record a video of the robot executing an IK solution.

    Args:
        sim: MuJoCo simulation object
        ee_sid: End-effector site ID
        target_sid: Target site ID (for visualization)
        start_qpos: Starting joint configuration
        target_pos: Target Cartesian position
        ik_params: Dictionary of IK solver parameters
        output_path: Path to save MP4 file
        width: Video width
        height: Video height
        fps: Video framerate
        camera_name: MuJoCo camera name for rendering

    Returns:
        result_dict: Dictionary with test results (same as run_ik_test)
    """
    if not SKVIDEO_AVAILABLE:
        print("Warning: skvideo not available, cannot record video")
        return None

    frames = []

    # Reset to starting configuration
    sim.data.qpos[:ARM_nJnt] = start_qpos
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
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

    # Generate smooth trajectory to execute IK solution
    sim_dt = sim.model.opt.timestep
    move_duration = 2.0  # 2 seconds to reach target
    waypoints = generate_joint_space_min_jerk(
        start=start_qpos,
        goal=ik_result.qpos[:ARM_nJnt],
        time_to_go=move_duration,
        dt=sim_dt
    )

    # Execute trajectory and record
    sim_fps = int(1.0 / sim_dt)
    steps_per_frame = max(1, sim_fps // fps)

    for idx, wp in enumerate(waypoints):
        sim.data.ctrl[:ARM_nJnt] = wp['position']
        sim.advance(render=False)
        sim.forward()

        if (idx % steps_per_frame) == 0:
            frame = sim.renderer.render_offscreen(
                width=width, height=height, camera_id=camera_name, device_id=0
            )
            frames.append(frame)

    # Hold at final position (0.5 seconds)
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
        'ik_err_norm': ik_result.err_norm,
        'ik_steps': ik_result.steps,
        'ik_success': ik_result.success,
        'joint_displacement': joint_displacement,
        'video_path': str(output_path)
    }


def run_ik_test(sim, ee_sid, start_qpos, target_pos, ik_params):
    """
    Test IK solver for a single target position.

    Args:
        sim: MuJoCo simulation object
        ee_sid: End-effector site ID
        start_qpos: Starting joint configuration
        target_pos: Target Cartesian position
        ik_params: Dictionary of IK solver parameters

    Returns:
        result_dict: Dictionary with test results
    """
    # Reset to starting configuration
    sim.data.qpos[:ARM_nJnt] = start_qpos
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
    sim.forward()

    # Get starting EE position
    start_ee_pos = sim.data.site_xpos[ee_sid].copy()

    # Calculate target distance
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

    # Apply IK solution and measure actual error
    sim.data.qpos[:ARM_nJnt] = ik_result.qpos[:ARM_nJnt]
    sim.forward()
    final_ee_pos = sim.data.site_xpos[ee_sid].copy()

    # Calculate errors
    absolute_error = np.linalg.norm(final_ee_pos - target_pos)
    relative_error = absolute_error / target_distance if target_distance > 0 else 0.0

    # Joint displacement
    joint_displacement = np.linalg.norm(ik_result.qpos[:ARM_nJnt] - start_qpos)

    return {
        'target_pos': target_pos.tolist(),
        'target_distance': target_distance,
        'start_ee_pos': start_ee_pos.tolist(),
        'final_ee_pos': final_ee_pos.tolist(),
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'ik_err_norm': ik_result.err_norm,
        'ik_steps': ik_result.steps,
        'ik_success': ik_result.success,
        'joint_displacement': joint_displacement
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
    default='/home/s185927/thesis/robohive/robohive/robohive/experiment_scripts/ik_error_results',
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
    '--plot',
    is_flag=True,
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
         seed, ik_tolerance, ik_max_steps, ik_regularization, plot,
         save_videos, video_width, video_height, video_fps, video_camera):
    """Run IK error analysis experiment."""

    # Set random seed
    np.random.seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INVERSE KINEMATICS ERROR ANALYSIS")
    print("=" * 80)
    print(f"Model: {sim_path}")
    print(f"Output directory: {output_dir}")
    print(f"Radii to test: {radii}")
    print(f"Samples per radius: {samples_per_radius}")
    print(f"Sampling mode: {sampling_mode}")
    print(f"Random seed: {seed}")
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

        for sample_idx in tqdm(range(samples_per_radius), desc=f"R={radius:.3f}m"):
            # Sample target position
            if sampling_mode == 'volume':
                target_pos = sample_sphere_target(ee_start, radius)
            else:  # surface
                target_pos = sample_sphere_surface(ee_start, radius)

            # Record video for first sample of each radius (if enabled)
            if save_videos and not video_recorded_for_radius and SKVIDEO_AVAILABLE:
                video_path = output_path / f"ik_video_radius_{radius:.3f}m.mp4"
                print(f"\n  Recording video: {video_path}")
                result = record_ik_video(
                    sim, ee_sid, target_sid, ARM_JNT0, target_pos, ik_params,
                    video_path, width=video_width, height=video_height,
                    fps=video_fps, camera_name=video_camera
                )
                video_recorded_for_radius = True
                if result:
                    print(f"  Video saved: {video_path}")
            else:
                # Run standard IK test (no video)
                result = run_ik_test(sim, ee_sid, ARM_JNT0, target_pos, ik_params)

            result['radius'] = radius
            result['sample_idx'] = sample_idx
            radius_results.append(result)
            all_results.append(result)

        # Print statistics for this radius
        abs_errors = [r['absolute_error'] for r in radius_results]
        rel_errors = [r['relative_error'] for r in radius_results]
        ik_steps = [r['ik_steps'] for r in radius_results]
        ik_successes = [r['ik_success'] for r in radius_results]

        print(f"\nResults for radius {radius:.3f}m:")
        print(f"  Absolute error: {np.mean(abs_errors):.6f} ± {np.std(abs_errors):.6f}m")
        print(f"                  [min: {np.min(abs_errors):.6f}, max: {np.max(abs_errors):.6f}]")
        print(f"  Relative error: {100 * np.mean(rel_errors):.3f} ± {100 * np.std(rel_errors):.3f}%")
        print(f"                  [min: {100 * np.min(rel_errors):.3f}%, max: {100 * np.max(rel_errors):.3f}%]")
        print(f"  IK iterations:  {np.mean(ik_steps):.1f} ± {np.std(ik_steps):.1f}")
        print(f"  Success rate:   {100 * np.mean(ik_successes):.1f}%")

    # Save results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print("=" * 80)

    # Save raw results as JSON
    json_path = output_path / "ik_error_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'experiment_config': {
                'sim_path': sim_path,
                'radii': list(radii),
                'samples_per_radius': samples_per_radius,
                'sampling_mode': sampling_mode,
                'seed': seed,
                'ik_params': ik_params,
                'starting_position': ee_start.tolist(),
                'starting_joints': ARM_JNT0.tolist()
            },
            'results': all_results
        }, f, indent=2)
    print(f"Saved raw results: {json_path}")

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_results)
    csv_path = output_path / "ik_error_results.csv"
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
            'mean_ik_steps': radius_data['ik_steps'].mean(),
            'std_ik_steps': radius_data['ik_steps'].std(),
            'success_rate': radius_data['ik_success'].mean()
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_csv_path = output_path / "ik_error_summary.csv"
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
            capsize=5,
            label='Mean ± Std'
        )
        ax.set_xlabel('Target Radius (m)', fontsize=12)
        ax.set_ylabel('Absolute Error (m)', fontsize=12)
        ax.set_title('IK Absolute Error vs Target Distance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot1_path = output_path / "absolute_error_vs_radius.png"
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
            capsize=5,
            color='orange',
            label='Mean ± Std'
        )
        ax.set_xlabel('Target Radius (m)', fontsize=12)
        ax.set_ylabel('Relative Error (%)', fontsize=12)
        ax.set_title('IK Relative Error vs Target Distance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot2_path = output_path / "relative_error_vs_radius.png"
        plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot2_path}")
        plt.close()

        # 3. Box plot of errors by radius
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Absolute error box plot
        data_by_radius = [df[df['radius'] == r]['absolute_error'].values for r in radii]
        bp1 = ax1.boxplot(data_by_radius, labels=[f"{r:.2f}" for r in radii])
        ax1.set_xlabel('Target Radius (m)', fontsize=12)
        ax1.set_ylabel('Absolute Error (m)', fontsize=12)
        ax1.set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Relative error box plot
        data_by_radius_rel = [100 * df[df['radius'] == r]['relative_error'].values for r in radii]
        bp2 = ax2.boxplot(data_by_radius_rel, labels=[f"{r:.2f}" for r in radii])
        ax2.set_xlabel('Target Radius (m)', fontsize=12)
        ax2.set_ylabel('Relative Error (%)', fontsize=12)
        ax2.set_title('Relative Error Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plot3_path = output_path / "error_distributions.png"
        plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot3_path}")
        plt.close()

        # 4. IK convergence statistics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # IK iterations
        ax1.errorbar(
            summary_df['radius'],
            summary_df['mean_ik_steps'],
            yerr=summary_df['std_ik_steps'],
            marker='s',
            capsize=5,
            color='green',
            label='Mean ± Std'
        )
        ax1.set_xlabel('Target Radius (m)', fontsize=12)
        ax1.set_ylabel('IK Iterations', fontsize=12)
        ax1.set_title('IK Convergence Iterations', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Success rate
        ax2.plot(
            summary_df['radius'],
            100 * summary_df['success_rate'],
            marker='o',
            color='red',
            linewidth=2
        )
        ax2.set_xlabel('Target Radius (m)', fontsize=12)
        ax2.set_ylabel('Success Rate (%)', fontsize=12)
        ax2.set_title('IK Success Rate', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])

        plot4_path = output_path / "ik_convergence_stats.png"
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
        ax.set_xlabel('Target Distance (m)', fontsize=12)
        ax.set_ylabel('Absolute Error (m)', fontsize=12)
        ax.set_title('Absolute Error vs Target Distance (All Samples)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot5_path = output_path / "error_scatter.png"
        plt.savefig(plot5_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot5_path}")
        plt.close()

    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {output_dir}")
    print("\nKey findings:")
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
