"""
Plot camera comparison analysis for V-JEPA CEM planning evaluation.

Creates a consolidated figure comparing planning performance across different
camera angles, with all cameras shown on the same axes for direct comparison.

Usage:
    python plot_camera_comparison.py --base_dir /path/to/experiments --cameras left_cam front_cam right_cam
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import click
import os
import glob
import sys
from PIL import Image
from collections import defaultdict

# Import shared plotting configuration
sys.path.insert(0, '/home/s185927/thesis')
from plot_config import PLOT_PARAMS, apply_plot_params

# Color palette for different cameras
CAMERA_COLORS = {
    'left_cam': '#1f77b4',       # Blue (baseline)
    'front_cam': '#ff7f0e',      # Orange
    'right_cam': '#2ca02c',      # Green
    'back_cam': '#d62728',       # Red
    'top_cam': '#9467bd',        # Purple
    'front_right_cam': '#8c564b', # Brown
    'back_left_cam': '#e377c2',   # Pink
    'left_high_cam': '#7f7f7f',   # Gray
    'left_low_cam': '#bcbd22',    # Yellow-green
}

# Display names for cameras
CAMERA_DISPLAY_NAMES = {
    'left_cam': 'Left cam',
    'front_cam': 'Right cam',
    'left_high_cam': 'High cam',
}

# Marker styles for different cameras
CAMERA_MARKERS = {
    'left_cam': 'o',
    'front_cam': 's',
    'right_cam': '^',
    'back_cam': 'v',
    'top_cam': 'D',
    'front_right_cam': 'p',
    'back_left_cam': 'h',
    'left_high_cam': '*',
    'left_low_cam': 'X',
}


def find_data_file(experiment_dir):
    """Find the distance summary npz file in the experiment directory."""
    pattern = os.path.join(experiment_dir, "*_distance_summary.npz")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def load_camera_data(base_dir, camera_name, experiment_type, use_camera_subdir=True):
    """Load data for a specific camera and experiment type."""
    if use_camera_subdir:
        experiment_dir = os.path.join(base_dir, camera_name, f"reach_along_{experiment_type}")
    else:
        experiment_dir = os.path.join(base_dir, f"reach_along_{experiment_type}")

    if not os.path.isdir(experiment_dir):
        print(f"  Warning: Directory not found: {experiment_dir}")
        return None

    data_path = find_data_file(experiment_dir)
    if not data_path:
        print(f"  Warning: No data file found in {experiment_dir}")
        return None

    data = np.load(data_path, allow_pickle=True)
    return data


def stack_episodes_to_mean_std(ep_list):
    """Convert object-array of per-episode 1D arrays into (mean, std, n_steps, n_episodes)."""
    cleaned = []
    for d in ep_list:
        if isinstance(d, np.ndarray) and d.dtype == object:
            inner = d[0] if hasattr(d[0], '__iter__') and not isinstance(d[0], str) else d
            arr = np.array(inner, dtype=float)
        else:
            arr = np.array(d, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            continue
        cleaned.append(arr)

    if len(cleaned) == 0:
        return None, None, 0, 0

    min_len = min(len(a) for a in cleaned)
    trimmed = np.stack([a[:min_len] for a in cleaned], axis=0)
    mean = np.nanmean(trimmed, axis=0)
    std = np.nanstd(trimmed, axis=0)
    return mean, std, min_len, len(trimmed)


def load_sample_images(base_dir, camera_name, experiment_type, episode=0, use_camera_subdir=True):
    """Load sample start/goal images for a camera."""
    if use_camera_subdir:
        experiment_dir = os.path.join(base_dir, camera_name, f"reach_along_{experiment_type}")
    else:
        experiment_dir = os.path.join(base_dir, f"reach_along_{experiment_type}")

    start_img = None
    goal_img = None

    # Try planning images directory first
    images_dir = os.path.join(experiment_dir, f"planning_images_ep{episode}")
    if os.path.isdir(images_dir):
        start_path = os.path.join(images_dir, "step00_raw_current.png")
        goal_path = os.path.join(images_dir, "goal.png")

        if os.path.exists(start_path):
            start_img = np.array(Image.open(start_path))
        if os.path.exists(goal_path):
            goal_img = np.array(Image.open(goal_path))

    # Fallback to experiment root
    if start_img is None:
        start_path = os.path.join(experiment_dir, f"run_{episode}_start.png")
        if os.path.exists(start_path):
            start_img = np.array(Image.open(start_path))

    if goal_img is None:
        goal_path = os.path.join(experiment_dir, f"run_{episode}_goal.png")
        if os.path.exists(goal_path):
            goal_img = np.array(Image.open(goal_path))

    return start_img, goal_img


@click.command()
@click.option('--base_dir', type=str, required=True, help='Base directory containing camera subdirectories')
@click.option('--cameras', type=str, multiple=True, required=True, help='Camera names to compare')
@click.option('--threshold', type=float, default=0.05, help='Success threshold in meters')
@click.option('--output_name', type=str, default='camera_comparison_analysis.png', help='Output filename')
@click.option('--camera_path', type=(str, str), multiple=True, help='Override path for a camera: --camera_path left_cam /path/to/data')
def main(base_dir, cameras, threshold, output_name, camera_path):
    """Generate comparison plot for multiple cameras."""

    print(f"Loading data from: {base_dir}")
    print(f"Comparing cameras: {list(cameras)}")

    # Build camera path overrides dict
    camera_path_overrides = dict(camera_path)
    if camera_path_overrides:
        print(f"Camera path overrides: {camera_path_overrides}")

    experiment_types = ['x', 'y', 'z']
    subplot_labels = ['(a)', '(b)', '(c)']

    # Data structure: {experiment_type: {camera: (mean, std, n_steps, n_episodes)}}
    all_data = defaultdict(dict)
    all_repr_data = defaultdict(dict)
    # sample_images[camera] = start_img (from first experiment type with data)
    sample_images = {}

    # Load data for all cameras and experiment types
    for camera in cameras:
        # Use override path if provided, otherwise use base_dir/camera
        if camera in camera_path_overrides:
            camera_base_dir = camera_path_overrides[camera]
            use_camera_subdir = False  # Override paths point directly to experiment data
            print(f"\nLoading data for camera: {camera} (from {camera_base_dir})")
        else:
            camera_base_dir = base_dir
            use_camera_subdir = True
            print(f"\nLoading data for camera: {camera}")
        for exp_type in experiment_types:
            data = load_camera_data(camera_base_dir, camera, exp_type, use_camera_subdir)
            if data is not None:
                # Position distances
                phase3_distances = data['phase3_distances_per_episode']
                mean, std, n_steps, n_episodes = stack_episodes_to_mean_std(phase3_distances)
                if mean is not None:
                    all_data[exp_type][camera] = (mean, std, n_steps, n_episodes)
                    print(f"  {exp_type}: {n_episodes} episodes, {n_steps} steps")

                # Representation distances
                if 'phase3_repr_l1_distances_per_episode' in data:
                    repr_distances = data['phase3_repr_l1_distances_per_episode']
                    r_mean, r_std, r_steps, r_eps = stack_episodes_to_mean_std(repr_distances)
                    if r_mean is not None:
                        all_repr_data[exp_type][camera] = (r_mean, r_std, r_steps, r_eps)

            # Load sample start and goal images for each camera (only need one per camera)
            if camera not in sample_images:
                start_img, goal_img = load_sample_images(camera_base_dir, camera, exp_type, use_camera_subdir=use_camera_subdir)
                if start_img is not None:
                    sample_images[camera] = (start_img, goal_img)

    # Compute shared y-axis limits for each row
    pos_max = 0
    for exp_type in experiment_types:
        for camera in cameras:
            if exp_type in all_data and camera in all_data[exp_type]:
                mean, std, _, _ = all_data[exp_type][camera]
                pos_max = max(pos_max, np.max(mean + std))

    repr_min = float('inf')
    repr_max = 0
    for exp_type in experiment_types:
        for camera in cameras:
            if exp_type in all_repr_data and camera in all_repr_data[exp_type]:
                mean, std, _, _ = all_repr_data[exp_type][camera]
                repr_max = max(repr_max, np.max(mean + std))
                repr_min = min(repr_min, np.min(mean - std))

    # Create main figure with 2 rows x 3 columns
    # Row 1: Position error for x, y, z
    # Row 2: Representation distance for x, y, z
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.2)

    # Track handles/labels for shared legend
    legend_handles = []
    legend_labels = []

    # Row 1: Position error plots
    for col, exp_type in enumerate(experiment_types):
        ax = fig.add_subplot(gs[0, col])

        if exp_type in all_data:
            for camera in cameras:
                if camera in all_data[exp_type]:
                    mean, std, n_steps, n_episodes = all_data[exp_type][camera]
                    steps = np.arange(n_steps)

                    color = CAMERA_COLORS.get(camera, '#000000')
                    marker = CAMERA_MARKERS.get(camera, 'o')
                    display_name = CAMERA_DISPLAY_NAMES.get(camera, camera)

                    line = ax.errorbar(
                        steps, mean, yerr=std,
                        fmt=f'-{marker}',
                        color=color,
                        linewidth=PLOT_PARAMS["euclid_linewidth"],
                        markersize=PLOT_PARAMS["euclid_markersize"],
                        capsize=PLOT_PARAMS["euclid_capsize"],
                        label=f'{display_name} (N={n_episodes})',
                    )

                    # Collect legend handles from first plot only
                    if col == 0:
                        legend_handles.append(line)
                        legend_labels.append(f'{display_name} (N={n_episodes})')

        # Add threshold line
        thresh_line = ax.axhline(
            y=threshold,
            color='r',
            linestyle='--',
            linewidth=PLOT_PARAMS["threshold_linewidth"],
            label=f'Threshold ({threshold}m)',
        )
        if col == 0:
            legend_handles.append(thresh_line)
            legend_labels.append(f'Threshold ({threshold}m)')

        if col == 0:
            ax.set_ylabel(r'$\|p_k - p_g\|_2$ (m)', fontsize=PLOT_PARAMS["label_size"])
        ax.set_title(f'{subplot_labels[col]} Reach along {exp_type}',
                     fontsize=PLOT_PARAMS["subtitle_size"])
        apply_plot_params(ax)
        ax.set_ylim(0, pos_max * 1.1)

    # Row 2: Representation distance plots
    for col, exp_type in enumerate(experiment_types):
        ax = fig.add_subplot(gs[1, col])

        if exp_type in all_repr_data:
            for camera in cameras:
                if camera in all_repr_data[exp_type]:
                    mean, std, n_steps, n_episodes = all_repr_data[exp_type][camera]
                    steps = np.arange(n_steps)

                    color = CAMERA_COLORS.get(camera, '#000000')
                    marker = CAMERA_MARKERS.get(camera, 'o')

                    ax.errorbar(
                        steps, mean, yerr=std,
                        fmt=f'-{marker}',
                        color=color,
                        linewidth=PLOT_PARAMS["repr_linewidth"],
                        markersize=PLOT_PARAMS["repr_markersize"],
                        capsize=PLOT_PARAMS["repr_capsize"],
                    )

        ax.set_xlabel('Step (k)', fontsize=PLOT_PARAMS["label_size"])
        if col == 0:
            ax.set_ylabel(r'$\|z_k - z_g\|_1$', fontsize=PLOT_PARAMS["label_size"])
        apply_plot_params(ax)
        if repr_max > 0:
            margin = (repr_max - repr_min) * 0.1
            ax.set_ylim(repr_min - margin, repr_max + margin)

    # Add shared legend at the bottom
    fig.legend(legend_handles, legend_labels,
               loc='lower center',
               ncol=len(legend_handles),
               fontsize=PLOT_PARAMS["legend_size"] + 2,
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Save main figure
    output_path = os.path.join(base_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot: {output_path}")

    # Create separate figure for camera images (2 rows: start x_0 and goal x_g)
    if sample_images:
        n_cameras = len(sample_images)
        fig_images, axes_images = plt.subplots(2, n_cameras, figsize=(4 * n_cameras, 8))
        if n_cameras == 1:
            axes_images = axes_images.reshape(2, 1)

        for idx, (camera, (start_img, goal_img)) in enumerate(sample_images.items()):
            display_name = CAMERA_DISPLAY_NAMES.get(camera, camera.replace('_', ' '))

            # Row 0: Start image (x_0)
            ax_start = axes_images[0, idx]
            if start_img is not None:
                ax_start.imshow(start_img)
            ax_start.axis('off')
            ax_start.set_title(display_name, fontsize=PLOT_PARAMS["subtitle_size"])

            # Row 1: Goal image (x_g)
            ax_goal = axes_images[1, idx]
            if goal_img is not None:
                ax_goal.imshow(goal_img)
            ax_goal.axis('off')

        # Add row labels
        fig_images.text(0.02, 0.75, '$x_0$', fontsize=PLOT_PARAMS["subtitle_size"], ha='center', va='center')
        fig_images.text(0.02, 0.25, '$x_g$', fontsize=PLOT_PARAMS["subtitle_size"], ha='center', va='center')

        # Add figure caption
        fig_images.text(0.5, 0.02, 'Starting frame ($x_0$) and goal frame ($x_g$) for planning along the x direction.',
                       fontsize=PLOT_PARAMS["legend_size"] + 2, ha='center', va='bottom', style='italic')

        plt.tight_layout(rect=[0.04, 0.06, 1, 1])

        # Save camera images figure
        images_output_name = output_name.replace('.png', '_camera_views.png')
        images_output_path = os.path.join(base_dir, images_output_name)
        plt.savefig(images_output_path, dpi=300, bbox_inches='tight')
        print(f"Saved camera views: {images_output_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    for exp_type in experiment_types:
        print(f"\n--- Reach along {exp_type.upper()} ---")
        if exp_type in all_data:
            for camera in cameras:
                if camera in all_data[exp_type]:
                    mean, std, n_steps, n_eps = all_data[exp_type][camera]
                    print(f"  {camera}:")
                    print(f"    Episodes: {n_eps}, Steps: {n_steps}")
                    print(f"    Initial: {mean[0]:.4f} +/- {std[0]:.4f} m")
                    print(f"    Final:   {mean[-1]:.4f} +/- {std[-1]:.4f} m")
                    print(f"    Improvement: {(mean[0] - mean[-1]):.4f} m ({100*(mean[0]-mean[-1])/mean[0]:.1f}%)")


if __name__ == '__main__':
    main()
