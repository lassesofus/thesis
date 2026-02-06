"""
Plot consolidated distance analysis for V-JEPA CEM planning evaluation across all three directions.
Modified version for 10-step (long) planning experiments.
Usage: python plot_consolidated_analysis_long.py --out_dir /path/to/experiment/folder
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import click
import os
import sys
import glob
from PIL import Image

# Import shared plotting configuration
sys.path.insert(0, '/home/s185927/thesis')
from plot_config import PLOT_PARAMS, apply_plot_params

# Additional parameters specific to this plot (scaled up for larger figure)
PLOT_PARAMS_LOCAL = {
    "figsize": (30, 15),
    "panel_label_size": 24,
    "step_label_size": 32,
    # Scaled font sizes for larger figure (30x15 vs 14x8)
    "label_size": 28,
    "legend_size": 20,
    "tick_label_size": 18,
    # Scaled line widths for larger figure
    "euclid_linewidth": 3.0,
    "euclid_markersize": 8,
    "euclid_capsize": 5,
    "repr_linewidth": 3.0,
    "repr_markersize": 8,
    "repr_capsize": 5,
    "threshold_linewidth": 3,
    # Match left_cam color from camera_comparison plot
    "line_color": '#1f77b4',
}

# Planning steps to show (steps 1-10, excluding step00 which is the start frame)
PLANNING_STEPS = list(range(1, 11))

# Panel labels for academic referencing
PANEL_LABELS = ['(a)', '(b)', '(c)']

DIRECTIONS = ['x', 'y', 'z']
DIRECTION_LABELS = {
    'x': 'Reach along x',
    'y': 'Reach along y',
    'z': 'Reach along z',
}


def find_data_file(experiment_dir):
    """Find the distance summary npz file in the experiment directory."""
    pattern = os.path.join(experiment_dir, "*_distance_summary.npz")
    matches = glob.glob(pattern)

    if matches:
        return matches[0]

    # Fallback: look for any .npz file
    pattern = os.path.join(experiment_dir, "*.npz")
    matches = glob.glob(pattern)

    if matches:
        return matches[0]

    return None


def find_planning_images_dir(experiment_dir, episode):
    """Find the planning images directory for a specific episode."""
    images_dir = os.path.join(experiment_dir, f"planning_images_ep{episode}")

    if os.path.isdir(images_dir):
        return images_dir

    return None


def stack_episodes_to_mean_std(ep_list):
    """Convert object-array of per-episode 1D arrays into (steps_mean, steps_std, n_steps, n_episodes)."""
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
        return None, None, None, 0
    # truncate to minimum length across episodes so we can stack
    min_len = min(len(a) for a in cleaned)
    trimmed = np.stack([a[:min_len] for a in cleaned], axis=0)
    mean = np.nanmean(trimmed, axis=0)
    std = np.nanstd(trimmed, axis=0)
    return mean, std, min_len, len(trimmed)


def load_direction_data(base_dir, direction, episode):
    """Load all data for a single direction."""
    experiment_dir = os.path.join(base_dir, f"reach_along_{direction}")

    if not os.path.isdir(experiment_dir):
        print(f"Warning: Directory not found: {experiment_dir}")
        return None

    data_path = find_data_file(experiment_dir)
    if not data_path:
        print(f"Warning: No data file found in {experiment_dir}")
        return None

    data = np.load(data_path, allow_pickle=True)

    # Compute mean/std for distances
    phase3_distances = data['phase3_distances_per_episode']
    phase3_repr_l1 = data.get('phase3_repr_l1_distances_per_episode', None)

    dist_mean, dist_std, dist_len, dist_n = stack_episodes_to_mean_std(phase3_distances)
    repr_mean, repr_std, repr_len, repr_n = (None, None, None, 0)
    if phase3_repr_l1 is not None:
        repr_mean, repr_std, repr_len, repr_n = stack_episodes_to_mean_std(phase3_repr_l1)

    # Load images
    start_img = None
    goal_img = None

    images_dir = find_planning_images_dir(experiment_dir, episode)
    if images_dir:
        start_path = os.path.join(images_dir, "step00_raw_current.png")
        if os.path.exists(start_path):
            start_img = np.array(Image.open(start_path))

        goal_path = os.path.join(images_dir, "goal.png")
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

    # Load final step image (step 10)
    final_img = None
    if images_dir:
        final_path = os.path.join(images_dir, "step10_final.png")
        if os.path.exists(final_path):
            final_img = np.array(Image.open(final_path))

    return {
        'dist_mean': dist_mean,
        'dist_std': dist_std,
        'dist_len': dist_len,
        'dist_n': dist_n,
        'repr_mean': repr_mean,
        'repr_std': repr_std,
        'repr_len': repr_len,
        'repr_n': repr_n,
        'start_img': start_img,
        'goal_img': goal_img,
        'final_img': final_img,
    }


@click.command()
@click.option('--out_dir', type=str, required=True, help='Base directory containing reach_along_{x,y,z} subdirectories')
@click.option('--episode', type=int, default=0, help='Episode number to use for picking images')
@click.option('--threshold', type=float, default=0.05, help='Success threshold in meters')
def main(out_dir, episode, threshold):
    print(f"Loading data from: {out_dir}")
    print(f"Episode (for images): {episode}")

    # Load data for all directions
    all_data = {}
    for direction in DIRECTIONS:
        print(f"\nLoading {direction} direction...")
        data = load_direction_data(out_dir, direction, episode)
        if data is None or data['dist_mean'] is None:
            print(f"Error: Could not load valid data for direction {direction}")
            return
        all_data[direction] = data

    # Compute shared y-axis limits for fair comparison
    all_dist_max = max(
        np.nanmax(all_data[d]['dist_mean'] + all_data[d]['dist_std'])
        for d in DIRECTIONS
    )
    all_dist_max = max(all_dist_max, threshold)

    all_repr_min = float('inf')
    all_repr_max = float('-inf')
    for d in DIRECTIONS:
        if all_data[d]['repr_mean'] is not None:
            rmin = np.nanmin(all_data[d]['repr_mean'] - all_data[d]['repr_std'])
            rmax = np.nanmax(all_data[d]['repr_mean'] + all_data[d]['repr_std'])
            all_repr_min = min(all_repr_min, rmin)
            all_repr_max = max(all_repr_max, rmax)

    # Create figure with GridSpec
    fig = plt.figure(figsize=(30, 12))

    # Use a 3-row outer layout: Start/Goal images, L2 plot, L1 plot
    outer_gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.8, 1.3, 1.3],
                                  hspace=0.25, wspace=0.15, top=0.92, bottom=0.06)

    for col_idx, direction in enumerate(DIRECTIONS):
        data = all_data[direction]

        # Create nested GridSpec for Start/Goal images (2 images side by side)
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0, col_idx], wspace=0.05)

        # Start image
        ax_start = fig.add_subplot(inner_gs[0, 0])
        if data['start_img'] is not None:
            ax_start.imshow(data['start_img'])
        ax_start.set_title('$x_0$', fontsize=PLOT_PARAMS_LOCAL["step_label_size"])
        ax_start.axis('off')

        # Goal image
        ax_goal = fig.add_subplot(inner_gs[0, 1])
        if data['goal_img'] is not None:
            ax_goal.imshow(data['goal_img'])
        ax_goal.set_title('$x_g$', fontsize=PLOT_PARAMS_LOCAL["step_label_size"])
        ax_goal.axis('off')

        # Add direction title with panel label above the image row
        # Calculate center x position for this column
        col_center = (inner_gs[0, 0].get_position(fig).x0 + inner_gs[0, 1].get_position(fig).x1) / 2
        panel_title = f'{PANEL_LABELS[col_idx]} {DIRECTION_LABELS[direction]}'
        fig.text(col_center, 0.96, panel_title, fontsize=36,
                 ha='center', va='bottom')

        # L2 Distance plot
        ax_euclidean = fig.add_subplot(outer_gs[1, col_idx])
        stat_steps = np.arange(data['dist_len'])
        ax_euclidean.errorbar(
            stat_steps,
            data['dist_mean'],
            yerr=data['dist_std'],
            fmt='-o',
            color=PLOT_PARAMS_LOCAL["line_color"],
            linewidth=PLOT_PARAMS_LOCAL["euclid_linewidth"],
            markersize=PLOT_PARAMS_LOCAL["euclid_markersize"],
            capsize=PLOT_PARAMS_LOCAL["euclid_capsize"],
            alpha=0.9,
            label=f'Mean ± std (N={data["dist_n"]})',
        )
        ax_euclidean.axhline(
            y=threshold,
            color='r',
            linestyle='--',
            linewidth=PLOT_PARAMS_LOCAL["threshold_linewidth"],
            label=f'Threshold ({threshold}m)',
        )
        if col_idx == 0:
            ax_euclidean.set_ylabel(r'$\|p_k - p_g\|_2$ (m)', fontsize=PLOT_PARAMS_LOCAL["label_size"])
            ax_euclidean.legend(fontsize=PLOT_PARAMS_LOCAL["legend_size"], loc='best')
        ax_euclidean.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
        ax_euclidean.set_xticks(stat_steps)

        # Shared y-axis limits with consistent decimal formatting
        y_step = PLOT_PARAMS["y_tick_step"]
        ax_euclidean.set_ylim(0.0, all_dist_max * 1.05)
        y_ticks = np.arange(0.0, all_dist_max + y_step * 0.5, y_step)
        ax_euclidean.set_yticks(y_ticks)
        ax_euclidean.set_yticklabels([f'{y:.2f}' for y in y_ticks])
        ax_euclidean.tick_params(axis='both', labelsize=PLOT_PARAMS_LOCAL["tick_label_size"], length=PLOT_PARAMS["tick_length"])

        # L1 Representation plot
        ax_repr = fig.add_subplot(outer_gs[2, col_idx])
        if data['repr_mean'] is not None:
            stat_repr_steps = np.arange(data['repr_len'])
            ax_repr.errorbar(
                stat_repr_steps,
                data['repr_mean'],
                yerr=data['repr_std'],
                fmt='-s',
                color=PLOT_PARAMS_LOCAL["line_color"],
                linewidth=PLOT_PARAMS_LOCAL["repr_linewidth"],
                markersize=PLOT_PARAMS_LOCAL["repr_markersize"],
                capsize=PLOT_PARAMS_LOCAL["repr_capsize"],
                alpha=0.9,
                label=f'Mean ± std (N={data["repr_n"]})',
            )
            ax_repr.set_xlabel('Step (k)', fontsize=PLOT_PARAMS_LOCAL["label_size"])
            if col_idx == 0:
                ax_repr.set_ylabel(r'$\|z_k - z_g\|_1$', fontsize=PLOT_PARAMS_LOCAL["label_size"])
                ax_repr.legend(fontsize=PLOT_PARAMS_LOCAL["legend_size"])
            ax_repr.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
            ax_repr.set_xticks(stat_repr_steps)

            # Shared y-axis limits for representation with consistent decimal formatting
            margin = 0.05 * abs(all_repr_max - all_repr_min)
            y_min_repr = all_repr_min - margin
            y_max_repr = all_repr_max + margin
            ax_repr.set_ylim(y_min_repr, y_max_repr)
            # Set y-ticks with 0.05 increments
            y_ticks_repr = np.arange(np.floor(y_min_repr / 0.05) * 0.05, y_max_repr + 0.025, 0.05)
            ax_repr.set_yticks(y_ticks_repr)
            ax_repr.set_yticklabels([f'{y:.2f}' for y in y_ticks_repr])
        else:
            ax_repr.text(0.5, 0.5, 'No repr data', ha='center', va='center', transform=ax_repr.transAxes)

        ax_repr.tick_params(axis='both', labelsize=PLOT_PARAMS_LOCAL["tick_label_size"], length=PLOT_PARAMS["tick_length"])

    # Save figure
    output_path = os.path.join(out_dir, 'consolidated_analysis_long_planning.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved consolidated plot: {output_path}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for direction in DIRECTIONS:
        data = all_data[direction]
        print(f"\n{DIRECTION_LABELS[direction]}:")
        print(f"  Episodes: {data['dist_n']}")
        print(f"  Initial distance: {data['dist_mean'][0]:.4f}m (std: {data['dist_std'][0]:.4f})")
        print(f"  Final distance: {data['dist_mean'][-1]:.4f}m (std: {data['dist_std'][-1]:.4f})")
        if data['repr_mean'] is not None:
            print(f"  Initial L1 repr: {data['repr_mean'][0]:.4f} (std: {data['repr_std'][0]:.4f})")
            print(f"  Final L1 repr: {data['repr_mean'][-1]:.4f} (std: {data['repr_std'][-1]:.4f})")


if __name__ == '__main__':
    main()
