"""
Plot consolidated distance analysis for V-JEPA CEM planning evaluation across all three directions.
Usage: python plot_consolidated_analysis.py --out_dir /path/to/experiment/folder
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import click
import os
import glob
from PIL import Image

# Central plotting configuration (matching plot_distance_analysis.py)
PLOT_PARAMS = {
    "figsize": (30, 15),  # Reduced height for tighter layout
    "title_size": 26,
    "subtitle_size": 22,
    "label_size": 20,
    "legend_size": 16,
    "grid_alpha": 0.3,
    "euclid_linewidth": 2.5,
    "euclid_markersize": 7,
    "euclid_capsize": 4,
    "repr_linewidth": 2.5,
    "repr_markersize": 7,
    "repr_capsize": 4,
    "threshold_linewidth": 2.5,
    "tick_label_size": 18,
    "tick_length": 5,
    "y_tick_step": 0.05,
    "panel_label_size": 24,
    "step_label_size": 26,
}

# Planning steps to show (excluding step00 which is the start frame)
PLANNING_STEPS = [1, 2, 3, 4, 5]

# Panel labels for academic referencing
PANEL_LABELS = ['(a)', '(b)', '(c)']

DIRECTIONS = ['x', 'y', 'z']
DIRECTION_LABELS = {
    'x': 'Reach Along X',
    'y': 'Reach Along Y',
    'z': 'Reach Along Z',
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

    # Load planning step images (steps 1-5, excluding step00 which is start frame)
    planning_imgs = []
    if images_dir:
        for step in PLANNING_STEPS:
            if step == 5:
                # Final step has different naming
                step_path = os.path.join(images_dir, "step05_final.png")
            else:
                step_path = os.path.join(images_dir, f"step{step:02d}_raw_current.png")
            if os.path.exists(step_path):
                planning_imgs.append(np.array(Image.open(step_path)))
            else:
                planning_imgs.append(None)

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
        'planning_imgs': planning_imgs,
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
    fig = plt.figure(figsize=PLOT_PARAMS["figsize"])

    # Use a 3-row outer layout: images (rows 0-1 combined), L2 plot, L1 plot
    # The first row will contain both Start/Goal and Planning sequence with tight spacing
    outer_gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[2.7, 1.3, 1.3],
                                  hspace=0.3, wspace=0.15, top=0.92, bottom=0.06)

    for col_idx, direction in enumerate(DIRECTIONS):
        data = all_data[direction]

        # Create nested GridSpec for image rows (Start/Goal + Planning) with tight spacing
        image_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0, col_idx],
                                                     height_ratios=[2.0, 0.7], hspace=0.15)

        # Create nested GridSpec for the Start/Goal image row (2 images side by side)
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=image_gs[0], wspace=0.05)

        # Start image
        ax_start = fig.add_subplot(inner_gs[0, 0])
        if data['start_img'] is not None:
            ax_start.imshow(data['start_img'])
        ax_start.set_title('$x_0$', fontsize=PLOT_PARAMS["step_label_size"])
        ax_start.axis('off')

        # Goal image
        ax_goal = fig.add_subplot(inner_gs[0, 1])
        if data['goal_img'] is not None:
            ax_goal.imshow(data['goal_img'])
        ax_goal.set_title('$x_g$', fontsize=PLOT_PARAMS["step_label_size"])
        ax_goal.axis('off')

        # Add direction title with panel label above the image row
        # Calculate center x position for this column
        col_center = (inner_gs[0, 0].get_position(fig).x0 + inner_gs[0, 1].get_position(fig).x1) / 2
        panel_title = f'{PANEL_LABELS[col_idx]} {DIRECTION_LABELS[direction]}'
        fig.text(col_center, 0.96, panel_title, fontsize=PLOT_PARAMS["title_size"],
                 ha='center', va='bottom', fontweight='bold')

        # Planning sequence row (steps 1-5)
        n_steps = len(PLANNING_STEPS)
        planning_gs = gridspec.GridSpecFromSubplotSpec(1, n_steps, subplot_spec=image_gs[1], wspace=0.02)
        for step_idx, step_num in enumerate(PLANNING_STEPS):
            ax_step = fig.add_subplot(planning_gs[0, step_idx])
            if data['planning_imgs'] and step_idx < len(data['planning_imgs']) and data['planning_imgs'][step_idx] is not None:
                ax_step.imshow(data['planning_imgs'][step_idx])
            ax_step.set_title(f'$x_{step_num}$', fontsize=PLOT_PARAMS["step_label_size"])
            ax_step.set_xticks([])
            ax_step.set_yticks([])
            for spine in ax_step.spines.values():
                spine.set_visible(False)

        # L2 Distance plot
        ax_euclidean = fig.add_subplot(outer_gs[1, col_idx])
        stat_steps = np.arange(data['dist_len'])
        ax_euclidean.errorbar(
            stat_steps,
            data['dist_mean'],
            yerr=data['dist_std'],
            fmt='k-o',
            linewidth=PLOT_PARAMS["euclid_linewidth"],
            markersize=PLOT_PARAMS["euclid_markersize"],
            capsize=PLOT_PARAMS["euclid_capsize"],
            alpha=0.9,
            label=f'Mean ± std (N={data["dist_n"]})',
        )
        ax_euclidean.axhline(
            y=threshold,
            color='r',
            linestyle='--',
            linewidth=PLOT_PARAMS["threshold_linewidth"],
            label=f'Threshold ({threshold}m)',
        )
        if col_idx == 0:
            ax_euclidean.set_ylabel('(m)', fontsize=PLOT_PARAMS["label_size"])
            ax_euclidean.legend(fontsize=PLOT_PARAMS["legend_size"], loc='best')
        ax_euclidean.set_title(r'$\Vert p_k - p_g \Vert_2$', fontsize=PLOT_PARAMS["title_size"])
        ax_euclidean.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
        ax_euclidean.set_xticks(stat_steps)

        # Shared y-axis limits with consistent decimal formatting
        y_step = PLOT_PARAMS["y_tick_step"]
        ax_euclidean.set_ylim(0.0, all_dist_max * 1.05)
        y_ticks = np.arange(0.0, all_dist_max + y_step * 0.5, y_step)
        ax_euclidean.set_yticks(y_ticks)
        ax_euclidean.set_yticklabels([f'{y:.2f}' for y in y_ticks])
        ax_euclidean.tick_params(axis='both', labelsize=PLOT_PARAMS["tick_label_size"], length=PLOT_PARAMS["tick_length"])

        # L1 Representation plot
        ax_repr = fig.add_subplot(outer_gs[2, col_idx])
        if data['repr_mean'] is not None:
            stat_repr_steps = np.arange(data['repr_len'])
            ax_repr.errorbar(
                stat_repr_steps,
                data['repr_mean'],
                yerr=data['repr_std'],
                fmt='k-s',
                linewidth=PLOT_PARAMS["repr_linewidth"],
                markersize=PLOT_PARAMS["repr_markersize"],
                capsize=PLOT_PARAMS["repr_capsize"],
                alpha=0.9,
                label=f'Mean ± std (N={data["repr_n"]})',
            )
            ax_repr.set_xlabel('Step', fontsize=PLOT_PARAMS["label_size"])
            if col_idx == 0:
                #ax_repr.set_ylabel('L1 Distance', fontsize=PLOT_PARAMS["label_size"])
                ax_repr.legend(fontsize=PLOT_PARAMS["legend_size"])
            ax_repr.set_title(r'$\Vert z_k - z_g \Vert_1$', fontsize=PLOT_PARAMS["title_size"])
            ax_repr.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
            ax_repr.set_xticks(stat_repr_steps)

            # Shared y-axis limits for representation with consistent decimal formatting
            margin = 0.05 * abs(all_repr_max - all_repr_min)
            ax_repr.set_ylim(all_repr_min - margin, all_repr_max + margin)
            # Format y-tick labels with consistent decimals
            ax_repr.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        else:
            ax_repr.text(0.5, 0.5, 'No repr data', ha='center', va='center', transform=ax_repr.transAxes)
            ax_repr.set_title(r'$\Vert z_k - z_g \Vert_1$', fontsize=PLOT_PARAMS["title_size"])

        ax_repr.tick_params(axis='both', labelsize=PLOT_PARAMS["tick_label_size"], length=PLOT_PARAMS["tick_length"])

    # Save figure
    output_path = os.path.join(out_dir, 'consolidated_analysis.png')
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
