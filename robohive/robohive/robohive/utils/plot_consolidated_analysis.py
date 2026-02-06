"""
Plot consolidated distance analysis for V-JEPA CEM planning evaluation across all three directions.
Usage: python plot_consolidated_analysis.py --out_dir /path/to/experiment/folder
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
    "figsize": (30, 18),
    "panel_label_size": 24,
    "step_label_size": 32,
    # Scaled font sizes for larger figure (30x15 vs 14x8)
    "label_size": 28,
    "legend_size": 26,
    "tick_label_size": 24,
    "tick_length": 8,
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

# Planning steps to show (excluding step00 which is the start frame)
PLANNING_STEPS = [1, 2, 3, 4, 5]

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
    phase3_predicted = data.get('phase3_predicted_energies_per_episode', None)

    dist_mean, dist_std, dist_len, dist_n = stack_episodes_to_mean_std(phase3_distances)
    repr_mean, repr_std, repr_len, repr_n = (None, None, None, 0)
    if phase3_repr_l1 is not None:
        repr_mean, repr_std, repr_len, repr_n = stack_episodes_to_mean_std(phase3_repr_l1)

    pred_mean, pred_std, pred_len, pred_n = (None, None, None, 0)
    if phase3_predicted is not None:
        pred_mean, pred_std, pred_len, pred_n = stack_episodes_to_mean_std(phase3_predicted)

    # Load actions for the specified episode
    phase3_actions = data.get('phase3_actions_raw_per_episode', None)
    episode_actions = None
    if phase3_actions is not None and len(phase3_actions) > episode:
        episode_actions = phase3_actions[episode]  # Shape: (n_steps, 7)

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
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'pred_len': pred_len,
        'pred_n': pred_n,
        'episode_actions': episode_actions,
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
    # Convert distances to cm for display
    all_dist_max = max(
        np.nanmax((all_data[d]['dist_mean'] + all_data[d]['dist_std']) * 100)
        for d in DIRECTIONS
    )

    all_repr_min = float('inf')
    all_repr_max = float('-inf')
    for d in DIRECTIONS:
        if all_data[d]['repr_mean'] is not None:
            rmin = np.nanmin(all_data[d]['repr_mean'] - all_data[d]['repr_std'])
            rmax = np.nanmax(all_data[d]['repr_mean'] + all_data[d]['repr_std'])
            all_repr_min = min(all_repr_min, rmin)
            all_repr_max = max(all_repr_max, rmax)
        if all_data[d]['pred_mean'] is not None:
            pmin = np.nanmin(all_data[d]['pred_mean'] - all_data[d]['pred_std'])
            pmax = np.nanmax(all_data[d]['pred_mean'] + all_data[d]['pred_std'])
            all_repr_min = min(all_repr_min, pmin)
            all_repr_max = max(all_repr_max, pmax)

    # Determine if we should show intermediate planning frames
    # Skip them when planning horizon is > 10 steps (long planning)
    max_planning_steps = max(all_data[d]['dist_len'] for d in DIRECTIONS)
    show_planning_frames = max_planning_steps <= 10

    # Create figure with GridSpec
    fig = plt.figure(figsize=PLOT_PARAMS_LOCAL["figsize"])

    # Use a 3-row outer layout: images (rows 0-1 combined), L2 plot, L1 plot
    # The first row will contain both Start/Goal and Planning sequence with tight spacing
    if show_planning_frames:
        outer_gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[3.0, 1.3, 1.3],
                                      hspace=0.12, wspace=0.15, top=0.96, bottom=0.05)
    else:
        # When not showing planning frames, adjust layout for just Start/Goal images
        outer_gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.8, 1.3, 1.3],
                                      hspace=0.15, wspace=0.15, top=0.92, bottom=0.06)

    for col_idx, direction in enumerate(DIRECTIONS):
        data = all_data[direction]

        if show_planning_frames:
            # Create nested GridSpec for image rows (Start/Goal + Planning) with tight spacing
            image_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0, col_idx],
                                                         height_ratios=[1.6, 1.0], hspace=-0.35)

            # Create nested GridSpec for the Start/Goal image row (2 images side by side)
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=image_gs[0], wspace=0.05)
        else:
            # No planning frames - use the full image row for Start/Goal only
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
        fig.text(col_center, 0.92, panel_title, fontsize=36,
                 ha='center', va='bottom')

        # Planning sequence row (steps 1-5) - only show when planning horizon <= 10
        if show_planning_frames:
            n_steps = len(PLANNING_STEPS)
            planning_gs = gridspec.GridSpecFromSubplotSpec(1, n_steps, subplot_spec=image_gs[1], wspace=0.02)
            for step_idx, step_num in enumerate(PLANNING_STEPS):
                ax_step = fig.add_subplot(planning_gs[0, step_idx])
                if data['planning_imgs'] and step_idx < len(data['planning_imgs']) and data['planning_imgs'][step_idx] is not None:
                    ax_step.imshow(data['planning_imgs'][step_idx])
                ax_step.set_title(f'$x_{step_num}$', fontsize=PLOT_PARAMS_LOCAL["step_label_size"])
                ax_step.set_xticks([])
                ax_step.set_yticks([])
                for spine in ax_step.spines.values():
                    spine.set_visible(False)
                # Add action delta text below the frame (column vector format with brackets)
                if data['episode_actions'] is not None and step_idx < len(data['episode_actions']):
                    action = data['episode_actions'][step_idx]
                    dx, dy, dz = action[0] * 100, action[1] * 100, action[2] * 100  # Convert to cm
                    action_text = f'⎛{dx:+.1f}⎞\n⎜{dy:+.1f}⎟\n⎝{dz:+.1f}⎠'
                    ax_step.set_xlabel(action_text, fontsize=24, labelpad=8, family='DejaVu Sans Mono', linespacing=0.85)
                    # Add label to the left of the first vector in the first column only
                    if step_idx == 0 and col_idx == 0:
                        ax_step.text(-0.40, -0.15, r'$a^*_{k-1}$ (cm)', fontsize=PLOT_PARAMS_LOCAL["label_size"],
                                    transform=ax_step.transAxes, ha='right', va='center', rotation=90)

        # L2 Distance plot (in cm)
        ax_euclidean = fig.add_subplot(outer_gs[1, col_idx])
        stat_steps = np.arange(data['dist_len'])
        ax_euclidean.errorbar(
            stat_steps,
            data['dist_mean'] * 100,  # Convert to cm
            yerr=data['dist_std'] * 100,  # Convert to cm
            fmt='-o',
            color=PLOT_PARAMS_LOCAL["line_color"],
            linewidth=PLOT_PARAMS_LOCAL["euclid_linewidth"],
            markersize=PLOT_PARAMS_LOCAL["euclid_markersize"],
            capsize=PLOT_PARAMS_LOCAL["euclid_capsize"],
            alpha=0.9,
            label=f'Mean ± std (N={data["dist_n"]})',
        )
        if col_idx == 0:
            ax_euclidean.set_ylabel(r'$\|p_k - p_g\|_2$ (cm)', fontsize=PLOT_PARAMS_LOCAL["label_size"])
        ax_euclidean.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
        ax_euclidean.set_xticks(stat_steps)

        # Shared y-axis limits with consistent formatting (in cm)
        y_step_cm = 5  # 5 cm steps
        ax_euclidean.set_ylim(0.0, all_dist_max * 1.05)
        y_ticks = np.arange(0.0, all_dist_max + y_step_cm * 0.5, y_step_cm)
        ax_euclidean.set_yticks(y_ticks)
        ax_euclidean.set_yticklabels([f'{y:.0f}' for y in y_ticks])
        ax_euclidean.tick_params(axis='both', labelsize=PLOT_PARAMS_LOCAL["tick_label_size"], length=PLOT_PARAMS_LOCAL["tick_length"])

        # L1 Representation plot
        ax_repr = fig.add_subplot(outer_gs[2, col_idx])
        if data['repr_mean'] is not None:
            stat_repr_steps = np.arange(data['repr_len'])
            # Plot actual measured energy
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
                label=r'$\hat{z}_k = z_k$',
            )
            # Plot predicted energy if available
            if data['pred_mean'] is not None:
                # Predicted energies are for steps 1-5 (after each action), so offset by 1
                stat_pred_steps = np.arange(1, data['pred_len'] + 1)
                ax_repr.errorbar(
                    stat_pred_steps,
                    data['pred_mean'],
                    yerr=data['pred_std'],
                    fmt='--^',
                    color='#ff7f0e',  # Orange
                    linewidth=PLOT_PARAMS_LOCAL["repr_linewidth"],
                    markersize=PLOT_PARAMS_LOCAL["repr_markersize"],
                    capsize=PLOT_PARAMS_LOCAL["repr_capsize"],
                    alpha=0.9,
                    label=r'$\hat{z}_k = P_\phi(a_{k-1}^\star; s_{k-1}, z_{k-1})$',
                )
            ax_repr.set_xlabel('Step (k)', fontsize=PLOT_PARAMS_LOCAL["label_size"])
            if col_idx == 0:
                ax_repr.set_ylabel(r'$\frac{1}{TD}\|\hat{z}_k - z_g\|_1$', fontsize=PLOT_PARAMS_LOCAL["label_size"])
                ax_repr.legend(fontsize=PLOT_PARAMS_LOCAL["legend_size"], loc='best')
            ax_repr.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
            ax_repr.set_xticks(stat_repr_steps)

            # Shared y-axis limits for representation
            margin = 0.05 * abs(all_repr_max - all_repr_min)
            y_min_repr = all_repr_min - margin
            y_max_repr = all_repr_max + margin
            ax_repr.set_ylim(y_min_repr, y_max_repr)
        else:
            ax_repr.text(0.5, 0.5, 'No repr data', ha='center', va='center', transform=ax_repr.transAxes)

        ax_repr.tick_params(axis='both', labelsize=PLOT_PARAMS_LOCAL["tick_label_size"], length=PLOT_PARAMS_LOCAL["tick_length"])

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
