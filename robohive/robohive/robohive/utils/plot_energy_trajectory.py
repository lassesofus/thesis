"""
Plot 2x6 grid showing transformed frames and energy landscapes for a single episode.

Top row: Transformed current frames (steps 0-5, where x5 is the final state)
Bottom row: Energy landscapes with arrows from center showing optimal direction (red) and chosen action (green)
           Note: No energy landscape for step 5 as no action is taken after the final state.

Usage: python -m utils.plot_energy_trajectory --experiment_dir /path/to/reach_along_x --episode 0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from PIL import Image
import click
import os
import glob

# Import shared plotting configuration
import sys
sys.path.insert(0, '/home/s185927/thesis')
from plot_config import PLOT_PARAMS

PLOT_PARAMS_LOCAL = {
    "figsize": (36, 14),
    "title_size": 20,
    "step_label_size": 54,
    "label_size": 40,
    "tick_size": 40,
    "optimal_marker_size": 46,
    "action_marker_size": 40,
    "colorbar_label_size": 46,
    "legend_fontsize": 36,
    "legend_marker_size": 44,
}

NUM_STEPS = 5  # Planning steps 0-4
NUM_FRAMES = 6  # Observation frames 0-5 (including final state)


def find_data_file(experiment_dir):
    """Find the distance summary npz file in the experiment directory."""
    pattern = os.path.join(experiment_dir, "*_distance_summary.npz")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def find_json_file(experiment_dir):
    """Find the distance summary JSON file in the experiment directory."""
    pattern = os.path.join(experiment_dir, "*_distance_summary.json")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def load_episode_data(experiment_dir, episode):
    """Load action data, images, and energy landscape data for a specific episode."""
    import json

    # Load npz data for actions
    data_path = find_data_file(experiment_dir)
    actions = None
    if data_path:
        data = np.load(data_path, allow_pickle=True)
        if 'phase3_actions_raw_per_episode' in data:
            actions_per_ep = data['phase3_actions_raw_per_episode']
            if episode < len(actions_per_ep):
                actions = actions_per_ep[episode]
                if isinstance(actions, np.ndarray) and actions.dtype == object:
                    actions = np.array(list(actions))

    # Load position error data from JSON
    position_errors = None
    json_path = find_json_file(experiment_dir)
    if json_path:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        if 'phase3_vjepa_distances' in json_data and episode < len(json_data['phase3_vjepa_distances']):
            position_errors = json_data['phase3_vjepa_distances'][episode]

    # Load images and energy data
    images_dir = os.path.join(experiment_dir, f"planning_images_ep{episode}")

    transformed_frames = []
    energy_data = []

    for step in range(NUM_FRAMES):
        # Raw frame (more consistent naming across all steps)
        if step < NUM_STEPS:
            frame_path = os.path.join(images_dir, f"step{step:02d}_raw_current.png")
        else:
            # Final frame (step 5) has different naming
            frame_path = os.path.join(images_dir, f"step{step:02d}_final.png")

        if os.path.exists(frame_path):
            transformed_frames.append(np.array(Image.open(frame_path)))
        else:
            transformed_frames.append(None)

        # Energy landscape data (only for steps 0-4, not for final state)
        if step < NUM_STEPS:
            ed_path = os.path.join(images_dir, f"step{step:02d}_energy_data.npz")
            if os.path.exists(ed_path):
                energy_data.append(np.load(ed_path, allow_pickle=True))
            else:
                energy_data.append(None)
        else:
            energy_data.append(None)  # No energy landscape for final state

    # Load goal frame
    goal_path = os.path.join(images_dir, "goal.png")
    goal_frame = None
    if os.path.exists(goal_path):
        goal_frame = np.array(Image.open(goal_path))

    return {
        'actions': actions,
        'transformed_frames': transformed_frames,
        'energy_data': energy_data,
        'goal_frame': goal_frame,
        'position_errors': position_errors,
    }


def extract_experiment_type(experiment_dir):
    """Extract experiment type (x, y, z) from directory name."""
    dirname = os.path.basename(experiment_dir)
    if 'reach_along_x' in dirname:
        return 'x'
    elif 'reach_along_y' in dirname:
        return 'y'
    elif 'reach_along_z' in dirname:
        return 'z'
    return 'x'  # default


def action_to_landscape_coords(action, experiment_type, grid_size=0.075):
    """
    Convert raw action (DROID frame) to energy landscape coordinates.

    Actions are [dx, dy, dz, droll, dpitch, dyaw, grip] in DROID frame.
    Energy landscape axes depend on experiment_type:
      - x, y: X-Y plane (axis1=X, axis2=Y)
      - z: X-Z plane (axis1=X, axis2=Z)
    """
    if action is None or len(action) < 3:
        return None

    if experiment_type in ('x', 'y'):
        # X-Y plane
        a1, a2 = action[0], action[1]
    else:  # 'z'
        # X-Z plane
        a1, a2 = action[0], action[2]

    # Clip to grid bounds
    a1 = np.clip(a1, -grid_size, grid_size)
    a2 = np.clip(a2, -grid_size, grid_size)

    return (a1, a2)


def plot_energy_landscape_with_action(ax, energy_data, action_coords, step_idx, norm, cmap,
                                       goal_frame=None, axis_labels=None, show_ylabel=True):
    """
    Plot energy landscape heatmap with arrows from center to optimal direction and chosen action.
    Uses shared normalization for consistent colormap across all panels.
    If energy_data is None and goal_frame is provided, show goal frame instead.

    Args:
        axis_labels: tuple of (xlabel, ylabel) for the heatmap axes
        show_ylabel: whether to show the y-axis label (only first column should show it)
    """
    from matplotlib.patches import FancyArrowPatch

    if energy_data is None:
        if goal_frame is not None:
            ax.imshow(goal_frame)
            ax.set_title('$x_g$', fontsize=PLOT_PARAMS_LOCAL["step_label_size"])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=PLOT_PARAMS_LOCAL["label_size"],
                    color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('lightgray')
        return None

    heatmap = energy_data['heatmap']
    axis1_edges = energy_data['axis1_edges']
    axis2_edges = energy_data['axis2_edges']
    optimal_action = energy_data['optimal_action']

    # Convert edges from meters to cm for display
    axis1_edges_cm = axis1_edges * 100
    axis2_edges_cm = axis2_edges * 100

    # Plot heatmap with shared normalization (extent in cm)
    im = ax.imshow(
        heatmap.T,
        origin='lower',
        extent=[axis1_edges_cm[0], axis1_edges_cm[-1], axis2_edges_cm[0], axis2_edges_cm[-1]],
        cmap=cmap,
        norm=norm,
        aspect='equal'
    )

    # Arrow styling parameters
    arrow_width = 8.0
    mutation_scale = 30

    # Plot arrow to optimal direction (red) - from center (0,0) to optimal action
    if optimal_action is not None and len(optimal_action) >= 2:
        opt_x, opt_y = optimal_action[0] * 100, optimal_action[1] * 100
        # Only draw arrow if optimal action is not at origin
        if abs(opt_x) > 0.1 or abs(opt_y) > 0.1:
            arrow_opt = FancyArrowPatch(
                (0, 0), (opt_x, opt_y),
                arrowstyle='->,head_width=0.4,head_length=0.3',
                color='red',
                linewidth=arrow_width,
                mutation_scale=mutation_scale,
                zorder=10
            )
            ax.add_patch(arrow_opt)

    # Plot arrow to chosen action (green) - from center (0,0) to chosen action
    if action_coords is not None:
        act_x, act_y = action_coords[0] * 100, action_coords[1] * 100
        # Only draw arrow if action is not at origin
        if abs(act_x) > 0.1 or abs(act_y) > 0.1:
            arrow_act = FancyArrowPatch(
                (0, 0), (act_x, act_y),
                arrowstyle='->,head_width=0.4,head_length=0.3',
                color='lime',
                linewidth=arrow_width,
                mutation_scale=mutation_scale,
                zorder=11
            )
            ax.add_patch(arrow_act)

    # Plot current position marker at origin (0,0)
    ax.plot(0, 0, 'o',
            markersize=20,
            color='white',
            markeredgecolor='black',
            markeredgewidth=3,
            label='Current', zorder=12)

    # Set tick labels to single digit (integer cm values)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    ax.tick_params(axis='both', labelsize=PLOT_PARAMS_LOCAL["tick_size"])

    # Add axis labels with units
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0], fontsize=PLOT_PARAMS_LOCAL["label_size"])
        if show_ylabel:
            ax.set_ylabel(axis_labels[1], fontsize=PLOT_PARAMS_LOCAL["label_size"])

    return im


def compute_global_colormap_range(energy_data_list):
    """Compute global min/max for consistent colormap across all energy landscapes."""
    vmin, vmax = float('inf'), float('-inf')

    for ed in energy_data_list:
        if ed is not None:
            heatmap = ed['heatmap']
            vmin = min(vmin, np.nanmin(heatmap))
            vmax = max(vmax, np.nanmax(heatmap))

    if vmin == float('inf'):
        vmin, vmax = 0, 1

    return vmin, vmax


@click.command()
@click.option('--experiment_dir', type=str, required=True,
              help='Directory containing reach_along_{x,y,z} experiment results')
@click.option('--episode', type=int, default=0, help='Episode number to visualize')
@click.option('--grid_size', type=float, default=0.075, help='Energy landscape grid size')
def main(experiment_dir, episode, grid_size):
    print(f"Loading data from: {experiment_dir}")
    print(f"Episode: {episode}")

    experiment_type = extract_experiment_type(experiment_dir)
    print(f"Experiment type: {experiment_type}")

    data = load_episode_data(experiment_dir, episode)

    if all(f is None for f in data['transformed_frames']):
        print("Error: No transformed frame images found")
        return

    if all(e is None for e in data['energy_data']):
        print("Error: No energy landscape data found. Re-run experiment with --visualize_energy_landscape")
        return

    # Compute global colormap range for consistent coloring
    vmin, vmax = compute_global_colormap_range(data['energy_data'])
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = 'viridis'

    # Determine axis labels based on experiment type
    if experiment_type in ('x', 'y'):
        axis_labels = (r'$\Delta x$ (cm)', r'$\Delta y$ (cm)')
    else:  # 'z'
        axis_labels = (r'$\Delta x$ (cm)', r'$\Delta z$ (cm)')

    # Create figure with 3 rows: frames, energy landscapes, position error
    # Third row spans all columns for position error plot
    has_position_errors = data['position_errors'] is not None
    if has_position_errors:
        fig = plt.figure(figsize=(PLOT_PARAMS_LOCAL["figsize"][0], PLOT_PARAMS_LOCAL["figsize"][1] + 5))
        # Use nested gridspec: outer has 2 rows (top 2 rows together, bottom row separate)
        gs_outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.1, 0.5],
                                      hspace=0.65, left=0.05, right=0.97, top=0.96, bottom=0.10)
        # Inner gridspec for top two rows (frames and heatmaps) with smaller hspace
        gs = gridspec.GridSpecFromSubplotSpec(2, NUM_FRAMES, subplot_spec=gs_outer[0],
                                               height_ratios=[1.0, 1.0], hspace=0.02, wspace=0.38)
        # Bottom row gridspec
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, NUM_FRAMES, subplot_spec=gs_outer[1], wspace=0.38)
    else:
        fig = plt.figure(figsize=PLOT_PARAMS_LOCAL["figsize"])
        gs = gridspec.GridSpec(2, NUM_FRAMES, figure=fig, height_ratios=[1.075, 1.1],
                               hspace=0.22, wspace=0.38,
                               left=0.05, right=0.97, top=0.95, bottom=0.2)

    last_im = None  # Track last image for colorbar

    for step in range(NUM_FRAMES):
        # Top row: transformed frames
        ax_frame = fig.add_subplot(gs[0, step])
        if data['transformed_frames'][step] is not None:
            ax_frame.imshow(data['transformed_frames'][step])
        ax_frame.set_title(f'$x_{step}$', fontsize=PLOT_PARAMS_LOCAL["step_label_size"])
        ax_frame.axis('off')

        # Bottom row: energy landscapes with action overlay
        ax_energy = fig.add_subplot(gs[1, step])

        # Get action taken at this step (if available)
        action_coords = None
        if step < NUM_STEPS and data['actions'] is not None and step < len(data['actions']):
            action_coords = action_to_landscape_coords(
                data['actions'][step], experiment_type, grid_size
            )

        # For the last column (step 5), show goal frame instead of energy landscape
        goal_frame_for_step = data['goal_frame'] if step == NUM_FRAMES - 1 else None

        im = plot_energy_landscape_with_action(
            ax_energy, data['energy_data'][step] if step < len(data['energy_data']) else None,
            action_coords, step, norm, cmap, goal_frame=goal_frame_for_step,
            axis_labels=axis_labels, show_ylabel=(step == 0)
        )
        if im is not None:
            last_im = im

    # Add third row for position error plot - will be repositioned later to align with frames
    if has_position_errors:
        ax_error = fig.add_subplot(gs_bottom[0, :])  # Span all columns in bottom gridspec

    # Add single shared horizontal colorbar (between heatmaps and position error plot)
    if last_im is not None:
        if has_position_errors:
            # Place colorbar closer to the heatmaps, above the position error plot
            fig.canvas.draw()
            error_pos = ax_error.get_position()
            cbar_ax = fig.add_axes([0.05, error_pos.y1 + 0.12, 0.76, 0.025])  # [left, bottom, width, height]
        else:
            # Position colorbar under the heatmaps (columns 0-4), leaving space for goal image
            cbar_ax = fig.add_axes([0.05, 0.03, 0.76, 0.04])  # [left, bottom, width, height]
        cbar = fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r'$\frac{1}{TD}\| P_\phi(\hat{a}_k; s_k, z_k) - z_g \|_1$',
                       fontsize=PLOT_PARAMS_LOCAL["colorbar_label_size"] + 4)
        cbar.ax.tick_params(labelsize=PLOT_PARAMS_LOCAL["tick_size"] + 2)

    # Continue with position error plot configuration
    if has_position_errors:
        num_steps = len(data['position_errors'])
        steps = list(range(num_steps))
        # Convert position errors from meters to cm for consistency with heatmap axes
        position_errors_cm = [e * 100 for e in data['position_errors']]
        ax_error.plot(steps, position_errors_cm, '-o', color='#1f77b4', markersize=22, linewidth=5)
        ax_error.axhline(y=5, color='r', linestyle='--', linewidth=2.5)  # 5cm threshold
        ax_error.set_ylabel(r'$\|p_k - p_g\|_2$ (cm)', fontsize=PLOT_PARAMS_LOCAL["label_size"])
        ax_error.set_xlabel('Step ($k$)', fontsize=PLOT_PARAMS_LOCAL["label_size"])
        ax_error.tick_params(axis='both', labelsize=PLOT_PARAMS_LOCAL["tick_size"])
        ax_error.set_xticks(steps)
        ax_error.set_ylim(bottom=0, top=30)  # 0-30cm
        ax_error.grid(True, alpha=0.3)

        # Reposition bottom plot so data points align with frame centers above
        # Use gridspec column positions instead of actual axes (which may be constrained by aspect ratio)
        # Get the intended column positions from gridspec
        gs_pos_first = gs[0, 0].get_position(fig)
        gs_pos_last = gs[0, NUM_FRAMES - 1].get_position(fig)
        frame_x_left = (gs_pos_first.x0 + gs_pos_first.x1) / 2
        frame_x_right = (gs_pos_last.x0 + gs_pos_last.x1) / 2

        # Add padding for tick labels while keeping data points aligned with frame centers
        padding = 0.3  # padding in data units
        data_range = num_steps - 1  # 0 to 5 = range of 5
        total_range = data_range + 2 * padding  # total range including padding

        # Calculate new plot width to accommodate padding while keeping data points aligned
        frame_width = frame_x_right - frame_x_left
        new_width = frame_width * total_range / data_range
        new_left = frame_x_left - frame_width * padding / data_range

        error_pos = ax_error.get_position()
        ax_error.set_position([new_left, error_pos.y0, new_width, error_pos.height])
        ax_error.set_xlim(-padding, num_steps - 1 + padding)

    # Add legend in lower right corner
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markersize=PLOT_PARAMS_LOCAL["legend_marker_size"] - 16, label='Current position',
               markeredgecolor='black', markeredgewidth=2, linestyle='None'),
        Line2D([0], [0], marker=r'$\rightarrow$', color='red',
               markersize=60, markeredgewidth=0, linestyle='None', label=r'Optimal $(\Delta x, \Delta y)$'),
        Line2D([0], [0], marker=r'$\rightarrow$', color='lime',
               markersize=60, markeredgewidth=0, linestyle='None', label=r'Planned $(\Delta x, \Delta y)$'),
    ]
    fig.legend(handles=legend_elements, loc='lower right',
               fontsize=PLOT_PARAMS_LOCAL["legend_fontsize"],
               bbox_to_anchor=(0.99, 0.27), frameon=False)

    # Save figure
    output_path = os.path.join(experiment_dir, f'energy_trajectory_ep{episode}.png')
    plt.savefig(output_path, dpi=200)
    print(f"\nSaved: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
