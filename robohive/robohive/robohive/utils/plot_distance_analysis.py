"""
Plot distance analysis for V-JEPA CEM planning evaluation.
Usage: python plot_distance_analysis.py --experiment_type x --episode 0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import click
import os
import glob
from PIL import Image

# Central plotting configuration
PLOT_PARAMS = {
    "figsize_with_images": (10, 14),
    "figsize_plots_only": (12, 10),
    "title_size": 28,
    "subtitle_size": 28,
    "label_size": 20,
    "legend_size": 12,
    "grid_alpha": 0.3,
    "euclid_linewidth": 2.0,
    "euclid_markersize": 5,
    "euclid_capsize": 3,
    "repr_linewidth": 2.0,
    "repr_markersize": 5,
    "repr_capsize": 3,
    "threshold_linewidth": 2,
    # NEW: tick sizes
    "tick_label_size": 12,
    "tick_length": 4,
    # NEW: y-axis tick spacing
    "y_tick_step": 0.05,
}

def find_data_file(experiment_dir):
    """Find the distance summary npz file in the experiment directory."""
    # Look for files matching pattern *_distance_summary.npz
    pattern = os.path.join(experiment_dir, "*_distance_summary.npz")
    matches = glob.glob(pattern)
    
    if matches:
        print(f"Found data file: {matches[0]}")
        return matches[0]
    
    # Fallback: look for any .npz file
    pattern = os.path.join(experiment_dir, "*.npz")
    matches = glob.glob(pattern)
    
    if matches:
        print(f"Found npz file: {matches[0]}")
        return matches[0]
    
    return None

def find_planning_images_dir(experiment_dir, episode):
    """Find the planning images directory for a specific episode."""
    # Look for directory matching pattern planning_images_ep{episode}
    images_dir = os.path.join(experiment_dir, f"planning_images_ep{episode}")
    
    if os.path.isdir(images_dir):
        print(f"Found planning images directory: {images_dir}")
        return images_dir
    
    return None

@click.command()
@click.option(
    '--experiment_type',
    type=click.Choice(['x', 'y', 'z']),
    required=True,
    help='Experiment type (x, y, or z) to select reach_along_{type} directory',
)
@click.option('--episode', type=int, default=0, help='Episode number to use for picking images')
@click.option('--threshold', type=float, default=0.05, help='Success threshold in meters')
@click.option('--out_dir', type=str, default=None, help='Base directory containing reach_along_{x,y,z} subdirectories')
def main(experiment_type, episode, threshold, out_dir):
    # Resolve experiment_dir from experiment_type
    if out_dir is None:
        base_root = "/home/s185927/thesis/robohive/robohive/robohive/experiments"
    else:
        base_root = out_dir
    experiment_dir = os.path.join(base_root, f"reach_along_{experiment_type}")

    # Validate experiment directory exists
    if not os.path.isdir(experiment_dir):
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return
    
    print(f"Analyzing experiment: {experiment_dir}")
    print(f"Experiment type: {experiment_type}")
    print(f"Episode (for images only): {episode}")
    
    # Find data file
    data_path = find_data_file(experiment_dir)
    if not data_path:
        print("Error: Could not find distance summary npz file!")
        return
    
    # Find images directory
    images_dir = find_planning_images_dir(experiment_dir, episode)
    if not images_dir:
        print(f"Warning: Could not find planning images for episode {episode}")
    
    # Output directory is the experiment directory
    out_dir = experiment_dir
    
    # Load data
    data = np.load(data_path, allow_pickle=True)
    
    # Debug: print available keys
    print("Available keys in data:", list(data.keys()))
    
    # Use correct key names
    phase3_distances_per_episode = data['phase3_distances_per_episode']
    phase3_repr_l1_distances_per_episode = data.get('phase3_repr_l1_distances_per_episode', None)
    phase3_final_distances = data['phase3_final_distance']
    phase1_final_distances = data['phase1_final_distance']
    
    # ---------- helper: stack episodes and compute mean/std ----------
    def _stack_episodes_to_mean_std(ep_list):
        """Convert object-array of per-episode 1D arrays into (steps_mean, steps_std, n_episodes)."""
        cleaned = []
        for d in ep_list:
            # each d may itself be a numpy array or list, possibly nested with dtype=object
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
        trimmed = np.stack([a[:min_len] for a in cleaned], axis=0)  # [episodes, steps]
        mean = np.nanmean(trimmed, axis=0)
        std = np.nanstd(trimmed, axis=0)
        return mean, std, min_len, len(trimmed)
    # ---------------------------------------------------------------

    # Compute across-episode mean/std for distances
    dist_mean, dist_std, dist_len, dist_n = _stack_episodes_to_mean_std(phase3_distances_per_episode)
    repr_mean, repr_std, repr_len, repr_n = (None, None, None, 0)
    if phase3_repr_l1_distances_per_episode is not None:
        repr_mean, repr_std, repr_len, repr_n = _stack_episodes_to_mean_std(phase3_repr_l1_distances_per_episode)

    if dist_mean is None or dist_len is None:
        print("No valid V-JEPA distance data across episodes!")
        return

    # Decide label for dimension based on experiment_type
    dim_label = {
        'x': 'x',
        'y': 'y',
        'z': 'z',
    }[experiment_type]

    # Load images if directory provided (still use one episode just to pick images)
    start_img = None
    goal_img = None
    images_dir = find_planning_images_dir(experiment_dir, episode)
    if images_dir and os.path.isdir(images_dir):
        # Try to find start image (first step) - specific to this episode's planning
        start_path = os.path.join(images_dir, "step00_raw_current.png")
        if os.path.exists(start_path):
            start_img = np.array(Image.open(start_path))
            print(f"Loaded start image: {start_path}")
        
        # Try to find goal image - specific to this episode's planning
        goal_path = os.path.join(images_dir, "goal.png")
        if os.path.exists(goal_path):
            goal_img = np.array(Image.open(goal_path))
            print(f"Loaded goal image: {goal_path}")
    
    # Fallback: try to load from experiment root if planning_images dir not found
    if (start_img is None or goal_img is None):
        # Look for episode-specific images in experiment root
        start_path = os.path.join(experiment_dir, f"run_{episode}_start.png")
        goal_path = os.path.join(experiment_dir, f"run_{episode}_goal.png")
        
        if start_img is None and os.path.exists(start_path):
            start_img = np.array(Image.open(start_path))
            print(f"Loaded start image from root: {start_path}")
        
        if goal_img is None and os.path.exists(goal_path):
            goal_img = np.array(Image.open(goal_path))
            print(f"Loaded goal image from root: {goal_path}")
    
    # Create figure with GridSpec for flexible layout
    if start_img is not None and goal_img is not None:
        # Layout: Start/Goal images on top, Distance plots below
        fig = plt.figure(figsize=PLOT_PARAMS["figsize_with_images"])
        gs = gridspec.GridSpec(
            3,
            2,
            figure=fig,
            # make middle row taller, bottom slightly smaller
            height_ratios=[2.5, 1.6, 1.6],
            # slightly tighter spacing overall
            hspace=0.35,
            wspace=0.05,
        )
        
        ax_start = fig.add_subplot(gs[0, 0])
        ax_start.imshow(start_img)
        ax_start.set_title('Start Frame', fontsize=PLOT_PARAMS["subtitle_size"])
        ax_start.axis('off')
        
        ax_goal = fig.add_subplot(gs[0, 1])
        ax_goal.imshow(goal_img)
        ax_goal.set_title('Goal Frame', fontsize=PLOT_PARAMS["subtitle_size"])
        ax_goal.axis('off')
        
        ax_euclidean = fig.add_subplot(gs[1, :])
        ax_repr = fig.add_subplot(gs[2, :])
    else:
        fig, (ax_euclidean, ax_repr) = plt.subplots(2, 1, figsize=PLOT_PARAMS["figsize_plots_only"])
        print("Warning: Could not load start/goal images. Showing plots only.")
    
    # ---------- Euclidean distance: summary only ----------
    stat_steps = np.arange(dist_len)
    ax_euclidean.errorbar(
        stat_steps,
        dist_mean,
        yerr=dist_std,
        fmt='k-o',
        linewidth=PLOT_PARAMS["euclid_linewidth"],
        markersize=PLOT_PARAMS["euclid_markersize"],
        capsize=PLOT_PARAMS["euclid_capsize"],
        alpha=0.9,
        label=f'Mean ± std (N={dist_n} episodes)',
    )
    ax_euclidean.axhline(
        y=threshold,
        color='r',
        linestyle='--',
        linewidth=PLOT_PARAMS["threshold_linewidth"],
        label=f'Success Threshold ({threshold}m)',
    )
    ax_euclidean.set_xlabel('Step', fontsize=PLOT_PARAMS["label_size"])
    ax_euclidean.set_ylabel('Position Error (m)', fontsize=PLOT_PARAMS["label_size"])
    ax_euclidean.set_title(
        f'L2 Distance to Goal',
        fontsize=PLOT_PARAMS["title_size"]
    )
    ax_euclidean.legend(fontsize=PLOT_PARAMS["legend_size"], loc='best')
    ax_euclidean.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
    ax_euclidean.set_xticks(stat_steps)
    # NEW: y-axis ticks every 0.05
    y_min = 0.0
    y_max = max(float(np.nanmax(dist_mean + dist_std)), float(threshold))
    y_step = PLOT_PARAMS["y_tick_step"]
    ax_euclidean.set_ylim(y_min, y_max * 1.05)
    ax_euclidean.set_yticks(np.arange(y_min, y_max + y_step * 0.5, y_step))
    # existing tick params
    ax_euclidean.tick_params(
        axis='both',
        which='both',
        labelsize=PLOT_PARAMS["tick_label_size"],
        length=PLOT_PARAMS["tick_length"],
    )

    # ---------- Representation L1: summary only ----------
    if repr_mean is not None and repr_len is not None:
        stat_repr_steps = np.arange(repr_len)
        ax_repr.errorbar(
            stat_repr_steps,
            repr_mean,
            yerr=repr_std,
            fmt='k-s',
            linewidth=PLOT_PARAMS["repr_linewidth"],
            markersize=PLOT_PARAMS["repr_markersize"],
            capsize=PLOT_PARAMS["repr_capsize"],
            alpha=0.9,
            label=f'Mean ± std (N={repr_n} episodes)',
        )
        ax_repr.set_xlabel('Step', fontsize=PLOT_PARAMS["label_size"])
        ax_repr.set_ylabel('L1 Distance', fontsize=PLOT_PARAMS["label_size"])
        ax_repr.set_title(
            'L1 Distance to Goal Representation',
            fontsize=PLOT_PARAMS["title_size"]
        )
        ax_repr.legend(fontsize=PLOT_PARAMS["legend_size"])
        ax_repr.grid(True, alpha=PLOT_PARAMS["grid_alpha"])
        ax_repr.set_xticks(stat_repr_steps)
        # Let y-axis ticks follow data automatically, only set sensible limits
        repr_y_min = float(np.nanmin(repr_mean - repr_std))
        repr_y_max = float(np.nanmax(repr_mean + repr_std))
        # Add a small margin; do not enforce a fixed step
        ax_repr.set_ylim(repr_y_min - 0.05 * abs(repr_y_max - repr_y_min),
                         repr_y_max + 0.05 * abs(repr_y_max - repr_y_min))
    else:
        ax_repr.text(
            0.5,
            0.5,
            'No representation distance data available',
            ha='center',
            va='center',
            transform=ax_repr.transAxes,
            fontsize=PLOT_PARAMS["label_size"],
        )
        ax_repr.set_title(
            'L1 Distance Between Representations',
            fontsize=PLOT_PARAMS["title_size"]
        )
        # Let matplotlib choose y-ticks; use a generic limit
        ax_repr.set_ylim(0.0, 1.0)

    # existing tick params for representation axis
    ax_repr.tick_params(
        axis='both',
        which='both',
        labelsize=PLOT_PARAMS["tick_label_size"],
        length=PLOT_PARAMS["tick_length"],
    )

    # Save figure
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'comprehensive_analysis_summary_{experiment_type}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comprehensive summary plot: {output_path}")

    # Print global summary statistics instead of per-episode
    print("\n=== Across-episode Summary Statistics ===")
    print(f"Episodes used (position): {dist_n}")
    print(f"Steps in summary (position): {dist_len}")
    print(f"Initial mean distance: {dist_mean[0]:.4f}m")
    print(f"Final mean distance: {dist_mean[-1]:.4f}m")
    print(f"Initial std distance: {dist_std[0]:.4f}m")
    print(f"Final std distance: {dist_std[-1]:.4f}m")

    if repr_mean is not None:
        print(f"\nRepresentation distances (across {repr_n} episodes):")
        print(f"Initial mean L1: {repr_mean[0]:.6f} (std {repr_std[0]:.6f})")
        print(f"Final mean L1: {repr_mean[-1]:.6f} (std {repr_std[-1]:.6f})")

if __name__ == '__main__':
    main()
