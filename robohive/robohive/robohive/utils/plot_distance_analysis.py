"""
Plot distance analysis for V-JEPA CEM planning evaluation.
Usage: python plot_distance_analysis.py --data_path ./distance_summary.npz
"""

import numpy as np
import matplotlib.pyplot as plt
import click
import os

@click.command()
@click.option('--data_path', type=str, required=True, help='Path to distance_summary.npz file')
@click.option('--out_dir', type=str, default='./', help='Directory to save plots')
@click.option('--threshold', type=float, default=0.05, help='Success threshold in meters')
def main(data_path, out_dir, threshold):
    # Load data
    data = np.load(data_path, allow_pickle=True)
    
    # Debug: print available keys
    print("Available keys in data:", list(data.keys()))
    
    # Use correct key names (singular to match the saved format)
    phase3_distances_per_episode = data['phase3_distances_per_episode']
    phase3_final_distances = data['phase3_final_distance']
    phase1_final_distances = data['phase1_final_distance']
    
    # Filter out NaN values - handle nested lists properly
    valid_episodes = []
    for i, d in enumerate(phase3_distances_per_episode):
        try:
            # Handle nested list structure
            if isinstance(d, np.ndarray) and d.dtype == object:
                # This is an array of lists, extract first element if it's a list
                if len(d) > 0:
                    inner = d[0] if hasattr(d[0], '__iter__') and not isinstance(d[0], str) else d
                    d_arr = np.array(inner, dtype=float)
                else:
                    continue
            else:
                d_arr = np.array(d, dtype=float)
            
            # Check if has valid data (not empty, not all NaN)
            if len(d_arr) > 0 and not np.all(np.isnan(d_arr)):
                valid_episodes.append(i)
        except (ValueError, TypeError) as e:
            # Skip episodes that can't be converted to float arrays
            print(f"Skipping episode {i}: {e}")
            continue
    
    if not valid_episodes:
        print("No valid V-JEPA planning data found!")
        return
    
    # Extract valid trajectories and convert to numpy arrays
    trajectories = []
    for i in valid_episodes:
        d = phase3_distances_per_episode[i]
        if isinstance(d, np.ndarray) and d.dtype == object:
            inner = d[0] if hasattr(d[0], '__iter__') and not isinstance(d[0], str) else d
            trajectories.append(np.array(inner, dtype=float))
        else:
            trajectories.append(np.array(d, dtype=float))
    
    max_steps = max(len(t) for t in trajectories)
    
    # Pad trajectories to same length with NaN
    padded_trajectories = np.full((len(trajectories), max_steps), np.nan)
    for i, traj in enumerate(trajectories):
        padded_trajectories[i, :len(traj)] = traj
    
    # Calculate mean and std across episodes
    mean_distance = np.nanmean(padded_trajectories, axis=0)
    std_distance = np.nanstd(padded_trajectories, axis=0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distance over planning steps (mean ± std)
    ax1 = axes[0, 0]
    steps = np.arange(len(mean_distance))
    ax1.plot(steps, mean_distance, 'b-', linewidth=2, label='Mean distance')
    ax1.fill_between(steps, mean_distance - std_distance, mean_distance + std_distance, 
                      alpha=0.3, color='b', label='±1 std')
    ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold}m)')
    ax1.set_xlabel('Planning Step')
    ax1.set_ylabel('Distance to Target (m)')
    ax1.set_title(f'V-JEPA Planning: Distance over Steps (n={len(valid_episodes)})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual trajectories
    ax2 = axes[0, 1]
    for i, traj in enumerate(trajectories[:min(10, len(trajectories))]):  # Plot first 10
        ax2.plot(np.arange(len(traj)), traj, alpha=0.5)
    ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold}m)')
    ax2.set_xlabel('Planning Step')
    ax2.set_ylabel('Distance to Target (m)')
    ax2.set_title('Individual Trajectories (up to 10)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final distance comparison (Phase 1 vs Phase 3)
    ax3 = axes[1, 0]
    valid_phase1 = phase1_final_distances[~np.isnan(phase1_final_distances)]
    valid_phase3 = phase3_final_distances[~np.isnan(phase3_final_distances)]
    
    if len(valid_phase1) > 0 and len(valid_phase3) > 0:
        positions = [1, 2]
        data_to_plot = [valid_phase1, valid_phase3]
        labels = ['IK Baseline', 'V-JEPA Planning']
        
        bp = ax3.boxplot(data_to_plot, positions=positions, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
            patch.set_facecolor(color)
        
        ax3.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold}m)')
        ax3.set_ylabel('Final Distance to Target (m)')
        ax3.set_title('Final Distance Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No comparison data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Final Distance Comparison (No Data)')
    
    # Plot 4: Success rate over planning steps
    ax4 = axes[1, 1]
    success_rate = np.zeros(max_steps)
    for step in range(max_steps):
        distances_at_step = padded_trajectories[:, step]
        valid_at_step = distances_at_step[~np.isnan(distances_at_step)]
        if len(valid_at_step) > 0:
            success_rate[step] = np.mean(valid_at_step <= threshold) * 100
        else:
            success_rate[step] = np.nan
    
    ax4.plot(steps, success_rate, 'g-', linewidth=2, marker='o')
    ax4.set_xlabel('Planning Step')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title(f'Success Rate (threshold={threshold}m)')
    ax4.set_ylim([0, 105])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, 'distance_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Number of valid episodes: {len(valid_episodes)}")
    print(f"Max planning steps: {max_steps}")
    
    if len(valid_phase1) > 0:
        print(f"\nPhase 1 (IK Baseline):")
        print(f"  Mean: {np.mean(valid_phase1):.4f}m ± {np.std(valid_phase1):.4f}m")
        print(f"  Success rate: {np.mean(valid_phase1 <= threshold)*100:.1f}%")
    
    if len(valid_phase3) > 0:
        print(f"\nPhase 3 (V-JEPA Planning):")
        print(f"  Mean: {np.mean(valid_phase3):.4f}m ± {np.std(valid_phase3):.4f}m")
        print(f"  Success rate: {np.mean(valid_phase3 <= threshold)*100:.1f}%")
        print(f"  Final step success rate: {success_rate[-1]:.1f}%")
    
    # Don't show interactive plot in non-interactive environments
    # plt.show()

if __name__ == '__main__':
    main()
