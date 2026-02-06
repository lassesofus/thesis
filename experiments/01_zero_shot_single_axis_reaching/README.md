# Single Goal Reach Experiments

## Overview

This experiment evaluates V-JEPA 2's ability to perform goal-conditioned reaching tasks in simulation using CEM planning in latent space.

## Experiment Design

### Three-Phase Protocol

For each episode:

1. **Phase 1 (Capture Goal)**: Use IK to move to target position and capture goal image
2. **Phase 2 (Return Home)**: Return robot to start position and capture start image
3. **Phase 3 (V-JEPA Planning)**: Use CEM planning to reach goal from start position
   - Plans action sequences in latent space
   - Minimizes representation distance to goal
   - Tracks both latent distance and physical distance

### Experiment Types

- **x-axis**: Reaching primarily along x-axis (DROID frame = RoboHive y-axis)
- **y-axis**: Reaching primarily along y-axis (DROID frame = -RoboHive x-axis)
- **z-axis**: Reaching primarily along z-axis (vertical movement)

## Scripts

### Main Experiment Runner

**`single_goal_reach_experiments.sh`**
- Runs all three experiments (x, y, z)
- 10 episodes per axis
- Saves distance metrics and videos
- Uses correct coordinate transform (`swap_xy_negate_x`)

```bash
./single_goal_reach_experiments.sh
```

### Plotting Script

**`single_goal_reach_experiments_plots.sh`**
- Generates distance plots showing V-JEPA planning performance
- Shows both representation distance and physical distance over time

```bash
./single_goal_reach_experiments_plots.sh
```

## Configuration

- **Planning steps**: 5 (CEM iterations)
- **Max episodes**: 10 per axis
- **Render mode**: offscreen
- **Fixed target**: Yes (reproducible)
- **Action transform**: `swap_xy_negate_x` (DROID → RoboHive)
- **V-JEPA Planning**: Enabled

## Outputs

Results are saved to:
```
/home/s185927/thesis/robohive/robohive/robohive/experiments/
├── reach_along_x/
├── reach_along_y/
└── reach_along_z/
```

Each directory contains:
- Distance metrics (JSON/NPZ)
- Video recordings (MP4)
- Planning visualizations

## Key Insights

This experiment helps answer:
- Can V-JEPA 2's latent space support effective goal-conditioned planning?
- Does minimizing representation distance correlate with reducing physical distance?
- How well does the model generalize to different movement directions?

## Requirements

- Conda environment: `vjepa2-312`
- V-JEPA 2 action-conditioned model checkpoint
- RoboHive simulation environment (FrankaReachRandom-v0)
- Offscreen rendering support

## Quick Start

From anywhere:
```bash
/home/s185927/thesis/experiments/single_goal_reach_experiments/single_goal_reach_experiments.sh
```

Or use the shortcut:
```bash
/home/s185927/thesis/scripts/reach/run_xyz_experiments.sh
```

## Related

- Main integration script: `/home/s185927/thesis/robohive/robohive/robohive/utils/robo_samples.py`
- Plot utility: `/home/s185927/thesis/robohive/robohive/robohive/utils/plot_distance_analysis.py`
- Coordinate transforms: `/home/s185927/thesis/robohive/robohive/robohive/utils/generate_droid_sim_data.py`
