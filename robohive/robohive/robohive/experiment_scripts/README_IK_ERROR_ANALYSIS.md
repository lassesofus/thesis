# Inverse Kinematics Error Analysis

This directory contains experiments for analyzing the error characteristics of RoboHive's inverse kinematics solver.

## Overview

The IK solver in RoboHive uses a **Jacobian-based iterative method** with damped least squares regularization. This experiment investigates how the IK error varies with target distance from the starting position.

## How IK Works in RoboHive

### Algorithm

1. **Jacobian Computation**: Calculate the Jacobian matrix `J` relating joint velocities to end-effector velocities
2. **Error Calculation**: Measure translational error `Δx = x_target - x_current`
3. **Update Step**: Compute joint position update using damped least squares:
   ```
   Δq = J^T (JJ^T + λI)^(-1) Δx
   ```
   where `λ` is the regularization strength
4. **Iteration**: Apply update, recompute forward kinematics, repeat until convergence

### Key Parameters

- **`tol`**: Convergence threshold for error norm (default: 1e-14)
- **`max_steps`**: Maximum iterations before giving up (default: 500)
- **`regularization_strength`**: Damping factor λ for numerical stability (default: 3e-2)
- **`max_update_norm`**: Caps joint update magnitude per iteration (default: 2.0)

### Convergence Criteria

The solver stops when:
1. `err_norm < tol` (success), or
2. `err_norm / update_norm > progress_thresh` (insufficient progress), or
3. `steps >= max_steps` (max iterations reached)

## Experiment: IK Error vs Target Distance

### Hypothesis

We investigate whether IK error scales with target distance, and whether there are workspace regions with higher error.

### Methodology

1. **Setup**: Start robot at default configuration
2. **Sampling**: For each radius M ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}m:
   - Sample N=100 target positions uniformly in/on sphere of radius M
   - Center sphere at starting end-effector position
3. **IK Test**: For each target:
   - Run IK solver to find joint configuration
   - Execute computed configuration
   - Measure final end-effector position
4. **Metrics**:
   - **Absolute error**: L2 distance from final EE position to target (meters)
   - **Relative error**: Absolute error / target distance (dimensionless)
   - **IK convergence**: Number of iterations, success/failure
   - **Joint displacement**: L2 norm of joint position change

### Output

The experiment produces:
- `ik_error_results.json`: Raw results for all trials
- `ik_error_results.csv`: Results in tabular format
- `ik_error_summary.csv`: Summary statistics by radius
- **Plots**:
  - `absolute_error_vs_radius.png`: Mean absolute error with error bars
  - `relative_error_vs_radius.png`: Relative error percentage
  - `error_distributions.png`: Box plots of error distributions
  - `ik_convergence_stats.png`: Iterations and success rate
  - `error_scatter.png`: All samples colored by radius
- **Videos** (if `--save_videos` enabled):
  - `ik_video_radius_0.050m.mp4`: Visual inspection for 0.05m radius
  - `ik_video_radius_0.100m.mp4`: Visual inspection for 0.10m radius
  - etc. (one video per radius tested)

## Usage

### Basic Usage

```bash
cd /home/s185927/thesis/robohive/robohive/robohive/experiment_scripts
python ik_error_analysis.py
```

This will run with default parameters:
- Radii: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] meters
- Samples per radius: 100
- IK tolerance: 1e-4
- Output: `./ik_error_results/`

### Custom Parameters

```bash
# Test different radii (note: each radius needs its own --radii flag)
python ik_error_analysis.py \
  --radii 0.05 \
  --radii 0.10 \
  --radii 0.20 \
  --radii 0.30 \
  --samples_per_radius 200

# Tighter IK tolerance
python ik_error_analysis.py \
  --ik_tolerance 1e-6 \
  --ik_max_steps 3000

# Sample on sphere surface only (not volume)
python ik_error_analysis.py \
  --sampling_mode surface

# Custom output directory
python ik_error_analysis.py \
  --output_dir /data/s185927/ik_experiments/run_001
```

### All Options

```
--sim_path TEXT              Path to MuJoCo XML model
--output_dir TEXT            Output directory for results
--radii FLOAT                Sphere radii to test (can specify multiple times)
--samples_per_radius INT     Number of samples per radius (default: 100)
--sampling_mode [volume|surface]  Sample mode (default: volume)
--seed INT                   Random seed (default: 42)
--ik_tolerance FLOAT         IK convergence tolerance (default: 1e-4)
--ik_max_steps INT           Maximum IK iterations (default: 2000)
--ik_regularization FLOAT    IK regularization strength (default: 1.0)
--plot / --no-plot           Generate plots (default: True)
--save_videos                Save one MP4 video per radius for visual inspection
--video_width INT            Video width in pixels (default: 640)
--video_height INT           Video height in pixels (default: 480)
--video_fps INT              Video framerate (default: 30)
--video_camera TEXT          MuJoCo camera name (default: left_cam)
```

### Recording Videos

To save one video per radius for visual inspection:

```bash
python ik_error_analysis.py \
  --save_videos \
  --video_width 1280 \
  --video_height 720 \
  --video_fps 60
```

This will create videos named `ik_video_radius_0.100m.mp4`, `ik_video_radius_0.200m.mp4`, etc.
Each video shows:
- Robot at starting position (0.5s hold)
- Smooth motion to target using IK solution (2s)
- Final position with target marker visible (0.5s hold)
- Total video duration: ~3 seconds per radius

## Expected Results

### Typical Findings

1. **Absolute Error**: Generally increases with target distance
   - Near targets (0.05-0.10m): ~10^-5 to 10^-4 m
   - Medium targets (0.15-0.20m): ~10^-4 to 10^-3 m
   - Far targets (0.25-0.30m): ~10^-3 to 10^-2 m

2. **Relative Error**: Often remains relatively constant (~0.1-1% of distance)
   - Some variation due to workspace singularities
   - May increase near joint limits

3. **IK Convergence**:
   - Near targets converge quickly (5-20 iterations)
   - Far targets may require more iterations (20-100+)
   - Success rate typically >95% for reachable targets

4. **Failure Modes**:
   - Targets beyond reachable workspace
   - Targets near singularities
   - Joint limit violations

## Interpreting Results

### What is "Good" IK Performance?

- **Absolute error < 1mm**: Excellent for most manipulation tasks
- **Absolute error < 5mm**: Acceptable for reaching tasks
- **Absolute error > 10mm**: May indicate solver issues or unreachable targets

### Sources of Error

1. **Numerical precision**: Solver tolerance limits minimum error
2. **Linearization error**: Jacobian is only locally accurate
3. **Joint limits**: May prevent reaching exact target
4. **Singularities**: Jacobian becomes rank-deficient
5. **Regularization**: Damping trades accuracy for stability

## Integration with Main Experiments

This analysis helps interpret results from:
- `generate_droid_sim_data.py`: Training data generation uses IK
- `robo_samples.py`: Phase 1 baseline uses IK to reach targets
- V-JEPA planning: Compares to IK baseline performance

**Key insight**: If IK error is ~1mm but V-JEPA achieves 10mm, the difference (9mm) is due to planning/control, not IK limitations.

## Dependencies

- `robohive` (with MuJoCo backend)
- `numpy`, `pandas`, `matplotlib`
- `click` for CLI
- `tqdm` for progress bars

## Troubleshooting

### "IK never succeeds"
- Check that targets are within robot workspace
- Increase `--ik_max_steps`
- Decrease `--ik_tolerance` if too strict

### "All errors are huge"
- Verify robot model loads correctly
- Check starting configuration is valid
- Ensure end-effector site exists in model

### "Script crashes with MuJoCo error"
- Verify MuJoCo rendering backend: `export MUJOCO_GL=egl`
- Check XML path is correct
- Ensure sufficient GPU memory for rendering

## Future Extensions

Possible experiment variations:
1. **Directional bias**: Test x/y/z directions separately
2. **Joint initialization**: Start from different configurations
3. **Orientation targets**: Include orientation constraints
4. **Solver comparison**: Test different IK methods
5. **Workspace mapping**: 3D error heatmap
