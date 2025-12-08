# IK Error Analysis Experiment - Summary

## What Was Created

### 1. Main Experiment Script
**Location**: `/home/s185927/thesis/robohive/robohive/robohive/experiment_scripts/ik_error_analysis.py`

A comprehensive script that investigates how IK solver error varies with target distance.

### 2. Documentation
**Location**: `/home/s185927/thesis/robohive/robohive/robohive/experiment_scripts/README_IK_ERROR_ANALYSIS.md`

Complete documentation including:
- How IK works in RoboHive
- Experiment methodology
- Usage examples
- Expected results and interpretation

### 3. Quick Test Script
**Location**: `/home/s185927/thesis/robohive/robohive/robohive/experiment_scripts/test_ik_quick.sh`

Fast test with minimal samples to verify installation works.

## How Inverse Kinematics Works in RoboHive

RoboHive uses **Jacobian-based iterative IK** with damped least squares:

### Algorithm Steps

1. **Compute Jacobian**: `J = ∂x/∂q` relates joint velocities to end-effector velocities
2. **Calculate Error**: `Δx = x_target - x_current`
3. **Compute Update**: `Δq = J^T (JJ^T + λI)^(-1) Δx`
   - Uses pseudoinverse with L2 regularization
   - `λ` (regularization_strength) provides damping for stability
4. **Apply Update**: Update joint positions and recompute forward kinematics
5. **Iterate**: Repeat until convergence or max iterations

### Key Features

- **Iterative refinement**: Repeatedly linearizes and solves
- **Regularization**: Adds damping when error is large for numerical stability
- **Early stopping**: Halts if progress is too slow (likely stuck in local minimum)
- **Update limiting**: Caps maximum joint movement per iteration

### Parameters You Can Control

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `tol` | 1e-14 | Convergence threshold (smaller = more accurate) |
| `max_steps` | 500 | Maximum iterations (higher = more attempts) |
| `regularization_strength` | 3e-2 | Damping factor (higher = more stable, less accurate) |
| `max_update_norm` | 2.0 | Maximum joint update per iteration |

## Experiment Design

### Setup
- Robot starts at standard home configuration
- End-effector position noted as sphere center

### For Each Radius M ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} meters:
1. Sample N=100 target positions uniformly in sphere of radius M
2. For each target:
   - Reset robot to home position
   - Run IK to solve for target position
   - Execute solution and measure actual end-effector position
   - Record absolute and relative errors

### Metrics Collected

1. **Absolute Error** (meters): `||x_final - x_target||_2`
2. **Relative Error** (dimensionless): `absolute_error / target_distance`
3. **IK Convergence**:
   - Number of iterations
   - Success/failure status
   - Final error norm reported by solver
4. **Joint Displacement**: `||q_final - q_start||_2`

## Quick Start

### Test Installation (Fast)

```bash
cd /home/s185927/thesis/robohive/robohive/robohive/experiment_scripts
./test_ik_quick.sh
```

This runs 2 radii × 10 samples = 20 tests (~30 seconds)

### Run Full Experiment

```bash
cd /home/s185927/thesis/robohive/robohive/robohive/experiment_scripts
python ik_error_analysis.py
```

This runs 6 radii × 100 samples = 600 tests (~5-10 minutes)

### Custom Parameters

```bash
# More radii, more samples (note: each radius needs its own --radii flag)
python ik_error_analysis.py \
  --radii 0.05 --radii 0.10 --radii 0.15 --radii 0.20 \
  --radii 0.25 --radii 0.30 --radii 0.35 \
  --samples_per_radius 200

# Tighter tolerance for higher accuracy
python ik_error_analysis.py \
  --ik_tolerance 1e-6 \
  --ik_max_steps 3000 \
  --ik_regularization 0.5

# Surface sampling (all targets exactly M meters away)
python ik_error_analysis.py \
  --sampling_mode surface
```

## Expected Output

### Files Generated

```
ik_error_results/
├── ik_error_results.json          # Raw data (all trials)
├── ik_error_results.csv           # Tabular format
├── ik_error_summary.csv           # Summary statistics by radius
├── absolute_error_vs_radius.png   # Error vs distance plot
├── relative_error_vs_radius.png   # Percentage error plot
├── error_distributions.png        # Box plots
├── ik_convergence_stats.png       # Iterations & success rate
├── error_scatter.png              # All samples scatter plot
└── ik_video_radius_*.mp4          # Videos (if --save_videos enabled)
```

#### Video Recording (Optional)

Add `--save_videos` to record one MP4 video per radius:
- Shows robot starting position
- Smooth motion to target using IK solution
- Final position with visible target marker
- ~3 seconds per video
- Useful for visual verification and presentations

### Sample Output

```
================================================================================
INVERSE KINEMATICS ERROR ANALYSIS
================================================================================
Model: .../franka_reach_v0.xml
Output directory: ik_error_results
Radii to test: (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
Samples per radius: 100
...

Results for radius 0.050m:
  Absolute error: 0.000123 ± 0.000045m
                  [min: 0.000034, max: 0.000287]
  Relative error: 0.246 ± 0.090%
                  [min: 0.068%, max: 0.574%]
  IK iterations:  8.3 ± 3.2
  Success rate:   100.0%

Results for radius 0.300m:
  Absolute error: 0.001456 ± 0.000823m
                  [min: 0.000234, max: 0.004123]
  Relative error: 0.485 ± 0.274%
                  [min: 0.078%, max: 1.374%]
  IK iterations:  45.7 ± 18.9
  Success rate:   98.0%
```

## Interpreting Results

### What to Look For

1. **Does error scale with distance?**
   - Absolute error typically increases with target distance
   - Relative error may remain relatively constant

2. **Are there failure modes?**
   - Check success rate - should be >95% for reachable targets
   - Failed samples may indicate workspace boundaries or singularities

3. **How does solver iterate?**
   - More iterations for distant targets is expected
   - Very high iterations (>100) may indicate difficult configurations

### Typical IK Performance

| Target Distance | Expected Absolute Error | Expected Relative Error |
|----------------|------------------------|------------------------|
| 0.05m | 10^-5 to 10^-4 m | 0.1% to 0.5% |
| 0.10m | 10^-4 to 5×10^-4 m | 0.2% to 0.8% |
| 0.20m | 5×10^-4 to 2×10^-3 m | 0.3% to 1.0% |
| 0.30m | 1×10^-3 to 5×10^-3 m | 0.5% to 1.5% |

**Note**: These are rough guidelines. Actual performance depends on:
- Robot kinematics and workspace
- IK solver parameters
- Proximity to singularities/joint limits

### Sources of Error

1. **Solver tolerance**: Can't converge below specified `tol`
2. **Linearization**: Jacobian is only locally accurate
3. **Regularization**: Damping trades accuracy for stability
4. **Joint limits**: May prevent exact target reaching
5. **Numerical precision**: Floating-point arithmetic limits

## Why This Matters

### For Your Thesis

This experiment provides a **baseline understanding** of IK performance:

1. **Training Data Quality** (`generate_droid_sim_data.py`):
   - Training data uses IK to reach targets
   - IK error affects quality of "expert demonstrations"
   - If IK has 1mm error, training data has ≥1mm error

2. **Baseline Comparison** (`robo_samples.py` Phase 1):
   - Phase 1 uses IK to reach targets (baseline)
   - Phase 3 uses V-JEPA planning
   - Can only claim V-JEPA improvement if it beats IK baseline

3. **Performance Attribution**:
   - If V-JEPA achieves 10mm error and IK achieves 1mm error:
     - 9mm is due to planning/control quality
     - 1mm is fundamental IK limitation
   - This experiment quantifies the IK component

### Key Question This Answers

**"When V-JEPA doesn't reach the exact target, is it because:**
1. **The planner chose wrong actions?** (V-JEPA's fault)
2. **The IK solver couldn't reach there?** (Not V-JEPA's fault)"

This experiment separates these factors.

## Next Steps

After running this experiment:

1. **Analyze results**: Look at plots and summary statistics
2. **Compare to V-JEPA**: Run `robo_samples.py` experiments
3. **Attribute performance**: Separate IK error from planning error
4. **Report findings**: Include in thesis discussion

Example thesis statement:
> "The IK solver achieves sub-millimeter accuracy (mean: 0.5mm) for targets within 0.2m. When V-JEPA planning achieves 8mm final distance, the 7.5mm discrepancy is attributable to the visual world model's planning accuracy, not the IK solver's limitations."

## Troubleshooting

### Script won't run
```bash
# Check dependencies
pip install numpy pandas matplotlib click tqdm

# Verify robohive installed
python -c "import robohive; print('OK')"

# Check MuJoCo backend
export MUJOCO_GL=egl
```

### Low success rate (<90%)
- Targets may be outside workspace
- Reduce radii: `--radii 0.05 --radii 0.10 --radii 0.15`
- Increase iterations: `--ik_max_steps 3000`

### All errors very large (>1cm)
- Verify model loads correctly
- Check starting configuration is valid
- Try lower radii first

## Files Overview

```
/home/s185927/thesis/robohive/robohive/robohive/experiment_scripts/
├── ik_error_analysis.py              # Main experiment script
├── README_IK_ERROR_ANALYSIS.md       # Detailed documentation
└── test_ik_quick.sh                  # Quick test script
```

Ready to run! Start with the quick test, then run the full experiment.
