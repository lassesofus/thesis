# Trajectory Diversity Improvements

## Issues Identified and Fixed

### Problem 1: Minimum Distance Was 0
**Issue:** The original code sampled distances uniformly from `[0, max_reach_distance]`, which meant:
- Some trajectories had essentially no movement (distance ≈ 0)
- Many trajectories looked identical because they barely moved

**Fix:** Added `--min_reach_distance` parameter (default: 0.05m)
- Now samples from `[min_reach_distance, max_reach_distance]`
- Ensures all trajectories have meaningful movement
- Default: 5cm to 30cm range

### Problem 2: All Movements in One Direction
**Issue:** All trajectories moved in the positive direction along the axis
- X-axis: only `(+delta, 0, 0)` movements
- Y-axis: only `(0, +delta, 0)` movements
- Z-axis: only `(0, 0, +delta)` movements

**Fix:** Added `--bidirectional` flag
- When enabled, samples from `[-max, -min] ∪ [min, max]`
- Allows both positive and negative movements
- Provides 2× more visual diversity

### Problem 3: No Random Seed Control
**Issue:** Could not reproduce exact same dataset for experiments

**Fix:** Added `--seed` parameter
- Set `--seed 42` for reproducible results
- Leave unset for random generation

## New Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--min_reach_distance` | float | `0.05` | Minimum distance (meters) |
| `--max_reach_distance` | float | `0.3` | Maximum distance (meters) |
| `--bidirectional` | flag | `False` | Allow positive AND negative movements |
| `--seed` | int | `None` | Random seed for reproducibility |

## How to Analyze Your Existing Data

Use the new analysis script to check if your existing trajectories have diversity issues:

```bash
cd /home/s185927/thesis/robohive/robohive/robohive/utils

python analyze_trajectories.py /data/s185927/droid_sim/x_axis/trajectory_metadata.json
```

**What to look for:**
1. **Unique values**: Should be close to total number of trajectories
2. **Min target distance**: If close to 0, you have trivial trajectories
3. **Direction distribution**: If all positive, you might want bidirectional
4. **Distance histogram**: Should be roughly uniform, not concentrated at 0

Example output:
```
Target Distance Statistics:
  Min:    0.0012m   ← ⚠️ BAD: Too close to zero!
  Max:    0.2987m
  Mean:   0.1501m
  Unique values: 998/1000

Direction distribution:
  Positive x: 1000 (100.0%)  ← ⚠️ Consider --bidirectional
  Negative x: 0 (0.0%)
```

## Recommended Settings for Different Use Cases

### Case 1: Basic Reaching (One Direction)
```bash
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/x_axis_v2 \
  --num_trajectories 1000 \
  --task reaching \
  --traj_dir x \
  --min_reach_distance 0.05 \
  --max_reach_distance 0.3 \
  --train_test_split 0.8 \
  --save_split_info \
  --seed 42
```

**Use when:**
- Testing unidirectional reaching (e.g., always extending arm forward)
- Simulating push/grasp tasks (one direction dominant)

**Diversity:** Moderate (distances vary, but all in same direction)

### Case 2: Bidirectional Reaching (Maximum Diversity)
```bash
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/x_axis_bidir \
  --num_trajectories 1000 \
  --task reaching \
  --traj_dir x \
  --min_reach_distance 0.05 \
  --max_reach_distance 0.3 \
  --bidirectional \
  --train_test_split 0.8 \
  --save_split_info \
  --seed 42
```

**Use when:**
- Training general-purpose reaching policies
- Testing if V-JEPA can handle reversed movements
- Maximizing training data diversity

**Diversity:** High (distances and directions both vary)

### Case 3: Small Precise Movements
```bash
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/x_axis_precise \
  --num_trajectories 1000 \
  --task reaching \
  --traj_dir x \
  --min_reach_distance 0.02 \
  --max_reach_distance 0.10 \
  --bidirectional \
  --train_test_split 0.8 \
  --save_split_info
```

**Use when:**
- Fine manipulation tasks
- Testing V-JEPA on small-scale movements
- High-precision reaching

**Diversity:** High within small range

### Case 4: Large Workspace Exploration
```bash
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/x_axis_large \
  --num_trajectories 1000 \
  --task reaching \
  --traj_dir x \
  --min_reach_distance 0.10 \
  --max_reach_distance 0.40 \
  --bidirectional \
  --train_test_split 0.8 \
  --save_split_info
```

**Use when:**
- Testing workspace limits
- Training for large-scale movements

**Diversity:** Maximum (large range, bidirectional)

## Comparison: Old vs New Behavior

### Example: 1000 Trajectories Along X-Axis

**OLD BEHAVIOR** (without fixes):
```
Distance range: [0.0003, 0.2998]m
- ~50 trajectories with distance < 0.01m (nearly stationary)
- All moving in positive X direction
- Visual appearance: Very similar (all "reach forward slightly")
```

**NEW BEHAVIOR** (with `--min_reach_distance 0.05 --bidirectional`):
```
Distance range: [0.0501, 0.2997]m
- 0 trajectories with distance < 0.05m
- ~500 moving in positive X, ~500 in negative X
- Visual appearance: Much more diverse (forward AND backward reaches)
```

## Migration Guide

If you already generated data with the old script:

1. **Analyze existing data:**
   ```bash
   python analyze_trajectories.py /data/s185927/droid_sim/x_axis/trajectory_metadata.json
   ```

2. **Check if you need to regenerate:**
   - If min target distance < 0.02m → Regenerate
   - If all trajectories in one direction AND you want bidirectional → Regenerate
   - Otherwise → Your data is probably fine

3. **Regenerate with improvements:**
   ```bash
   # Use same seed for reproducibility if needed
   python generate_droid_sim_data.py \
     --out_dir /data/s185927/droid_sim/x_axis_improved \
     --num_trajectories 1000 \
     --task reaching \
     --traj_dir x \
     --min_reach_distance 0.05 \
     --max_reach_distance 0.3 \
     --bidirectional \
     --save_split_info \
     --seed 42
   ```

## Expected Results After Fix

After regenerating with the new parameters, you should see:

### In `trajectory_metadata.json`:
```json
{
  "trajectory_index": 0,
  "target_distance": 0.213,  // ✓ >= min_reach_distance
  ...
},
{
  "trajectory_index": 1,
  "target_distance": -0.087,  // ✓ Negative (if --bidirectional)
  ...
}
```

### Visual Inspection:
- Videos should show clear movement (not tiny jiggles)
- With `--bidirectional`: Half reach forward, half reach backward
- Distance variation should be obvious between trajectories

### Analysis Script Output:
```
Target Distance Statistics:
  Min:    0.0502m   ✓ GOOD: No trivial movements
  Max:    0.2995m   ✓ GOOD: Using full range
  Unique values: 999/1000   ✓ GOOD: High diversity

Direction distribution:
  Positive x: 503 (50.3%)   ✓ GOOD: Balanced
  Negative x: 497 (49.7%)   ✓ GOOD: Balanced
```

## Troubleshooting

### Issue: "Trajectories still look similar"
**Reason:** They're all moving along the same axis (expected for `--traj_dir x`)

**Solutions:**
1. Use `--bidirectional` for more variety
2. Generate separate datasets for X, Y, Z axes
3. Use `--traj_dir sphere` for 3D random targets

### Issue: "Some targets unreachable (IK failures)"
**Reason:** `max_reach_distance` too large for robot workspace

**Solutions:**
1. Reduce `--max_reach_distance` (try 0.25m or 0.2m)
2. Check workspace limits: `python -m robohive.utils.examine_env -e FrankaReachRandom-v0`
3. Use smaller values for Z-axis (vertical movements often more limited)

### Issue: "Want even more diversity"
**Solutions:**
1. Combine multiple axes:
   ```bash
   # Generate X, Y, Z separately, then combine CSVs
   cat x_axis/train_trajectories.csv \
       y_axis/train_trajectories.csv \
       z_axis/train_trajectories.csv > combined_train.csv
   ```
2. Use `--traj_dir sphere` for true 3D diversity
3. Use `--task multi_target` for complex trajectories

## Summary

**Before:** Trajectories sampled from [0, 0.3]m → many trivial movements, all one direction
**After:** Trajectories sampled from [0.05, 0.3]m with optional bidirectional → meaningful, diverse movements

**Recommendation:** Regenerate your dataset with:
- `--min_reach_distance 0.05` (minimum)
- `--bidirectional` (recommended for max diversity)
- `--seed 42` (for reproducibility)
