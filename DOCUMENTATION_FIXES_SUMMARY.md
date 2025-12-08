# Documentation Fixes Summary

## Problem Found

The documentation in `generate_droid_sim_data.py` and `verify_coordinate_transform.py` claimed the coordinate transformation was `swap_xy_negate_x` with negation:
- **Documented:** `DROID_x = -RoboHive_y` (with negation)
- **Actual code:** `DROID_x = RoboHive_y` (no negation)

This mismatch could lead to confusion and incorrect usage.

## Files Fixed

### 1. `/home/s185927/thesis/robohive/robohive/robohive/utils/generate_droid_sim_data.py`

**Changes:**
- Updated header documentation (lines 11-27) to remove references to `swap_xy_negate_x`
- Fixed `transform_pose_to_droid_frame()` docstring (lines 116-151) to match implementation
- Corrected transformation description from `swap_xy_negate_x` → `swap_xy`
- Updated recommended action_transform flag from `swap_xy_negate_y` → `swap_xy`

**Result:** Documentation now correctly states:
```
DROID_x = RoboHive_y  (no negation)
DROID_y = RoboHive_x
DROID_z = RoboHive_z
```

### 2. `/home/s185927/thesis/robohive/robohive/robohive/utils/verify_coordinate_transform.py`

**Changes:**
- Fixed `transform_pose_to_droid_frame()` implementation (lines 15-34) to remove negation
- Fixed `inverse_transform_pose()` implementation (lines 37-59) to match (swap is its own inverse)
- Updated print statements (lines 129-141) to show correct transformation
- Corrected transformation name from `swap_xy_negate_x` → `swap_xy`

**Before (incorrect):**
```python
transformed_pose[0] = -pose[1]  # new_x = -old_y (WRONG!)
```

**After (correct):**
```python
transformed_pose[0] = pose[1]   # DROID_x = RoboHive_y (matches generate_droid_sim_data.py)
```

### 3. `/home/s185927/thesis/robohive/robohive/robohive/utils/robo_samples.py`

**Already updated in previous step:**
- Added `robohive_to_droid_pos()` helper function
- Remapped `--experiment_type` flags to DROID conventions
- Enhanced debug output to show both coordinate frames
- Updated comments to clarify DROID ↔ RoboHive transformation

### 4. `/home/s185927/thesis/COORDINATE_SYSTEM_CHANGES.md`

**Updated:**
- Added section about files modified
- Clarified that `swap_xy` is its own inverse (bidirectional)
- Noted that documentation now matches implementation

## Correct Usage

### For Training Data Generation:
```bash
python generate_droid_sim_data.py \
  --out_dir /data/sim_droid \
  --num_trajectories 100 \
  --traj_dir x
```
Data is automatically saved in DROID frame using `swap_xy` transformation.

### For Evaluation/Deployment:
```bash
python robo_samples.py \
  --render offscreen \
  --experiment_type x \
  --enable_vjepa_planning \
  --action_transform swap_xy \
  --planning_steps 10
```
Use `--action_transform swap_xy` (NOT `swap_xy_negate_x` or `swap_xy_negate_y`)

### For Verification:
```bash
python verify_coordinate_transform.py /path/to/trajectory --verbose
```
Will now correctly verify that data uses `swap_xy` transformation.

## Key Takeaways

1. **Simple swap:** The actual transformation is a simple x↔y swap, no negation
2. **Bidirectional:** `swap_xy` is its own inverse (applying it twice gets you back to the original)
3. **Consistency:** All three files now have matching documentation and implementation
4. **DROID standard:** User-facing flags now consistently use DROID coordinate conventions

## Verification

To verify the fix worked, check that:
1. `generate_droid_sim_data.py` header says "swap_xy" (line 13)
2. `verify_coordinate_transform.py` prints "swap_xy" when run (line 129)
3. Both files implement: `transformed[0] = pose[1]` (no negation)
4. `robo_samples.py` recommends using `--action_transform swap_xy`

All documentation now matches the actual code implementation! ✅
