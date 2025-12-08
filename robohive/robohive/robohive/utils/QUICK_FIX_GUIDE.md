# Quick Fix: Trajectory Diversity Issues

## TL;DR - What Changed

Your trajectories look identical because:
1. ❌ **OLD:** Sampled distances from `[0, 0.3]m` → many had distance ≈ 0 (no movement)
2. ❌ **OLD:** All moved in positive direction only
3. ✅ **NEW:** Sample from `[0.05, 0.3]m` → all have meaningful movement
4. ✅ **NEW:** Optional bidirectional movement (positive AND negative)

## Step 1: Analyze Your Current Data

```bash
cd /home/s185927/thesis/robohive/robohive/robohive/utils

python analyze_trajectories.py /data/s185927/droid_sim/x_axis/trajectory_metadata.json
```

Look for:
- **Min distance < 0.02m** → You have trivial trajectories
- **All positive or all negative** → Missing half the movement space

## Step 2: Regenerate with Better Diversity

### Recommended Command (Maximum Diversity):

```bash
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/x_axis_v2 \
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

**Key differences:**
- `--min_reach_distance 0.05` (NEW) → No more trivial movements
- `--bidirectional` (NEW) → Both directions (+x and -x)
- `--seed 42` (NEW) → Reproducible

### Alternative: One Direction Only (If That's What You Want)

```bash
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/x_axis_v2 \
  --num_trajectories 1000 \
  --task reaching \
  --traj_dir x \
  --min_reach_distance 0.05 \
  --max_reach_distance 0.3 \
  --train_test_split 0.8 \
  --save_split_info
```

(Same as above but without `--bidirectional`)

## Step 3: Verify Improvement

```bash
python analyze_trajectories.py /data/s185927/droid_sim/x_axis_v2/trajectory_metadata.json
```

You should see:
```
Target Distance Statistics:
  Min:    0.0501m   ✓ No trivial movements
  Max:    0.2995m   ✓ Using full range

Direction distribution:
  Positive x: 503 (50.3%)   ✓ Balanced (if --bidirectional)
  Negative x: 497 (49.7%)   ✓ Balanced (if --bidirectional)
```

## New Parameters Summary

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `--min_reach_distance 0.05` | 0.05m | Minimum movement (avoids trivial trajectories) |
| `--max_reach_distance 0.3` | 0.3m | Maximum movement |
| `--bidirectional` | Off | Enable negative movements too |
| `--seed 42` | Random | Set seed for reproducibility |

## Before vs After

**BEFORE (your current data):**
- Distance range: [0.0001, 0.2998]m
- ~5% trajectories barely move (< 0.01m)
- All move in same direction

**AFTER (with new defaults):**
- Distance range: [0.0500, 0.2998]m
- 0% trivial trajectories
- Both directions (if `--bidirectional`)

## Questions?

- **Q: Do I need to regenerate?**
  A: Run the analysis script first. If min distance < 0.02m, yes.

- **Q: Should I use --bidirectional?**
  A: Yes, for maximum diversity. No, if you specifically want one direction only.

- **Q: Will this work with my V-JEPA training?**
  A: Yes! The data format is identical, just better diversity.

- **Q: Can I use the same seed to reproduce my data?**
  A: No, the old data didn't use seeds. But you can use `--seed` going forward.

## For More Details

- Full documentation: `README_DIMENSION_TRAJECTORIES.md`
- Detailed fixes: `DIVERSITY_FIX_SUMMARY.md`
- Analysis tool: `analyze_trajectories.py`
