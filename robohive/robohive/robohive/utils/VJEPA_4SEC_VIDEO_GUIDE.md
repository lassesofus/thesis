# Generating 4+ Second Videos for V-JEPA Training

## What Was Fixed

The script had a **critical bug**: trajectories were being truncated before the robot could reach the target!

### The Problem
- Robot needs ~2-3 seconds to complete reach movements (`reach_horizon`)
- Videos need to be 4+ seconds for V-JEPA training
- Old script was cutting execution short because `trajectory_length=300` frames

### The Solution
Added **proper downsampling** (like `robo_samples.py` has):
- Robot executes FULL trajectory (e.g., 2-3 seconds)
- Frames are downsampled to target video FPS (30)
- Videos are exactly the duration you specify

## Recommended Command for V-JEPA Training

### For 4-5 Second Videos (Ideal for V-JEPA):

```bash
cd /home/s185927/thesis/robohive/robohive/robohive/utils

python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/x_axis_vjepa \
  --num_trajectories 10 \
  --task reaching \
  --traj_dir y \
  --min_reach_distance 0.05 \
  --max_reach_distance 0.3 \
  --train_test_split 0.8 \
  --save_split_info \
  --seed 42 \
  --reach_horizon 3.0 \
  --fps 30
```

**What you get:**
- Reach time: 3.0 seconds (robot completes movement)
- Simulation steps: 3.0 / 0.002 = 1500 steps
- Downsampling: Every 17th frame (sim_fps=500, video_fps=30)
- Video frames: 1500 / 17 ≈ 88 frames
- Video duration: 88 frames / 30 fps ≈ **2.9 seconds**

Wait, that's only 2.9 seconds! To get 4+ seconds:

### Corrected Command for 4+ Second Videos:

```bash
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/x_axis_vjepa \
  --num_trajectories 1000 \
  --task reaching \
  --traj_dir x \
  --min_reach_distance 0.05 \
  --max_reach_distance 0.3 \
  --bidirectional \
  --train_test_split 0.8 \
  --save_split_info \
  --seed 42 \
  --reach_horizon 4.5 \
  --fps 30
```

**What you get:**
- Reach time: 4.5 seconds
- Simulation steps: 4.5 / 0.002 = 2250 steps
- Video frames: 2250 / 17 ≈ 132 frames
- Video duration: 132 / 30 ≈ **4.4 seconds** ✓

## Key Parameters

| Parameter | Recommended | Purpose |
|-----------|-------------|---------|
| `--reach_horizon` | `4.5` | Time for robot to reach target (seconds) |
| `--fps` | `30` | Video framerate (matches V-JEPA expectations) |
| `--min_reach_distance` | `0.05` | Minimum movement (meters) |
| `--max_reach_distance` | `0.3` | Maximum movement (meters) |
| `--bidirectional` | ✓ | Enable both +/- movements for diversity |
| `--seed` | `42` | Reproducible random generation |

## How Downsampling Works

### Old Behavior (BROKEN):
```
1. Generate 1500 waypoints for 3s reach
2. Truncate to trajectory_length=300  ← BUG!
3. Robot only executes 20% of movement
4. Saves 300 frames
5. Video: 300/30 = 10 seconds
```

### New Behavior (FIXED):
```
1. Generate 2250 waypoints for 4.5s reach
2. Execute ALL waypoints (robot completes reach) ✓
3. Downsample: save every 17th frame
4. Saves ~132 frames
5. Video: 132/30 ≈ 4.4 seconds ✓
```

## Testing the Fix

Run with 10 trajectories first:

```bash
python generate_droid_sim_data.py \
  --out_dir /tmp/test_vjepa_video \
  --num_trajectories 10 \
  --task reaching \
  --traj_dir x \
  --min_reach_distance 0.05 \
  --max_reach_distance 0.3 \
  --bidirectional \
  --train_test_split 0.8 \
  --save_split_info \
  --seed 42 \
  --reach_horizon 4.5
```

Then check results:

```bash
python analyze_trajectories.py /tmp/test_vjepa_video/trajectory_metadata.json
```

**What to look for:**
- ✓ Final distances should be < 0.05m (SUCCESS)
- ✓ Min target distance >= 0.05m
- ✓ Unique target distances ≈ num_trajectories

### Check video duration:

```bash
# Install ffmpeg if needed
ffmpeg -i /tmp/test_vjepa_video/episode_0000/recordings/MP4/99999999.mp4 2>&1 | grep Duration
```

Should show: `Duration: 00:00:04.xx` (4+ seconds)

## Full Workflow for V-JEPA Training

### 1. Generate Data

```bash
# X-axis (1000 trajectories, ~4.4 second videos)
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/x_axis_vjepa \
  --num_trajectories 1000 \
  --task reaching \
  --traj_dir x \
  --min_reach_distance 0.05 \
  --max_reach_distance 0.3 \
  --bidirectional \
  --train_test_split 0.8 \
  --save_split_info \
  --seed 42 \
  --reach_horizon 4.5
```

### 2. Update V-JEPA Config

Edit `vjepa2/configs/train/vitg16/droid-256px-8f.yaml`:

```yaml
data:
  dataset_type: VideoDataset
  datasets: ['DROIDVideoDataset']
  droid_train_paths: '/data/s185927/droid_sim/x_axis_vjepa/train_trajectories.csv'

  # V-JEPA uses 8-frame clips
  frames_per_clip: 8
  frame_step: 4  # Skip frames between sampled frames
  num_clips: 1
```

### 3. Train V-JEPA

```bash
cd /home/s185927/thesis/vjepa2

python -m app.main \
  --fname configs/train/vitg16/droid-256px-8f.yaml \
  --devices cuda:0
```

### 4. Evaluate on Test Set

```bash
cd /home/s185927/thesis/robohive/robohive/robohive/utils

python eval_vjepa_planning.py \
  --test_csv /data/s185927/droid_sim/x_axis_vjepa/test_trajectories.csv \
  --metadata /data/s185927/droid_sim/x_axis_vjepa/trajectory_metadata.json \
  --out_dir /data/s185927/eval_results/x_axis \
  --planning_steps 10 \
  --action_transform swap_xy_negate_x
```

## Different Video Durations

Adjust `--reach_horizon` to control video length:

| reach_horizon | Sim Steps | Video Frames | Video Duration |
|---------------|-----------|--------------|----------------|
| 2.0s | 1000 | ~59 | ~2.0s |
| 3.0s | 1500 | ~88 | ~2.9s |
| 4.5s | 2250 | ~132 | ~4.4s ✓ |
| 6.0s | 3000 | ~176 | ~5.9s |
| 10.0s | 5000 | ~294 | ~9.8s |

**Formula:**
```
sim_steps = reach_horizon / 0.002
video_frames = sim_steps / 17  (downsampling factor)
video_duration = video_frames / 30
```

**For exact 4.0 seconds:**
```
reach_horizon = 4.08s
→ 2040 steps → 120 frames → 4.0 seconds
```

## Troubleshooting

### Issue: Trajectories still failing
**Check:** Run `analyze_trajectories.py` - final distances should be < threshold

**Fix:** Increase `--reach_horizon` to give robot more time

### Issue: Videos too short/long
**Check:** Use `ffmpeg -i <video> 2>&1 | grep Duration`

**Fix:** Adjust `--reach_horizon`:
- Too short → increase reach_horizon
- Too long → decrease reach_horizon

### Issue: File sizes too large
**Solution:** 4-5 second videos at 640x480 are reasonable (~5-10MB each)

For smaller files:
- Reduce `--width 480 --height 360`
- Or reduce `--reach_horizon` (but keep >= 4s for V-JEPA)

## Summary of Changes

**Fixed:**
1. ✓ Added proper downsampling (matches robo_samples.py)
2. ✓ Robot now completes full reach before stopping
3. ✓ Videos are correct duration (4+ seconds)
4. ✓ All trajectories should succeed (distance < threshold)

**New parameters:**
- `--video_duration` (optional): Directly specify target video duration
- Deprecated `--trajectory_length`: Use `--reach_horizon` instead

**Recommended for V-JEPA:**
```bash
--reach_horizon 4.5  # For ~4.4 second videos
--fps 30              # Standard video framerate
--bidirectional       # Maximum diversity
--seed 42             # Reproducibility
```
