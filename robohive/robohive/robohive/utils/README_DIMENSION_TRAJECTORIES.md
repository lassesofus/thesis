# Dimension-Specific Trajectory Generation for V-JEPA Training

This guide explains how to generate trajectories along specific dimensions (x, y, or z axes) with train/test splits for V-JEPA training and evaluation.

## Overview

The enhanced `generate_droid_sim_data.py` script can now:
- Generate trajectories along specific dimensions (x, y, z, or random sphere)
- Sample target distances uniformly within a specified range
- Automatically split data into training and test sets
- Save DROID-compatible format for V-JEPA training
- Track success metrics based on distance thresholds

The companion `eval_vjepa_planning.py` script evaluates V-JEPA planning performance on test trajectories.

## Quick Start

### Step 1: Generate Training Data Along X-Axis

```bash
cd /home/s185927/thesis/robohive/robohive/robohive/utils

python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/x_axis_data \
  --num_trajectories 1000 \
  --task reaching \
  --traj_dir x \
  --max_reach_distance 0.3 \
  --train_test_split 0.8 \
  --success_threshold 0.05 \
  --save_split_info \
  --trajectory_length 300 \
  --camera_name 99999999 \
  --mujoco_camera left_cam
```

This generates:
- 800 training trajectories (80%)
- 200 test trajectories (20%)
- Targets sampled uniformly along x-axis: offset = (delta, 0, 0) where delta ∈ [0, 0.3]m

### Step 2: Train V-JEPA on Generated Data

Update your V-JEPA config (`configs/train/vitg16/droid-256px-8f.yaml`):

```yaml
data:
  dataset_type: VideoDataset
  datasets: ['DROIDVideoDataset']
  droid_train_paths: '/data/s185927/droid_sim/x_axis_data/train_trajectories.csv'
  # ... other data settings
```

Then train:

```bash
cd /home/s185927/thesis/vjepa2
python -m app.main --fname configs/train/vitg16/droid-256px-8f.yaml --devices cuda:0
```

### Step 3: Evaluate V-JEPA Planning on Test Set

```bash
cd /home/s185927/thesis/robohive/robohive/robohive/utils

python eval_vjepa_planning.py \
  --test_csv /data/s185927/droid_sim/x_axis_data/test_trajectories.csv \
  --metadata /data/s185927/droid_sim/x_axis_data/trajectory_metadata.json \
  --out_dir /data/s185927/eval_results/x_axis \
  --planning_steps 10 \
  --action_transform swap_xy_negate_x \
  --success_threshold 0.05
```

## Command-Line Arguments

### generate_droid_sim_data.py

#### New Arguments for Dimension-Specific Sampling

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--traj_dir` | choice | `sphere` | Direction: `x`, `y`, `z`, or `sphere` (random) |
| `--max_reach_distance` | float | `0.3` | Maximum distance along specified dimension (meters) |
| `--train_test_split` | float | `0.8` | Fraction for training (0.8 = 80% train, 20% test) |
| `--success_threshold` | float | `0.05` | Distance threshold to consider target reached (meters) |
| `--save_split_info` | flag | `False` | Save separate train/test CSV files |

#### Existing Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--out_dir` | str | *required* | Output directory for trajectories |
| `--num_trajectories` | int | `10` | Total number of trajectories |
| `--task` | choice | `reaching` | Task type: `reaching`, `random_exploration`, `multi_target` |
| `--trajectory_length` | int | `300` | Length in timesteps |
| `--camera_name` | str | `99999999` | Camera identifier for DROID format |
| `--mujoco_camera` | str | `left_cam` | MuJoCo camera name |
| `--csv_output` | str | `None` | Path for CSV list of all trajectories |

### eval_vjepa_planning.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test_csv` | str | *required* | CSV file with test trajectory paths |
| `--metadata` | str | *required* | JSON file with trajectory metadata |
| `--out_dir` | str | *required* | Output directory for results |
| `--planning_steps` | int | `10` | Number of CEM planning steps |
| `--action_transform` | str | `swap_xy_negate_x` | Action transformation type |
| `--success_threshold` | float | `0.05` | Success distance threshold (meters) |
| `--horizon` | float | `3.0` | Time per planning step (seconds) |

## Usage Examples

### Example 1: Generate 1000 Trajectories Along Y-Axis

```bash
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/y_axis_1000 \
  --num_trajectories 1000 \
  --task reaching \
  --traj_dir y \
  --max_reach_distance 0.3 \
  --train_test_split 0.8 \
  --save_split_info \
  --success_threshold 0.05
```

**Output:**
- `/data/s185927/droid_sim/y_axis_1000/episode_0000/` through `episode_0999/`
- `/data/s185927/droid_sim/y_axis_1000/train_trajectories.csv` (800 trajectories)
- `/data/s185927/droid_sim/y_axis_1000/test_trajectories.csv` (200 trajectories)
- `/data/s185927/droid_sim/y_axis_1000/trajectory_metadata.json`

### Example 2: Generate Along Z-Axis with 90/10 Split

```bash
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/z_axis_5000 \
  --num_trajectories 5000 \
  --task reaching \
  --traj_dir z \
  --max_reach_distance 0.25 \
  --train_test_split 0.9 \
  --save_split_info \
  --success_threshold 0.03
```

**Result:** 4500 training trajectories, 500 test trajectories along z-axis

### Example 3: Evaluate with Different Planning Parameters

```bash
# Short planning (5 steps)
python eval_vjepa_planning.py \
  --test_csv /data/s185927/droid_sim/x_axis_data/test_trajectories.csv \
  --metadata /data/s185927/droid_sim/x_axis_data/trajectory_metadata.json \
  --out_dir /data/s185927/eval_results/x_axis_5steps \
  --planning_steps 5

# Long planning (20 steps)
python eval_vjepa_planning.py \
  --test_csv /data/s185927/droid_sim/x_axis_data/test_trajectories.csv \
  --metadata /data/s185927/droid_sim/x_axis_data/trajectory_metadata.json \
  --out_dir /data/s185927/eval_results/x_axis_20steps \
  --planning_steps 20
```

## Output Files

### Generated by generate_droid_sim_data.py

```
out_dir/
├── episode_0000/
│   ├── trajectory.h5           # Robot state, camera extrinsics
│   ├── metadata_sim.json       # Video path, resolution, fps
│   └── recordings/
│       └── MP4/
│           └── 99999999.mp4    # RGB video
├── episode_0001/
│   └── ...
├── ...
├── trajectory_metadata.json     # Full metadata for all trajectories
├── train_trajectories.csv       # Training set paths (if --save_split_info)
└── test_trajectories.csv        # Test set paths (if --save_split_info)
```

### trajectory_metadata.json Format

```json
[
  {
    "trajectory_index": 0,
    "trajectory_path": "/data/.../episode_0000",
    "target_position": [0.489, 0.0, 0.434],
    "target_distance": 0.213,
    "final_distance": 0.003,
    "success": true,
    "split": "train",
    "trajectory_direction": "x",
    "task": "reaching"
  },
  ...
]
```

### Generated by eval_vjepa_planning.py

```
out_dir/
└── eval_results.json            # Evaluation results for all test trajectories
```

### eval_results.json Format

```json
[
  {
    "trajectory_index": 42,
    "trajectory_path": "/data/.../episode_0042",
    "target_position": [0.489, 0.0, 0.434],
    "target_distance": 0.213,
    "trajectory_direction": "x",
    "final_distance": 0.067,
    "success": false,
    "distances_per_step": [0.213, 0.189, 0.145, ...],
    "repr_distances_per_step": [0.0234, 0.0198, 0.0156, ...],
    "actions_taken": [[0.05, 0.01, -0.02, ...], ...],
    "planning_steps": 10
  },
  ...
]
```

## Workflow for Full Experiment

### 1. Generate Data for All Axes

```bash
# X-axis
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/x_axis \
  --num_trajectories 1000 --traj_dir x \
  --max_reach_distance 0.3 --save_split_info

# Y-axis
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/y_axis \
  --num_trajectories 1000 --traj_dir y \
  --max_reach_distance 0.3 --save_split_info

# Z-axis
python generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/z_axis \
  --num_trajectories 1000 --traj_dir z \
  --max_reach_distance 0.3 --save_split_info
```

### 2. Combine Training Sets (Optional)

```bash
cat /data/s185927/droid_sim/x_axis/train_trajectories.csv \
    /data/s185927/droid_sim/y_axis/train_trajectories.csv \
    /data/s185927/droid_sim/z_axis/train_trajectories.csv \
    > /data/s185927/droid_sim/combined_train.csv
```

### 3. Fine-tune V-JEPA

```bash
cd /home/s185927/thesis/vjepa2
python -m app.main \
  --fname configs/train/vitg16/droid-256px-8f.yaml \
  --devices cuda:0
```

### 4. Evaluate on Each Axis

```bash
# Evaluate X-axis
python eval_vjepa_planning.py \
  --test_csv /data/s185927/droid_sim/x_axis/test_trajectories.csv \
  --metadata /data/s185927/droid_sim/x_axis/trajectory_metadata.json \
  --out_dir /data/s185927/eval/x_axis

# Evaluate Y-axis
python eval_vjepa_planning.py \
  --test_csv /data/s185927/droid_sim/y_axis/test_trajectories.csv \
  --metadata /data/s185927/droid_sim/y_axis/trajectory_metadata.json \
  --out_dir /data/s185927/eval/y_axis

# Evaluate Z-axis
python eval_vjepa_planning.py \
  --test_csv /data/s185927/droid_sim/z_axis/test_trajectories.csv \
  --metadata /data/s185927/droid_sim/z_axis/trajectory_metadata.json \
  --out_dir /data/s185927/eval/z_axis
```

### 5. Analyze Results

```python
import json
import numpy as np

# Load results
with open('/data/s185927/eval/x_axis/eval_results.json') as f:
    x_results = json.load(f)

# Calculate success rate
successes = [r['success'] for r in x_results]
print(f"X-axis success rate: {sum(successes)/len(successes)*100:.1f}%")

# Plot distance vs success
import matplotlib.pyplot as plt
target_dists = [r['target_distance'] for r in x_results]
final_dists = [r['final_distance'] for r in x_results]

plt.scatter(target_dists, final_dists)
plt.xlabel('Target Distance (m)')
plt.ylabel('Final Distance (m)')
plt.title('V-JEPA Planning Performance')
plt.show()
```

## Tips and Best Practices

1. **Start Small**: Test with `--num_trajectories 10` first to verify everything works

2. **Choose Appropriate Distances**:
   - Start with `--max_reach_distance 0.3` for Franka arm
   - Too large distances may be unreachable due to workspace limits

3. **Success Threshold**:
   - `0.05m` (5cm) is reasonable for reaching tasks
   - Use `0.03m` (3cm) for more precise evaluation

4. **Train/Test Split**:
   - `0.8` (80/20) is standard
   - Use `0.9` (90/10) if you need more training data

5. **Action Transform**:
   - `swap_xy_negate_x` typically works well for left_cam
   - Verify transform by checking if actions move in expected direction

6. **Planning Steps**:
   - Start with 10 steps
   - Increase to 20 for harder targets
   - More steps = longer execution time

## Troubleshooting

### Issue: IK failures during generation

**Solution**: Targets may be outside workspace. Reduce `--max_reach_distance`

### Issue: Low success rate in evaluation

**Solutions**:
- Increase `--planning_steps`
- Adjust `--action_transform`
- Reduce `--success_threshold` temporarily to understand actual performance
- Check if V-JEPA model trained long enough

### Issue: CUDA out of memory

**Solution**: Reduce batch size in V-JEPA training config or use smaller model

### Issue: Videos not saving

**Solution**: Install scikit-video: `pip install scikit-video`

## Integration with Existing Scripts

The generated data is fully compatible with:
- V-JEPA 2 DROIDVideoDataset
- Existing robo_samples.py visualization
- Standard DROID data processing pipelines

## Citation

If you use this code for research, please cite:
- V-JEPA 2: [arXiv link]
- RoboHive: [arXiv link]
- Your thesis work
