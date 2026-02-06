# DROID-Compatible Simulation Data Generation

This directory contains tools for generating synthetic robot trajectories in DROID format for V-JEPA2 training.

## Files

- **`generate_droid_sim_data.py`**: Main data generation script
- **`verify_droid_data.py`**: Verification script to test data loading
- **`README_DROID_DATA_GEN.md`**: This file

## Quick Start

### 1. Generate Training Data

```bash
cd /home/s185927/thesis/robohive/robohive/robohive

# Generate 100 reaching trajectories
python utils/generate_droid_sim_data.py \
  --out_dir /data/s185927/sim_droid \
  --num_trajectories 100 \
  --task reaching \
  --trajectory_length 300 \
  --camera_name 99999999 \
  --csv_output /data/s185927/sim_droid_train.csv
```

### 2. Verify Data

```bash
cd /home/s185927/thesis

python robohive/robohive/robohive/utils/verify_droid_data.py \
  --csv_path /data/s185927/sim_droid_train.csv \
  --num_samples 5
```

### 3. Train V-JEPA2

Update your V-JEPA2 config (`configs/train/vitg16/droid-sim-256px-8f.yaml`):

```yaml
data:
  dataset_type: VideoDataset
  datasets: ['DROIDVideoDataset']
  droid_train_paths: '/data/s185927/sim_droid_train.csv'
  batch_size: 8
  frames_per_clip: 8
  tubelet_size: 2
  sampling_rate: 4
```

Then train:

```bash
cd /home/s185927/thesis/vjepa2
python -m app.main --fname configs/train/vitg16/droid-sim-256px-8f.yaml --devices cuda:0
```

## Data Format

Each generated trajectory follows the DROID spec:

```
/data/s185927/sim_droid/episode_0000/
├── trajectory.h5                    # Robot state data
├── metadata_sim.json                # Video metadata
└── recordings/MP4/99999999.mp4      # RGB video
```

### trajectory.h5 Structure

```
observation/
├── robot_state/
│   ├── cartesian_position    [T, 6]  float64  [x, y, z, roll, pitch, yaw]
│   └── gripper_position      [T]     float64  scalar gripper state
└── camera_extrinsics/
    └── 99999999_left         [T, 6]  float64  [x, y, z, roll, pitch, yaw]
```

### metadata_sim.json

```json
{
  "trajectory_length": 300,
  "left_mp4_path": "recordings/MP4/99999999.mp4",
  "camera_name": "99999999",
  "fps": 30,
  "resolution": [640, 480],
  "source": "robohive_simulation"
}
```

## Task Types

### Reaching
Single reach to a random target within a sphere:
```bash
--task reaching --reach_radius 0.2 --reach_horizon 3.0
```

### Multi-Target
Multiple consecutive reaches (2-4 targets):
```bash
--task multi_target --reach_radius 0.2
```

### Random Exploration
Random joint-space exploration:
```bash
--task random_exploration --trajectory_length 300
```

## Important Parameters

### Trajectory Length

**Minimum length**: Must satisfy `T >= nframes` where:
- `nframes = frames_per_clip × ceil(video_fps / target_fps)`
- For V-JEPA2: `frames_per_clip=8`, `video_fps=30`, `target_fps=4`
- Therefore: `nframes = 8 × ceil(30/4) = 8 × 8 = 64`

**Recommended**: Use `--trajectory_length 300` for training data

### Camera Configuration

- `--camera_name`: Identifier used in HDF5 datasets (e.g., `99999999`)
- `--mujoco_camera`: MuJoCo camera to render from (e.g., `left_cam`)
- `--width`, `--height`: Video resolution (default: 640×480)
- `--fps`: Video framerate (default: 30)

### Output

- `--out_dir`: Base directory for trajectories
- `--csv_output`: CSV file listing all trajectory paths (for V-JEPA2)

## Example: Large-Scale Data Generation

Generate 1000 trajectories for pre-training:

```bash
python utils/generate_droid_sim_data.py \
  --out_dir /data/s185927/sim_droid_1k \
  --num_trajectories 1000 \
  --task multi_target \
  --trajectory_length 300 \
  --camera_name 99999999 \
  --mujoco_camera left_cam \
  --width 640 \
  --height 480 \
  --fps 30 \
  --csv_output /data/s185927/sim_droid_1k_train.csv
```

This will take approximately 30-60 minutes depending on hardware.

## Synchronization

The script ensures one-to-one correspondence between:
- HDF5 timestep `i` ↔ Video frame `i`

This is critical for V-JEPA2's `loadvideo_decord` method, which samples frames and states using the same indices.

## Troubleshooting

### Error: "Video is too short"

Increase `--trajectory_length` to at least 100 for testing, 300 for training.

### Error: "gladLoadGL error"

The script automatically sets `MUJOCO_GL=egl` for headless rendering. If issues persist, ensure EGL is available:
```bash
python -c "import mujoco; print('MuJoCo GL:', mujoco.mj_getRenderContext())"
```

### Data Quality Issues

Verify data quality:
```bash
python utils/verify_droid_data.py --csv_path <your_csv> --num_samples 10
```

## Integration with Existing Code

To generate data during evaluation/rollouts in `robo_samples.py`:

1. Import the save function:
   ```python
   from robohive.utils.generate_droid_sim_data import save_trajectory
   ```

2. Collect data during execution:
   ```python
   data = {
       'frames': frames_list,
       'cartesian_position': positions_array,
       'gripper_position': gripper_array,
       'camera_extrinsics': extrinsics_array
   }
   ```

3. Save after trajectory:
   ```python
   save_trajectory(traj_dir, data, camera_name='99999999', fps=30)
   ```

## Citation

If you use this data generation pipeline, please cite:

```
@misc{robohive_droid_sim,
  title={DROID-Compatible Simulation Data for V-JEPA2},
  author={Your Name},
  year={2025},
  howpublished={RoboHive + V-JEPA2 Integration}
}
```

## License

Follows RoboHive (Apache 2.0) and V-JEPA2 (CC BY-NC 4.0) licenses.
