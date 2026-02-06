#!/bin/bash
# Generate 1000 trajectories along DROID x-axis (RoboHive y-axis) with RobotiQ gripper
# Train/test split: 80/20
# Output: /data/s185927/droid_sim/axis_aligned/x_axis

set -e  # Exit on error

# Enable headless rendering
export MUJOCO_GL=egl

echo "Generating 1000 x-axis aligned trajectories with RobotiQ gripper..."

python /home/s185927/thesis/robohive/robohive/robohive/utils/generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/axis_aligned/x_axis \
  --num_trajectories 1000 \
  --task reaching \
  --traj_dir x \
  --train_test_split 0.8 \
  --save_split_info \
  --seed 42 \
  --reach_horizon 4.5 \
  --gripper robotiq

echo "Done! Check output at: /data/s185927/droid_sim/axis_aligned/x_axis"
echo "Train set: /data/s185927/droid_sim/axis_aligned/x_axis/train_trajectories.csv"
echo "Test set: /data/s185927/droid_sim/axis_aligned/x_axis/test_trajectories.csv"
