#!/bin/bash
# Generate 1000 trajectories along DROID x-axis (RoboHive y-axis)
# Train/test split: 80/20
# Output: /data/s185927/droid_sim/axis_aligned

set -e  # Exit on error

echo "Generating 1000 x-axis aligned trajectories..."

python /home/s185927/thesis/robohive/robohive/robohive/utils/generate_droid_sim_data.py \
  --out_dir /data/s185927/droid_sim/axis_aligned \
  --num_trajectories 1000 \
  --task reaching \
  --traj_dir x \
  --train_test_split 0.8 \
  --save_split_info \
  --seed 42 \
  --reach_horizon 4.5

echo "Done! Check output at: /data/s185927/droid_sim/axis_aligned"
echo "Train set: /data/s185927/droid_sim/axis_aligned/train_trajectories.csv"
echo "Test set: /data/s185927/droid_sim/axis_aligned/test_trajectories.csv"
