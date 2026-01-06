#!/bin/bash
# Generate trajectories along x, y, and z axes for zero-shot latent-physical correlation analysis
# Output: /data/s185927/droid_sim/zero_shot_correlation/{x,y,z}_axis
#
# All trajectories have a fixed 0.2m offset to match the zero-shot reaching experiments.
# These trajectories will be used to analyze how well the pretrained V-JEPA 2 encoder's
# latent space correlates with physical distances in different movement directions.

set -e  # Exit on error

BASE_OUT_DIR="/data/s185927/droid_sim/zero_shot_correlation"
NUM_TRAJECTORIES=50  # 50 test trajectories per axis (matching finetuned analysis)
SEED=42
REACH_DISTANCE_MIN=0.2  # Fixed 0.2m offset matching zero-shot experiments
REACH_DISTANCE_MAX=0.2001  # Tiny epsilon to satisfy max > min validation

echo "Generating axis-aligned trajectories for zero-shot correlation analysis..."
echo "Output directory: $BASE_OUT_DIR"
echo "Fixed reach distance: ${REACH_DISTANCE_MIN}m"
echo ""

# Generate x-axis trajectories
echo "=== Generating X-axis trajectories ==="
python /home/s185927/thesis/robohive/robohive/robohive/utils/generate_droid_sim_data.py \
  --out_dir "${BASE_OUT_DIR}/x_axis" \
  --num_trajectories $NUM_TRAJECTORIES \
  --task reaching \
  --traj_dir x \
  --min_reach_distance $REACH_DISTANCE_MIN \
  --max_reach_distance $REACH_DISTANCE_MAX \
  --train_test_split 0.0 \
  --save_split_info \
  --seed $SEED \
  --reach_horizon 4.5

echo ""

# Generate y-axis trajectories
echo "=== Generating Y-axis trajectories ==="
python /home/s185927/thesis/robohive/robohive/robohive/utils/generate_droid_sim_data.py \
  --out_dir "${BASE_OUT_DIR}/y_axis" \
  --num_trajectories $NUM_TRAJECTORIES \
  --task reaching \
  --traj_dir y \
  --min_reach_distance $REACH_DISTANCE_MIN \
  --max_reach_distance $REACH_DISTANCE_MAX \
  --train_test_split 0.0 \
  --save_split_info \
  --seed $((SEED + 1)) \
  --reach_horizon 4.5

echo ""

# Generate z-axis trajectories
echo "=== Generating Z-axis trajectories ==="
python /home/s185927/thesis/robohive/robohive/robohive/utils/generate_droid_sim_data.py \
  --out_dir "${BASE_OUT_DIR}/z_axis" \
  --num_trajectories $NUM_TRAJECTORIES \
  --task reaching \
  --traj_dir z \
  --min_reach_distance $REACH_DISTANCE_MIN \
  --max_reach_distance $REACH_DISTANCE_MAX \
  --train_test_split 0.0 \
  --save_split_info \
  --seed $((SEED + 2)) \
  --reach_horizon 4.5

echo ""
echo "Done! Generated trajectories for all three axes:"
echo "  X-axis: ${BASE_OUT_DIR}/x_axis/test_trajectories.csv"
echo "  Y-axis: ${BASE_OUT_DIR}/y_axis/test_trajectories.csv"
echo "  Z-axis: ${BASE_OUT_DIR}/z_axis/test_trajectories.csv"
