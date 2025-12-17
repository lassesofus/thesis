#!/usr/bin/env bash
# filepath: /home/s185927/thesis/single_goal_reach_experiments.sh

set -euo pipefail

# Load conda from /opt/conda
source /opt/conda/etc/profile.d/conda.sh

# Activate environment
conda activate vjepa2-312

# Enable headless rendering
export MUJOCO_GL=egl

# Go to project root
cd /home/s185927/thesis/robohive/robohive/robohive

OUT_DIR="/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/left_cam"
COMMON_ARGS="--render offscreen --out_dir ${OUT_DIR} --fixed_target --planning_steps 5 --save_distance_data --max_episodes 10 --action_transform swap_xy_negate_x --enable_vjepa_planning --save_planning_images --camera_name left_cam"

# Run sampling for x, y, z
python utils/robo_samples.py ${COMMON_ARGS} --experiment_type x
python utils/robo_samples.py ${COMMON_ARGS} --experiment_type y
python utils/robo_samples.py ${COMMON_ARGS} --experiment_type z

# Run distance analysis plots for x, y, z
python -m utils.plot_distance_analysis --experiment_type x --out_dir ${OUT_DIR}
python -m utils.plot_distance_analysis --experiment_type y --out_dir ${OUT_DIR}
python -m utils.plot_distance_analysis --experiment_type z --out_dir ${OUT_DIR}