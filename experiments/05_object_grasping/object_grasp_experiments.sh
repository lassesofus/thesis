#!/usr/bin/env bash
# filepath: /home/s185927/thesis/experiments/05_object_grasping/object_grasp_experiments.sh
# Object grasping experiments using V-JEPA planning
# Based on single_goal_reach_experiments.sh but with a graspable object in the scene

set -euo pipefail

# Load conda from /opt/conda
source /opt/conda/etc/profile.d/conda.sh

# Activate environment
conda activate vjepa2-312

# Enable headless rendering
export MUJOCO_GL=egl

# Enable CEM planning optimizations (GPU pose computation, AMP, cudnn.benchmark, TF32)
export OPT_AMP=0 VJEPA_OPTIMIZE=0

# Go to project root
cd /home/s185927/thesis/robohive/robohive/robohive

# Scene with graspable object (robotiq gripper version)
SIM_PATH="/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_grasp_robotiq_v0.xml"

OUT_DIR="/home/s185927/thesis/experiments/05_object_grasping/results"
COMMON_ARGS="--sim_path ${SIM_PATH} --render offscreen --out_dir ${OUT_DIR} --fixed_target --planning_steps 5 --save_distance_data --max_episodes 10 --action_transform swap_xy_negate_x --enable_vjepa_planning --save_planning_images --visualize_energy_landscape --camera_name left_cam --gripper robotiq --reset_sim --teleport_actions"

# Run sampling for x, y, z (reaching toward/away from object along each axis)
python utils/robo_samples.py ${COMMON_ARGS} --experiment_type x
python utils/robo_samples.py ${COMMON_ARGS} --experiment_type y
python utils/robo_samples.py ${COMMON_ARGS} --experiment_type z

# Run distance analysis plots for x, y, z
python -m utils.plot_distance_analysis --experiment_type x --out_dir ${OUT_DIR}
python -m utils.plot_distance_analysis --experiment_type y --out_dir ${OUT_DIR}
python -m utils.plot_distance_analysis --experiment_type z --out_dir ${OUT_DIR}
