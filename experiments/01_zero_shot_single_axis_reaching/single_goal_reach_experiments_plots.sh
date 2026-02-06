#!/usr/bin/env bash
# single_goal_reach_experiments_plots.sh

set -euo pipefail

# Load conda from /opt/conda
source /opt/conda/etc/profile.d/conda.sh

# Activate environment
conda activate vjepa2-312

# Enable headless rendering
export MUJOCO_GL=egl

# Go to project root
cd /home/s185927/thesis/robohive/robohive/robohive

OUT_DIR="/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching"


# Run distance analysis plots for x, y, z
python -m utils.plot_distance_analysis --experiment_type x --out_dir ${OUT_DIR}
python -m utils.plot_distance_analysis --experiment_type y --out_dir ${OUT_DIR}
python -m utils.plot_distance_analysis --experiment_type z --out_dir ${OUT_DIR}