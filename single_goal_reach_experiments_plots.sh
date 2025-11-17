#!/usr/bin/env bash
# single_goal_reach_experiments_plots.sh

set -euo pipefail

# Load conda from /opt/conda
source /opt/conda/etc/profile.d/conda.sh

# Activate environment
conda activate vjepa2-312

# Go to project root
cd /home/s185927/thesis/robohive/robohive/robohive

# Run distance analysis plots for x, y, z
python -m utils.plot_distance_analysis --experiment_type x
python -m utils.plot_distance_analysis --experiment_type y
python -m utils.plot_distance_analysis --experiment_type z