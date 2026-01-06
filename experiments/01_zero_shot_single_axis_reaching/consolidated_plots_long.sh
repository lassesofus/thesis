#!/usr/bin/env bash
# consolidated_plots_long.sh
# Generate consolidated analysis plots for 10-step (long) planning experiments

set -euo pipefail

# Load conda from /opt/conda
source /opt/conda/etc/profile.d/conda.sh

# Activate environment
conda activate vjepa2-312

# Enable headless rendering
export MUJOCO_GL=egl

# Go to project root
cd /home/s185927/thesis/robohive/robohive/robohive

OUT_DIR="/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/left_cam_long_planning"

echo "Generating consolidated plot for long planning: $OUT_DIR"
python -m utils.plot_consolidated_analysis_long --out_dir "$OUT_DIR"

echo "Done generating consolidated plot for long planning."
