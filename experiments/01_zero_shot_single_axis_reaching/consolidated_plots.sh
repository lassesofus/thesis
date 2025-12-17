#!/usr/bin/env bash
# consolidated_plots.sh
# Generate consolidated analysis plots for all camera configurations

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

# Run consolidated analysis for each camera configuration
for cam_dir in "${OUT_DIR}"/left_cam*; do
    if [ -d "$cam_dir" ]; then
        echo "Generating consolidated plot for: $cam_dir"
        python -m utils.plot_consolidated_analysis --out_dir "$cam_dir"
    fi
done

echo "Done generating consolidated plots."
