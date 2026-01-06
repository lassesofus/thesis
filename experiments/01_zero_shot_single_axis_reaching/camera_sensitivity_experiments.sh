#!/usr/bin/env bash
# filepath: /home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/camera_sensitivity_experiments.sh
#
# Experiment: Camera Angle Sensitivity for V-JEPA Planning
# =========================================================
# This script runs the same reaching experiments with different camera angles
# to test if the V-JEPA world model's planning is sensitive to viewpoint.
#
# Baseline: left_cam (original camera angle from DROID-like training data)
# Test cameras: front_cam, right_cam (different viewpoints)

set -euo pipefail

# Load conda from /opt/conda
source /opt/conda/etc/profile.d/conda.sh

# Activate environment
conda activate vjepa2-312

# Enable headless rendering
export MUJOCO_GL=egl

# Go to project root
cd /home/s185927/thesis/robohive/robohive/robohive

# ============================================================================
# Configuration
# ============================================================================

# Base output directory for this experiment set
BASE_OUT_DIR="/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/camera_sensitivity"

# Cameras to test (modify as needed)
# Available cameras in franka_reach_v0.xml:
#   left_cam      - Original baseline (from training distribution)
#   front_cam     - Front view (azimuth=0, elevation=15)
#   right_cam     - Right side view (azimuth=270, elevation=15)
#   back_cam      - Back view (azimuth=180, elevation=15)
#   top_cam       - Top-down view (elevation=80)
#   front_right_cam - Diagonal view (azimuth=315, elevation=25)
#   back_left_cam   - Diagonal view (azimuth=135, elevation=25)
#   left_high_cam   - High angle left view (azimuth=110, elevation=45)
#   left_low_cam    - Low angle left view (azimuth=110, elevation=0)

CAMERAS=("left_cam" "front_cam" "left_high_cam")

# Number of episodes per experiment
MAX_EPISODES=10

# Common arguments shared across all experiments
COMMON_ARGS="--render offscreen --fixed_target --planning_steps 5 --save_distance_data --action_transform swap_xy_negate_x --enable_vjepa_planning --save_planning_images"

# ============================================================================
# Run experiments for each camera
# ============================================================================

for CAMERA in "${CAMERAS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Running experiments with camera: ${CAMERA}"
    echo "========================================================================"

    OUT_DIR="${BASE_OUT_DIR}/${CAMERA}"
    CAMERA_ARGS="--camera_name ${CAMERA} --out_dir ${OUT_DIR} --max_episodes ${MAX_EPISODES}"

    # Run for x, y, z axes
    echo ""
    echo "--- X-axis reaching ---"
    python utils/robo_samples.py ${COMMON_ARGS} ${CAMERA_ARGS} --experiment_type x

    echo ""
    echo "--- Y-axis reaching ---"
    python utils/robo_samples.py ${COMMON_ARGS} ${CAMERA_ARGS} --experiment_type y

    echo ""
    echo "--- Z-axis reaching ---"
    python utils/robo_samples.py ${COMMON_ARGS} ${CAMERA_ARGS} --experiment_type z

    echo ""
    echo "Completed experiments for ${CAMERA}"
    echo "Results saved to: ${OUT_DIR}"
done

echo ""
echo "========================================================================"
echo "All camera experiments completed!"
echo "========================================================================"
echo ""
echo "Now generating comparison plots..."

# Generate comparison plot with all cameras on same axes
python /home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/plot_camera_comparison.py \
    --base_dir "${BASE_OUT_DIR}" \
    --cameras left_cam --cameras front_cam --cameras left_high_cam

echo ""
echo "Done! Check ${BASE_OUT_DIR}/camera_comparison_analysis.png"
