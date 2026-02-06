#!/bin/bash
# =============================================================================
# Metric Sensitivity Experiment: Energy Function Ablation
# =============================================================================
#
# This script runs the metric sensitivity experiment to test whether zero-shot
# reaching performance is sensitive to the choice of energy function in CEM.
#
# Conditions:
#   - 3 energy metrics: l1 (baseline), cosine (scale-invariant), norm_l1 (DROID-normalized)
#   - 3 axes: x, y, z (+20cm offset each)
#   - 10 episodes per condition
#   - Total: 90 episodes
#
# IMPORTANT: This experiment uses NO latent alignment (--latent_alignment none)
#            to isolate the effect of the distance metric.
# =============================================================================

set -e  # Exit on error

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate vjepa2-312

# Set rendering mode for headless execution
export MUJOCO_GL=egl

# Enable V-JEPA optimizations
export VJEPA_OPTIMIZE=1
export OPT_CUDNN=1

# Paths
ROBOHIVE_DIR="/home/s185927/thesis/robohive/robohive/robohive"
RESULTS_DIR="/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/metric_sensitivity/results"
STATS_DIR="/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/metric_sensitivity/stats"

# Change to RoboHive directory
cd "$ROBOHIVE_DIR"

# Common arguments (same as baseline experiments)
COMMON_ARGS="--render offscreen \
    --fixed_target \
    --planning_steps 5 \
    --save_distance_data \
    --max_episodes 10 \
    --action_transform swap_xy_negate_x \
    --enable_vjepa_planning \
    --camera_name left_cam \
    --gripper robotiq \
    --seed 42 \
    --latent_alignment none"

echo "=============================================="
echo "METRIC SENSITIVITY EXPERIMENT"
echo "=============================================="
echo "Results directory: $RESULTS_DIR"
echo "Stats directory: $STATS_DIR"
echo ""

# Function to run experiments for a given metric
run_metric_experiments() {
    local METRIC=$1
    local METRIC_DIR="${RESULTS_DIR}/${METRIC}"

    echo "=============================================="
    echo "Running experiments with energy metric: ${METRIC}"
    echo "=============================================="

    for AXIS in x y z; do
        echo ""
        echo "--- Axis: ${AXIS} ---"

        python utils/robo_samples.py ${COMMON_ARGS} \
            --experiment_type ${AXIS} \
            --out_dir "${METRIC_DIR}" \
            --energy_metric ${METRIC} \
            --energy_stats_path "${STATS_DIR}"

        echo "Completed ${METRIC} / ${AXIS}"
    done

    echo ""
    echo "Completed all axes for metric: ${METRIC}"
}

# =============================================================================
# Run experiments for each energy metric
# =============================================================================

# 1. L1 (baseline) - SKIPPED: Using existing baseline results
# The baseline L1 results should be copied from the previous experiment:
#   /home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/left_cam_fixed_cem_optimized/
# to:
#   ${RESULTS_DIR}/l1/
echo ""
echo "########################################"
echo "# METRIC 1: L1 (baseline) - SKIPPED"
echo "# Using existing baseline results"
echo "########################################"
# run_metric_experiments "l1"

# 2. Cosine (scale-invariant)
echo ""
echo "########################################"
echo "# METRIC 2: Cosine (scale-invariant)"
echo "########################################"
run_metric_experiments "cosine"

# 3. Normalized L1 (DROID-normalized)
echo ""
echo "########################################"
echo "# METRIC 3: Normalized L1 (DROID-normalized)"
echo "########################################"
run_metric_experiments "norm_l1"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Directory structure:"
echo "  ${RESULTS_DIR}/"
echo "    l1/        (copy from baseline experiment)"
echo "    cosine/"
echo "    norm_l1/"
echo ""
echo "Total NEW episodes: 60 (2 metrics x 3 axes x 10 episodes)"
echo ""
echo "NOTE: Copy L1 baseline results before running analyze_results.py:"
echo "  cp -r /home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/left_cam_fixed_cem_optimized/reach_along_* ${RESULTS_DIR}/l1/"
echo ""
echo "Next step: Run analyze_results.py to generate plots and statistics"
