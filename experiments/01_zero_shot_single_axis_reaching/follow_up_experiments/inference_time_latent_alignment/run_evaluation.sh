#!/usr/bin/env bash
# filepath: run_evaluation.sh
# Run the latent alignment evaluation experiment
# Evaluates baseline (none), mean-only, and CORAL alignment for x, y, z axes

set -euo pipefail

# Load conda from /opt/conda
source /opt/conda/etc/profile.d/conda.sh

# Activate environment
conda activate vjepa2-312

# Enable headless rendering
export MUJOCO_GL=egl

# Enable CEM planning optimizations
export OPT_AMP=0 VJEPA_OPTIMIZE=1

# Go to project root
cd /home/s185927/thesis/robohive/robohive/robohive

# Configuration
OUT_DIR="/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/inference_time_latent_alignment/results"
STATS_PATH="/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/inference_time_latent_alignment/stats"

# Common arguments (matching original zero-shot reaching protocol)
COMMON_ARGS="--render offscreen --fixed_target --planning_steps 5 --save_distance_data --max_episodes 10 --action_transform swap_xy_negate_x --enable_vjepa_planning --camera_name left_cam --gripper robotiq"

echo "============================================================"
echo "LATENT ALIGNMENT EVALUATION EXPERIMENT"
echo "============================================================"
echo "Output directory: ${OUT_DIR}"
echo "Stats path: ${STATS_PATH}"
echo ""

# Function to run experiments for a given alignment method
run_alignment_experiments() {
    local alignment_method=$1
    local run_dir="${OUT_DIR}/${alignment_method}"

    echo ""
    echo "============================================================"
    echo "Running experiments with alignment: ${alignment_method}"
    echo "Output: ${run_dir}"
    echo "============================================================"

    # Run for each axis
    for axis in x y z; do
        echo ""
        echo "--- Axis: ${axis} ---"
        python utils/robo_samples.py ${COMMON_ARGS} \
            --experiment_type ${axis} \
            --out_dir ${run_dir} \
            --latent_alignment ${alignment_method} \
            --latent_alignment_stats_path ${STATS_PATH}
    done

    echo ""
    echo "Completed ${alignment_method} alignment experiments"
}

# Create output directory
mkdir -p ${OUT_DIR}

# Run experiments for each alignment method
run_alignment_experiments "none"
run_alignment_experiments "mean"
run_alignment_experiments "coral"

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: ${OUT_DIR}"
echo "  - none/  (baseline)"
echo "  - mean/  (mean-only alignment)"
echo "  - coral/ (CORAL whitening-coloring)"
echo ""
echo "Next steps:"
echo "  1. Run analysis script to consolidate results"
echo "  2. Generate comparison plots"
echo ""
