#!/bin/bash
# =============================================================================
# IK Error Analysis with Actuator Dynamics
# =============================================================================
# This script runs the actuator-based IK error analysis experiment.
# It measures the realistic tracking error when executing IK solutions
# via actuator control (matching real robot behavior).
#
# Usage:
#   ./run_experiment.sh          # Run with default parameters
#   ./run_experiment.sh --quick  # Quick test run (10 samples per radius)
# =============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"

# Path to the analysis script
ANALYSIS_SCRIPT="/home/s185927/thesis/robohive/robohive/robohive/experiment_scripts/ik_error_analysis_with_actuators.py"

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate vjepa2-312

# Set rendering backend
export MUJOCO_GL=egl

# Check for quick mode
if [[ "$1" == "--quick" ]]; then
    echo "Running in QUICK mode (10 samples per radius)"
    SAMPLES=10
else
    echo "Running FULL experiment (100 samples per radius)"
    SAMPLES=100
fi

# Run the experiment
echo "============================================================================="
echo "IK Error Analysis with Actuator Dynamics"
echo "============================================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Samples per radius: $SAMPLES"
echo "============================================================================="

python "$ANALYSIS_SCRIPT" \
    --radii 0.05 --radii 0.10 --radii 0.15 --radii 0.20 --radii 0.25 --radii 0.30 --radii 0.35 --radii 0.4\
    --samples_per_radius "$SAMPLES" \
    --sampling_mode surface \
    --z_min 0.75 \
    --seed 42 \
    --horizon 3.0 \
    --hold_time 0.5 \
    --output_dir "$OUTPUT_DIR" \
    --plot

echo ""
echo "============================================================================="
echo "Experiment complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================================="
