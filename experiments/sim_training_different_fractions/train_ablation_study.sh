#!/bin/bash
#
# Train V-JEPA models on varying fractions of training data.
#
# This script trains 4 different models (25%, 50%, 75%, 100% of training data)
# sequentially on a single GPU.
# The 0% baseline uses the pretrained checkpoint directly (no training).
#

set -e  # Exit on error

# Configuration
DEVICE="cuda:0"
CONFIG_DIR="/home/s185927/thesis/vjepa2/configs/train/vitg16/ablation"
LOG_DIR="/home/s185927/thesis/vjepa2/logs/ablation"
PERCENTAGES=(25 50 75 100)

# Create log directory
mkdir -p "${LOG_DIR}"

# Activate conda environment
echo "Activating conda environment: vjepa2-312"
eval "$(conda shell.bash hook)"
conda activate vjepa2-312

# Navigate to vjepa2 directory
cd /home/s185927/thesis/vjepa2

echo "=========================================="
echo "V-JEPA Ablation Study Training"
echo "=========================================="
echo "Device: ${DEVICE}"
echo "Config directory: ${CONFIG_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Percentages: ${PERCENTAGES[@]}"
echo "=========================================="
echo ""

# Train each model
TIMING_FILE="${LOG_DIR}/training_times.txt"
echo "Training Times" > "${TIMING_FILE}"
echo "==============" >> "${TIMING_FILE}"
echo "" >> "${TIMING_FILE}"

OVERALL_START=$(date +%s)

for PCT in "${PERCENTAGES[@]}"; do
    CONFIG_FILE="${CONFIG_DIR}/droid-256px-8f_$(printf '%03d' ${PCT})pct.yaml"
    LOG_FILE="${LOG_DIR}/train_$(printf '%03d' ${PCT})pct.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training model with ${PCT}% of data..."
    echo "Config: ${CONFIG_FILE}"
    echo "Log: ${LOG_FILE}"

    # Check if config exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        exit 1
    fi

    # Record start time for this model
    MODEL_START=$(date +%s)
    START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')

    # Train the model with early stopping
    python /home/s185927/thesis/experiments/sim_training_different_fractions/train_single_model_early_stopping.py \
        --fname "${CONFIG_FILE}" \
        --devices "${DEVICE}" \
        2>&1 | tee "${LOG_FILE}"

    # Record end time and calculate duration
    MODEL_END=$(date +%s)
    END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
    DURATION=$((MODEL_END - MODEL_START))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed ${PCT}% model"
    echo "Training time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${DURATION} seconds)"
    echo ""

    # Save timing info
    echo "${PCT}% model:" >> "${TIMING_FILE}"
    echo "  Start:    ${START_TIME_STR}" >> "${TIMING_FILE}"
    echo "  End:      ${END_TIME_STR}" >> "${TIMING_FILE}"
    echo "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s (${DURATION} seconds)" >> "${TIMING_FILE}"
    echo "" >> "${TIMING_FILE}"
done

# Calculate overall time
OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
OVERALL_HOURS=$((OVERALL_DURATION / 3600))
OVERALL_MINUTES=$(((OVERALL_DURATION % 3600) / 60))
OVERALL_SECONDS=$((OVERALL_DURATION % 60))

echo "==============" >> "${TIMING_FILE}"
echo "Total time: ${OVERALL_HOURS}h ${OVERALL_MINUTES}m ${OVERALL_SECONDS}s (${OVERALL_DURATION} seconds)" >> "${TIMING_FILE}"

echo "Timing information saved to: ${TIMING_FILE}"

echo "=========================================="
echo "All models trained successfully!"
echo "=========================================="
echo "Total training time: ${OVERALL_HOURS}h ${OVERALL_MINUTES}m ${OVERALL_SECONDS}s"
echo ""
echo "Logs saved to: ${LOG_DIR}"
echo "Timing info saved to: ${TIMING_FILE}"
echo ""
echo "Per-model timing:"
cat "${TIMING_FILE}"
echo ""
echo "Trained models: 25%, 50%, 75%, 100%"
echo "0% baseline: Use pretrained checkpoint /data/s185927/vjepa2/weights/vitg-384.pt"
echo ""
echo "Next steps:"
echo "1. Evaluate each model (including 0% baseline) on test trajectories"
echo "2. Generate plots comparing performance (0%, 25%, 50%, 75%, 100%)"
echo "3. Analyze how planning performance improves with training data"
