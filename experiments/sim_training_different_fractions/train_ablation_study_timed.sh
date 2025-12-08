#!/bin/bash
# Wrapper script that times the ablation study training

echo "Starting timed ablation study training at: $(date)"
echo "========================================"

# Record start time
START_TIME=$(date +%s)

# Run the training script
/home/s185927/thesis/vjepa2/scripts/train_ablation_study.sh

# Record end time
END_TIME=$(date +%s)

# Calculate duration
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Total seconds: ${DURATION}s"
echo "========================================"

# Save timing info
TIMING_FILE="/home/s185927/thesis/vjepa2/logs/ablation/training_time.txt"
echo "Ablation study training completed" > "${TIMING_FILE}"
echo "Start: $(date -d @${START_TIME})" >> "${TIMING_FILE}"
echo "End: $(date -d @${END_TIME})" >> "${TIMING_FILE}"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s (${DURATION} seconds)" >> "${TIMING_FILE}"
echo "Saved timing info to: ${TIMING_FILE}"
