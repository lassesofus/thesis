#!/bin/bash
#
# Re-run 100% fine-tuning with checkpoints saved every 5 epochs
# for checkpoint-trajectory diagnostics.
#
# Uses SAME hyperparameters as Section 4.2:
# - lr=1e-4, wd=0.04, cosine schedule, warmup=5 epochs
# - encoder frozen (enc_lr_scale=0.0)
# - no augmentations
# - early stopping DISABLED (patience=100)
# - epochs: 60 (fixed endpoint)
# - save_every_freq: 5 (saves e0.pt, e5.pt, ..., e55.pt, e60.pt)
#
export WANDB_API_KEY="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"

set -e

# Load conda
source /opt/conda/etc/profile.d/conda.sh
conda activate vjepa2-312

# Configuration
DEVICE="cuda:0"
CONFIG_FILE="/home/s185927/thesis/experiments/03_checkpoint_diagnostics/configs/x_axis_finetune_100pct_diagnostics.yaml"
LOG_DIR="/home/s185927/thesis/experiments/03_checkpoint_diagnostics/logs"

# Create log directory
mkdir -p "${LOG_DIR}"

cd /home/s185927/thesis/vjepa2

echo "=========================================="
echo "Checkpoint Diagnostics Training"
echo "=========================================="
echo "Device: ${DEVICE}"
echo "Config: ${CONFIG_FILE}"
echo "Log dir: ${LOG_DIR}"
echo ""
echo "Key settings:"
echo "  - save_every_freq: 5 (saves every 5 epochs)"
echo "  - epochs: 60"
echo "  - early_stopping_patience: 100 (effectively disabled)"
echo "  - encoder frozen (enc_lr_scale=0.0)"
echo "=========================================="
echo ""

# Record start time
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
echo "[${START_TIME_STR}] Starting training..."

# Train the model
python /home/s185927/thesis/experiments/sim_training_different_fractions/train_single_model_early_stopping.py \
    --fname "${CONFIG_FILE}" \
    --devices "${DEVICE}" \
    2>&1 | tee "${LOG_DIR}/train.log"

# Record end time
END_TIME=$(date +%s)
END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "[${END_TIME_STR}] Training completed!"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Checkpoints saved to:"
echo "  /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/"
echo ""
echo "Expected checkpoints: e0.pt, e5.pt, e10.pt, ..., e55.pt, e60.pt"
echo "=========================================="
