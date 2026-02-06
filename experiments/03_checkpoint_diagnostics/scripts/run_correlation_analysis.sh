#!/bin/bash
# Run Probe A with correlation analysis on all checkpoints
# Usage: ./run_correlation_analysis.sh

set -e

# Load conda from /opt/conda
source /opt/conda/etc/profile.d/conda.sh

# Activate environment
conda activate vjepa2-312

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT_DIR="/data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics"
CONFIG_PATH="$BASE_DIR/configs/x_axis_finetune_100pct_diagnostics.yaml"
PROBE_IDS_FILE="$BASE_DIR/diagnostics/probe_ids.txt"
ACTIONS_FILE="$BASE_DIR/diagnostics/actions_seed_42_M512.npy"
OUTPUT_DIR="$BASE_DIR/diagnostics/correlation_analysis"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Epochs to process
EPOCHS=(0 5 10 15 20 25 30 35 40 45 50 55)

echo "=============================================="
echo "Running Probe A with Correlation Analysis"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoints: ${EPOCHS[*]}"
echo ""

for epoch in "${EPOCHS[@]}"; do
    checkpoint="$CHECKPOINT_DIR/e${epoch}.pt"
    output="$OUTPUT_DIR/epoch_${epoch}_correlation.json"

    if [ ! -f "$checkpoint" ]; then
        echo "WARNING: Checkpoint not found: $checkpoint"
        continue
    fi

    if [ -f "$output" ]; then
        echo "SKIP: Output already exists: $output"
        continue
    fi

    echo "Processing epoch $epoch..."
    python "$SCRIPT_DIR/probe_a_with_correlation.py" \
        --checkpoint "$checkpoint" \
        --config "$CONFIG_PATH" \
        --probe_ids_file "$PROBE_IDS_FILE" \
        --actions_file "$ACTIONS_FILE" \
        --output "$output" \
        --device cuda:0

    echo "Done: epoch $epoch"
    echo ""
done

echo "=============================================="
echo "All checkpoints processed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
