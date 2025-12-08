#!/bin/bash
# Convenience script to run V-JEPA model evaluation

METADATA="/data/s185927/droid_sim/y_axis/trajectory_metadata.json"
MODEL_DIR="/data/s185927/vjepa2/weights/droid"
OUT_DIR="/data/s185927/vjepa_eval_results"
PLANNING_STEPS=5

echo "==============================================="
echo "V-JEPA Model Evaluation Pipeline"
echo "==============================================="
echo ""
echo "Metadata: $METADATA"
echo "Models: $MODEL_DIR"
echo "Output: $OUT_DIR"
echo "Planning steps: $PLANNING_STEPS"
echo ""

# Run evaluation
echo "Running evaluation..."
python /home/s185927/thesis/robohive/robohive/robohive/utils/eval_vjepa_models.py \
    --metadata "$METADATA" \
    --model_dir "$MODEL_DIR" \
    --planning_steps "$PLANNING_STEPS" \
    --out_dir "$OUT_DIR" \
    --checkpoint_name "best.pt"

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation complete! Creating plots..."
    
    # Create plots
    python /home/s185927/thesis/robohive/robohive/robohive/utils/plot_model_comparison.py \
        --eval_dir "$OUT_DIR" \
        --out_path "$OUT_DIR/model_comparison.png" \
        --plot_evolution
    
    echo ""
    echo "==============================================="
    echo "Done! Results saved to:"
    echo "  $OUT_DIR/evaluation_summary.json"
    echo "  $OUT_DIR/model_comparison.png"
    echo "  $OUT_DIR/model_comparison_evolution.png"
    echo "==============================================="
else
    echo ""
    echo "ERROR: Evaluation failed!"
    exit 1
fi
