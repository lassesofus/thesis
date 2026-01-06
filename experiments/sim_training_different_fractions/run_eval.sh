#!/bin/bash
# Evaluate V-JEPA models on axis_aligned/x test set
#
# This script evaluates all models in the model_dir on the x-axis test set.
# Results are saved incrementally every 25 samples.
#
# Usage:
#   ./run_eval.sh                    # Evaluate all models
#   ./run_eval.sh --max_samples 10   # Quick test with 10 samples

set -e

# Activate conda environment
echo "Activating conda environment: vjepa2-312"
eval "$(conda shell.bash hook)"
conda activate vjepa2-312

# Enable headless rendering
export MUJOCO_GL=egl

cd /home/s185927/thesis/robohive/robohive/robohive/utils

python eval_vjepa_models.py \
    --metadata /data/s185927/droid_sim/axis_aligned/x/trajectory_metadata.json \
    --model_dir /data/s185927/vjepa2/weights/droid/from_scratch_non_frozen_encoder \
    --planning_steps 5 \
    --out_dir /home/s185927/thesis/experiments/sim_training_different_fractions/eval_results_full_v2 \
    --action_transform swap_xy_negate_x \
    "$@"
