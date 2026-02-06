#!/bin/bash
# Evaluate fine-tuned V-JEPA models (25%, 50%, 75%, 100%) vs Meta baseline
# on x-axis test set from simulation data
#
# Usage:
#   ./run_eval.sh                      # Evaluate with 10 samples (default)
#   ./run_eval.sh --max_samples 50     # Evaluate with 50 samples
#   ./run_eval.sh --max_samples 200    # Full test set (200 samples)
#   ./run_eval.sh --no-save-images     # Disable image saving

set -e

# Activate conda environment
echo "Activating conda environment: vjepa2-312"
eval "$(conda shell.bash hook)"
conda activate vjepa2-312

# Enable headless rendering
export MUJOCO_GL=egl

# Enable CEM planning optimizations (GPU pose computation, AMP, cudnn.benchmark, TF32)
export OPT_AMP=0 VJEPA_OPTIMIZE=1

cd /home/s185927/thesis/robohive/robohive/robohive/utils

python eval_vjepa_models.py \
    --metadata /data/s185927/droid_sim/axis_aligned/x_axis/trajectory_metadata.json \
    --model_dir /data/s185927/vjepa2/weights/droid \
    --planning_steps 5 \
    --out_dir /home/s185927/thesis/experiments/sim_training_different_fractions_x_axis_finetune/eval_results_new \
    --action_transform swap_xy_negate_x \
    --gripper robotiq \
    --max_samples 10 \
    --save_images \
    "$@"
