#!/bin/bash
# Evaluate fine-tuned V-JEPA models (25%, 50%, 75%, 100%) vs Meta baseline
# on x-axis TRAINING set from simulation data
#
# This evaluates on training trajectories to check if finetuning improved
# planning on the data the models were trained on.
#
# Usage:
#   ./run_eval_train.sh                      # Evaluate with 10 samples (default)
#   ./run_eval_train.sh --max_samples 50     # Evaluate with 50 samples

set -e

# Activate conda environment
echo "Activating conda environment: vjepa2-312"
eval "$(conda shell.bash hook)"
conda activate vjepa2-312

# Enable headless rendering
export MUJOCO_GL=egl

cd /home/s185927/thesis/robohive/robohive/robohive/utils

python eval_vjepa_models.py \
    --metadata /data/s185927/droid_sim/axis_aligned/trajectory_metadata.json \
    --model_dir /data/s185927/vjepa2/weights/droid/fine_tuned \
    --planning_steps 5 \
    --out_dir /home/s185927/thesis/experiments/sim_training_different_fractions_x_axis_finetune/eval_results_train \
    --action_transform swap_xy_negate_x \
    --max_samples 10 \
    --split train \
    --save_images \
    "$@"
