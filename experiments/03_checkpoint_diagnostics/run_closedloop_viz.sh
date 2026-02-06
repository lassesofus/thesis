#!/usr/bin/env bash
set -euo pipefail

# Load conda
source /opt/conda/etc/profile.d/conda.sh
conda activate vjepa2-312
export MUJOCO_GL=egl

cd /home/s185927/thesis/experiments/03_checkpoint_diagnostics

python scripts/visualize_energy_landscape_grid_closedloop.py \
    --checkpoints /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e0.pt \
                  /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e15.pt \
                  /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e35.pt \
                  /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e55.pt \
    --epochs 0 15 35 55 \
    --config configs/x_axis_finetune_100pct_diagnostics.yaml \
    --output_dir plots \
    --n_steps 5
