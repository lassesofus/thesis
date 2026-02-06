#!/bin/bash
# Analyze correlation between Euclidean distance and latent distance
# Processes all test trajectories and generates scatter plot

set -e  # Exit on error

echo "Analyzing latent-physical correlation..."

cd /home/s185927/thesis/experiments/sim_training_different_fractions

python analyze_latent_physical_correlation.py \
  --test_csv /data/s185927/droid_sim/y_axis/test_trajectories.csv \
  --checkpoint /data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_100pct/best.pt \
  --config /home/s185927/thesis/vjepa2/configs/train/vitg16/ablation/droid-256px-8f_100pct.yaml \
  --device cuda:0 \
  --output_plot /home/s185927/thesis/experiments/sim_training_different_fractions/latent_physical_correlation.png

echo "Done! Plot saved to: latent_physical_correlation.png"
echo "Data saved to: latent_physical_correlation.npz"
