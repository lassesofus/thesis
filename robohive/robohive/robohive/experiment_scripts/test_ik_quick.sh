#!/bin/bash
#
# Quick test of IK error analysis script
# Runs with minimal samples to verify everything works
#

echo "========================================"
echo "Quick IK Error Analysis Test"
echo "========================================"
echo ""
echo "This will run a quick test with minimal samples."
echo "Full experiment should use --samples_per_radius 100"
echo ""

cd /home/s185927/thesis/robohive/robohive/robohive/experiment_scripts

python ik_error_analysis.py \
  --radii 0.05 \
  --radii 0.10 \
  --radii 0.15 \
  --radii 0.20 \
  --radii 0.25 \
  --radii 0.30 \
  --radii 0.35 \
  --radii 0.40 \
  --radii 0.45 \
  --radii 0.50 \
  --radii 0.55 \
  --radii 0.60 \
  --radii 0.65 \
  --radii 0.70 \
  --radii 0.75 \
  --radii 0.80 \
  --radii 0.85 \
  --radii 0.90 \
  --radii 0.95 \
  --radii 1.00 \
  --samples_per_radius 100 \
  --output_dir ./ik_error_results_test \
  --seed 42 \
  --save_videos

echo ""
echo "========================================"
echo "Test complete!"
echo "Check results in: ./ik_error_results_test"
echo "========================================"
