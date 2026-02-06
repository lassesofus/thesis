#!/bin/bash
# Test script to verify coordinate axes visualization and transformation

set -e  # Exit on error

echo "========================================="
echo "Testing Coordinate Axes Visualization"
echo "========================================="

TEST_DIR="/tmp/test_coordinate_axes_$(date +%s)"
echo "Creating test directory: $TEST_DIR"
mkdir -p "$TEST_DIR"

cd /home/s185927/thesis/robohive/robohive/robohive

echo ""
echo "Generating 2 test trajectories with coordinate axes..."
echo ""

python utils/generate_droid_sim_data.py \
    --out_dir "$TEST_DIR" \
    --num_trajectories 2 \
    --task reaching \
    --traj_dir z \
    --max_reach_distance 0.2 \
    --reach_horizon 2.0 \
    --width 640 \
    --height 480 \
    --fps 15 \
    --seed 42

echo ""
echo "========================================="
echo "Test completed successfully!"
echo "========================================="
echo ""
echo "Generated trajectories in: $TEST_DIR"
echo ""
echo "To verify coordinate axes visualization:"
echo "  1. Check the MP4 videos in: $TEST_DIR/episode_*/recordings/MP4/"
echo "  2. Look for RGB-colored axes (Red=X, Green=Y, Blue=Z) in the video"
echo ""
echo "To verify coordinate transformation:"
echo "  3. Check trajectory.h5 files to ensure poses are in DROID frame"
echo "  4. Run: python -c \"import h5py; f=h5py.File('$TEST_DIR/episode_0000/trajectory.h5'); print(f['observation']['robot_state']['cartesian_position'][:])\""
echo ""
echo "Clean up test data with: rm -rf $TEST_DIR"
echo ""
