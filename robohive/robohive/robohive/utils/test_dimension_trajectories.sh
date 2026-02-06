#!/bin/bash
# Quick test script for dimension-specific trajectory generation
# This creates a small dataset (10 trajectories) to verify functionality

set -e  # Exit on error

echo "========================================"
echo "Testing Dimension-Specific Trajectories"
echo "========================================"

# Test directory
TEST_DIR="/tmp/test_droid_sim_$(date +%s)"
echo "Test directory: $TEST_DIR"

# Navigate to utils directory
cd "$(dirname "$0")"

echo ""
echo "Step 1: Generating 10 test trajectories along X-axis..."
python generate_droid_sim_data.py \
  --out_dir "$TEST_DIR/x_axis" \
  --num_trajectories 10 \
  --task reaching \
  --traj_dir x \
  --max_reach_distance 0.2 \
  --train_test_split 0.8 \
  --success_threshold 0.05 \
  --save_split_info \
  --trajectory_length 100 \
  --camera_name 99999999 \
  --mujoco_camera left_cam

echo ""
echo "Step 2: Verifying generated files..."

# Check if files exist
if [ ! -f "$TEST_DIR/x_axis/trajectory_metadata.json" ]; then
    echo "ERROR: trajectory_metadata.json not found!"
    exit 1
fi

if [ ! -f "$TEST_DIR/x_axis/train_trajectories.csv" ]; then
    echo "ERROR: train_trajectories.csv not found!"
    exit 1
fi

if [ ! -f "$TEST_DIR/x_axis/test_trajectories.csv" ]; then
    echo "ERROR: test_trajectories.csv not found!"
    exit 1
fi

# Count trajectories
NUM_TRAIN=$(wc -l < "$TEST_DIR/x_axis/train_trajectories.csv")
NUM_TEST=$(wc -l < "$TEST_DIR/x_axis/test_trajectories.csv")

echo "✓ Found train_trajectories.csv with $NUM_TRAIN trajectories"
echo "✓ Found test_trajectories.csv with $NUM_TEST trajectories"

# Check if episode directories exist
FIRST_EPISODE="$TEST_DIR/x_axis/episode_0000"
if [ ! -d "$FIRST_EPISODE" ]; then
    echo "ERROR: Episode directory not found: $FIRST_EPISODE"
    exit 1
fi

if [ ! -f "$FIRST_EPISODE/trajectory.h5" ]; then
    echo "ERROR: trajectory.h5 not found in $FIRST_EPISODE"
    exit 1
fi

echo "✓ Found episode directories with trajectory.h5 files"

# Parse and display metadata summary
echo ""
echo "Step 3: Analyzing metadata..."
python -c "
import json
import numpy as np

with open('$TEST_DIR/x_axis/trajectory_metadata.json') as f:
    metadata = json.load(f)

print(f'Total trajectories: {len(metadata)}')

train = [m for m in metadata if m['split'] == 'train']
test = [m for m in metadata if m['split'] == 'test']

print(f'Train: {len(train)}, Test: {len(test)}')

target_distances = [m['target_distance'] for m in metadata]
print(f'Target distances: min={min(target_distances):.3f}m, max={max(target_distances):.3f}m, mean={np.mean(target_distances):.3f}m')

successes = sum(1 for m in metadata if m['success'])
print(f'IK success rate: {successes}/{len(metadata)} ({100*successes/len(metadata):.1f}%)')

# Check that all are X-axis
directions = set(m['trajectory_direction'] for m in metadata)
if directions == {'x'}:
    print('✓ All trajectories are along X-axis')
else:
    print(f'WARNING: Found unexpected directions: {directions}')
"

echo ""
echo "========================================"
echo "✓ All tests passed!"
echo "========================================"
echo ""
echo "Test data location: $TEST_DIR"
echo ""
echo "To test evaluation (requires V-JEPA model):"
echo "  python eval_vjepa_planning.py \\"
echo "    --test_csv $TEST_DIR/x_axis/test_trajectories.csv \\"
echo "    --metadata $TEST_DIR/x_axis/trajectory_metadata.json \\"
echo "    --out_dir $TEST_DIR/eval_results \\"
echo "    --planning_steps 5"
echo ""
echo "To clean up test data:"
echo "  rm -rf $TEST_DIR"
