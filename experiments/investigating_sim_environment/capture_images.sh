#!/bin/bash
# Capture simulation images with DROID coordinate frame visualization
#
# Usage: ./capture_images.sh [franka|robotiq]
#   franka  - Use original Franka parallel-jaw gripper (default)
#   robotiq - Use RobotiQ 2F-85 gripper

set -e

# Parse gripper argument (default: franka)
GRIPPER="${1:-franka}"

if [[ "$GRIPPER" != "franka" && "$GRIPPER" != "robotiq" ]]; then
    echo "Usage: $0 [franka|robotiq]"
    echo "  franka  - Use original Franka parallel-jaw gripper (default)"
    echo "  robotiq - Use RobotiQ 2F-85 gripper"
    exit 1
fi

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate vjepa2-312

# Set rendering backend for headless operation
export MUJOCO_GL=egl

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Capturing simulation images with $GRIPPER gripper..."
echo "Output directory: $SCRIPT_DIR"

python capture_sim_images.py \
    --width 640 \
    --height 480 \
    --output_dir "$SCRIPT_DIR" \
    --cameras left_cam front_cam left_high_cam rear_cam \
    --gripper "$GRIPPER"

echo ""
echo "Images saved to: $SCRIPT_DIR"
ls -la "$SCRIPT_DIR"/*.png 2>/dev/null || echo "No PNG files found"
