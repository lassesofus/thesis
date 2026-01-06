#!/bin/bash
# Capture simulation images with DROID coordinate frame visualization
#
# Usage: ./capture_images.sh

set -e

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate vjepa2-312

# Set rendering backend for headless operation
export MUJOCO_GL=egl

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Capturing simulation images..."
echo "Output directory: $SCRIPT_DIR"

python capture_sim_images.py \
    --width 640 \
    --height 480 \
    --output_dir "$SCRIPT_DIR" \
    --cameras left_cam front_cam left_high_cam rear_cam

echo ""
echo "Images saved to: $SCRIPT_DIR"
ls -la "$SCRIPT_DIR"/*.png 2>/dev/null || echo "No PNG files found"
