#!/bin/bash
# =============================================================================
# Plot IK Actuator Error Analysis Results
# =============================================================================
# This script regenerates plots from existing experiment results.
# Useful for adjusting plot styling without re-running the experiment.
#
# Usage:
#   ./plot_results.sh
# =============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate vjepa2-312

echo "============================================================================="
echo "Plotting IK Actuator Error Analysis Results"
echo "============================================================================="
echo "Input directory: $SCRIPT_DIR"
echo "============================================================================="

python "$SCRIPT_DIR/plot_results.py" --input_dir "$SCRIPT_DIR"

echo ""
echo "============================================================================="
echo "Plotting complete!"
echo "============================================================================="
