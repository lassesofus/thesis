# Experiment Scripts

This directory contains reproducible experiment scripts for quick execution.

## Directory Structure

```
scripts/
├── data_generation/    # Scripts for generating training/test data
├── training/           # Scripts for training models
├── evaluation/         # Scripts for evaluating trained models
├── analysis/           # Scripts for analyzing results
└── README.md          # This file
```

## Usage

All scripts are executable shell scripts. Run them from anywhere:

```bash
# Generate x-axis aligned data
/home/s185927/thesis/scripts/data_generation/generate_x_axis_1000.sh

# Or navigate to scripts directory first
cd /home/s185927/thesis/scripts
./data_generation/generate_x_axis_1000.sh
```

## Naming Convention

Use descriptive names that indicate:
- What the script does
- Key parameters (e.g., axis direction, number of samples)

Examples:
- `generate_x_axis_1000.sh` - Generate 1000 x-axis trajectories
- `generate_y_axis_bidirectional_500.sh` - Generate 500 bidirectional y-axis trajectories
- `train_vitg_100pct.sh` - Train ViT-g on 100% data
- `analyze_latent_correlation.sh` - Analyze latent-physical correlation

## Best Practices

1. **Start with shebang**: `#!/bin/bash`
2. **Exit on error**: `set -e`
3. **Add comments**: Describe what the script does at the top
4. **Echo progress**: Print what's happening
5. **Print outputs**: Show where results are saved
6. **Use absolute paths**: Avoid confusion about working directory
7. **Version control**: Commit scripts to git for reproducibility

## Creating New Scripts

Template:

```bash
#!/bin/bash
# Brief description of what this script does
# Key parameters and expected outputs

set -e  # Exit on error

echo "Starting [experiment name]..."

# Your commands here
python /path/to/script.py \
  --arg1 value1 \
  --arg2 value2

echo "Done! Results saved to: /path/to/output"
```

Make it executable:
```bash
chmod +x your_script.sh
```

## Tips

- **Quick edits**: Edit parameters in the script instead of remembering CLI args
- **Variations**: Copy script and modify for variations (e.g., different axes)
- **Documentation**: Scripts serve as documentation of experiments
- **Sharing**: Easy to share exact commands with collaborators
