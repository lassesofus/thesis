# Simulation Training Data Ablation Study

## Overview

This experiment investigates how the amount of simulation training data affects V-JEPA 2's performance. It trains multiple models on different fractions of the dataset (25%, 50%, 75%, 100%) to understand the data efficiency of the approach.

## Experiment Design

### Training Configurations

Models trained on varying data fractions:
- **25%**: Quarter of data
- **50%**: Half of data
- **75%**: Three quarters of data
- **100%**: Full dataset

All models use identical architecture (ViT-g/16, 256px, 8 frames) and hyperparameters, only varying the training data size.

## Scripts

### Data Preparation

**`create_data_splits.py`**
- Creates train/val/test splits for different data fractions
- Ensures consistent splits across experiments

**`create_configs.py`**
- Generates training configuration files for each data fraction
- Based on template config with early stopping

### Training

**`train_ablation_study.sh`**
- Trains all models sequentially
- Standard training (no timing)

**`train_ablation_study_timed.sh`**
- Times each training run
- Useful for measuring training efficiency

**`train_with_validation.py`**
- Training script with validation loop
- Supports early stopping based on validation loss

**`train_single_model_early_stopping.py`**
- Train individual models with early stopping

### Evaluation

**`evaluate_validation_loss.py`**
- Compute validation loss for trained models
- Compare generalization across data fractions

**`analyze_training_times.py`**
- Analyze and plot training time vs data fraction
- Measure training efficiency

### Analysis

**`analyze_latent_physical_correlation.py`**
- **NEW**: Analyzes correlation between 3D Euclidean distance and latent distance
- Processes test trajectories through trained encoder
- Generates scatter plots with correlation coefficient
- Includes progress bar for long-running analysis
- Outputs: PNG plot + NPZ data file

**`plot_latent_physical_correlation.py`**
- **NEW**: Quick plotting script for pre-computed correlation data
- Loads NPZ file and regenerates plot with thesis styling
- Useful for adjusting plot appearance without recomputing

**`plot_comparison.py`**
- Compare performance metrics across data fractions
- Validation loss, training time, etc.

**`plot_eval_results.py`**
- Visualize evaluation results
- Performance vs data fraction curves

### Utilities

**`add_early_stopping_to_configs.py`**
- Add early stopping parameters to existing configs

**`verify_transformations.py`**
- Verify coordinate transformations are correct
- Debug tool for axis alignment issues

## Configuration

Training parameters:
- **Model**: ViT-g/16 (1B parameters)
- **Resolution**: 256×256
- **Frames**: 8 frames per clip
- **Batch size**: 8 per GPU
- **Dataset**: DROID simulation data (axis-aligned reaching)
- **Early stopping**: Validation-based with patience

## Outputs

Results structure:
```
/data/s185927/vjepa2/weights/droid/
├── vitg16-256px-8f_25pct/
├── vitg16-256px-8f_50pct/
├── vitg16-256px-8f_75pct/
└── vitg16-256px-8f_100pct/
```

Each directory contains:
- Model checkpoints (best.pt, latest.pt)
- Training logs
- Validation metrics

Analysis outputs:
```
/home/s185927/thesis/experiments/sim_training_different_fractions/
├── latent_physical_correlation.png  # Correlation scatter plot
├── latent_physical_correlation.npz  # Correlation data
├── training_times.png               # Training efficiency plot
└── comparison_plots.png             # Performance comparison
```

## Key Research Questions

1. **Data Efficiency**: How much simulation data is needed for effective learning?
2. **Scaling Laws**: How does performance scale with data size?
3. **Latent Space Quality**: Does latent distance correlate with physical distance?
4. **Generalization**: Do models trained on less data generalize differently?

## Requirements

- Conda environment: `vjepa2-312`
- DROID simulation dataset splits
- Multi-GPU training setup (distributed training)
- Test trajectory CSV files for correlation analysis

## Quick Start

### Generate data splits
```bash
python create_data_splits.py
```

### Create training configs
```bash
python create_configs.py
```

### Train all models
```bash
./train_ablation_study.sh
```

### Analyze latent-physical correlation
```bash
python analyze_latent_physical_correlation.py \
  --test_csv /data/s185927/droid_sim/y_axis/test_trajectories.csv \
  --checkpoint /data/s185927/vjepa2/weights/droid/vitg16-256px-8f_100pct/best.pt \
  --config /home/s185927/thesis/vjepa2/configs/train/vitg16/ablation/droid-256px-8f_100pct.yaml
```

### Re-plot correlation from saved data
```bash
python plot_latent_physical_correlation.py \
  --data_path latent_physical_correlation.npz
```

Or use shortcuts:
```bash
/home/s185927/thesis/scripts/analysis/analyze_latent_correlation.sh
/home/s185927/thesis/scripts/analysis/plot_latent_correlation.sh
```

## Related

- Training configs: `/home/s185927/thesis/vjepa2/configs/train/vitg16/ablation/`
- Data generation: `/home/s185927/thesis/robohive/robohive/robohive/utils/generate_droid_sim_data.py`
- Plot styling: `/home/s185927/thesis/plot_config.py`
