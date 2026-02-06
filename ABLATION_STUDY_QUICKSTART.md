# V-JEPA Ablation Study - Quick Start Guide

## What's Been Created

Your ablation study infrastructure is ready! Here's what was set up:

### ✓ Data Splits (80/20 train/val split)
- **640 training episodes** split into 10 fractions (10%, 20%, ..., 100%)
- **160 validation episodes** (constant across all experiments)
- Location: `/data/s185927/droid_sim/y_axis/splits/`

### ✓ Training Configurations
- **10 YAML configs** for each data percentage
- Location: `/home/s185927/thesis/vjepa2/configs/train/vitg16/ablation/`
- Each outputs to a unique checkpoint directory

### ✓ Training Scripts
- **Master script**: `vjepa2/scripts/train_ablation_study.sh`
- Trains all 10 models sequentially on single GPU
- Logs saved to `vjepa2/logs/ablation/`

### ✓ Evaluation Scripts
- **Validation loss**: `vjepa2/scripts/evaluate_validation_loss.py`
- **Comprehensive guide**: `vjepa2/scripts/ABLATION_STUDY_GUIDE.md`

## Quick Start: Train All Models

```bash
cd /home/s185927/thesis/vjepa2
./scripts/train_ablation_study.sh
```

This trains 10 models with 64, 128, 192, ..., 640 episodes.

## Quick Start: Train Single Model

```bash
cd /home/s185927/thesis/vjepa2

# Example: Train 10% model
python -m app.main \
  --fname configs/train/vitg16/ablation/droid-256px-8f_010pct.yaml \
  --devices cuda:0
```

## Data Breakdown

| Percentage | Episodes | Train CSV File | Output Directory |
|------------|----------|----------------|------------------|
| 10% | 64 | `splits/train_010pct.csv` | `weights/.../vitg16-256px-8f_010pct/` |
| 20% | 128 | `splits/train_020pct.csv` | `weights/.../vitg16-256px-8f_020pct/` |
| 30% | 192 | `splits/train_030pct.csv` | `weights/.../vitg16-256px-8f_030pct/` |
| 40% | 256 | `splits/train_040pct.csv` | `weights/.../vitg16-256px-8f_040pct/` |
| 50% | 320 | `splits/train_050pct.csv` | `weights/.../vitg16-256px-8f_050pct/` |
| 60% | 384 | `splits/train_060pct.csv` | `weights/.../vitg16-256px-8f_060pct/` |
| 70% | 448 | `splits/train_070pct.csv` | `weights/.../vitg16-256px-8f_070pct/` |
| 80% | 512 | `splits/train_080pct.csv` | `weights/.../vitg16-256px-8f_080pct/` |
| 90% | 576 | `splits/train_090pct.csv` | `weights/.../vitg16-256px-8f_090pct/` |
| 100% | 640 | `splits/train_100pct.csv` | `weights/.../vitg16-256px-8f_100pct/` |

**Validation**: 160 episodes (constant for all experiments)

## Workflow

1. **Train models**: Run `train_ablation_study.sh` or train individually
2. **Evaluate validation**: Run `evaluate_validation_loss.py` to get validation metrics
3. **Test planning**: Integrate with your test set from `trajectory_metadata.json`
4. **Plot results**: Create plots showing distance vs. training data size

## Important Notes

### Early Stopping
The original V-JEPA code doesn't have built-in early stopping. Options:

1. **Reduce epochs**: Edit configs to use fewer epochs (e.g., 100 instead of 315)
2. **Manual stopping**: Monitor logs and stop training when loss plateaus
3. **Checkpoint evaluation**: Train fully, then evaluate all checkpoints and pick best

To reduce epochs for all configs:
```bash
cd /home/s185927/thesis/vjepa2/configs/train/vitg16/ablation/
for config in *.yaml; do
  sed -i 's/epochs: 315/epochs: 100/' "$config"
done
```

### Adjusting Training
You can modify configs to:
- Change batch size (line 8: `batch_size: 8`)
- Change number of epochs (line 63: `epochs: 315`)
- Change learning rate (line 67: `lr: 0.000425`)
- Change checkpoint frequency (line 48: `save_every_freq: 25`)

### Monitoring Training
```bash
# Watch training log
tail -f /home/s185927/thesis/vjepa2/logs/ablation/train_010pct.log

# Check checkpoint directory
ls -lh /data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_010pct/

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Expected Output

After training completes, you should have:
- **10 trained models**: One for each data percentage
- **Checkpoints**: `latest.pt` in each output directory
- **Logs**: CSV logs with training metrics in each checkpoint directory
- **Training logs**: Text logs in `vjepa2/logs/ablation/`

## Next Steps After Training

1. **Evaluate validation loss**:
   ```bash
   cd /home/s185927/thesis/vjepa2
   conda activate vjepa2-312
   python scripts/evaluate_validation_loss.py --device cuda:0
   ```

2. **Evaluate planning performance**: Adapt your existing planning evaluation code to test each model on the test set

3. **Generate plots**: Plot average distance to target vs. training data percentage

4. **Analyze results**: Confirm your hypothesis that performance improves with more training data

## File Locations

```
/home/s185927/thesis/
├── vjepa2/
│   ├── scripts/
│   │   ├── create_data_splits.py
│   │   ├── create_configs.py
│   │   ├── train_ablation_study.sh          ← Run this to train all
│   │   ├── evaluate_validation_loss.py
│   │   └── ABLATION_STUDY_GUIDE.md          ← Detailed guide
│   ├── configs/train/vitg16/ablation/
│   │   └── droid-256px-8f_*pct.yaml         ← 10 config files
│   └── logs/ablation/
│       └── train_*pct.log                    ← Training logs
└── ABLATION_STUDY_QUICKSTART.md             ← This file

/data/s185927/
├── droid_sim/y_axis/splits/
│   ├── train_*pct.csv                        ← Training data splits
│   ├── val_trajectories.csv                  ← Validation data
│   └── split_metadata.txt
└── vjepa2/weights/droid/
    └── 4.8.vitg16-256px-8f_*pct/             ← Model checkpoints
```

## Need Help?

- **Detailed guide**: `vjepa2/scripts/ABLATION_STUDY_GUIDE.md`
- **Split creation**: `vjepa2/scripts/create_data_splits.py --help`
- **Config creation**: `vjepa2/scripts/create_configs.py --help`
- **Validation eval**: `vjepa2/scripts/evaluate_validation_loss.py --help`

## Ready to Start?

```bash
cd /home/s185927/thesis/vjepa2
./scripts/train_ablation_study.sh
```

Happy training! 🚀
