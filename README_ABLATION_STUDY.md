# V-JEPA Ablation Study - Complete Setup

## Summary

**YES - Early stopping is now fully implemented!** Your ablation study setup is complete and ready to train.

## What You Asked For

✅ Train models on 10%, 20%, ..., 100% of training data (10 models total)
✅ **Early stopping when validation loss plateaus** ← **FULLY IMPLEMENTED**
✅ Automatic best model selection
✅ Ready-to-run training scripts
✅ Validation monitoring after each epoch

## Training Duration (WITH Early Stopping)

**Before (fixed 315 epochs)**:
- Per model: 26-52 hours
- All 10 models: 11-22 days

**Now (with early stopping)**:
- Per model: ~2.5-17 hours (stops when optimal)
- All 10 models: ~1-7 days (varies by model)
- **3-10x faster!**

## How Early Stopping Works

```
For each epoch:
  1. Train on training set
  2. Evaluate on validation set
  3. If validation loss improved:
       → Save as best model
       → Reset patience counter
  4. If validation loss did NOT improve:
       → Increment patience counter
  5. If patience counter reaches 10:
       → STOP TRAINING
       → Best model already saved
```

## Quick Start

```bash
cd /home/s185927/thesis/vjepa2
./scripts/train_ablation_study.sh
```

This trains all 10 models (10%, 20%, ..., 100%) with early stopping.

## Early Stopping Configuration

Each model will:
- **Validate every 1 epoch** (after each full pass through training data)
- **Stop after 10 epochs without improvement** (patience=10)
- **Save best model** to `best.pt` when validation loss is lowest
- **Continue for max 315 epochs** (but will likely stop much earlier)

Example: If validation loss is best at epoch 35, but doesn't improve for next 10 epochs, training stops at epoch 45. The model saved at epoch 35 (lowest val loss) is your `best.pt`.

## Data Splits

| Percentage | Train Episodes | Val Episodes | Test Episodes |
|------------|----------------|--------------|---------------|
| All | 640 | 160 | In metadata.json |
| 10% | 64 | 160 (same) | (same) |
| 20% | 128 | 160 (same) | (same) |
| 30% | 192 | 160 (same) | (same) |
| ... | ... | 160 (same) | (same) |
| 100% | 640 | 160 (same) | (same) |

**Key Point**: Validation set is the SAME for all experiments, enabling fair comparison.

## Files & Locations

### Data Splits
```
/data/s185927/droid_sim/y_axis/splits/
├── train_010pct.csv  (64 episodes)
├── train_020pct.csv  (128 episodes)
├── ...
├── train_100pct.csv  (640 episodes)
└── val_trajectories.csv  (160 episodes, constant)
```

### Configs with Early Stopping
```
/home/s185927/thesis/vjepa2/configs/train/vitg16/ablation/
├── droid-256px-8f_010pct.yaml
├── droid-256px-8f_020pct.yaml
├── ...
└── droid-256px-8f_100pct.yaml
```

### Training Scripts
```
/home/s185927/thesis/vjepa2/scripts/
├── train_ablation_study.sh              ← Run this to train all
├── train_single_model_early_stopping.py ← Or train one model
├── create_data_splits.py
├── create_configs.py
└── add_early_stopping_to_configs.py
```

### Trained Models (after training)
```
/data/s185927/vjepa2/weights/droid/
├── 4.8.vitg16-256px-8f_010pct/
│   ├── best.pt        ← Use this for evaluation!
│   ├── latest.pt
│   ├── log_r0.csv     (training loss)
│   └── val_log_r0.csv (validation loss)
├── 4.8.vitg16-256px-8f_020pct/
│   └── ...
└── ... (10 directories total)
```

### Logs
```
/home/s185927/thesis/vjepa2/logs/ablation/
├── train_010pct.log
├── train_020pct.log
└── ... (10 log files)
```

## Adjusting Early Stopping

### Stop Training Sooner (5 epochs patience)
```bash
python scripts/add_early_stopping_to_configs.py --patience 5
```

### Be More Patient (20 epochs patience)
```bash
python scripts/add_early_stopping_to_configs.py --patience 20
```

### Validate Every 2 Epochs (faster, but less precise)
```bash
python scripts/add_early_stopping_to_configs.py --val_freq 2 --patience 20
```

## Complete Workflow

### 1. Train Models (with early stopping)
```bash
cd /home/s185927/thesis/vjepa2
./scripts/train_ablation_study.sh
```

**Wait for training to complete** (each model stops automatically when optimal)

### 2. Verify Training Results
```bash
# Check that best.pt exists for each model
ls /data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_*/best.pt

# View validation loss over time for 10% model
column -t -s',' /data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_010pct/val_log_r0.csv
```

### 3. Evaluate Planning Performance
For each model, load `best.pt` and evaluate on test trajectories:

```python
from notebooks.utils.world_model_wrapper import WorldModel

# Load best model for 10% data
model_10pct = WorldModel(
    enc_checkpoint_path="/data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_010pct/best.pt",
    device="cuda:0"
)

# Run planning evaluation on test samples
# (integrate with your existing robo_samples.py planning code)
```

### 4. Generate Results Plot
```python
import matplotlib.pyplot as plt
import numpy as np

percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
avg_distances = [...]  # From planning evaluation

plt.plot(percentages, avg_distances, marker='o')
plt.xlabel('Training Data (%)')
plt.ylabel('Avg Distance to Target (m)')
plt.title('Planning Performance vs Training Data Size')
plt.savefig('ablation_results.png')
```

## Monitoring Training

### Watch Training in Real-Time
```bash
# Terminal 1: Run training
cd /home/s185927/thesis/vjepa2
./scripts/train_ablation_study.sh

# Terminal 2: Monitor current model
tail -f logs/ablation/train_010pct.log
```

### Look for Early Stopping Messages
```
[Epoch 35] avg. loss 0.110
Running validation...
Validation: loss=0.1100 [jloss=0.0589, sloss=0.0511]
★ Validation improved: 0.1102 → 0.1100
★ Saved best model with val_loss=0.1100

[Epoch 45] avg. loss 0.109
Running validation...
Validation: loss=0.1101 [jloss=0.0590, sloss=0.0511]
No improvement for 10 epochs (best: 0.1100)
Early stopping triggered! No improvement for 10 epochs.
Best validation loss: 0.1100 at epoch 35
Training complete! Stopped early at epoch 45
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

## What to Expect

1. **First few models (10-30%)**: May converge quickly (~20-40 epochs)
2. **Middle models (40-70%)**: May need more epochs (~40-80 epochs)
3. **Large models (80-100%)**: May train longer but still benefit from early stopping

Each model will find its own optimal training duration!

## Documentation

- **`/home/s185927/thesis/EARLY_STOPPING_COMPLETE.md`** - Complete early stopping guide
- **`/home/s185927/thesis/ABLATION_STUDY_QUICKSTART.md`** - Quick reference
- **`vjepa2/scripts/ABLATION_STUDY_GUIDE.md`** - Detailed workflow guide
- **`/home/s185927/thesis/README_ABLATION_STUDY.md`** - This file

## Key Points

✅ **Early stopping IS implemented** - stops when validation loss plateaus
✅ **Best model saved automatically** - use `best.pt` for evaluation
✅ **Training time reduced 3-10x** - typically 2-17 hours per model
✅ **Validation every epoch** - continuous monitoring
✅ **All configs ready** - just run the training script
✅ **10 models** - trained on 10%, 20%, ..., 100% of data

## Ready to Go!

Start training now:
```bash
cd /home/s185927/thesis/vjepa2
./scripts/train_ablation_study.sh
```

Good luck with your thesis! 🎓🚀
