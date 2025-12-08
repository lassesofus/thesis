# V-JEPA Ablation Study: Training Data Size vs. Performance

This guide explains how to train V-JEPA models on different fractions of training data (10%, 20%, ..., 100%) and evaluate their planning performance.

## Overview

The ablation study evaluates your hypothesis:
> **Performance improves with more training data**

We'll train 10 models on increasing amounts of training data and measure their ability to reach target positions in the test set.

## Setup Complete ✓

The following has been created for you:

### 1. Data Splits ✓
- **Location**: `/data/s185927/droid_sim/y_axis/splits/`
- **Training sets**: 10 files (64, 128, 192, ..., 640 episodes)
  - `train_010pct.csv` (64 episodes, 10% of 640)
  - `train_020pct.csv` (128 episodes, 20% of 640)
  - ...
  - `train_100pct.csv` (640 episodes, 100% of 640)
- **Validation set**: `val_trajectories.csv` (160 episodes, constant across all experiments)
- **Test set**: Available in `/data/s185927/droid_sim/y_axis/trajectory_metadata.json` (marked with `"split": "test"`)

### 2. Training Configs ✓
- **Location**: `/home/s185927/thesis/vjepa2/configs/train/vitg16/ablation/`
- **Files**: 10 configs, one for each percentage
  - `droid-256px-8f_010pct.yaml`
  - `droid-256px-8f_020pct.yaml`
  - ...
  - `droid-256px-8f_100pct.yaml`
- Each config:
  - Uses the corresponding training split
  - Uses the same validation split
  - Outputs to a unique checkpoint directory

### 3. Training Script ✓
- **Location**: `/home/s185927/thesis/vjepa2/scripts/train_ablation_study.sh`
- Trains all 10 models sequentially on a single GPU
- Logs output for each model separately

### 4. Validation Evaluation Script ✓
- **Location**: `/home/s185927/thesis/vjepa2/scripts/evaluate_validation_loss.py`
- Evaluates validation loss for each trained model
- Saves results to JSON for analysis

## Workflow

### Step 1: Train All Models

**Option A: Train all models sequentially (recommended for single GPU)**

```bash
cd /home/s185927/thesis/vjepa2
./scripts/train_ablation_study.sh
```

This will:
- Train each model sequentially (10%, 20%, ..., 100%)
- Save logs to `/home/s185927/thesis/vjepa2/logs/ablation/`
- Save checkpoints to `/data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_XXXpct/`

**Estimated time**: ~10-50 hours per model (depending on epochs and hardware)

**Option B: Train individual models (for parallel training or debugging)**

```bash
cd /home/s185927/thesis/vjepa2

# Train 10% model
python -m app.main \
  --fname configs/train/vitg16/ablation/droid-256px-8f_010pct.yaml \
  --devices cuda:0

# Train 20% model
python -m app.main \
  --fname configs/train/vitg16/ablation/droid-256px-8f_020pct.yaml \
  --devices cuda:0

# ... etc for other percentages
```

### Step 2: Evaluate Validation Loss (Optional)

After training, evaluate how well each model performs on the validation set:

```bash
cd /home/s185927/thesis/vjepa2

conda activate vjepa2-312

python scripts/evaluate_validation_loss.py \
  --percentages 10 20 30 40 50 60 70 80 90 100 \
  --device cuda:0 \
  --output results/ablation_validation_results.json
```

This creates a JSON file with validation losses for each model.

### Step 3: Evaluate Planning Performance on Test Set

Now evaluate each model's ability to plan actions that reach test targets.

Create a script `scripts/evaluate_planning_performance.py`:

```python
#!/usr/bin/env python3
"""
Evaluate planning performance for each trained model.

For each model:
1. Load the trained checkpoint
2. For each test trajectory in trajectory_metadata.json (split="test")
3. Run V-JEPA-based CEM planning to reach the target
4. Record final Euclidean distance to target
5. Compute average distance across all test samples
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/home/s185927/thesis/robohive/robohive/robohive')

from notebooks.utils.world_model_wrapper import WorldModel
from utils.inverse_kinematics import qpos_from_site_pose


def evaluate_planning(
    checkpoint_path: str,
    test_samples: list,
    device: str = 'cuda:0',
    planning_steps: int = 5,
):
    """
    Evaluate planning performance for a model.

    Args:
        checkpoint_path: Path to trained V-JEPA checkpoint
        test_samples: List of test trajectory metadata dicts
        device: Device to use
        planning_steps: Number of CEM planning steps

    Returns:
        Dictionary with results
    """
    # Load world model
    world_model = WorldModel(
        enc_checkpoint_path=checkpoint_path,
        device=device,
    )

    distances = []

    for sample in test_samples:
        target_pos = np.array(sample['target_position'])
        trajectory_path = sample['trajectory_path']

        # Load start and goal images
        # (You'll need to implement this based on your data format)
        # For DROID sim data, images are in trajectory_path/images/

        # Run CEM planning (similar to robo_samples.py logic)
        # ...

        # Execute planned actions and measure final distance
        # final_distance = ...

        distances.append(final_distance)

    avg_distance = np.mean(distances)
    std_distance = np.std(distances)

    return {
        'checkpoint': checkpoint_path,
        'avg_distance': avg_distance,
        'std_distance': std_distance,
        'num_samples': len(distances),
        'all_distances': distances,
    }


def main():
    # Load test samples
    with open('/data/s185927/droid_sim/y_axis/trajectory_metadata.json', 'r') as f:
        all_metadata = json.load(f)

    test_samples = [s for s in all_metadata if s.get('split') == 'test']
    print(f"Found {len(test_samples)} test samples")

    percentages = list(range(10, 101, 10))
    results = []

    for pct in percentages:
        print(f"\nEvaluating {pct}% model...")

        checkpoint_path = (
            f"/data/s185927/vjepa2/weights/droid/"
            f"4.8.vitg16-256px-8f_{pct:03d}pct/latest.pt"
        )

        if not Path(checkpoint_path).exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        result = evaluate_planning(
            checkpoint_path=checkpoint_path,
            test_samples=test_samples,
            device='cuda:0',
            planning_steps=5,
        )
        result['percentage'] = pct
        results.append(result)

        print(f"  Avg distance: {result['avg_distance']:.4f} ± {result['std_distance']:.4f}")

    # Save results
    output_path = Path('/home/s185927/thesis/vjepa2/results/ablation_planning_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
```

**Note**: You'll need to integrate this with your existing planning code from `robohive/robohive/robohive/utils/robo_samples.py`.

### Step 4: Generate Plots

Create a plotting script `scripts/plot_ablation_results.py`:

```python
#!/usr/bin/env python3
"""Plot ablation study results."""

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_results():
    # Load planning results
    with open('/home/s185927/thesis/vjepa2/results/ablation_planning_results.json', 'r') as f:
        results = json.load(f)

    percentages = [r['percentage'] for r in results]
    avg_distances = [r['avg_distance'] for r in results]
    std_distances = [r['std_distance'] for r in results]

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        percentages,
        avg_distances,
        yerr=std_distances,
        marker='o',
        capsize=5,
        linewidth=2,
        markersize=8,
    )

    plt.xlabel('Training Data Size (%)', fontsize=14)
    plt.ylabel('Average Euclidean Distance to Target (m)', fontsize=14)
    plt.title('V-JEPA Planning Performance vs. Training Data Size', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = '/home/s185927/thesis/vjepa2/results/ablation_plot.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

    plt.show()


if __name__ == '__main__':
    plot_results()
```

Run it:

```bash
python scripts/plot_ablation_results.py
```

## Early Stopping Strategy

Since the original training code doesn't have built-in early stopping, here are practical approaches:

### Option 1: Fixed Epochs with Shorter Training
Modify the configs to use fewer epochs (e.g., 50-100 instead of 315):

```bash
# Edit each config file
for pct in 010 020 030 040 050 060 070 080 090 100; do
  sed -i 's/epochs: 315/epochs: 100/' \
    configs/train/vitg16/ablation/droid-256px-8f_${pct}pct.yaml
done
```

### Option 2: Manual Early Stopping
Monitor training logs and stop manually when validation loss plateaus:

```bash
# In one terminal, train
python -m app.main --fname configs/train/vitg16/ablation/droid-256px-8f_010pct.yaml --devices cuda:0

# In another terminal, monitor loss
tail -f /data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_010pct/log_r0.csv
```

Stop training (Ctrl+C) when loss stops improving for 10-20 epochs.

### Option 3: Checkpoint-Based Evaluation
Train for a fixed number of epochs, save checkpoints frequently (`save_every_freq: 5` in config), then evaluate all checkpoints on validation set and pick the best one.

## Expected Results

If your hypothesis is correct, you should see:
- **Validation loss decreases** as training data increases
- **Average distance to target decreases** as training data increases
- **Planning performance improves** with more data

## Files Created

```
/home/s185927/thesis/vjepa2/
├── scripts/
│   ├── create_data_splits.py         # Generate train/val splits
│   ├── create_configs.py              # Generate config files
│   ├── train_ablation_study.sh        # Master training script
│   ├── evaluate_validation_loss.py    # Validation evaluation
│   └── ABLATION_STUDY_GUIDE.md        # This file
├── configs/train/vitg16/ablation/
│   ├── droid-256px-8f_010pct.yaml
│   ├── droid-256px-8f_020pct.yaml
│   └── ... (10 config files total)
└── logs/ablation/
    └── (training logs will go here)

/data/s185927/droid_sim/y_axis/splits/
├── train_010pct.csv
├── train_020pct.csv
├── ...
├── train_100pct.csv
├── val_trajectories.csv
└── split_metadata.txt
```

## Quick Start

To start training immediately:

```bash
cd /home/s185927/thesis/vjepa2
./scripts/train_ablation_study.sh
```

## Troubleshooting

1. **Out of memory**: Reduce `batch_size` in configs (currently 8)
2. **Training too slow**: Reduce `epochs` in configs (currently 315)
3. **Checkpoint not found**: Check output folder in config matches actual location
4. **Validation script fails**: Ensure checkpoint exists at expected path

## Next Steps

1. **Start training**: Run `train_ablation_study.sh`
2. **Monitor progress**: Check logs in `/home/s185927/thesis/vjepa2/logs/ablation/`
3. **Evaluate validation loss**: Run `evaluate_validation_loss.py` after training
4. **Evaluate planning**: Integrate planning evaluation with your test set
5. **Plot results**: Create plots showing performance vs. data size

Good luck with your ablation study!
