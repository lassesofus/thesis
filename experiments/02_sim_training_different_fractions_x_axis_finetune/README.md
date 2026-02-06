# X-Axis Predictor Finetuning Experiment

This experiment finetunes the V-JEPA 2 action-conditioned **predictor module only** on x-axis simulation data, with **frozen encoders**.

## Key Difference from Previous Training

In previous training (`sim_training_different_fractions/`), the encoder was inadvertently trained due to `enc_lr_scale` defaulting to 1.0. This experiment fixes that:

- **`enc_lr_scale: 0.0`** - Freezes both context and target encoders
- **`load_predictor: true`** - Loads pretrained action-conditioned predictor
- Only the predictor module parameters receive gradient updates

## Data

- **Source**: `/data/s185927/droid_sim/axis_aligned/x/`
- **Splits**: `/data/s185927/droid_sim/axis_aligned/x/splits/`
  - `train_025pct.csv` (25% of training data)
  - `train_050pct.csv` (50% of training data)
  - `train_075pct.csv` (75% of training data)
  - `train_100pct.csv` (100% of training data)
  - `val_trajectories.csv` (validation set)

## Pretrained Model

The finetuning starts from the pretrained action-conditioned model:
- **Checkpoint**: `/home/s185927/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt`

## How Encoder Freezing Works

In `app/vjepa_droid/utils.py`, the `init_opt` function creates separate parameter groups:

```python
param_groups = [
    {
        "params": (encoder params),
        "lr_scale": enc_lr_scale,  # When 0.0, encoder gets no learning rate
    },
    {
        "params": (predictor params),
        # No lr_scale = uses full learning rate
    },
    ...
]
```

When `enc_lr_scale=0.0`:
- Encoder parameters are still in optimizer but with learning rate = 0
- Gradients may flow through encoder but weights don't update
- Only predictor parameters are updated

## Usage

1. **Generate configs** (already done):
   ```bash
   python create_configs.py
   ```

2. **Run finetuning**:
   ```bash
   ./train_predictor_finetune_timed.sh
   ```

   Or without timing wrapper:
   ```bash
   ./train_predictor_finetune.sh
   ```

## Output

Finetuned models will be saved to:
- `/data/s185927/vjepa2/weights/droid/x_axis_finetune_025pct/`
- `/data/s185927/vjepa2/weights/droid/x_axis_finetune_050pct/`
- `/data/s185927/vjepa2/weights/droid/x_axis_finetune_075pct/`
- `/data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct/`

Each contains:
- `latest.pt` - Most recent checkpoint
- `best.pt` - Best validation loss checkpoint

## Logs

Training logs saved to:
- `/home/s185927/thesis/vjepa2/logs/x_axis_finetune/`

## Comparison

| Experiment | Encoder | Predictor | Data |
|------------|---------|-----------|------|
| Previous (y-axis) | **Trained** (bug) | Trained from scratch | y-axis |
| This (x-axis) | **Frozen** | Finetuned from pretrained | x-axis |

## Expected Behavior

Since we're finetuning an already-trained predictor:
- Training should converge faster (starting from good initialization)
- May require fewer epochs
- Early stopping should trigger earlier
- Should maintain encoder quality while adapting predictor to x-axis domain
