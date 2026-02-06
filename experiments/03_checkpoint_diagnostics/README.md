# Checkpoint-Trajectory Diagnostics for 100% Predictor Fine-Tuning

Diagnose why predictor-only fine-tuning (100% data) degrades planning by tracking diagnostic metrics across training checkpoints.

## Quick Start

```bash
cd /home/s185927/thesis/experiments/03_checkpoint_diagnostics

# 1. Train with frequent checkpoints (saves every 5 epochs)
./scripts/train_with_frequent_checkpoints.sh

# 2. Run all diagnostics on checkpoints
python scripts/run_all_diagnostics.py \
    --checkpoint_dir /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics \
    --output_dir diagnostics

# 3. Generate plots
python scripts/plot_diagnostics.py \
    --results diagnostics/results.csv \
    --output_dir plots
```

## Directory Structure

```
03_checkpoint_diagnostics/
├── configs/
│   └── x_axis_finetune_100pct_diagnostics.yaml  # Training config
├── scripts/
│   ├── train_with_frequent_checkpoints.sh       # Step 1: Training
│   ├── run_all_diagnostics.py                   # Step 2: Run all probes
│   ├── probe_a_energy_landscape.py              # Sharpness + alignment
│   ├── probe_a_with_correlation.py              # Extended: + E(a) vs D(a) correlation
│   ├── probe_b_onpolicy_loss.py                 # On-policy prediction loss
│   ├── probe_c_planning.py                      # Planning evaluation
│   ├── plot_diagnostics.py                      # Step 3: Visualization
│   ├── run_correlation_analysis.sh              # Run correlation analysis on all epochs
│   ├── plot_correlation_analysis.py             # Plot correlation results
│   ├── visualize_energy_landscape.py            # Energy heatmaps (epochs side-by-side)
│   └── visualize_energy_landscape_grid.py       # Energy heatmaps (epochs × planning steps)
├── diagnostics/
│   ├── probe_ids.txt                            # 25 fixed test trajectory IDs
│   ├── actions_seed_42_M512.npy                 # 512 fixed sampled actions
│   ├── results.csv                              # Per-epoch metrics (output)
│   ├── raw/                                     # Per-epoch raw data (output)
│   └── correlation_analysis/                    # Correlation analysis output
└── plots/                                       # Generated plots (output)
```

## Diagnostic Probes

| Probe | Metrics | What it measures |
|-------|---------|------------------|
| A | `sharpness_stdE`, `sharpness_rangeE`, `align_cos` | Energy landscape quality and gradient-goal alignment |
| A+ | `spearman_corr`, `pearson_corr` + full E(a), D(a) arrays | Whether energy ranking matches true action quality |
| B | `onpolicy_pred_loss` | Prediction accuracy on visited states (OOD test) |
| C | `planning_final_dist_mean`, `planning_delta_per_step` | Actual planning performance |

## Running Individual Probes

```bash
# Probe A only (energy landscape)
python scripts/probe_a_energy_landscape.py \
    --checkpoint /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e10.pt \
    --config configs/x_axis_finetune_100pct_diagnostics.yaml \
    --probe_ids_file diagnostics/probe_ids.txt \
    --actions_file diagnostics/actions_seed_42_M512.npy \
    --output diagnostics/probe_a_e10.json

# Probe B only (on-policy loss)
python scripts/probe_b_onpolicy_loss.py \
    --checkpoint /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e10.pt \
    --config configs/x_axis_finetune_100pct_diagnostics.yaml \
    --probe_ids_file diagnostics/probe_ids.txt \
    --output diagnostics/probe_b_e10.json

# Probe C only (planning)
python scripts/probe_c_planning.py \
    --checkpoint /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e10.pt \
    --config configs/x_axis_finetune_100pct_diagnostics.yaml \
    --probe_ids_file diagnostics/probe_ids.txt \
    --output diagnostics/probe_c_e10.json
```

## Expected Output

`diagnostics/results.csv`:
```
epoch,val_loss_dataset,sharpness_stdE,sharpness_rangeE,align_cos,onpolicy_pred_loss,planning_final_dist_mean,planning_delta_per_step
0,0.52,0.034,0.18,0.72,0.048,0.21,0.012
5,0.49,0.031,0.15,0.65,0.051,0.23,0.008
...
```

## Interpreting Results

**Driver 1 (Landscape Flattening)**: `sharpness_*` decreases → CEM cannot discriminate actions

**Driver 2 (Misalignment)**: `align_cos` drops toward 0 → gradient points away from goal

**Driver 3 (OOD Brittleness)**: `val_loss` improves but `onpolicy_pred_loss` worsens

The metric that changes **before** planning degrades is likely the causal driver.

## Energy-Action Correlation Analysis

The original Probe A measures energy landscape **spread** (σ(E), range) but not whether the energy **ranking** correctly identifies good actions. Since CEM is a ranking-based optimizer, we need to verify that lower energy actually corresponds to better actions.

### The Key Question

For sampled actions, does the energy E(a) correctly predict the true outcome D(a)?

```
E(a) = ||f(z_k, a) - z_goal||₁    # Energy: predicted latent distance to goal
D(a) = ||s_k + a[:3] - s_goal||   # True distance after applying action (geometric)
```

If `correlation(E, D) ≈ 1`: CEM's ranking is correct, spread is the only issue
If `correlation(E, D) → 0`: CEM's ranking is wrong, energy is noise

### Running the Correlation Analysis

```bash
cd /home/s185927/thesis/experiments/03_checkpoint_diagnostics/scripts

# Run extended Probe A on all checkpoints (saves full E(a) and D(a) arrays)
./run_correlation_analysis.sh

# Generate plots after completion
python plot_correlation_analysis.py \
    --input_dir ../diagnostics/correlation_analysis \
    --output_dir ../diagnostics/correlation_analysis
```

### Output

- `correlation_analysis/epoch_*_correlation.json` - Full data per checkpoint
- `correlation_analysis/correlation_vs_epoch.png` - Spearman/Pearson correlation over training
- `correlation_analysis/energy_distance_scatter.png` - E(a) vs D(a) scatter plots
- `correlation_analysis/combined_metrics.png` - Correlation alongside sharpness
- `correlation_analysis/correlation_summary.csv` - Summary table

### Interpreting Correlation Results

| σ(E) | Spearman corr | Interpretation |
|------|---------------|----------------|
| High | High (~1.0) | Healthy: good spread AND correct ranking |
| Low | High (~1.0) | Spread collapse: ranking correct but signal too weak |
| High | Low (~0) | Ranking corruption: spread exists but is noise |
| Low | Low (~0) | Complete failure: no spread AND wrong ranking |

### Why Spearman (Rank) Correlation?

CEM only uses **ranking** to select top-k actions—it doesn't care about absolute energy values. Spearman correlation measures ranking agreement:

- Spearman = 1.0: Perfect ranking (E and D order actions identically)
- Spearman = 0.0: Random ranking (E provides no signal for CEM)
- Spearman < 0: Inverted ranking (E actively misleads CEM)

Pearson correlation is also computed but is sensitive to non-linear relationships that don't affect CEM.

## Training Config Changes (vs original 100%)

- `save_every_freq: 5` (checkpoints every 5 epochs)
- `early_stopping_patience: 100` (effectively disabled)
- `epochs: 60` (fixed endpoint)
- Output: `/data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/`

## Energy Landscape Visualization

### Side-by-side heatmaps (epochs comparison)

Visualize how the energy landscape flattens across training epochs:

```bash
python scripts/visualize_energy_landscape.py \
    --checkpoints \
        /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e0.pt \
        /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e25.pt \
        /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e55.pt \
    --epochs 0 25 55 \
    --config configs/x_axis_finetune_100pct_diagnostics.yaml \
    --output_dir plots
```

Output: `plots/energy_landscape_heatmaps.png`

### Grid visualization (epochs × planning steps)

Visualize the energy landscape across both training epochs (rows) and planning steps (columns):

```bash
# First run: compute grids and save cache (slow)
python scripts/visualize_energy_landscape_grid.py \
    --checkpoints \
        /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e0.pt \
        /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e15.pt \
        /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e35.pt \
        /data/s185927/vjepa2/weights/droid/x_axis_finetune_100pct_diagnostics/e55.pt \
    --epochs 0 15 35 55 \
    --config configs/x_axis_finetune_100pct_diagnostics.yaml \
    --output_dir plots \
    --n_steps 5

# Iterate on plotting (fast, loads from cache)
python scripts/visualize_energy_landscape_grid.py \
    --output_dir plots \
    --plot_only
```

Output:
- `plots/energy_landscape_grid.png` - Grid visualization
- `plots/energy_grid_cache.npz` - Cached data for fast re-plotting

The grid shows how:
- **Rows** (top to bottom): Landscape flattens as training progresses
- **Columns** (left to right): Landscape evolves during predictor-based planning rollout
