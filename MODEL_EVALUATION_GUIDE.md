# V-JEPA Model Evaluation Guide

This guide explains how to evaluate and compare multiple V-JEPA models on reaching task test samples.

## Overview

The evaluation system tests trained V-JEPA models against the Meta baseline using CEM (Cross-Entropy Method) planning in simulation. For each test sample, it:

1. **Captures goal image**: Moves robot to target position using IK
2. **Returns to start**: Resets to initial position  
3. **Runs CEM planning**: Executes N planning steps using the model
4. **Measures performance**: Tracks distance to target at each step

## Files Created

### Evaluation Scripts

- **`eval_vjepa_models.py`**: Main evaluation script
  - Loads models from checkpoints or PyTorch Hub
  - Runs CEM planning on test samples
  - Saves detailed results per model
  - Location: `robohive/robohive/robohive/utils/eval_vjepa_models.py`

- **`plot_model_comparison.py`**: Visualization script
  - Creates bar plots comparing mean final distances
  - Shows success rates across models
  - Optional: Plots distance evolution over planning steps
  - Location: `robohive/robohive/robohive/utils/plot_model_comparison.py`

- **`run_model_evaluation.sh`**: Convenience runner script
  - Runs full evaluation pipeline
  - Creates all plots
  - Location: `/home/s185927/thesis/run_model_evaluation.sh`

## Quick Start

### Option 1: Run Everything (Recommended)

```bash
cd /home/s185927/thesis
./run_model_evaluation.sh
```

This will:
- Evaluate all 4 trained models + Meta baseline
- Save results to `/data/s185927/vjepa_eval_results/`
- Generate comparison plots automatically

### Option 2: Manual Execution

#### Step 1: Run Evaluation

```bash
python robohive/robohive/robohive/utils/eval_vjepa_models.py \
    --metadata /data/s185927/droid_sim/y_axis/trajectory_metadata.json \
    --model_dir /data/s185927/vjepa2/weights/droid \
    --planning_steps 5 \
    --out_dir /data/s185927/vjepa_eval_results \
    --checkpoint_name best.pt
```

#### Step 2: Create Plots

```bash
python robohive/robohive/robohive/utils/plot_model_comparison.py \
    --eval_dir /data/s185927/vjepa_eval_results \
    --out_path /data/s185927/vjepa_eval_results/model_comparison.png \
    --plot_evolution
```

### Option 3: Test Run (Limited Samples)

```bash
python robohive/robohive/robohive/utils/eval_vjepa_models.py \
    --metadata /data/s185927/droid_sim/y_axis/trajectory_metadata.json \
    --model_dir /data/s185927/vjepa2/weights/droid \
    --planning_steps 5 \
    --out_dir /tmp/test_eval \
    --max_samples 10 \
    --checkpoint_name best.pt
```

## Models Being Evaluated

The evaluation compares 5 models:

1. **4.8.vitg16-256px-8f_025pct** - Trained on 25% of data
2. **4.8.vitg16-256px-8f_050pct** - Trained on 50% of data
3. **4.8.vitg16-256px-8f_075pct** - Trained on 75% of data
4. **4.8.vitg16-256px-8f_100pct** - Trained on 100% of data
5. **meta_baseline** - Pre-trained Meta model (no fine-tuning)

**Hypothesis**: Models trained on more diverse data should achieve lower final distances.

## Test Dataset

- **Location**: `/data/s185927/droid_sim/y_axis/trajectory_metadata.json`
- **Split**: 200 test samples (20% of 1000 total trajectories)
- **Task**: Reaching targets along y-axis
- **Distance range**: 0.05m to 0.3m from start position

## Output Files

After running evaluation, you'll find:

```
/data/s185927/vjepa_eval_results/
├── evaluation_summary.json              # Overall statistics per model
├── 4.8.vitg16-256px-8f_025pct_results.json
├── 4.8.vitg16-256px-8f_050pct_results.json
├── 4.8.vitg16-256px-8f_075pct_results.json
├── 4.8.vitg16-256px-8f_100pct_results.json
├── meta_baseline_results.json
├── model_comparison.png                 # Bar plots comparing models
└── model_comparison_evolution.png       # Distance evolution over steps
```

### Result JSON Structure

Each model's result file contains per-sample metrics:

```json
[
  {
    "sample_idx": 0,
    "trajectory_index": 2,
    "final_distance": 0.0234,
    "success": true,
    "target_distance": 0.1234,
    "distances_per_step": [0.1234, 0.0856, 0.0512, 0.0345, 0.0234, 0.0234],
    "repr_distances_per_step": [0.0456, 0.0312, 0.0198, 0.0145, 0.0123, 0.0110]
  },
  ...
]
```

### Summary JSON Structure

The `evaluation_summary.json` contains aggregated statistics:

```json
{
  "model_name": {
    "mean_final_distance": 0.0456,
    "std_final_distance": 0.0234,
    "median_final_distance": 0.0412,
    "success_rate": 0.85,
    "num_samples": 200
  },
  ...
}
```

## Visualization Outputs

### 1. Bar Comparison (`model_comparison.png`)

Two-panel figure:
- **Left panel**: Mean final distance after 5 planning steps (with std error bars)
- **Right panel**: Success rate (% of samples reaching < 5cm from target)

Models sorted by performance (best to worst).

### 2. Distance Evolution (`model_comparison_evolution.png`)

Line plot showing how distance to target changes over planning steps for each model.
- Shaded regions show ± 1 standard deviation
- Red dashed line indicates success threshold (5cm)

## Configuration Options

### Evaluation Script Options

```bash
--metadata PATH          # Path to trajectory_metadata.json (required)
--model_dir PATH         # Directory with model subdirectories (required)
--planning_steps N       # Number of CEM planning steps (default: 5)
--out_dir PATH          # Output directory (required)
--horizon FLOAT         # Time per action execution in seconds (default: 3.0)
--max_samples N         # Limit test samples (default: all)
--checkpoint_name NAME  # Which checkpoint to load (default: best.pt)
--sim_path PATH         # MuJoCo XML path (default: franka_reach_v0.xml)
```

### Plotting Script Options

```bash
--eval_dir PATH         # Directory with evaluation results (required)
--out_path PATH         # Output plot path (required)
--plot_evolution        # Also create distance evolution plot (flag)
```

## Interpreting Results

### Success Criteria

- **Success**: Final distance < 5cm (0.05m) from target
- **Failure**: Final distance ≥ 5cm

### Expected Patterns

If the hypothesis is correct, you should see:
- Lower mean final distances for models trained on more data (100% < 75% < 50% < 25%)
- Higher success rates for models with more training diversity
- Meta baseline may underperform due to lack of domain-specific fine-tuning

### Key Metrics

- **Mean final distance**: Average error after 5 planning steps across all test samples
- **Success rate**: Percentage of test samples reaching target (< 5cm)
- **Distance evolution**: How quickly the model converges to the target

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:
```bash
# Reduce batch size in evaluation or run on CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU
```

### Missing Checkpoints

Ensure checkpoint files exist:
```bash
ls -lh /data/s185927/vjepa2/weights/droid/*/best.pt
```

### Test Samples Not Found

Verify metadata file:
```bash
jq 'map(select(.split == "test")) | length' /data/s185927/droid_sim/y_axis/trajectory_metadata.json
```

Should output: `200`

## Next Steps

After evaluation:

1. **Analyze results**: Check `evaluation_summary.json` for key statistics
2. **Compare plots**: Visual inspection of `model_comparison.png`
3. **Deep dive**: Examine per-sample results for failure cases
4. **Statistical tests**: Perform significance tests between models
5. **Ablation studies**: Test different planning_steps values (3, 5, 10)

## Notes

- Evaluation runs headless (offscreen rendering)
- Each model evaluation takes ~30-60 minutes for 200 samples (depending on hardware)
- Results are saved incrementally (safe to interrupt and resume)
- GPU strongly recommended (10-20x faster than CPU)

---

**Created**: 2025-11-24
**Models**: 4 trained checkpoints + 1 Meta baseline
**Test Samples**: 200 reaching trajectories (y-axis)
**Task**: Compare planning performance across training data diversity
