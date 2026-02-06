# V-JEPA Model Evaluation - Quick Reference

## What Was Created

### 1. Main Evaluation Script (`eval_vjepa_models.py`)
**Location**: `robohive/robohive/robohive/utils/eval_vjepa_models.py`

Evaluates all models on 200 test samples using CEM planning:
- Loads trained checkpoints OR Meta baseline from PyTorch Hub
- For each test sample: Move to target → Return home → Plan for 5 steps
- Tracks distance at each step + final distance
- Saves JSON results per model

### 2. Plotting Script (`plot_model_comparison.py`)
**Location**: `robohive/robohive/robohive/utils/plot_model_comparison.py`

Creates two visualizations:
- **Bar chart**: Mean final distance + success rate across models
- **Evolution plot**: How distance changes over planning steps

### 3. Convenience Runner (`run_model_evaluation.sh`)
**Location**: `/home/s185927/thesis/run_model_evaluation.sh`

One-command execution:
```bash
./run_model_evaluation.sh
```
Runs evaluation + creates all plots automatically.

### 4. Comprehensive Guide (`MODEL_EVALUATION_GUIDE.md`)
**Location**: `/home/s185927/thesis/MODEL_EVALUATION_GUIDE.md`

Full documentation with examples, troubleshooting, and interpretation.

## Quick Start

### Run Full Evaluation (All 200 Test Samples)

```bash
cd /home/s185927/thesis
./run_model_evaluation.sh
```

**Expected time**: 30-60 minutes (with GPU)  
**Output**: `/data/s185927/vjepa_eval_results/`

### Quick Test (10 Samples Only)

```bash
python robohive/robohive/robohive/utils/eval_vjepa_models.py \
    --metadata /data/s185927/droid_sim/y_axis/trajectory_metadata.json \
    --model_dir /data/s185927/vjepa2/weights/droid \
    --planning_steps 5 \
    --out_dir /tmp/quick_test \
    --max_samples 10
```

**Expected time**: 3-5 minutes

## Models Being Tested

| Model | Training Data | Checkpoint |
|-------|--------------|------------|
| 4.8.vitg16-256px-8f_025pct | 25% | `/data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_025pct/best.pt` |
| 4.8.vitg16-256px-8f_050pct | 50% | `/data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_050pct/best.pt` |
| 4.8.vitg16-256px-8f_075pct | 75% | `/data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_075pct/best.pt` |
| 4.8.vitg16-256px-8f_100pct | 100% | `/data/s185927/vjepa2/weights/droid/4.8.vitg16-256px-8f_100pct/best.pt` |
| meta_baseline | Pre-trained (no fine-tuning) | PyTorch Hub |

## Expected Output

```
/data/s185927/vjepa_eval_results/
├── evaluation_summary.json                    # Aggregate statistics
├── 4.8.vitg16-256px-8f_025pct_results.json   # Per-sample results
├── 4.8.vitg16-256px-8f_050pct_results.json
├── 4.8.vitg16-256px-8f_075pct_results.json
├── 4.8.vitg16-256px-8f_100pct_results.json
├── meta_baseline_results.json
├── model_comparison.png                       # Bar charts
└── model_comparison_evolution.png             # Line plot
```

## Key Metrics

- **Mean Final Distance**: Average error after 5 planning steps (meters)
- **Success Rate**: % of samples reaching < 5cm from target
- **Distance Evolution**: Convergence speed over planning steps

## Hypothesis

Models trained on more diverse data (higher %) should achieve:
- ✓ Lower mean final distances
- ✓ Higher success rates
- ✓ Faster convergence

## Next Steps After Evaluation

1. Check `evaluation_summary.json` for quick statistics
2. View `model_comparison.png` for visual comparison
3. Analyze failure cases in detailed result JSONs
4. Run statistical significance tests if needed
5. Create thesis plots/tables from results

## Troubleshooting

**CUDA OOM**: Reduce samples or use CPU  
**Missing checkpoints**: Verify paths with `ls /data/s185927/vjepa2/weights/droid/*/best.pt`  
**No test samples**: Check metadata with `jq 'map(select(.split == "test"))' <metadata.json>`

---

**Date**: 2025-11-24  
**Test Samples**: 200 (y-axis reaching tasks)  
**Planning Steps**: 5  
**Success Threshold**: 5cm
