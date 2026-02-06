# Batch Scheduler Commands

Uses OpenAI Batch API: https://platform.openai.com/docs/guides/batch

## Overview

Processes ~80K frame pairs through OpenAI's Batch API with 40M tokens/day rate limit.

| Metric | Value |
|--------|-------|
| Total pairs | 80,614 (~80,348 valid after filtering) |
| Total tokens | ~321M (80K × 4K tokens/pair) |
| Daily limit | 40M tokens |
| Batches | 269 (300 pairs each) |
| Batches/day | 33 |
| Estimated days | ~9 |

### Token breakdown per pair (actual from batch 1)
- Input: ~2,617 tokens (prompt + 2 base64 images)
- Output: ~1,199 tokens (5 instructions + descriptions)
- Total: ~3,816 tokens/pair

### Cost estimate (gpt-5-mini batch pricing)
| Type | Rate | Total tokens | Cost |
|------|------|--------------|------|
| Input | $0.125/1M | ~210M | ~$26 |
| Output | $1.00/1M | ~96M | ~$96 |
| **Total** | | ~306M | **~$122** |

### Batch API limits (from OpenAI docs)
- File size: **200 MB max** per batch file
- Requests: **50,000 max** per batch
- Completion window: **24 hours**
- We use: ~300 requests/batch, ~150-200 MB/file

## Quick Status Check

```bash
cd /home/s185927/thesis/experiments/04_language_grounding
python batch_scheduler.py status --output_dir ./batch_run
```

## Monitor Logs

```bash
# Real-time log monitoring
tail -f batch_run/logs/scheduler.log

# Last 50 log entries
tail -50 batch_run/logs/scheduler.log
```

## Manual Commands

### Run a scheduler tick (check status, download results, submit new batches)
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate vjepa2-312
python batch_scheduler.py tick --output_dir ./batch_run
```

### Force submit one batch (bypass daily limit check)
```bash
python batch_scheduler.py tick --output_dir ./batch_run --force
```

### Merge completed results into final output
```bash
python batch_scheduler.py merge --output_dir ./batch_run
```

### Retry failed batches
```bash
python batch_scheduler.py retry-failed --output_dir ./batch_run
```

## Output Files

| File | Description |
|------|-------------|
| `batch_run/batch_state.json` | Master state tracking all batches |
| `batch_run/batches/batch_part_XXX.jsonl` | JSONL request files (269 total) |
| `batch_run/results/results_part_XXX.json` | Raw API results per batch |
| `batch_run/final/all_instructions.json` | Merged results with full metadata |
| `batch_run/logs/scheduler.log` | Execution logs |

## Cron Jobs (already configured)

```bash
# View current cron jobs
crontab -l

# Edit cron jobs
crontab -e
```

Current schedule:
- `4 0 * * *` - Daily at 00:04 UTC: submit new batches
- `5 */6 * * *` - Every 6 hours: check status & download results

## Data Paths

| Path | Description |
|------|-------------|
| `/data/s185927/droid_raw/sampled_pairs/frame_pairs.json` | Original 80K frame pairs |
| `/data/s185927/droid_raw/sampled_pairs/` | Frame images (traj_XXX/frames/) |
| `/data/s185927/droid_raw/vjepa_features/` | VJEPA features |

## Result Format

After running `merge`, each entry in `all_instructions.json` contains:

```json
{
  "pair_index": 0,
  "trajectory_id": "AUTOLab+5d05c5aa+2023-07-07-09h-42m-23s",
  "frame_k": 90,
  "frame_k_d": 120,
  "d": 30,
  "frame_k_path": "traj_000/frames/frame_00090.jpg",
  "frame_k_d_path": "traj_000/frames/frame_00120.jpg",
  "position_delta": 0.124,
  "gripper_delta": 0.0,
  "z_delta": -0.115,
  "rotation_delta": 0.119,
  "salience_tier": "A",
  "instructions": [
    "Grab the yellow cup",
    "Pick up the yellow cup",
    "Grasp the small yellow cup by the rim",
    "Move the yellow cup into the gripper",
    "Close the gripper on that yellow cup"
  ],
  "what_changed": "Robot grabbed the yellow cup",
  "frame1_description": "...",
  "frame2_description": "..."
}
```

## Troubleshooting

### Check if tick is still running
```bash
ps aux | grep batch_scheduler
```

### Check OpenAI API status for a specific batch
```bash
python -c "
from instruction_generator_v3 import OpenAIBatchGenerator
gen = OpenAIBatchGenerator()
print(gen.check_batch_status('BATCH_ID_HERE'))
"
```

### Reinitialize from scratch (WARNING: deletes all progress)
```bash
rm -rf batch_run
python batch_scheduler.py init \
  --pairs_file /data/s185927/droid_raw/sampled_pairs/frame_pairs.json \
  --base_dir /data/s185927/droid_raw/sampled_pairs \
  --output_dir ./batch_run \
  --model gpt-5-mini \
  --daily_limit 40000000 \
  --pairs_per_batch 300

source /opt/conda/etc/profile.d/conda.sh && conda activate vjepa2-312
python batch_scheduler.py create-batches --output_dir ./batch_run
```
