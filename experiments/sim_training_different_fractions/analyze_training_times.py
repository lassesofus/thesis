#!/usr/bin/env python3
"""
Analyze training times from log files and checkpoints.

Extracts timing information per model and generates summary statistics.
"""

import json
import re
from pathlib import Path
from datetime import datetime, timedelta


def parse_log_timestamps(log_path):
    """Extract start and end timestamps from training log."""
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Find first and last timestamp lines
    timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]'
    
    start_time = None
    end_time = None
    
    for line in lines:
        match = re.search(timestamp_pattern, line)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            if start_time is None:
                start_time = timestamp
            end_time = timestamp  # Keep updating to get last one
    
    return start_time, end_time


def get_training_info_from_checkpoint(checkpoint_path):
    """Get epoch and iteration info from checkpoint."""
    import torch
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return {
        'epoch': ckpt.get('epoch', 'N/A'),
        'iteration': ckpt.get('itr', 'N/A'),
        'loss': ckpt.get('loss', 'N/A'),
        'val_loss': ckpt.get('val_loss', 'N/A'),
    }


def main():
    log_dir = Path('/home/s185927/thesis/vjepa2/logs/ablation')
    weights_dir = Path('/data/s185927/vjepa2/weights/droid')
    
    percentages = [25, 50, 75, 100]
    results = {}
    
    print("=" * 80)
    print("Training Time Analysis")
    print("=" * 80)
    print()
    
    for pct in percentages:
        pct_str = f"{pct:03d}"
        log_file = log_dir / f"train_{pct_str}pct.log"
        checkpoint = weights_dir / f"4.8.vitg16-256px-8f_{pct_str}pct" / "best.pt"
        
        if not log_file.exists():
            print(f"{pct}% model: Log not found")
            continue
        
        # Parse timestamps
        start_time, end_time = parse_log_timestamps(log_file)
        
        if start_time and end_time:
            duration = end_time - start_time
            hours = duration.total_seconds() / 3600
            
            result = {
                'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': int(duration.total_seconds()),
                'duration_hours': round(hours, 2),
                'duration_str': str(duration).split('.')[0],  # Remove microseconds
            }
            
            # Get checkpoint info if available
            if checkpoint.exists():
                try:
                    ckpt_info = get_training_info_from_checkpoint(checkpoint)
                    result.update(ckpt_info)
                except Exception as e:
                    result['checkpoint_error'] = str(e)
            
            results[f"{pct}pct"] = result
            
            print(f"{pct}% model:")
            print(f"  Start:    {result['start']}")
            print(f"  End:      {result['end']}")
            print(f"  Duration: {result['duration_str']} ({result['duration_hours']:.2f} hours)")
            if 'epoch' in result:
                print(f"  Epochs:   {result['epoch']}")
                print(f"  Loss:     {result.get('loss', 'N/A'):.4f}")
            print()
        else:
            print(f"{pct}% model: Could not parse timestamps")
            print()
    
    # Summary statistics
    if results:
        total_seconds = sum(r['duration_seconds'] for r in results.values())
        total_hours = total_seconds / 3600
        avg_hours = total_hours / len(results)
        
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Models trained: {len(results)}")
        print(f"Total time: {total_hours:.2f} hours ({total_seconds / 86400:.2f} days)")
        print(f"Average per model: {avg_hours:.2f} hours")
        print()
        
        # Save to JSON
        output_file = log_dir / "training_times.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved detailed timing info to: {output_file}")


if __name__ == '__main__':
    main()
