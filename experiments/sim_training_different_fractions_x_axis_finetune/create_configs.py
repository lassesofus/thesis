#!/usr/bin/env python3
"""
Create training config files for finetuning the action-conditioned predictor.

This script generates YAML configs for finetuning with different fractions
of training data. Key differences from previous training:
- Loads the pretrained action-conditioned predictor from torch cache
- Freezes both context and target encoders (enc_lr_scale: 0.0)
- Only trains the predictor module
"""

import argparse
import os
from pathlib import Path

import yaml


def create_configs(
    base_config: str,
    output_dir: str,
    splits_dir: str,
    percentages: list = None,
    pretrain_checkpoint: str = None,
):
    """
    Create config files for each data percentage.

    Args:
        base_config: Path to base config YAML
        output_dir: Directory to save new configs
        splits_dir: Directory containing data split CSVs
        percentages: List of percentages (default [25, 50, 75, 100])
        pretrain_checkpoint: Path to pretrained action-conditioned model
    """
    if percentages is None:
        percentages = [25, 50, 75, 100]

    if pretrain_checkpoint is None:
        pretrain_checkpoint = "/home/s185927/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt"

    # Load base config
    print(f"Loading base config: {base_config}")
    with open(base_config, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits_path = Path(splits_dir)

    for pct in percentages:
        # Create a deep copy of base config
        cfg = yaml.safe_load(yaml.dump(base_cfg))

        # Update datasets to use the split CSVs
        train_csv = str(splits_path / f"train_{pct:03d}pct.csv")
        val_csv = str(splits_path / "val_trajectories.csv")

        # Update data section
        cfg['data']['datasets'] = [train_csv]

        # Add validation dataset
        if 'val_data' not in cfg:
            cfg['val_data'] = yaml.safe_load(yaml.dump(cfg['data']))
        cfg['val_data']['datasets'] = [val_csv]

        # Update output folder to include percentage and indicate finetuning
        cfg['folder'] = f"/data/s185927/vjepa2/weights/droid/x_axis_finetune_{pct:03d}pct"

        # === KEY CHANGES FOR PREDICTOR FINETUNING ===

        # 1. Use the action-conditioned pretrained checkpoint
        cfg['meta']['pretrain_checkpoint'] = pretrain_checkpoint

        # 2. Load the pretrained predictor (not train from scratch)
        cfg['meta']['load_predictor'] = True

        # 3. Load encoders from checkpoint
        cfg['meta']['load_encoder'] = True

        # 4. Use target_encoder for both context and target (as in AC model)
        cfg['meta']['context_encoder_key'] = 'target_encoder'
        cfg['meta']['target_encoder_key'] = 'target_encoder'

        # 5. FREEZE ENCODERS: Set enc_lr_scale to 0.0
        # This is the critical setting that was missing before!
        cfg['optimization']['enc_lr_scale'] = 0.0

        # 6. FINETUNING-SPECIFIC LR SETTINGS
        # Use lower learning rate for finetuning pretrained predictor
        cfg['optimization']['lr'] = 0.0001          # 4x lower than original (was 0.000425)
        cfg['optimization']['start_lr'] = 2e-5     # Lower start LR
        cfg['optimization']['warmup'] = 5           # Shorter warmup (was 15)
        cfg['optimization']['epochs'] = 100         # Fewer max epochs (early stopping will likely trigger first)

        # 7. Early stopping settings
        cfg['meta']['early_stopping_patience'] = 50
        cfg['meta']['val_freq'] = 5
        cfg['meta']['early_stopping_min_delta'] = 0.001

        # 7. Save checkpoints more frequently for analysis
        cfg['meta']['save_every_freq'] = 25

        # Save config
        output_file = output_path / f"x_axis_finetune_{pct:03d}pct.yaml"
        with open(output_file, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        print(f"Created config for {pct}%: {output_file}")
        print(f"  Train: {train_csv}")
        print(f"  Val: {val_csv}")
        print(f"  Output: {cfg['folder']}")
        print(f"  Pretrained: {pretrain_checkpoint}")
        print(f"  load_predictor: {cfg['meta']['load_predictor']}")
        print(f"  enc_lr_scale: {cfg['optimization']['enc_lr_scale']} (encoders frozen)")
        print()

    print(f"\nAll configs saved to: {output_dir}")
    print("\nIMPORTANT: enc_lr_scale=0.0 ensures encoders are frozen!")
    print("Only the predictor module will be trained.")


def main():
    parser = argparse.ArgumentParser(description="Create V-JEPA predictor finetuning configs")
    parser.add_argument(
        "--base_config",
        type=str,
        default="/home/s185927/thesis/vjepa2/configs/train/vitg16/droid-256px-8f.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/s185927/thesis/vjepa2/configs/train/vitg16/x_axis_finetune",
        help="Directory to save new configs",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/data/s185927/droid_sim/axis_aligned/x/splits",
        help="Directory containing data split CSVs",
    )
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default="/home/s185927/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt",
        help="Path to pretrained action-conditioned model",
    )
    parser.add_argument(
        "--percentages",
        type=int,
        nargs="+",
        default=None,
        help="Percentages to create configs for (default: 25 50 75 100)",
    )

    args = parser.parse_args()

    create_configs(
        base_config=args.base_config,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        percentages=args.percentages,
        pretrain_checkpoint=args.pretrain_checkpoint,
    )


if __name__ == "__main__":
    main()
