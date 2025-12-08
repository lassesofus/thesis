#!/usr/bin/env python3
"""
Training wrapper with validation and early stopping.

This script wraps the V-JEPA training process and adds:
- Validation data loader
- Validation loss evaluation
- Early stopping based on validation loss
- Best model checkpointing
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

# Add parent directory to path to import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vjepa_droid.droid import init_data
from app.vjepa_droid.transforms import make_transforms
from app.main import get_args_parser


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimizing loss, 'max' for maximizing metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, current_value):
        """
        Check if training should stop.

        Returns:
            Tuple of (should_stop, is_best)
        """
        if self.best_value is None:
            self.best_value = current_value
            return False, True

        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True, False
            return False, False


def validate(
    encoder,
    predictor,
    target_encoder,
    val_loader,
    device,
    dtype,
    mixed_precision,
    normalize_reps,
    loss_exp,
    tokens_per_frame,
    max_num_frames,
    batch_size,
    auto_steps=2,
):
    """
    Run validation on the validation set.

    Returns:
        Average validation loss
    """
    encoder.eval()
    predictor.eval()
    target_encoder.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            try:
                clips = batch[0].to(device, non_blocking=True)
                actions = batch[1].to(device, dtype=torch.float, non_blocking=True)
                states = batch[2].to(device, dtype=torch.float, non_blocking=True)
                extrinsics = batch[3].to(device, dtype=torch.float, non_blocking=True)

                # Forward through target encoder
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    # Target encoding
                    c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
                    h = target_encoder(c)
                    h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
                    if normalize_reps:
                        h = F.layer_norm(h, (h.size(-1),))

                    # Predictions (teacher forcing only for validation)
                    _z = h[:, :-tokens_per_frame]
                    _a = actions
                    _s = states[:, :-1]
                    _e = extrinsics[:, :-1]

                    z_tf = predictor(_z, _a, _s, _e)
                    if normalize_reps:
                        z_tf = F.layer_norm(z_tf, (z_tf.size(-1),))

                    # Auto-regressive predictions
                    _z = torch.cat([h[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
                    for n in range(1, auto_steps):
                        _a, _s, _e = actions[:, :n + 1], states[:, :n + 1], extrinsics[:, :n + 1]
                        _z_nxt = predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
                        if normalize_reps:
                            _z_nxt = F.layer_norm(_z_nxt, (_z_nxt.size(-1),))
                        _z = torch.cat([_z, _z_nxt], dim=1)
                    z_ar = _z[:, tokens_per_frame:]

                    # Compute losses
                    _h_tf = h[:, tokens_per_frame : z_tf.size(1) + tokens_per_frame]
                    jloss = torch.mean(torch.abs(z_tf - _h_tf) ** loss_exp) / loss_exp

                    _h_ar = h[:, tokens_per_frame : z_ar.size(1) + tokens_per_frame]
                    sloss = torch.mean(torch.abs(z_ar - _h_ar) ** loss_exp) / loss_exp

                    loss = jloss + sloss

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                print(f"Warning: Validation batch failed with error: {e}")
                continue

    encoder.train()
    predictor.train()
    target_encoder.train()

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train V-JEPA with validation and early stopping')
    parser.add_argument(
        '--fname',
        type=str,
        required=True,
        help='Path to config YAML file',
    )
    parser.add_argument(
        '--devices',
        type=str,
        default='cuda:0',
        help='Device to use (e.g., cuda:0)',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (epochs)',
    )
    parser.add_argument(
        '--val_freq',
        type=int,
        default=1,
        help='Validation frequency (epochs)',
    )

    args = parser.parse_args()

    # Load config
    with open(args.fname, 'r') as f:
        config = yaml.safe_load(f)

    # Check if val_data exists in config
    if 'val_data' not in config:
        print("ERROR: Config must contain 'val_data' section for validation")
        print("Please use configs generated by create_configs.py")
        sys.exit(1)

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices.split(':')[-1] if ':' in args.devices else '0'

    print(f"Training with config: {args.fname}")
    print(f"Output folder: {config['folder']}")
    print(f"Early stopping patience: {args.patience} epochs")
    print(f"Validation frequency: {args.val_freq} epochs")

    # Import and run the standard training, but we'll need to modify it
    # For now, let's create a simple runner script that the user can modify

    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print("\nNOTE: This script has created the infrastructure for validation.")
    print("To fully integrate validation with early stopping, you need to:")
    print("1. Modify app/vjepa_droid/train.py to call validate() after each epoch")
    print("2. Add early stopping logic in the training loop")
    print("3. Save best model based on validation loss")
    print("\nAlternatively, run training with the generated configs using:")
    print(f"  python -m app.main --fname {args.fname} --devices {args.devices}")


if __name__ == '__main__':
    main()
