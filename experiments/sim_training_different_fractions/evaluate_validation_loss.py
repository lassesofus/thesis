#!/usr/bin/env python3
"""
Evaluate validation loss for trained models.

This script loads each trained checkpoint and evaluates its
performance on the validation set.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import yaml

# Add parent directory to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vjepa_droid.droid import init_data
from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_video_model


def evaluate_model(
    checkpoint_path,
    val_data_path,
    config,
    device='cuda:0',
):
    """
    Evaluate a trained model on validation set.

    Args:
        checkpoint_path: Path to model checkpoint
        val_data_path: Path to validation CSV
        config: Training configuration dict
        device: Device to use

    Returns:
        Dictionary with validation metrics
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Extract config parameters
    cfgs_data = config.get('data', {})
    cfgs_model = config.get('model', {})
    cfgs_loss = config.get('loss', {})
    cfgs_meta = config.get('meta', {})

    batch_size = cfgs_data.get('batch_size', 8)
    crop_size = cfgs_data.get('crop_size', 256)
    patch_size = cfgs_data.get('patch_size', 16)
    tubelet_size = cfgs_data.get('tubelet_size', 2)
    fps = cfgs_data.get('fps', 4)
    camera_views = cfgs_data.get('camera_views', ['left_mp4_path'])
    num_workers = cfgs_data.get('num_workers', 4)
    dataset_fpcs = cfgs_data.get('dataset_fpcs', [8])
    max_num_frames = max(dataset_fpcs)

    model_name = cfgs_model.get('model_name', 'vit_giant_xformers')
    pred_depth = cfgs_model.get('pred_depth', 24)
    pred_num_heads = cfgs_model.get('pred_num_heads', 16)
    pred_embed_dim = cfgs_model.get('pred_embed_dim', 1024)
    pred_is_frame_causal = cfgs_model.get('pred_is_frame_causal', True)
    uniform_power = cfgs_model.get('uniform_power', True)
    use_rope = cfgs_model.get('use_rope', True)
    use_extrinsics = cfgs_model.get('use_extrinsics', False)
    use_silu = cfgs_model.get('use_silu', False)
    use_pred_silu = cfgs_model.get('use_pred_silu', False)
    wide_silu = cfgs_model.get('wide_silu', True)

    loss_exp = cfgs_loss.get('loss_exp', 1.0)
    normalize_reps = cfgs_loss.get('normalize_reps', True)
    auto_steps = cfgs_loss.get('auto_steps', 2)

    use_sdpa = cfgs_meta.get('use_sdpa', True)
    which_dtype = cfgs_meta.get('dtype', 'bfloat16')

    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    tokens_per_frame = int((crop_size // patch_size) ** 2)

    # Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device)
        torch.cuda.set_device(device)

    # Initialize model
    print("Initializing model...")
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=512,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=7,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=False,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'], strict=False)
    predictor.load_state_dict(checkpoint['predictor'], strict=False)

    target_encoder = torch.nn.Module()
    if 'target_encoder' in checkpoint:
        import copy
        target_encoder = copy.deepcopy(encoder)
        target_encoder.load_state_dict(checkpoint['target_encoder'], strict=False)
    else:
        target_encoder = encoder

    encoder.eval()
    predictor.eval()
    target_encoder.eval()

    # Initialize validation data loader
    print("Loading validation data...")
    transform = make_transforms(
        random_horizontal_flip=False,  # No augmentation for validation
        random_resize_aspect_ratio=[1.0, 1.0],
        random_resize_scale=[1.777, 1.777],
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    val_loader, _ = init_data(
        data_path=val_data_path,
        batch_size=batch_size,
        frames_per_clip=max_num_frames,
        tubelet_size=1,
        fps=fps,
        camera_views=camera_views,
        camera_frame=False,
        stereo_view=False,
        transform=transform,
        collator=torch.utils.data.default_collate,
        num_workers=num_workers,
        world_size=1,
        pin_mem=True,
        persistent_workers=False,
        rank=0,
    )

    # Evaluate
    print("Evaluating on validation set...")
    total_loss = 0.0
    total_jloss = 0.0
    total_sloss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                clips = batch[0].to(device, non_blocking=True)
                actions = batch[1].to(device, dtype=torch.float, non_blocking=True)
                states = batch[2].to(device, dtype=torch.float, non_blocking=True)
                extrinsics = batch[3].to(device, dtype=torch.float, non_blocking=True)

                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    # Target encoding
                    c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
                    h = target_encoder(c)
                    h = h.view(clips.size(0), max_num_frames, -1, h.size(-1)).flatten(1, 2)
                    if normalize_reps:
                        h = F.layer_norm(h, (h.size(-1),))

                    # Teacher-forced predictions
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
                        _a = actions[:, :n + 1]
                        _s = states[:, :n + 1]
                        _e = extrinsics[:, :n + 1]
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
                total_jloss += jloss.item()
                total_sloss += sloss.item()
                num_batches += 1

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches...")

            except Exception as e:
                print(f"Warning: Batch {batch_idx} failed with error: {e}")
                continue

    avg_loss = total_loss / max(num_batches, 1)
    avg_jloss = total_jloss / max(num_batches, 1)
    avg_sloss = total_sloss / max(num_batches, 1)

    print(f"\nValidation Results:")
    print(f"  Total Loss: {avg_loss:.4f}")
    print(f"  Teacher-Forced Loss: {avg_jloss:.4f}")
    print(f"  Auto-Regressive Loss: {avg_sloss:.4f}")
    print(f"  Batches evaluated: {num_batches}")

    return {
        'checkpoint': checkpoint_path,
        'val_loss': avg_loss,
        'val_jloss': avg_jloss,
        'val_sloss': avg_sloss,
        'num_batches': num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate validation loss')
    parser.add_argument(
        '--percentages',
        type=int,
        nargs='+',
        default=list(range(10, 101, 10)),
        help='Percentages to evaluate',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/s185927/thesis/vjepa2/results/ablation_validation_results.json',
        help='Output JSON file for results',
    )

    args = parser.parse_args()

    config_dir = Path('/home/s185927/thesis/vjepa2/configs/train/vitg16/ablation')
    val_data_path = '/data/s185927/droid_sim/y_axis/splits/val_trajectories.csv'
    weights_base = Path('/data/s185927/vjepa2/weights/droid')

    results = []

    for pct in args.percentages:
        print(f"\n{'=' * 70}")
        print(f"Evaluating {pct}% model")
        print(f"{'=' * 70}")

        # Load config
        config_file = config_dir / f'droid-256px-8f_{pct:03d}pct.yaml'
        if not config_file.exists():
            print(f"Config not found: {config_file}")
            continue

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Find checkpoint (use latest.pt or best checkpoint)
        folder_name = f'4.8.vitg16-256px-8f_{pct:03d}pct'
        checkpoint_dir = weights_base / folder_name
        checkpoint_path = checkpoint_dir / 'latest.pt'

        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            # Try to find any .pt file
            checkpoints = list(checkpoint_dir.glob('*.pt'))
            if checkpoints:
                checkpoint_path = checkpoints[0]
                print(f"Using checkpoint: {checkpoint_path}")
            else:
                print(f"No checkpoints found in {checkpoint_dir}")
                continue

        # Evaluate
        result = evaluate_model(
            checkpoint_path=str(checkpoint_path),
            val_data_path=val_data_path,
            config=config,
            device=args.device,
        )
        result['percentage'] = pct
        results.append(result)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 70}")

    # Print summary
    print("\nSummary:")
    print(f"{'Percentage':<12} {'Val Loss':<12} {'TF Loss':<12} {'AR Loss':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['percentage']:<12} {r['val_loss']:<12.4f} {r['val_jloss']:<12.4f} {r['val_sloss']:<12.4f}")


if __name__ == '__main__':
    main()
