#!/usr/bin/env python3
"""
Verify that encoder parameters are not being updated during training.

This script loads a trained model and compares encoder weights with the
original pretrained checkpoint to confirm they are identical.
"""

import argparse
import sys
from pathlib import Path

import torch


def compare_state_dicts(dict1, dict2, prefix="", rtol=1e-5, atol=1e-8):
    """Compare two state dicts and return differences."""
    differences = []

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    missing_in_2 = keys1 - keys2
    missing_in_1 = keys2 - keys1
    common_keys = keys1 & keys2

    if missing_in_2:
        differences.append(f"Keys in first but not second: {missing_in_2}")
    if missing_in_1:
        differences.append(f"Keys in second but not first: {missing_in_1}")

    for key in common_keys:
        if prefix and not key.startswith(prefix):
            continue

        t1 = dict1[key]
        t2 = dict2[key]

        if t1.shape != t2.shape:
            differences.append(f"{key}: shape mismatch {t1.shape} vs {t2.shape}")
            continue

        if not torch.allclose(t1, t2, rtol=rtol, atol=atol):
            max_diff = (t1 - t2).abs().max().item()
            mean_diff = (t1 - t2).abs().mean().item()
            differences.append(f"{key}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")

    return differences


def main():
    parser = argparse.ArgumentParser(description="Verify encoder is frozen")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="/home/s185927/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt",
        help="Path to pretrained checkpoint",
    )
    parser.add_argument(
        "--trained",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g., best.pt or latest.pt)",
    )
    parser.add_argument(
        "--encoder_key",
        type=str,
        default="target_encoder",
        help="Key for encoder in pretrained checkpoint",
    )

    args = parser.parse_args()

    print(f"Loading pretrained checkpoint: {args.pretrained}")
    pretrained = torch.load(args.pretrained, map_location="cpu")

    print(f"Loading trained checkpoint: {args.trained}")
    trained = torch.load(args.trained, map_location="cpu")

    # Get encoder state dicts
    pretrained_encoder = pretrained.get(args.encoder_key, pretrained.get("encoder", {}))
    trained_encoder = trained.get("encoder", {})

    # Clean up keys (remove 'module.' prefix if present)
    def clean_keys(d):
        return {k.replace("module.", "").replace("backbone.", ""): v for k, v in d.items()}

    pretrained_encoder = clean_keys(pretrained_encoder)
    trained_encoder = clean_keys(trained_encoder)

    print(f"\nPretrained encoder keys: {len(pretrained_encoder)}")
    print(f"Trained encoder keys: {len(trained_encoder)}")

    # Compare encoder weights
    print("\n" + "="*60)
    print("Comparing ENCODER weights...")
    print("="*60)

    differences = compare_state_dicts(pretrained_encoder, trained_encoder)

    if not differences:
        print("\n✓ ENCODER weights are IDENTICAL!")
        print("  The encoder was successfully frozen during training.")
    else:
        print(f"\n✗ Found {len(differences)} differences in encoder weights:")
        for diff in differences[:10]:  # Show first 10
            print(f"  - {diff}")
        if len(differences) > 10:
            print(f"  ... and {len(differences) - 10} more")
        print("\n  WARNING: Encoder weights changed during training!")

    # Also check predictor to confirm it was trained
    print("\n" + "="*60)
    print("Comparing PREDICTOR weights...")
    print("="*60)

    pretrained_predictor = pretrained.get("predictor", {})
    trained_predictor = trained.get("predictor", {})

    pretrained_predictor = clean_keys(pretrained_predictor)
    trained_predictor = clean_keys(trained_predictor)

    predictor_diffs = compare_state_dicts(pretrained_predictor, trained_predictor)

    if not predictor_diffs:
        print("\n  Predictor weights are IDENTICAL (no training occurred?)")
    else:
        print(f"\n✓ Predictor has {len(predictor_diffs)} changed parameters")
        print("  The predictor was successfully trained.")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    encoder_frozen = len(differences) == 0
    predictor_trained = len(predictor_diffs) > 0

    if encoder_frozen and predictor_trained:
        print("✓ SUCCESS: Encoder frozen, predictor trained")
        return 0
    elif not encoder_frozen:
        print("✗ FAILURE: Encoder was modified during training")
        return 1
    else:
        print("? UNCLEAR: Predictor not trained")
        return 2


if __name__ == "__main__":
    sys.exit(main())
