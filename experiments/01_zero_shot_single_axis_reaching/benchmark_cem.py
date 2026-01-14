#!/usr/bin/env python3
"""
Benchmark CEM planning performance with controlled timing.

Usage:
    # Baseline (no optimizations)
    python benchmark_cem.py

    # Test cudnn.benchmark + TF32
    OPT_CUDNN=1 python benchmark_cem.py

    # Test torch.compile
    OPT_COMPILE=1 python benchmark_cem.py

    # Test AMP (mixed precision)
    OPT_AMP=1 python benchmark_cem.py

    # Test all optimizations
    OPT_CUDNN=1 OPT_COMPILE=1 OPT_AMP=1 python benchmark_cem.py
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

# Setup paths
_vjepa_root = "/home/s185927/thesis/vjepa2"
if _vjepa_root not in sys.path:
    sys.path.insert(0, _vjepa_root)

from notebooks.utils.world_model_wrapper import WorldModel
from app.vjepa_droid.transforms import make_transforms


def benchmark_planning(world_model, z_n, s_n, z_goal, warmup=2, runs=5):
    """Run multiple planning iterations and report timing stats."""
    times = []

    with torch.no_grad():
        # Warmup runs (important for torch.compile JIT compilation)
        print(f"Warmup ({warmup} runs)...")
        for i in range(warmup):
            _ = world_model.infer_next_action(z_n, s_n, z_goal)
            torch.cuda.synchronize()
            print(f"  Warmup {i+1} complete")

        # Timed runs
        print(f"\nTimed runs ({runs} runs):")
        for i in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = world_model.infer_next_action(z_n, s_n, z_goal)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s")

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"\n{'='*40}")
    print(f"Mean: {mean_time:.3f}s, Std: {std_time:.3f}s")
    print(f"Min: {np.min(times):.3f}s, Max: {np.max(times):.3f}s")
    print(f"{'='*40}")
    return times


def forward_target(encoder, clips, tokens_per_frame, normalize_reps=True):
    """Encode video clips using the encoder (matches notebook implementation)."""
    B, C, T, H, W = clips.size()
    # Reshape: [B, C, T, H, W] -> [B*T, C, 2, H, W] (duplicate for temporal dim)
    c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def main():
    device = torch.device("cuda:0")

    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()

    # Check which optimizations are enabled
    opt_cudnn = os.environ.get('OPT_CUDNN', '0') == '1'
    opt_compile = os.environ.get('OPT_COMPILE', '0') == '1'
    opt_amp = os.environ.get('OPT_AMP', '0') == '1'

    print("Optimization flags:")
    print(f"  OPT_CUDNN={int(opt_cudnn)} (cudnn.benchmark + TF32)")
    print(f"  OPT_COMPILE={int(opt_compile)} (torch.compile)")
    print(f"  OPT_AMP={int(opt_amp)} (mixed precision)")
    print()

    # Apply cudnn optimizations BEFORE model loading
    if opt_cudnn:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled: cudnn.benchmark + TF32")

    # Load models
    print("\nLoading V-JEPA models...")
    encoder, predictor = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_ac_vit_giant"
    )
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    print("Models loaded.")

    # Apply torch.compile AFTER model loading
    if opt_compile:
        print("Applying torch.compile (this may take a moment)...")
        predictor = torch.compile(predictor, mode="reduce-overhead")
        print("Enabled: torch.compile")

    # Create world model
    crop_size = 256
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    # Use same MPC args as robo_samples.py
    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args={
            "rollout": 1,
            "samples": 800,
            "topk": 10,
            "cem_steps": 10,
            "momentum_mean": 0.15,
            "momentum_mean_gripper": 0.15,
            "momentum_std": 0.75,
            "momentum_std_gripper": 0.15,
            "maxnorm": 0.075,
            "verbose": False
        },
        normalize_reps=True,
        device=str(device)
    )

    # Store AMP flag for potential future use in world_model_wrapper
    # (Currently AMP needs to be integrated into the wrapper)
    if opt_amp:
        print("Note: OPT_AMP requires integration into world_model_wrapper.py")
        print("      This flag is prepared for future use.")

    # Create dummy inputs matching the notebook pattern
    # clips: [B=1, C=3, T=2, H=256, W=256] - two frames (current + goal)
    # states: [B=1, T=1, 7] - current pose
    print(f"\nCreating dummy inputs (crop_size={crop_size}, tokens_per_frame={tokens_per_frame})...")

    with torch.no_grad():
        # Create dummy video clips [B, C, T, H, W]
        dummy_clips = torch.randn(1, 3, 2, crop_size, crop_size, device=device)

        # Encode using forward_target (same as notebook)
        h = forward_target(encoder, dummy_clips, tokens_per_frame, normalize_reps=True)

        # Split into current frame rep and goal frame rep
        z_n = h[:, :tokens_per_frame].unsqueeze(1).clone()      # [1, 1, tokens_per_frame, D]
        z_goal = h[:, -tokens_per_frame:].unsqueeze(1).clone()  # [1, 1, tokens_per_frame, D]

        # Current pose [B, T, 7]
        s_n = torch.zeros(1, 1, 7, device=device, dtype=z_n.dtype)

        # Free encoder memory after encoding (we only need predictor for CEM)
        del h, dummy_clips
        # Also remove encoder reference from world_model (infer_next_action doesn't use it)
        del world_model.encoder
        del encoder
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("  Freed encoder from GPU memory")

    print(f"  z_n shape: {z_n.shape}")
    print(f"  z_goal shape: {z_goal.shape}")
    print(f"  s_n shape: {s_n.shape}")

    print("\n" + "="*50)
    print("=== CEM Planning Benchmark Results ===")
    print("="*50)
    print(f"Config: samples=800, cem_steps=10, rollout=1")
    print(f"Expected iterations: 800 * 10 * 1 = 8,000 predictor calls")
    print()

    benchmark_planning(world_model, z_n, s_n, z_goal, warmup=2, runs=5)


if __name__ == "__main__":
    main()
