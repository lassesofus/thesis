#!/usr/bin/env python3
"""
Collect V-JEPA-2 embeddings from DROID and simulation for latent alignment experiment.

This script:
1. Samples N frames from DROID (left exocentric camera)
2. Samples N frames from RoboHive simulation (left camera)
3. Encodes all frames using frozen V-JEPA-2 encoder
4. Computes and saves mean/covariance statistics for both domains
5. Runs sanity checks (norms, condition numbers, PCA scatter)

Usage:
    python collect_embeddings.py --n_frames 1000 --device cuda:0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add paths for imports
sys.path.insert(0, "/home/s185927/thesis/vjepa2")
sys.path.insert(0, "/home/s185927/thesis")

from app.vjepa_droid.utils import init_video_model
from app.vjepa_droid.transforms import make_transforms

# Paths
DROID_INDEX = "/data/s185927/droid_raw/droid_index.csv"
SIM_DATA_DIR = "/data/s185927/droid_sim/zero_shot_correlation"
OUTPUT_DIR = Path("/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/inference_time_latent_alignment/stats")

# Model config (pretrained V-JEPA-2-AC encoder)
MODEL_CONFIG = {
    "checkpoint": "/home/s185927/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt",
    "crop_size": 256,
    "patch_size": 16,
    "model_name": "vit_giant_xformers",
}

# Experiment config
MIN_CLIP_DURATION_SECONDS = 4.0


def load_encoder(device: str = "cuda:0"):
    """Load the frozen V-JEPA-2 encoder."""
    print(f"Loading V-JEPA-2 encoder from {MODEL_CONFIG['checkpoint']}")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device)
        torch.cuda.set_device(device)

    encoder, predictor = init_video_model(
        uniform_power=True,
        device=device,
        patch_size=MODEL_CONFIG["patch_size"],
        max_num_frames=512,
        tubelet_size=2,
        model_name=MODEL_CONFIG["model_name"],
        crop_size=MODEL_CONFIG["crop_size"],
        pred_depth=24,
        pred_num_heads=16,
        pred_embed_dim=1024,
        action_embed_dim=7,
        pred_is_frame_causal=True,
        use_rope=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
    )

    # Load checkpoint
    checkpoint = torch.load(MODEL_CONFIG["checkpoint"], map_location=device)

    # Use target encoder (EMA weights, more stable)
    if "target_encoder" in checkpoint:
        encoder_dict = checkpoint["target_encoder"]
    else:
        encoder_dict = checkpoint["encoder"]

    encoder_dict = {k.replace("module.", ""): v for k, v in encoder_dict.items()}
    encoder.load_state_dict(encoder_dict, strict=False)
    encoder.eval()

    print(f"Encoder loaded. Param checksum: {sum(p.sum().item() for p in encoder.parameters()):.2f}")

    return encoder, device


def create_transform(crop_size: int = 256):
    """Create deterministic transform matching inference setup."""
    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )


def encode_single_frame(encoder, frame: np.ndarray, transform, device) -> np.ndarray:
    """
    Encode a single RGB frame using V-JEPA-2 encoder.

    Args:
        encoder: V-JEPA-2 encoder
        frame: RGB frame as numpy array (H, W, C), uint8
        transform: Preprocessing transform
        device: torch device

    Returns:
        embedding: Mean-pooled frame embedding (D,) as numpy array
    """
    # Transform expects [T, H, W, C], so add time dimension
    frame_batch = frame[np.newaxis, ...]  # [1, H, W, C]

    # Apply transform: outputs [C, 1, H, W], then unsqueeze to [1, C, 1, H, W]
    clips = transform(frame_batch).unsqueeze(0).to(device)

    B, C, T, H, W = clips.size()

    with torch.no_grad():
        # Permute to [B*T, C, H, W]
        c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)

        # Model expects tubelet_size=2, repeat temporal dim
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)

        # Encode: output shape [B*T, num_tokens, D]
        h = encoder(c)

        # Apply layer norm (matching planning code)
        h = F.layer_norm(h, (h.size(-1),))

        # Mean pool over tokens to get frame-level embedding
        embedding = h.mean(dim=1).squeeze(0).cpu().numpy()  # (D,)

    return embedding


def get_droid_video_path(episode_path: str, metadata: dict) -> Optional[Path]:
    """Resolve the left camera video path from DROID metadata."""
    if "left_mp4_path" not in metadata:
        return None

    episode_path = Path(episode_path)
    left_mp4_rel = metadata["left_mp4_path"]
    video_filename = Path(left_mp4_rel).name
    video_path = episode_path / "recordings" / "MP4" / video_filename

    if video_path.exists():
        return video_path

    return None


def get_droid_episodes(min_duration_seconds: float = 4.0) -> List[str]:
    """Get list of DROID episode paths with clips longer than min_duration."""
    with open(DROID_INDEX, "r") as f:
        all_episodes = [line.strip() for line in f if line.strip()]

    valid_episodes = []
    print(f"Filtering {len(all_episodes)} DROID episodes for duration >= {min_duration_seconds}s...")

    for episode_path in tqdm(all_episodes, desc="Checking DROID episodes"):
        try:
            metadata_files = list(Path(episode_path).glob("metadata_*.json"))
            if not metadata_files:
                continue

            with open(metadata_files[0], "r") as f:
                metadata = json.load(f)

            traj_length = metadata.get("trajectory_length", 0)
            duration = traj_length / 30.0

            if duration >= min_duration_seconds:
                video_path = get_droid_video_path(episode_path, metadata)
                if video_path is not None:
                    valid_episodes.append(episode_path)
        except Exception:
            continue

    print(f"Found {len(valid_episodes)} valid DROID episodes (>= {min_duration_seconds}s)")
    return valid_episodes


def sample_droid_frames(
    episodes: List[str],
    n_frames: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """Randomly sample RGB frames from DROID episodes (left exocentric camera)."""
    from decord import VideoReader, cpu

    rng = np.random.default_rng(seed)
    selected_episodes = rng.choice(episodes, size=n_frames, replace=True)

    frames = []
    print(f"Sampling {n_frames} frames from DROID left exocentric cameras...")

    for episode_path in tqdm(selected_episodes, desc="Sampling DROID frames"):
        try:
            metadata_files = list(Path(episode_path).glob("metadata_*.json"))
            with open(metadata_files[0], "r") as f:
                metadata = json.load(f)

            video_path = get_droid_video_path(episode_path, metadata)
            if video_path is None:
                continue
            vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))

            frame_idx = rng.integers(0, len(vr))
            frame = vr[frame_idx].asnumpy()
            frames.append(frame)

        except Exception:
            continue

    print(f"Successfully sampled {len(frames)} DROID frames")
    return frames


def get_sim_episodes() -> List[str]:
    """Get list of simulation episode paths."""
    episodes = []

    for axis in ["x_axis", "y_axis", "z_axis"]:
        axis_dir = Path(SIM_DATA_DIR) / axis
        if axis_dir.exists():
            for ep_dir in axis_dir.iterdir():
                if ep_dir.is_dir() and ep_dir.name.startswith("episode_"):
                    episodes.append(str(ep_dir))

    print(f"Found {len(episodes)} simulation episodes")
    return episodes


def sample_sim_frames(
    episodes: List[str],
    n_frames: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """Randomly sample RGB frames from simulation episodes."""
    import imageio

    rng = np.random.default_rng(seed)
    selected_episodes = rng.choice(episodes, size=n_frames, replace=True)

    frames = []
    print(f"Sampling {n_frames} frames from simulation...")

    for episode_path in tqdm(selected_episodes, desc="Sampling simulation frames"):
        try:
            with open(Path(episode_path) / "metadata_sim.json", "r") as f:
                metadata = json.load(f)

            video_path = Path(episode_path) / metadata["left_mp4_path"]
            reader = imageio.get_reader(str(video_path))
            all_frames = [f for f in reader]
            reader.close()

            if len(all_frames) == 0:
                continue

            frame_idx = rng.integers(0, len(all_frames))
            frame = all_frames[frame_idx]
            frames.append(frame)

        except Exception:
            continue

    print(f"Successfully sampled {len(frames)} simulation frames")
    return frames


def encode_frames(
    encoder,
    frames: List[np.ndarray],
    transform,
    device,
    desc: str = "Encoding",
) -> np.ndarray:
    """Encode a list of frames using V-JEPA-2 encoder."""
    embeddings = []

    for frame in tqdm(frames, desc=desc):
        emb = encode_single_frame(encoder, frame, transform, device)
        embeddings.append(emb)

    return np.stack(embeddings, axis=0)


def compute_and_save_statistics(
    droid_embeddings: np.ndarray,
    sim_embeddings: np.ndarray,
    output_dir: Path,
) -> dict:
    """Compute mean and covariance statistics for both domains."""
    print("\n=== Computing Statistics ===")

    # Compute means
    mu_droid = droid_embeddings.mean(axis=0)
    mu_sim = sim_embeddings.mean(axis=0)

    # Compute covariances (use float64 for numerical stability)
    droid_centered = droid_embeddings - mu_droid
    sim_centered = sim_embeddings - mu_sim

    # Unbiased covariance (N-1)
    cov_droid = np.cov(droid_embeddings.T).astype(np.float64)
    cov_sim = np.cov(sim_embeddings.T).astype(np.float64)

    # Save statistics
    np.save(output_dir / "mu_droid.npy", mu_droid)
    np.save(output_dir / "mu_sim.npy", mu_sim)
    np.save(output_dir / "cov_droid.npy", cov_droid)
    np.save(output_dir / "cov_sim.npy", cov_sim)

    # Save raw embeddings for validation
    np.save(output_dir / "z_droid.npy", droid_embeddings)
    np.save(output_dir / "z_sim.npy", sim_embeddings)

    print(f"Saved mu_droid.npy: shape {mu_droid.shape}")
    print(f"Saved mu_sim.npy: shape {mu_sim.shape}")
    print(f"Saved cov_droid.npy: shape {cov_droid.shape}")
    print(f"Saved cov_sim.npy: shape {cov_sim.shape}")

    # Sanity checks
    print("\n=== Sanity Checks ===")

    # Mean offset
    mean_offset_norm = np.linalg.norm(mu_droid - mu_sim)
    print(f"||mu_droid - mu_sim||_2: {mean_offset_norm:.4f}")

    # Covariance traces
    trace_droid = np.trace(cov_droid)
    trace_sim = np.trace(cov_sim)
    print(f"trace(cov_droid): {trace_droid:.4f}")
    print(f"trace(cov_sim): {trace_sim:.4f}")

    # Eigenvalues and condition numbers
    eig_droid = np.linalg.eigvalsh(cov_droid)
    eig_sim = np.linalg.eigvalsh(cov_sim)

    print(f"cov_droid eigenvalues: min={eig_droid.min():.6f}, max={eig_droid.max():.4f}")
    print(f"cov_sim eigenvalues: min={eig_sim.min():.6f}, max={eig_sim.max():.4f}")

    # Condition numbers (ratio of max to min positive eigenvalue)
    pos_eig_droid = eig_droid[eig_droid > 1e-10]
    pos_eig_sim = eig_sim[eig_sim > 1e-10]

    if len(pos_eig_droid) > 0:
        cond_droid = pos_eig_droid.max() / pos_eig_droid.min()
        print(f"cov_droid condition number: {cond_droid:.2e}")
    else:
        cond_droid = float('inf')

    if len(pos_eig_sim) > 0:
        cond_sim = pos_eig_sim.max() / pos_eig_sim.min()
        print(f"cov_sim condition number: {cond_sim:.2e}")
    else:
        cond_sim = float('inf')

    # Return stats for logging
    stats = {
        "n_droid": len(droid_embeddings),
        "n_sim": len(sim_embeddings),
        "embedding_dim": droid_embeddings.shape[1],
        "mean_offset_norm": float(mean_offset_norm),
        "trace_cov_droid": float(trace_droid),
        "trace_cov_sim": float(trace_sim),
        "min_eig_droid": float(eig_droid.min()),
        "max_eig_droid": float(eig_droid.max()),
        "min_eig_sim": float(eig_sim.min()),
        "max_eig_sim": float(eig_sim.max()),
        "cond_droid": float(cond_droid) if cond_droid != float('inf') else "inf",
        "cond_sim": float(cond_sim) if cond_sim != float('inf') else "inf",
    }

    return stats


def create_pca_scatter(
    droid_embeddings: np.ndarray,
    sim_embeddings: np.ndarray,
    output_dir: Path,
):
    """Create PCA scatter plot for sanity check."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    print("\n=== Creating PCA Scatter Plot ===")

    # Combine embeddings
    all_embeddings = np.vstack([droid_embeddings, sim_embeddings])

    # Fit PCA
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_embeddings)

    droid_pca = all_pca[:len(droid_embeddings)]
    sim_pca = all_pca[len(droid_embeddings):]

    var_explained = pca.explained_variance_ratio_

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(droid_pca[:, 0], droid_pca[:, 1], alpha=0.5, s=30, label="DROID", c="tab:blue")
    ax.scatter(sim_pca[:, 0], sim_pca[:, 1], alpha=0.5, s=30, label="Simulation", c="tab:orange")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
    ax.set_title("PCA: V-JEPA 2 Latent Space (DROID vs Simulation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pca_sanity_check.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved PCA plot to {output_dir / 'pca_sanity_check.png'}")


def main():
    parser = argparse.ArgumentParser(description="Collect embeddings for latent alignment experiment")
    parser.add_argument("--n_frames", type=int, default=1000, help="Frames per domain")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_encoding", action="store_true", help="Skip encoding, load from cache")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / "embeddings_cache.npz"

    if args.skip_encoding and cache_file.exists():
        print(f"Loading cached embeddings from {cache_file}")
        data = np.load(cache_file)
        droid_embeddings = data["droid_embeddings"]
        sim_embeddings = data["sim_embeddings"]
    else:
        # Load encoder
        encoder, device = load_encoder(args.device)
        transform = create_transform(MODEL_CONFIG["crop_size"])

        # Get episodes
        droid_episodes = get_droid_episodes(MIN_CLIP_DURATION_SECONDS)
        sim_episodes = get_sim_episodes()

        # Sample frames
        droid_frames = sample_droid_frames(droid_episodes, args.n_frames, args.seed)
        sim_frames = sample_sim_frames(sim_episodes, args.n_frames, args.seed + 1)

        # Balance counts
        n_balanced = min(len(droid_frames), len(sim_frames))
        print(f"\nBalancing to {n_balanced} frames per domain")
        droid_frames = droid_frames[:n_balanced]
        sim_frames = sim_frames[:n_balanced]

        # Encode frames
        print("\nEncoding frames...")
        droid_embeddings = encode_frames(encoder, droid_frames, transform, device, "Encoding DROID")
        sim_embeddings = encode_frames(encoder, sim_frames, transform, device, "Encoding Simulation")

        # Clean up GPU
        del encoder
        torch.cuda.empty_cache()

        # Cache embeddings
        np.savez(
            cache_file,
            droid_embeddings=droid_embeddings,
            sim_embeddings=sim_embeddings,
        )
        print(f"Cached embeddings to {cache_file}")

    print(f"\nDROID embeddings shape: {droid_embeddings.shape}")
    print(f"Simulation embeddings shape: {sim_embeddings.shape}")

    # Compute and save statistics
    stats = compute_and_save_statistics(droid_embeddings, sim_embeddings, output_dir)

    # Create PCA scatter plot
    create_pca_scatter(droid_embeddings, sim_embeddings, output_dir)

    # Save config
    config = {
        "n_frames": args.n_frames,
        "seed": args.seed,
        "model_checkpoint": MODEL_CONFIG["checkpoint"],
        "crop_size": MODEL_CONFIG["crop_size"],
        "camera": "left_exocentric (DROID) / left_cam (simulation)",
        **stats,
    }

    with open(output_dir / "collection_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print("EMBEDDING COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Files saved:")
    print(f"  - mu_droid.npy, mu_sim.npy")
    print(f"  - cov_droid.npy, cov_sim.npy")
    print(f"  - z_droid.npy, z_sim.npy")
    print(f"  - collection_config.json")
    print(f"  - pca_sanity_check.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
