#!/usr/bin/env python3
"""
Compare V-JEPA-2 latent space distributions between real DROID and simulated RoboHive frames.

This experiment answers: Are simulated observations embedded into the same regions
of latent space as real DROID observations?

Approach:
- Randomly sample individual RGB frames from DROID trajectories (left exocentric camera)
- Randomly sample RGB frames from simulation rollouts (start, intermediate, goal)
- Encode all frames with frozen V-JEPA-2 encoder
- Compare distributions via PCA, UMAP, and distance statistics
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
import imageio

# Add paths for imports
sys.path.insert(0, "/home/s185927/thesis/vjepa2")
sys.path.insert(0, "/home/s185927/thesis")

from plot_config import PLOT_PARAMS, apply_plot_params, configure_axis

from app.vjepa_droid.utils import init_video_model
from app.vjepa_droid.transforms import make_transforms

# Paths
DROID_INDEX = "/data/s185927/droid_raw/droid_index.csv"
SIM_DATA_DIR = "/data/s185927/droid_sim/zero_shot_correlation"
OUTPUT_DIR = Path("/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/latent_distribution_comparison")

# Model config (pretrained V-JEPA-2-AC encoder)
MODEL_CONFIG = {
    "checkpoint": "/home/s185927/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt",
    "crop_size": 256,
    "patch_size": 16,
    "model_name": "vit_giant_xformers",
}

# Experiment config
MIN_CLIP_DURATION_SECONDS = 4.0  # Only use DROID clips longer than this


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

        # Apply layer norm
        h = F.layer_norm(h, (h.size(-1),))

        # Mean pool over tokens to get frame-level embedding
        embedding = h.mean(dim=1).squeeze(0).cpu().numpy()  # (D,)

    return embedding


def get_droid_video_path(episode_path: str, metadata: dict) -> Optional[Path]:
    """
    Resolve the left camera video path from DROID metadata.

    The left_mp4_path in metadata is relative to the lab root directory,
    e.g., "failure/2023-07-07/.../recordings/MP4/xxx.mp4"
    and episode_path is like "/data/droid_raw/1.0.1/AUTOLab/failure/2023-07-07/..."

    So we need to find the lab root and construct the full path.
    """
    if "left_mp4_path" not in metadata:
        return None

    episode_path = Path(episode_path)
    left_mp4_rel = metadata["left_mp4_path"]

    # The video file is in episode_path/recordings/MP4/
    # Extract just the filename from left_mp4_path
    video_filename = Path(left_mp4_rel).name
    video_path = episode_path / "recordings" / "MP4" / video_filename

    if video_path.exists():
        return video_path

    return None


def get_droid_episodes(min_duration_seconds: float = 4.0) -> List[str]:
    """
    Get list of DROID episode paths with clips longer than min_duration.

    Filters based on trajectory_length from metadata (assuming ~30fps video).
    """
    with open(DROID_INDEX, "r") as f:
        all_episodes = [line.strip() for line in f if line.strip()]

    valid_episodes = []
    print(f"Filtering {len(all_episodes)} DROID episodes for duration >= {min_duration_seconds}s...")

    for episode_path in tqdm(all_episodes, desc="Checking DROID episodes"):
        try:
            # Find metadata file
            metadata_files = list(Path(episode_path).glob("metadata_*.json"))
            if not metadata_files:
                continue

            with open(metadata_files[0], "r") as f:
                metadata = json.load(f)

            # Check trajectory length (trajectory_length is number of frames)
            traj_length = metadata.get("trajectory_length", 0)

            # Assume ~30fps for DROID videos
            duration = traj_length / 30.0

            if duration >= min_duration_seconds:
                # Verify left camera video exists
                video_path = get_droid_video_path(episode_path, metadata)
                if video_path is not None:
                    valid_episodes.append(episode_path)
        except Exception as e:
            continue

    print(f"Found {len(valid_episodes)} valid DROID episodes (>= {min_duration_seconds}s)")
    return valid_episodes


def sample_droid_frames(
    episodes: List[str],
    n_frames: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Randomly sample RGB frames from DROID episodes (left exocentric camera).

    Args:
        episodes: List of episode directory paths
        n_frames: Total number of frames to sample
        seed: Random seed

    Returns:
        List of RGB frames as numpy arrays (H, W, C)
    """
    rng = np.random.default_rng(seed)

    # Randomly select episodes (with replacement if needed)
    n_episodes = min(len(episodes), n_frames)
    selected_episodes = rng.choice(episodes, size=n_frames, replace=True)

    frames = []
    print(f"Sampling {n_frames} frames from DROID left exocentric cameras...")

    for episode_path in tqdm(selected_episodes, desc="Sampling DROID frames"):
        try:
            # Load metadata
            metadata_files = list(Path(episode_path).glob("metadata_*.json"))
            with open(metadata_files[0], "r") as f:
                metadata = json.load(f)

            # Load video from left camera
            video_path = get_droid_video_path(episode_path, metadata)
            if video_path is None:
                continue
            vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))

            # Sample random frame
            frame_idx = rng.integers(0, len(vr))
            frame = vr[frame_idx].asnumpy()  # (H, W, C)
            frames.append(frame)

        except Exception as e:
            # Skip problematic episodes
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
    """
    Randomly sample RGB frames from simulation episodes.
    Samples from start, intermediate, and goal frames.

    Args:
        episodes: List of episode directory paths
        n_frames: Total number of frames to sample
        seed: Random seed

    Returns:
        List of RGB frames as numpy arrays (H, W, C)
    """
    rng = np.random.default_rng(seed)

    # Select episodes (with replacement if needed)
    selected_episodes = rng.choice(episodes, size=n_frames, replace=True)

    frames = []
    print(f"Sampling {n_frames} frames from simulation...")

    for episode_path in tqdm(selected_episodes, desc="Sampling simulation frames"):
        try:
            # Load metadata
            with open(Path(episode_path) / "metadata_sim.json", "r") as f:
                metadata = json.load(f)

            # Load video
            video_path = Path(episode_path) / metadata["left_mp4_path"]
            reader = imageio.get_reader(str(video_path))
            all_frames = [f for f in reader]
            reader.close()

            if len(all_frames) == 0:
                continue

            # Sample random frame (including start, intermediate, goal)
            frame_idx = rng.integers(0, len(all_frames))
            frame = all_frames[frame_idx]  # (H, W, C)
            frames.append(frame)

        except Exception as e:
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
    """
    Encode a list of frames using V-JEPA-2 encoder.

    Returns:
        embeddings: (N, D) array of frame embeddings
    """
    embeddings = []

    for frame in tqdm(frames, desc=desc):
        emb = encode_single_frame(encoder, frame, transform, device)
        embeddings.append(emb)

    return np.stack(embeddings, axis=0)


def run_pca_analysis(
    droid_embeddings: np.ndarray,
    sim_embeddings: np.ndarray,
    output_dir: Path,
):
    """Run PCA analysis and create visualizations."""
    from sklearn.decomposition import PCA

    print("\n=== PCA Analysis ===")

    # Combine embeddings
    all_embeddings = np.vstack([droid_embeddings, sim_embeddings])
    labels = np.array(["DROID"] * len(droid_embeddings) + ["Simulation"] * len(sim_embeddings))

    # Fit PCA
    pca = PCA(n_components=min(50, all_embeddings.shape[0], all_embeddings.shape[1]))
    pca.fit(all_embeddings)

    # Transform
    all_pca = pca.transform(all_embeddings)
    droid_pca = all_pca[:len(droid_embeddings)]
    sim_pca = all_pca[len(droid_embeddings):]

    # Variance explained
    var_explained = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_explained)

    print(f"Variance explained by PC1: {var_explained[0]*100:.1f}%")
    print(f"Variance explained by PC2: {var_explained[1]*100:.1f}%")
    print(f"Cumulative variance (PC1-2): {cumvar[1]*100:.1f}%")
    print(f"Cumulative variance (PC1-10): {cumvar[9]*100:.1f}%")

    # Plot 2D PCA
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: 2D scatter
    ax = axes[0]
    ax.scatter(droid_pca[:, 0], droid_pca[:, 1], alpha=0.5, s=30, label="DROID", c="tab:blue")
    ax.scatter(sim_pca[:, 0], sim_pca[:, 1], alpha=0.5, s=30, label="Simulation", c="tab:orange")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)", fontsize=PLOT_PARAMS["label_size"])
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)", fontsize=PLOT_PARAMS["label_size"])
    ax.set_title("(a) PCA: V-JEPA 2 Latent Space", fontsize=PLOT_PARAMS["subtitle_size"])
    ax.legend(fontsize=PLOT_PARAMS["legend_size"], loc="best")
    apply_plot_params(ax)

    # Right: Variance explained
    ax = axes[1]
    ax.bar(range(1, 11), var_explained[:10] * 100, alpha=0.7, label="Individual", color="tab:blue")
    ax.plot(range(1, 11), cumvar[:10] * 100, "r-o", label="Cumulative",
            linewidth=PLOT_PARAMS["euclid_linewidth"], markersize=PLOT_PARAMS["euclid_markersize"])
    ax.set_xlabel("Principal Component", fontsize=PLOT_PARAMS["label_size"])
    ax.set_ylabel("Variance Explained (%)", fontsize=PLOT_PARAMS["label_size"])
    ax.set_title("(b) PCA Variance Explained", fontsize=PLOT_PARAMS["subtitle_size"])
    ax.legend(fontsize=PLOT_PARAMS["legend_size"], loc="best")
    apply_plot_params(ax)

    plt.tight_layout()
    plt.savefig(output_dir / "pca_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved PCA plot to {output_dir / 'pca_analysis.png'}")

    return {
        "var_explained": var_explained,
        "cumvar": cumvar,
        "droid_pca": droid_pca,
        "sim_pca": sim_pca,
    }


def run_umap_analysis(
    droid_embeddings: np.ndarray,
    sim_embeddings: np.ndarray,
    output_dir: Path,
):
    """Run UMAP visualization."""
    try:
        import umap
    except ImportError:
        print("UMAP not installed, skipping UMAP analysis")
        return None

    print("\n=== UMAP Analysis ===")

    # Combine embeddings
    all_embeddings = np.vstack([droid_embeddings, sim_embeddings])

    # Fit UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    all_umap = reducer.fit_transform(all_embeddings)

    droid_umap = all_umap[:len(droid_embeddings)]
    sim_umap = all_umap[len(droid_embeddings):]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(droid_umap[:, 0], droid_umap[:, 1], alpha=0.5, s=30, label="DROID", c="tab:blue")
    ax.scatter(sim_umap[:, 0], sim_umap[:, 1], alpha=0.5, s=30, label="Simulation", c="tab:orange")
    ax.set_xlabel("UMAP 1", fontsize=PLOT_PARAMS["label_size"])
    ax.set_ylabel("UMAP 2", fontsize=PLOT_PARAMS["label_size"])
    ax.set_title("UMAP: V-JEPA 2 Latent Space", fontsize=PLOT_PARAMS["subtitle_size"])
    ax.legend(fontsize=PLOT_PARAMS["legend_size"], loc="best")
    apply_plot_params(ax)

    plt.tight_layout()
    plt.savefig(output_dir / "umap_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved UMAP plot to {output_dir / 'umap_analysis.png'}")

    return {"droid_umap": droid_umap, "sim_umap": sim_umap}


def compute_distance_statistics(
    droid_embeddings: np.ndarray,
    sim_embeddings: np.ndarray,
) -> dict:
    """Compute embedding distance statistics."""
    print("\n=== Distance Statistics ===")

    # Mean embeddings
    droid_mean = droid_embeddings.mean(axis=0)
    sim_mean = sim_embeddings.mean(axis=0)

    # Inter-domain: distance between domain means
    inter_domain_mean_dist = np.linalg.norm(droid_mean - sim_mean)

    # Intra-domain: average distance to domain centroid
    droid_intra_dists = np.linalg.norm(droid_embeddings - droid_mean, axis=1)
    sim_intra_dists = np.linalg.norm(sim_embeddings - sim_mean, axis=1)

    droid_intra_mean = droid_intra_dists.mean()
    sim_intra_mean = sim_intra_dists.mean()

    # Sample pairwise distances (for efficiency, sample if large)
    n_samples = min(1000, len(droid_embeddings), len(sim_embeddings))
    rng = np.random.default_rng(42)

    droid_sample_idx = rng.choice(len(droid_embeddings), size=n_samples, replace=False)
    sim_sample_idx = rng.choice(len(sim_embeddings), size=n_samples, replace=False)

    # Pairwise inter-domain distances
    inter_pairwise = []
    for i in range(n_samples):
        d = np.linalg.norm(droid_embeddings[droid_sample_idx[i]] - sim_embeddings[sim_sample_idx[i]])
        inter_pairwise.append(d)
    inter_pairwise = np.array(inter_pairwise)

    # Pairwise intra-domain distances (within DROID)
    droid_intra_pairwise = []
    idx1 = rng.choice(len(droid_embeddings), size=n_samples, replace=True)
    idx2 = rng.choice(len(droid_embeddings), size=n_samples, replace=True)
    for i in range(n_samples):
        if idx1[i] != idx2[i]:
            d = np.linalg.norm(droid_embeddings[idx1[i]] - droid_embeddings[idx2[i]])
            droid_intra_pairwise.append(d)
    droid_intra_pairwise = np.array(droid_intra_pairwise)

    # Pairwise intra-domain distances (within Simulation)
    sim_intra_pairwise = []
    idx1 = rng.choice(len(sim_embeddings), size=n_samples, replace=True)
    idx2 = rng.choice(len(sim_embeddings), size=n_samples, replace=True)
    for i in range(n_samples):
        if idx1[i] != idx2[i]:
            d = np.linalg.norm(sim_embeddings[idx1[i]] - sim_embeddings[idx2[i]])
            sim_intra_pairwise.append(d)
    sim_intra_pairwise = np.array(sim_intra_pairwise)

    stats = {
        "inter_domain_mean_dist": inter_domain_mean_dist,
        "droid_intra_centroid_dist_mean": droid_intra_mean,
        "droid_intra_centroid_dist_std": droid_intra_dists.std(),
        "sim_intra_centroid_dist_mean": sim_intra_mean,
        "sim_intra_centroid_dist_std": sim_intra_dists.std(),
        "inter_pairwise_mean": inter_pairwise.mean(),
        "inter_pairwise_std": inter_pairwise.std(),
        "droid_intra_pairwise_mean": droid_intra_pairwise.mean(),
        "droid_intra_pairwise_std": droid_intra_pairwise.std(),
        "sim_intra_pairwise_mean": sim_intra_pairwise.mean(),
        "sim_intra_pairwise_std": sim_intra_pairwise.std(),
    }

    print(f"Inter-domain mean distance: {inter_domain_mean_dist:.4f}")
    print(f"DROID intra-domain (to centroid): {droid_intra_mean:.4f} +/- {droid_intra_dists.std():.4f}")
    print(f"Simulation intra-domain (to centroid): {sim_intra_mean:.4f} +/- {sim_intra_dists.std():.4f}")
    print(f"Inter-domain pairwise: {inter_pairwise.mean():.4f} +/- {inter_pairwise.std():.4f}")
    print(f"DROID intra-domain pairwise: {droid_intra_pairwise.mean():.4f} +/- {droid_intra_pairwise.std():.4f}")
    print(f"Simulation intra-domain pairwise: {sim_intra_pairwise.mean():.4f} +/- {sim_intra_pairwise.std():.4f}")

    # Domain gap ratio: inter / avg(intra)
    avg_intra = (droid_intra_pairwise.mean() + sim_intra_pairwise.mean()) / 2
    gap_ratio = inter_pairwise.mean() / avg_intra
    stats["domain_gap_ratio"] = gap_ratio
    print(f"\nDomain gap ratio (inter/intra): {gap_ratio:.4f}")
    if gap_ratio < 1.5:
        print("  -> Domains overlap significantly (ratio < 1.5)")
    elif gap_ratio < 2.0:
        print("  -> Domains are moderately separated (1.5 <= ratio < 2.0)")
    else:
        print("  -> Domains are well separated (ratio >= 2.0)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compare V-JEPA 2 latent distributions")
    parser.add_argument("--n_frames", type=int, default=2000, help="Frames per domain")
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

    # Run analyses
    pca_results = run_pca_analysis(droid_embeddings, sim_embeddings, output_dir)
    umap_results = run_umap_analysis(droid_embeddings, sim_embeddings, output_dir)
    distance_stats = compute_distance_statistics(droid_embeddings, sim_embeddings)

    # Save all results
    results = {
        "n_droid_frames": len(droid_embeddings),
        "n_sim_frames": len(sim_embeddings),
        "embedding_dim": droid_embeddings.shape[1],
        "pca_var_explained_pc1": float(pca_results["var_explained"][0]),
        "pca_var_explained_pc2": float(pca_results["var_explained"][1]),
        "pca_cumvar_10": float(pca_results["cumvar"][9]),
        **{k: float(v) for k, v in distance_stats.items()},
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Frames analyzed: {len(droid_embeddings)} DROID, {len(sim_embeddings)} Simulation")
    print(f"Embedding dimension: {droid_embeddings.shape[1]}")
    print(f"Domain gap ratio: {distance_stats['domain_gap_ratio']:.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
