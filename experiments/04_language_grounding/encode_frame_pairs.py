"""
Encode sampled frame pairs using V-JEPA encoder.

This script:
1. Loads frame pairs from the sampled pairs JSON
2. Encodes each frame SEPARATELY using V-JEPA (not as a video pair)
3. Computes the latent difference: delta = features_kd - features_k
4. Saves features and metadata with full traceability

Memory-efficient: Uses streaming encoding and memory-mapped files to handle
large datasets without running out of RAM.

Output format:
- features_k.pt: Tensor of shape [N, num_patches, embed_dim] for frame_k
- features_kd.pt: Tensor of shape [N, num_patches, embed_dim] for frame_k+d
- features_delta.pt: Tensor of shape [N, num_patches, embed_dim] (kd - k)
- metadata.json: List of dicts with trajectory_id, lab, frame indices, etc.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Add vjepa2 to path
VJEPA_PATH = Path("/home/s185927/thesis/vjepa2")
sys.path.insert(0, str(VJEPA_PATH))

from app.vjepa_droid.transforms import make_transforms


# Known DROID labs for metadata extraction
KNOWN_LABS = {"AUTOLab", "CLVR", "GuptaLab", "ILIAD", "IPRL", "IRIS"}


def extract_lab(trajectory_id: str) -> str:
    """Extract lab name from trajectory ID."""
    parts = trajectory_id.split("+")
    if len(parts) >= 2 and parts[0] in KNOWN_LABS:
        return parts[0]
    return "unknown"


class FramePairDataset(Dataset):
    """Dataset for loading frame pairs (k and k+d) together."""

    def __init__(
        self,
        pairs_data: list[dict],
        base_dir: Path,
        transform,
    ):
        self.pairs_data = pairs_data
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs_data)

    def __getitem__(self, idx):
        pair = self.pairs_data[idx]

        # Load frame_k
        frame_k_path = self.base_dir / pair["frame_k_path"]
        img_k = np.array(Image.open(frame_k_path).convert("RGB"))

        # Load frame_k+d
        frame_kd_path = self.base_dir / pair["frame_k_d_path"]
        img_kd = np.array(Image.open(frame_kd_path).convert("RGB"))

        # V-JEPA expects video input, so we duplicate each frame
        video_k = np.stack([img_k, img_k], axis=0)
        video_kd = np.stack([img_kd, img_kd], axis=0)

        tensor_k = self.transform(video_k)
        tensor_kd = self.transform(video_kd)

        return tensor_k, tensor_kd, idx


def load_encoder(model_name: str = "vjepa2_ac_vit_giant", device: str = "cuda"):
    """Load V-JEPA encoder from torch hub.

    Uses action-conditioned model by default since it's already cached from
    the reaching experiments. The encoder is identical to the standard model.
    """
    print(f"Loading V-JEPA encoder: {model_name}")

    encoder, predictor = torch.hub.load(
        "facebookresearch/vjepa2",
        model_name,
    )

    encoder = encoder.to(device).eval()
    print(f"Encoder loaded on {device}")

    return encoder


def encode_frame_pairs_streaming(
    pairs_file: str,
    base_dir: str,
    output_dir: str,
    model_name: str = "vjepa2_ac_vit_giant",
    batch_size: int = 16,
    num_workers: int = 8,
    device: str = "cuda",
    limit: Optional[int] = None,
    normalize_features: bool = True,
    use_float16: bool = False,
    delta_only: bool = False,
    skip_kd: bool = False,
):
    """
    Encode all frame pairs using streaming approach with memory-mapped output.

    This version processes pairs in batches and writes directly to disk using
    memory-mapped files to avoid OOM on large datasets.

    Args:
        pairs_file: Path to frame_pairs.json
        base_dir: Base directory containing frame images
        output_dir: Directory to save encoded features
        model_name: V-JEPA model name
        batch_size: Batch size for encoding
        num_workers: Number of data loader workers
        device: Device to use (cuda/cpu)
        limit: Limit number of pairs to process (for testing)
        normalize_features: Whether to apply layer normalization to features
        use_float16: Whether to save features in half precision (reduces storage by 50%)
        delta_only: Whether to only save delta features (skip features_k and features_kd)
        skip_kd: Whether to skip saving features_kd (saves features_k and features_delta only)
    """
    # Load pairs data
    print(f"Loading pairs from {pairs_file}")
    with open(pairs_file) as f:
        pairs_data = json.load(f)

    # Filter out pairs with missing paths
    pairs_data = [p for p in pairs_data if p.get("frame_k_path") and p.get("frame_k_d_path")]
    print(f"Loaded {len(pairs_data)} valid pairs")

    if limit:
        pairs_data = pairs_data[:limit]
        print(f"Limited to {len(pairs_data)} pairs")

    n_pairs = len(pairs_data)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    encoder = load_encoder(model_name, device)

    # Get crop size from model
    crop_size = 256
    if "384" in model_name:
        crop_size = 384

    # Create transform (no augmentation for inference)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    base_dir = Path(base_dir)

    # Create dataset and dataloader
    dataset = FramePairDataset(
        pairs_data=pairs_data,
        base_dir=base_dir,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # Determine output dtype
    np_dtype = np.float16 if use_float16 else np.float32
    dtype_str = "float16" if use_float16 else "float32"

    print(f"\nEncoding {n_pairs} frame pairs (streaming mode)...")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")
    print(f"Normalize: {normalize_features}")
    print(f"Float16: {use_float16}, Delta only: {delta_only}, Skip kd: {skip_kd}")

    # First pass: determine feature shape from first batch
    print("\nDetermining feature dimensions...")
    with torch.inference_mode():
        sample_k, sample_kd, _ = next(iter(dataloader))
        sample_k = sample_k.to(device)
        sample_features = encoder(sample_k)
        n_patches = sample_features.shape[1]
        embed_dim = sample_features.shape[2]
        print(f"  Feature shape per pair: [{n_patches}, {embed_dim}]")
        del sample_k, sample_kd, sample_features
        torch.cuda.empty_cache()

    # Create memory-mapped output files
    mmap_paths = {}
    mmap_files = {}

    # Determine which files to save
    save_k = not delta_only
    save_kd = not delta_only and not skip_kd

    if save_k:
        mmap_paths["features_k"] = output_dir / "features_k.npy"
        mmap_files["features_k"] = np.memmap(
            mmap_paths["features_k"], dtype=np_dtype, mode='w+',
            shape=(n_pairs, n_patches, embed_dim)
        )
    if save_kd:
        mmap_paths["features_kd"] = output_dir / "features_kd.npy"
        mmap_files["features_kd"] = np.memmap(
            mmap_paths["features_kd"], dtype=np_dtype, mode='w+',
            shape=(n_pairs, n_patches, embed_dim)
        )

    mmap_paths["features_delta"] = output_dir / "features_delta.npy"
    mmap_files["features_delta"] = np.memmap(
        mmap_paths["features_delta"], dtype=np_dtype, mode='w+',
        shape=(n_pairs, n_patches, embed_dim)
    )

    print(f"\nStreaming encoding to memory-mapped files...")

    # Process all pairs
    with torch.inference_mode():
        for batch_k, batch_kd, batch_indices in tqdm(dataloader, desc="Encoding pairs"):
            batch_k = batch_k.to(device)
            batch_kd = batch_kd.to(device)

            # Encode frame_k
            features_k = encoder(batch_k)
            if normalize_features:
                features_k = torch.nn.functional.layer_norm(features_k, (features_k.size(-1),))

            # Encode frame_k+d
            features_kd = encoder(batch_kd)
            if normalize_features:
                features_kd = torch.nn.functional.layer_norm(features_kd, (features_kd.size(-1),))

            # Compute delta
            features_delta = features_kd - features_k

            # Convert to numpy with correct dtype
            features_k_np = features_k.cpu().numpy().astype(np_dtype)
            features_kd_np = features_kd.cpu().numpy().astype(np_dtype)
            features_delta_np = features_delta.cpu().numpy().astype(np_dtype)

            # Write to memory-mapped files
            indices = batch_indices.numpy()
            for i, idx in enumerate(indices):
                if save_k:
                    mmap_files["features_k"][idx] = features_k_np[i]
                if save_kd:
                    mmap_files["features_kd"][idx] = features_kd_np[i]
                mmap_files["features_delta"][idx] = features_delta_np[i]

            # Flush periodically to ensure data is written
            if batch_indices[0] % (batch_size * 10) == 0:
                for mmap in mmap_files.values():
                    mmap.flush()

    # Final flush
    for mmap in mmap_files.values():
        mmap.flush()
        del mmap

    print("\nConverting to PyTorch format...")

    # Convert memory-mapped numpy arrays to PyTorch tensors
    saved_files = {}

    if save_k:
        print("  Loading features_k...")
        features_k_mmap = np.memmap(mmap_paths["features_k"], dtype=np_dtype, mode='r',
                                     shape=(n_pairs, n_patches, embed_dim))
        torch.save(torch.from_numpy(np.array(features_k_mmap)), output_dir / "features_k.pt")
        del features_k_mmap
        os.remove(mmap_paths["features_k"])
        saved_files["features_k"] = "features_k.pt"

    if save_kd:
        print("  Loading features_kd...")
        features_kd_mmap = np.memmap(mmap_paths["features_kd"], dtype=np_dtype, mode='r',
                                      shape=(n_pairs, n_patches, embed_dim))
        torch.save(torch.from_numpy(np.array(features_kd_mmap)), output_dir / "features_kd.pt")
        del features_kd_mmap
        os.remove(mmap_paths["features_kd"])
        saved_files["features_kd"] = "features_kd.pt"

    print("  Loading features_delta...")
    features_delta_mmap = np.memmap(mmap_paths["features_delta"], dtype=np_dtype, mode='r',
                                     shape=(n_pairs, n_patches, embed_dim))
    torch.save(torch.from_numpy(np.array(features_delta_mmap)), output_dir / "features_delta.pt")
    del features_delta_mmap
    os.remove(mmap_paths["features_delta"])
    saved_files["features_delta"] = "features_delta.pt"

    print(f"  Output directory: {output_dir}")

    # Build comprehensive metadata
    metadata = []
    for i, p in enumerate(pairs_data):
        lab = extract_lab(p["trajectory_id"])
        metadata.append({
            "index": i,
            "lab": lab,
            "trajectory_id": p["trajectory_id"],
            "frame_k": p["frame_k"],
            "frame_k_d": p["frame_k_d"],
            "d": p["d"],
            "frame_k_path": p["frame_k_path"],
            "frame_k_d_path": p["frame_k_d_path"],
            "position_delta": p.get("position_delta"),
            "gripper_delta": p.get("gripper_delta"),
            "z_delta": p.get("z_delta"),
            "rotation_delta": p.get("rotation_delta"),
            "salience_tier": p.get("salience_tier"),
        })

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    # Save summary stats
    from collections import Counter
    lab_counts = Counter(m["lab"] for m in metadata)

    stats = {
        "num_pairs": len(metadata),
        "feature_shape": [n_pairs, n_patches, embed_dim],
        "embed_dim": embed_dim,
        "num_patches": n_patches,
        "dtype": dtype_str,
        "delta_only": delta_only,
        "skip_kd": skip_kd,
        "model_name": model_name,
        "crop_size": crop_size,
        "normalized": normalize_features,
        "pairs_file": str(pairs_file),
        "base_dir": str(base_dir),
        "labs": dict(lab_counts),
        "files": {**saved_files, "metadata": "metadata.json"},
    }
    stats_path = output_dir / "encoding_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to: {stats_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ENCODING COMPLETE")
    print("=" * 60)
    print(f"Total pairs: {len(metadata)}")
    print(f"Feature shape: [{n_pairs}, {n_patches}, {embed_dim}]")
    print(f"Dtype: {dtype_str}")
    print(f"Saved: features_k={save_k}, features_kd={save_kd}, features_delta=True")
    print(f"By lab:")
    for lab, count in sorted(lab_counts.items()):
        print(f"  {lab}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Encode frame pairs with V-JEPA")
    parser.add_argument(
        "--pairs_file",
        type=str,
        default="/data/s185927/droid_raw/sampled_pairs/frame_pairs.json",
        help="Path to frame pairs JSON file",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/data/s185927/droid_raw/sampled_pairs",
        help="Base directory for frame images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/s185927/droid_raw/vjepa_features",
        help="Output directory for encoded features",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vjepa2_ac_vit_giant",
        choices=["vjepa2_vit_large", "vjepa2_vit_huge", "vjepa2_vit_giant", "vjepa2_vit_giant_384", "vjepa2_ac_vit_giant"],
        help="V-JEPA model to use (default: vjepa2_ac_vit_giant, already cached)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pairs (for testing)",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip layer normalization of features",
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Save features in float16 (half precision) to reduce storage by 50%%",
    )
    parser.add_argument(
        "--delta_only",
        action="store_true",
        help="Only save features_delta.pt (skip features_k and features_kd)",
    )
    parser.add_argument(
        "--skip_kd",
        action="store_true",
        help="Skip saving features_kd.pt (saves features_k and features_delta only)",
    )
    args = parser.parse_args()

    encode_frame_pairs_streaming(
        pairs_file=args.pairs_file,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        limit=args.limit,
        normalize_features=not args.no_normalize,
        use_float16=args.float16,
        delta_only=args.delta_only,
        skip_kd=args.skip_kd,
    )


if __name__ == "__main__":
    main()
