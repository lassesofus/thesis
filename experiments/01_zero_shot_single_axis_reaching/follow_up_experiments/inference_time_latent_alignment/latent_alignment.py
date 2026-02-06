#!/usr/bin/env python3
"""
Latent alignment transforms for sim-to-DROID embedding alignment.

This module provides:
1. LatentAligner class for applying mean-only or CORAL alignment
2. Functions to load alignment parameters from disk
3. PyTorch-compatible transforms for efficient GPU computation

Usage in planning:
    aligner = LatentAligner.from_path(stats_dir, method='coral', device='cuda:0')
    z_aligned = aligner(z_raw)  # Apply alignment to embeddings
"""

from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn


class LatentAligner(nn.Module):
    """
    Latent space alignment transform.

    Supports:
    - 'none': No alignment (identity transform)
    - 'mean': Mean-only alignment: z_aligned = z - mu_sim + mu_droid
    - 'coral': CORAL (whitening-coloring): z_aligned = (z - mu_sim) @ A + mu_droid
               where A = C_sim^(-1/2) @ C_droid^(1/2)
    """

    def __init__(
        self,
        method: Literal['none', 'mean', 'coral'] = 'none',
        mu_sim: Optional[np.ndarray] = None,
        mu_droid: Optional[np.ndarray] = None,
        coral_matrix: Optional[np.ndarray] = None,
        device: str = 'cpu',
    ):
        """
        Initialize the aligner.

        Args:
            method: Alignment method ('none', 'mean', 'coral')
            mu_sim: Mean of simulation embeddings [D]
            mu_droid: Mean of DROID embeddings [D]
            coral_matrix: CORAL transform matrix [D, D] (for 'coral' method)
            device: Device to place tensors on
        """
        super().__init__()
        self.method = method
        self.device_str = device

        if method == 'none':
            # No parameters needed
            self.register_buffer('mu_sim', None)
            self.register_buffer('mu_droid', None)
            self.register_buffer('coral_matrix', None)
        elif method in ['mean', 'coral']:
            if mu_sim is None or mu_droid is None:
                raise ValueError(f"mu_sim and mu_droid required for method '{method}'")

            self.register_buffer('mu_sim', torch.from_numpy(mu_sim).float())
            self.register_buffer('mu_droid', torch.from_numpy(mu_droid).float())

            if method == 'coral':
                if coral_matrix is None:
                    raise ValueError("coral_matrix required for method 'coral'")
                self.register_buffer('coral_matrix', torch.from_numpy(coral_matrix).float())
            else:
                self.register_buffer('coral_matrix', None)
        else:
            raise ValueError(f"Unknown alignment method: {method}")

        self.to(device)

    @classmethod
    def from_path(
        cls,
        stats_dir: str,
        method: Literal['none', 'mean', 'coral'] = 'none',
        device: str = 'cpu',
    ) -> 'LatentAligner':
        """
        Load aligner from saved statistics directory.

        Args:
            stats_dir: Directory containing alignment statistics
            method: Alignment method to use
            device: Device to place tensors on

        Returns:
            LatentAligner instance
        """
        if method == 'none':
            return cls(method='none', device=device)

        stats_dir = Path(stats_dir)

        # Load mean vectors
        mu_sim_path = stats_dir / "mu_sim.npy"
        mu_droid_path = stats_dir / "mu_droid.npy"

        if not mu_sim_path.exists() or not mu_droid_path.exists():
            raise FileNotFoundError(f"Mean vectors not found in {stats_dir}")

        mu_sim = np.load(mu_sim_path)
        mu_droid = np.load(mu_droid_path)

        coral_matrix = None
        if method == 'coral':
            coral_path = stats_dir / "coral_matrix.npy"
            if not coral_path.exists():
                raise FileNotFoundError(f"CORAL matrix not found: {coral_path}")
            coral_matrix = np.load(coral_path)

        return cls(
            method=method,
            mu_sim=mu_sim,
            mu_droid=mu_droid,
            coral_matrix=coral_matrix,
            device=device,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply alignment transform to embeddings.

        Args:
            z: Input embeddings, shape [..., D] or [..., T, D]
               Can be any batch shape as long as last dim is embedding dim

        Returns:
            z_aligned: Aligned embeddings, same shape as input
        """
        if self.method == 'none':
            return z

        # Handle different input shapes
        original_shape = z.shape
        D = self.mu_sim.shape[0]

        # Flatten to [N, D] for alignment
        if z.shape[-1] != D:
            raise ValueError(f"Expected embedding dim {D}, got {z.shape[-1]}")

        z_flat = z.reshape(-1, D)

        if self.method == 'mean':
            # Mean-only: z_aligned = z - mu_sim + mu_droid
            z_aligned = z_flat - self.mu_sim + self.mu_droid

        elif self.method == 'coral':
            # CORAL: z_aligned = (z - mu_sim) @ A + mu_droid
            z_centered = z_flat - self.mu_sim
            z_aligned = z_centered @ self.coral_matrix + self.mu_droid

        # Restore original shape
        z_aligned = z_aligned.reshape(original_shape)

        return z_aligned

    def __repr__(self) -> str:
        if self.method == 'none':
            return "LatentAligner(method='none')"
        elif self.method == 'mean':
            return f"LatentAligner(method='mean', dim={self.mu_sim.shape[0]})"
        else:
            return f"LatentAligner(method='coral', dim={self.mu_sim.shape[0]})"


def apply_alignment_numpy(
    z: np.ndarray,
    method: str,
    mu_sim: np.ndarray,
    mu_droid: np.ndarray,
    coral_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply alignment transform to numpy embeddings.

    Args:
        z: Input embeddings [..., D]
        method: 'none', 'mean', or 'coral'
        mu_sim: Simulation mean [D]
        mu_droid: DROID mean [D]
        coral_matrix: CORAL transform matrix [D, D]

    Returns:
        z_aligned: Aligned embeddings
    """
    if method == 'none':
        return z

    original_shape = z.shape
    D = mu_sim.shape[0]
    z_flat = z.reshape(-1, D)

    if method == 'mean':
        z_aligned = z_flat - mu_sim + mu_droid
    elif method == 'coral':
        if coral_matrix is None:
            raise ValueError("coral_matrix required for CORAL alignment")
        z_centered = z_flat - mu_sim
        z_aligned = z_centered @ coral_matrix + mu_droid
    else:
        raise ValueError(f"Unknown method: {method}")

    return z_aligned.reshape(original_shape)


# Test the aligner
if __name__ == "__main__":
    import sys

    stats_dir = Path("/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/inference_time_latent_alignment/stats")

    # Check if stats exist
    if not (stats_dir / "mu_sim.npy").exists():
        print("Stats not found. Run fit_alignment.py first.")
        sys.exit(1)

    # Test loading
    print("Testing LatentAligner...")

    for method in ['none', 'mean', 'coral']:
        aligner = LatentAligner.from_path(stats_dir, method=method, device='cpu')
        print(f"  {aligner}")

        # Test forward pass
        z = torch.randn(10, 256, 1408)  # [B, T, D]
        z_aligned = aligner(z)
        print(f"    Input: {z.shape}, Output: {z_aligned.shape}")

        if method != 'none':
            # Check that mean is shifted
            z_mean = z.mean(dim=(0, 1)).numpy()
            z_aligned_mean = z_aligned.mean(dim=(0, 1)).numpy()
            print(f"    Input mean norm: {np.linalg.norm(z_mean):.4f}")
            print(f"    Output mean norm: {np.linalg.norm(z_aligned_mean):.4f}")

    print("\nAll tests passed!")
