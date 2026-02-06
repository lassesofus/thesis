"""
Energy functions for CEM planning in latent space.

This module implements different distance metrics for scoring rollouts in CEM:
1. L1 (baseline): Mean absolute difference
2. Cosine: Scale-invariant cosine distance
3. Normalized L1: L1 after per-dimension normalization using DROID statistics

All functions expect token-wise representations of shape [..., T, D] where:
- T = number of tokens (e.g., 256)
- D = feature dimension (e.g., 1408 for V-JEPA-2 ViT-Giant)
"""

import torch
import numpy as np
from pathlib import Path


def l1_energy(z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 (mean absolute) energy between predicted and goal representations.

    This is the baseline V-JEPA-2 energy function.

    Args:
        z_pred: Predicted representation [..., T, D] or [T, D]
        z_goal: Goal representation [..., T, D] or [T, D]

    Returns:
        Energy values [...] - scalar for each batch element
    """
    # Compute mean absolute difference over tokens and dimensions
    return torch.mean(torch.abs(z_pred - z_goal), dim=(-2, -1))


def cosine_energy(z_pred: torch.Tensor, z_goal: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute cosine distance energy (scale-invariant).

    Computes cosine distance token-wise then averages over tokens:
        cos_sim_t = dot(z_pred[t], z_goal[t]) / (||z_pred[t]|| * ||z_goal[t]|| + eps)
        cos_dist_t = 1 - cos_sim_t
        Energy = mean_t(cos_dist_t)

    Args:
        z_pred: Predicted representation [..., T, D] or [T, D]
        z_goal: Goal representation [..., T, D] or [T, D]
        eps: Small constant for numerical stability

    Returns:
        Energy values [...] - scalar for each batch element, in range [0, 2]
    """
    # Compute norms along feature dimension D (last dim)
    norm_pred = torch.norm(z_pred, dim=-1, keepdim=True)  # [..., T, 1]
    norm_goal = torch.norm(z_goal, dim=-1, keepdim=True)  # [..., T, 1]

    # Compute dot product along feature dimension
    dot_product = torch.sum(z_pred * z_goal, dim=-1, keepdim=True)  # [..., T, 1]

    # Compute cosine similarity with eps for stability
    cos_sim = dot_product / (norm_pred * norm_goal + eps)  # [..., T, 1]

    # Convert to distance (1 - similarity), squeeze last dim
    cos_dist = 1.0 - cos_sim.squeeze(-1)  # [..., T]

    # Average over tokens
    return torch.mean(cos_dist, dim=-1)  # [...]


def normalized_l1_energy(
    z_pred: torch.Tensor,
    z_goal: torch.Tensor,
    mu_droid: torch.Tensor,
    sigma_droid: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute normalized L1 energy using DROID statistics.

    Normalizes latent vectors using per-dimension statistics from DROID embeddings:
        z_norm = (z - mu_DROID) / (sigma_DROID + eps)
        Energy = mean(abs(z_pred_norm - z_goal_norm))

    Note: Even though mu/sigma are computed from mean-pooled embeddings,
    normalization is applied token-wise by broadcasting over T.

    Args:
        z_pred: Predicted representation [..., T, D] or [T, D]
        z_goal: Goal representation [..., T, D] or [T, D]
        mu_droid: DROID embedding mean [D]
        sigma_droid: DROID embedding std [D]
        eps: Small constant for numerical stability in denominator

    Returns:
        Energy values [...] - scalar for each batch element
    """
    # Ensure statistics are on same device as inputs
    device = z_pred.device
    mu = mu_droid.to(device)
    sigma = sigma_droid.to(device)

    # Normalize (broadcasting: [D] -> [..., T, D])
    z_pred_norm = (z_pred - mu) / (sigma + eps)
    z_goal_norm = (z_goal - mu) / (sigma + eps)

    # Compute L1 over normalized representations
    return torch.mean(torch.abs(z_pred_norm - z_goal_norm), dim=(-2, -1))


class EnergyFunction:
    """
    Wrapper class for energy functions with optional DROID statistics.

    Provides a unified interface for all energy functions:
        energy = energy_fn(z_pred, z_goal)
    """

    def __init__(
        self,
        metric: str = 'l1',
        stats_path: str = None,
        device: str = 'cuda'
    ):
        """
        Initialize energy function.

        Args:
            metric: One of 'l1', 'cosine', 'norm_l1'
            stats_path: Path to directory containing DROID statistics
                       (required for 'norm_l1')
            device: Torch device
        """
        self.metric = metric
        self.device = device
        self.mu_droid = None
        self.sigma_droid = None

        if metric == 'norm_l1':
            if stats_path is None:
                raise ValueError("stats_path required for norm_l1 metric")
            self._load_droid_stats(stats_path)

    def _load_droid_stats(self, stats_path: str):
        """Load DROID statistics from file."""
        stats_dir = Path(stats_path)

        # Try to load mu_droid
        mu_path = stats_dir / 'mu_droid.npy'
        if mu_path.exists():
            self.mu_droid = torch.from_numpy(np.load(mu_path)).float().to(self.device)
        else:
            raise FileNotFoundError(f"Could not find mu_droid.npy in {stats_dir}")

        # Try to load sigma_droid (or compute from covariance)
        sigma_path = stats_dir / 'sigma_droid.npy'
        if sigma_path.exists():
            self.sigma_droid = torch.from_numpy(np.load(sigma_path)).float().to(self.device)
        else:
            # Try to compute from covariance
            cov_path = stats_dir / 'cov_droid.npy'
            if cov_path.exists():
                cov_droid = np.load(cov_path)
                sigma_droid = np.sqrt(np.diag(cov_droid))
                self.sigma_droid = torch.from_numpy(sigma_droid).float().to(self.device)

                # Save for future use
                np.save(stats_dir / 'sigma_droid.npy', sigma_droid)
                print(f"Computed and saved sigma_droid from covariance")
            else:
                raise FileNotFoundError(
                    f"Could not find sigma_droid.npy or cov_droid.npy in {stats_dir}"
                )

        print(f"Loaded DROID stats from {stats_dir}")
        print(f"  mu_droid shape: {self.mu_droid.shape}")
        print(f"  sigma_droid shape: {self.sigma_droid.shape}")
        print(f"  sigma_droid range: [{self.sigma_droid.min():.4f}, {self.sigma_droid.max():.4f}]")

    def __call__(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """
        Compute energy between predicted and goal representations.

        Args:
            z_pred: Predicted representation [..., T, D] or [T, D]
            z_goal: Goal representation [..., T, D] or [T, D]

        Returns:
            Energy values [...] - scalar for each batch element
        """
        if self.metric == 'l1':
            return l1_energy(z_pred, z_goal)
        elif self.metric == 'cosine':
            return cosine_energy(z_pred, z_goal)
        elif self.metric == 'norm_l1':
            return normalized_l1_energy(
                z_pred, z_goal,
                self.mu_droid, self.sigma_droid
            )
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def __repr__(self):
        return f"EnergyFunction(metric='{self.metric}')"


def compute_energy(
    z_pred: torch.Tensor,
    z_goal: torch.Tensor,
    metric: str,
    mu_droid: torch.Tensor = None,
    sigma_droid: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute energy using specified metric.

    Convenience function that dispatches to appropriate energy function.

    Args:
        z_pred: Predicted representation [..., T, D] or [T, D]
        z_goal: Goal representation [..., T, D] or [T, D]
        metric: One of 'l1', 'cosine', 'norm_l1'
        mu_droid: DROID embedding mean [D] (required for norm_l1)
        sigma_droid: DROID embedding std [D] (required for norm_l1)

    Returns:
        Energy values [...] - scalar for each batch element
    """
    if metric == 'l1':
        return l1_energy(z_pred, z_goal)
    elif metric == 'cosine':
        return cosine_energy(z_pred, z_goal)
    elif metric == 'norm_l1':
        if mu_droid is None or sigma_droid is None:
            raise ValueError("mu_droid and sigma_droid required for norm_l1 metric")
        return normalized_l1_energy(z_pred, z_goal, mu_droid, sigma_droid)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# For backward compatibility with mpc_utils.py
def l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Original L1 function matching mpc_utils.py signature.

    Args:
        a: Flattened representation [B, T*D]
        b: Flattened representation [B, T*D]

    Returns:
        L1 distances [B]
    """
    return torch.mean(torch.abs(a - b), dim=-1)


if __name__ == "__main__":
    # Test energy functions
    print("Testing energy functions...")

    # Create dummy data
    B, T, D = 10, 256, 1408
    z_pred = torch.randn(B, T, D)
    z_goal = torch.randn(B, T, D)

    # Test L1
    e_l1 = l1_energy(z_pred, z_goal)
    print(f"L1 energy shape: {e_l1.shape}, range: [{e_l1.min():.4f}, {e_l1.max():.4f}]")

    # Test cosine
    e_cos = cosine_energy(z_pred, z_goal)
    print(f"Cosine energy shape: {e_cos.shape}, range: [{e_cos.min():.4f}, {e_cos.max():.4f}]")

    # Test normalized L1 (with dummy stats)
    mu_droid = torch.zeros(D)
    sigma_droid = torch.ones(D)
    e_norm = normalized_l1_energy(z_pred, z_goal, mu_droid, sigma_droid)
    print(f"Normalized L1 energy shape: {e_norm.shape}, range: [{e_norm.min():.4f}, {e_norm.max():.4f}]")

    # Verify shapes are correct
    assert e_l1.shape == (B,), f"Expected ({B},), got {e_l1.shape}"
    assert e_cos.shape == (B,), f"Expected ({B},), got {e_cos.shape}"
    assert e_norm.shape == (B,), f"Expected ({B},), got {e_norm.shape}"

    print("\nAll tests passed!")
