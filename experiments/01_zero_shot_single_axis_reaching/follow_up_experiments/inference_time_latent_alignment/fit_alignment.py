#!/usr/bin/env python3
"""
Fit latent alignment transforms (mean-only and CORAL) from pre-collected embeddings.

This script:
1. Loads pre-computed embeddings from DROID and simulation
2. Computes mean/covariance statistics
3. Fits CORAL alignment transform (whitening-coloring)
4. Validates alignment on held-out embeddings
5. Saves all statistics and alignment parameters

Usage:
    python fit_alignment.py --epsilon 1e-3
"""

import argparse
import json
from pathlib import Path

import numpy as np


# Paths
EMBEDDINGS_CACHE = Path("/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/latent_distribution_comparison/embeddings_cache.npz")
OUTPUT_DIR = Path("/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/inference_time_latent_alignment/stats")


def compute_matrix_sqrt(cov: np.ndarray, epsilon: float = 1e-3) -> tuple:
    """
    Compute matrix square root and inverse square root using eigen-decomposition.

    Args:
        cov: Covariance matrix [D, D]
        epsilon: Regularization term

    Returns:
        sqrt_cov: C^(1/2)
        inv_sqrt_cov: C^(-1/2)
    """
    # Add regularization
    D = cov.shape[0]
    cov_reg = cov + epsilon * np.eye(D)

    # Eigen-decomposition (use float64 for numerical stability)
    cov_reg = cov_reg.astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)

    # Clip negative eigenvalues (should be small due to numerical issues)
    eigenvalues = np.maximum(eigenvalues, epsilon)

    # Compute sqrt and inverse sqrt of eigenvalues
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    inv_sqrt_eigenvalues = 1.0 / sqrt_eigenvalues

    # Reconstruct matrices: C^(1/2) = U * diag(sqrt(lambda)) * U^T
    sqrt_cov = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
    inv_sqrt_cov = eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T

    return sqrt_cov.astype(np.float32), inv_sqrt_cov.astype(np.float32)


def fit_alignment(
    droid_embeddings: np.ndarray,
    sim_embeddings: np.ndarray,
    epsilon: float = 1e-3,
    n_fit: int = 800,
) -> dict:
    """
    Fit mean-only and CORAL alignment transforms.

    Args:
        droid_embeddings: [N_D, D] DROID embeddings
        sim_embeddings: [N_S, D] simulation embeddings
        epsilon: Regularization for covariance inversion
        n_fit: Number of samples to use for fitting (rest for validation)

    Returns:
        dict with alignment parameters and validation metrics
    """
    print(f"\n{'='*60}")
    print("FITTING LATENT ALIGNMENT TRANSFORMS")
    print(f"{'='*60}")

    # Split into fit and validation sets
    droid_fit = droid_embeddings[:n_fit]
    droid_val = droid_embeddings[n_fit:]
    sim_fit = sim_embeddings[:n_fit]
    sim_val = sim_embeddings[n_fit:]

    print(f"Fit set: {len(droid_fit)} DROID, {len(sim_fit)} sim")
    print(f"Validation set: {len(droid_val)} DROID, {len(sim_val)} sim")

    # Compute means
    mu_droid = droid_fit.mean(axis=0)
    mu_sim = sim_fit.mean(axis=0)

    # Compute covariances (unbiased, N-1)
    cov_droid = np.cov(droid_fit.T).astype(np.float64)
    cov_sim = np.cov(sim_fit.T).astype(np.float64)

    print(f"\nMean offset ||mu_D - mu_S||_2: {np.linalg.norm(mu_droid - mu_sim):.4f}")
    print(f"trace(cov_D): {np.trace(cov_droid):.4f}")
    print(f"trace(cov_S): {np.trace(cov_sim):.4f}")

    # Compute eigenvalues for diagnostics
    eig_droid = np.linalg.eigvalsh(cov_droid)
    eig_sim = np.linalg.eigvalsh(cov_sim)

    print(f"\ncov_droid eigenvalues: min={eig_droid.min():.6f}, max={eig_droid.max():.4f}")
    print(f"cov_sim eigenvalues: min={eig_sim.min():.6f}, max={eig_sim.max():.4f}")

    # Compute matrix square roots for CORAL
    print(f"\nComputing matrix square roots with epsilon={epsilon}...")
    sqrt_cov_droid, _ = compute_matrix_sqrt(cov_droid, epsilon)
    _, inv_sqrt_cov_sim = compute_matrix_sqrt(cov_sim, epsilon)

    # CORAL transform matrix: A = C_S^(-1/2) @ C_D^(1/2)
    # Transform: z_aligned = (z - mu_S) @ A + mu_D
    coral_matrix = inv_sqrt_cov_sim @ sqrt_cov_droid

    print(f"CORAL matrix shape: {coral_matrix.shape}")
    print(f"CORAL matrix norm: {np.linalg.norm(coral_matrix):.4f}")

    # Validate alignment on held-out data
    print(f"\n{'='*60}")
    print("VALIDATION ON HELD-OUT EMBEDDINGS")
    print(f"{'='*60}")

    # Apply mean-only alignment to sim validation set
    sim_val_mean_aligned = sim_val - mu_sim + mu_droid

    # Apply CORAL alignment to sim validation set
    sim_val_coral_aligned = (sim_val - mu_sim) @ coral_matrix + mu_droid

    # Compute validation metrics
    # 1. Mean error: ||mean(aligned) - mu_D_fit||
    mean_err_raw = np.linalg.norm(sim_val.mean(axis=0) - mu_droid)
    mean_err_mean_aligned = np.linalg.norm(sim_val_mean_aligned.mean(axis=0) - mu_droid)
    mean_err_coral_aligned = np.linalg.norm(sim_val_coral_aligned.mean(axis=0) - mu_droid)

    print(f"\nMean error (||mean(aligned) - mu_D||_2):")
    print(f"  Raw sim:        {mean_err_raw:.4f}")
    print(f"  Mean-aligned:   {mean_err_mean_aligned:.4f}")
    print(f"  CORAL-aligned:  {mean_err_coral_aligned:.4f}")

    # 2. Covariance error: ||cov(aligned) - cov_D||_F / ||cov_D||_F
    cov_val_raw = np.cov(sim_val.T)
    cov_val_mean = np.cov(sim_val_mean_aligned.T)
    cov_val_coral = np.cov(sim_val_coral_aligned.T)

    cov_droid_norm = np.linalg.norm(cov_droid, 'fro')
    cov_err_raw = np.linalg.norm(cov_val_raw - cov_droid, 'fro') / cov_droid_norm
    cov_err_mean = np.linalg.norm(cov_val_mean - cov_droid, 'fro') / cov_droid_norm
    cov_err_coral = np.linalg.norm(cov_val_coral - cov_droid, 'fro') / cov_droid_norm

    print(f"\nCovariance error (||cov(aligned) - cov_D||_F / ||cov_D||_F):")
    print(f"  Raw sim:        {cov_err_raw:.4f}")
    print(f"  Mean-aligned:   {cov_err_mean:.4f}")
    print(f"  CORAL-aligned:  {cov_err_coral:.4f}")

    # 3. Pairwise distance to DROID validation set
    # Sample random pairs for efficiency
    n_pairs = min(200, len(droid_val), len(sim_val))
    rng = np.random.default_rng(42)
    droid_idx = rng.choice(len(droid_val), size=n_pairs, replace=False)
    sim_idx = rng.choice(len(sim_val), size=n_pairs, replace=False)

    dist_raw = np.mean([np.linalg.norm(sim_val[sim_idx[i]] - droid_val[droid_idx[i]])
                        for i in range(n_pairs)])
    dist_mean = np.mean([np.linalg.norm(sim_val_mean_aligned[sim_idx[i]] - droid_val[droid_idx[i]])
                         for i in range(n_pairs)])
    dist_coral = np.mean([np.linalg.norm(sim_val_coral_aligned[sim_idx[i]] - droid_val[droid_idx[i]])
                          for i in range(n_pairs)])

    print(f"\nMean pairwise distance to DROID (L2):")
    print(f"  Raw sim:        {dist_raw:.4f}")
    print(f"  Mean-aligned:   {dist_mean:.4f}")
    print(f"  CORAL-aligned:  {dist_coral:.4f}")

    # Package results
    results = {
        "epsilon": epsilon,
        "n_fit": n_fit,
        "n_val": len(sim_val),
        "embedding_dim": droid_embeddings.shape[1],
        "mean_offset_norm": float(np.linalg.norm(mu_droid - mu_sim)),
        "validation": {
            "mean_error": {
                "raw": float(mean_err_raw),
                "mean_aligned": float(mean_err_mean_aligned),
                "coral_aligned": float(mean_err_coral_aligned),
            },
            "cov_error_relative": {
                "raw": float(cov_err_raw),
                "mean_aligned": float(cov_err_mean),
                "coral_aligned": float(cov_err_coral),
            },
            "pairwise_dist_to_droid": {
                "raw": float(dist_raw),
                "mean_aligned": float(dist_mean),
                "coral_aligned": float(dist_coral),
            },
        },
        "cov_stats": {
            "trace_droid": float(np.trace(cov_droid)),
            "trace_sim": float(np.trace(cov_sim)),
            "min_eig_droid": float(eig_droid.min()),
            "max_eig_droid": float(eig_droid.max()),
            "min_eig_sim": float(eig_sim.min()),
            "max_eig_sim": float(eig_sim.max()),
        },
    }

    # Save alignment parameters
    alignment_params = {
        "mu_droid": mu_droid.astype(np.float32),
        "mu_sim": mu_sim.astype(np.float32),
        "cov_droid": cov_droid.astype(np.float32),
        "cov_sim": cov_sim.astype(np.float32),
        "sqrt_cov_droid": sqrt_cov_droid,
        "inv_sqrt_cov_sim": inv_sqrt_cov_sim,
        "coral_matrix": coral_matrix.astype(np.float32),
    }

    return results, alignment_params


def create_validation_plot(
    droid_embeddings: np.ndarray,
    sim_embeddings: np.ndarray,
    alignment_params: dict,
    output_dir: Path,
    n_fit: int = 800,
):
    """Create PCA visualization of raw vs aligned embeddings."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    print("\n=== Creating Validation Plot ===")

    # Get validation set
    sim_val = sim_embeddings[n_fit:]
    droid_val = droid_embeddings[n_fit:]

    # Apply alignments
    mu_sim = alignment_params["mu_sim"]
    mu_droid = alignment_params["mu_droid"]
    coral_matrix = alignment_params["coral_matrix"]

    sim_val_mean = sim_val - mu_sim + mu_droid
    sim_val_coral = (sim_val - mu_sim) @ coral_matrix + mu_droid

    # Fit PCA on DROID data
    pca = PCA(n_components=2)
    pca.fit(droid_embeddings)

    # Transform all embeddings
    droid_val_pca = pca.transform(droid_val)
    sim_val_pca = pca.transform(sim_val)
    sim_mean_pca = pca.transform(sim_val_mean)
    sim_coral_pca = pca.transform(sim_val_coral)

    var_explained = pca.explained_variance_ratio_

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Raw
    ax = axes[0]
    ax.scatter(droid_val_pca[:, 0], droid_val_pca[:, 1], alpha=0.5, s=30, label="DROID", c="tab:blue")
    ax.scatter(sim_val_pca[:, 0], sim_val_pca[:, 1], alpha=0.5, s=30, label="Sim (raw)", c="tab:orange")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
    ax.set_title("(a) Raw")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean-aligned
    ax = axes[1]
    ax.scatter(droid_val_pca[:, 0], droid_val_pca[:, 1], alpha=0.5, s=30, label="DROID", c="tab:blue")
    ax.scatter(sim_mean_pca[:, 0], sim_mean_pca[:, 1], alpha=0.5, s=30, label="Sim (mean)", c="tab:green")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
    ax.set_title("(b) Mean-aligned")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CORAL-aligned
    ax = axes[2]
    ax.scatter(droid_val_pca[:, 0], droid_val_pca[:, 1], alpha=0.5, s=30, label="DROID", c="tab:blue")
    ax.scatter(sim_coral_pca[:, 0], sim_coral_pca[:, 1], alpha=0.5, s=30, label="Sim (CORAL)", c="tab:red")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
    ax.set_title("(c) CORAL-aligned")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Validation: Latent Alignment (PCA projection)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "alignment_validation_pca.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved validation plot to {output_dir / 'alignment_validation_pca.png'}")


def main():
    parser = argparse.ArgumentParser(description="Fit latent alignment transforms")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Regularization for covariance")
    parser.add_argument("--n_fit", type=int, default=800, help="Number of samples for fitting (rest for validation)")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cached embeddings
    print(f"Loading embeddings from {EMBEDDINGS_CACHE}")
    data = np.load(EMBEDDINGS_CACHE)
    droid_embeddings = data["droid_embeddings"]
    sim_embeddings = data["sim_embeddings"]

    print(f"DROID embeddings: {droid_embeddings.shape}")
    print(f"Simulation embeddings: {sim_embeddings.shape}")

    # Fit alignment
    results, alignment_params = fit_alignment(
        droid_embeddings,
        sim_embeddings,
        epsilon=args.epsilon,
        n_fit=args.n_fit,
    )

    # Save alignment parameters
    np.savez(
        output_dir / "alignment_params.npz",
        **{k: v for k, v in alignment_params.items()}
    )
    print(f"\nSaved alignment parameters to {output_dir / 'alignment_params.npz'}")

    # Save individual arrays for easy loading
    np.save(output_dir / "mu_droid.npy", alignment_params["mu_droid"])
    np.save(output_dir / "mu_sim.npy", alignment_params["mu_sim"])
    np.save(output_dir / "cov_droid.npy", alignment_params["cov_droid"])
    np.save(output_dir / "cov_sim.npy", alignment_params["cov_sim"])
    np.save(output_dir / "coral_matrix.npy", alignment_params["coral_matrix"])

    # Save results
    results["method"] = "CORAL (whitening-coloring)"
    with open(output_dir / "alignment_config.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved alignment config to {output_dir / 'alignment_config.json'}")

    # Create validation plot
    create_validation_plot(droid_embeddings, sim_embeddings, alignment_params, output_dir, args.n_fit)

    print(f"\n{'='*60}")
    print("ALIGNMENT FITTING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Files saved:")
    print(f"  - alignment_params.npz (all parameters)")
    print(f"  - mu_droid.npy, mu_sim.npy")
    print(f"  - cov_droid.npy, cov_sim.npy")
    print(f"  - coral_matrix.npy")
    print(f"  - alignment_config.json")
    print(f"  - alignment_validation_pca.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
