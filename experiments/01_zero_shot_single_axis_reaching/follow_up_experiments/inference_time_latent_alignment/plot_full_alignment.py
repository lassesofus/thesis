#!/usr/bin/env python3
"""
Create PCA visualization of ALL embeddings (not just validation set).

Shows raw sim vs DROID, and the effect of mean-only and CORAL alignment
on the full 1000-sample datasets.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, "/home/s185927/thesis")
try:
    from plot_config import PLOT_PARAMS, apply_plot_params
except ImportError:
    PLOT_PARAMS = {"label_size": 12, "subtitle_size": 14, "legend_size": 10}
    def apply_plot_params(ax): pass

# Paths
EMBEDDINGS_CACHE = Path("/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/latent_distribution_comparison/embeddings_cache.npz")
STATS_DIR = Path("/home/s185927/thesis/experiments/01_zero_shot_single_axis_reaching/follow_up_experiments/inference_time_latent_alignment/stats")
OUTPUT_DIR = STATS_DIR


def main():
    # Load all embeddings
    print(f"Loading embeddings from {EMBEDDINGS_CACHE}")
    data = np.load(EMBEDDINGS_CACHE)
    droid_embeddings = data["droid_embeddings"]
    sim_embeddings = data["sim_embeddings"]

    print(f"DROID: {droid_embeddings.shape}")
    print(f"Simulation: {sim_embeddings.shape}")

    # Load alignment parameters
    mu_sim = np.load(STATS_DIR / "mu_sim.npy")
    mu_droid = np.load(STATS_DIR / "mu_droid.npy")
    coral_matrix = np.load(STATS_DIR / "coral_matrix.npy")

    # Apply alignments to ALL sim embeddings
    sim_mean_aligned = sim_embeddings - mu_sim + mu_droid
    sim_coral_aligned = (sim_embeddings - mu_sim) @ coral_matrix + mu_droid

    # Fit PCA on combined raw data (to get consistent axes)
    print("\nFitting PCA on combined raw embeddings...")
    all_raw = np.vstack([droid_embeddings, sim_embeddings])
    pca = PCA(n_components=2)
    pca.fit(all_raw)

    var_explained = pca.explained_variance_ratio_
    print(f"Variance explained: PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}%")

    # Transform all embeddings
    droid_pca = pca.transform(droid_embeddings)
    sim_raw_pca = pca.transform(sim_embeddings)
    sim_mean_pca = pca.transform(sim_mean_aligned)
    sim_coral_pca = pca.transform(sim_coral_aligned)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Common plot settings
    alpha = 0.4
    s = 20

    # (a) Raw - before alignment
    ax = axes[0]
    ax.scatter(droid_pca[:, 0], droid_pca[:, 1], alpha=alpha, s=s,
               label=f"DROID (n={len(droid_embeddings)})", c="tab:blue")
    ax.scatter(sim_raw_pca[:, 0], sim_raw_pca[:, 1], alpha=alpha, s=s,
               label=f"Simulation (n={len(sim_embeddings)})", c="tab:orange")
    ax.set_xlabel("PC1", fontsize=PLOT_PARAMS["label_size"])
    ax.set_ylabel("PC2", fontsize=PLOT_PARAMS["label_size"])
    ax.set_title("(a) Raw (no alignment)", fontsize=PLOT_PARAMS["subtitle_size"])
    ax.legend(fontsize=PLOT_PARAMS["legend_size"], loc="lower left")
    apply_plot_params(ax)

    # (b) Mean-only alignment
    ax = axes[1]
    ax.scatter(droid_pca[:, 0], droid_pca[:, 1], alpha=alpha, s=s,
               label="DROID", c="tab:blue")
    ax.scatter(sim_mean_pca[:, 0], sim_mean_pca[:, 1], alpha=alpha, s=s,
               label="Sim (mean-aligned)", c="tab:green")
    ax.set_xlabel("PC1", fontsize=PLOT_PARAMS["label_size"])
    ax.set_ylabel("PC2", fontsize=PLOT_PARAMS["label_size"])
    ax.set_title("(b) Mean-only alignment", fontsize=PLOT_PARAMS["subtitle_size"])
    ax.legend(fontsize=PLOT_PARAMS["legend_size"], loc="lower left")
    apply_plot_params(ax)

    # (c) CORAL alignment
    ax = axes[2]
    ax.scatter(droid_pca[:, 0], droid_pca[:, 1], alpha=alpha, s=s,
               label="DROID", c="tab:blue")
    ax.scatter(sim_coral_pca[:, 0], sim_coral_pca[:, 1], alpha=alpha, s=s,
               label="Sim (CORAL-aligned)", c="tab:red")
    ax.set_xlabel("PC1", fontsize=PLOT_PARAMS["label_size"])
    ax.set_ylabel("PC2", fontsize=PLOT_PARAMS["label_size"])
    ax.set_title("(c) CORAL alignment", fontsize=PLOT_PARAMS["subtitle_size"])
    ax.legend(fontsize=PLOT_PARAMS["legend_size"], loc="lower left")
    apply_plot_params(ax)

    # Set same axis limits for all subplots
    all_x = np.concatenate([droid_pca[:, 0], sim_raw_pca[:, 0], sim_mean_pca[:, 0], sim_coral_pca[:, 0]])
    all_y = np.concatenate([droid_pca[:, 1], sim_raw_pca[:, 1], sim_mean_pca[:, 1], sim_coral_pca[:, 1]])
    x_margin = (all_x.max() - all_x.min()) * 0.05
    y_margin = (all_y.max() - all_y.min()) * 0.05

    for ax in axes:
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
        ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "alignment_pca_full.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {output_path}")

    # Also compute and print statistics
    print("\n" + "=" * 60)
    print("ALIGNMENT STATISTICS (all 1000 samples)")
    print("=" * 60)

    # Mean distances
    raw_mean_dist = np.linalg.norm(sim_embeddings.mean(axis=0) - droid_embeddings.mean(axis=0))
    mean_aligned_dist = np.linalg.norm(sim_mean_aligned.mean(axis=0) - droid_embeddings.mean(axis=0))
    coral_aligned_dist = np.linalg.norm(sim_coral_aligned.mean(axis=0) - droid_embeddings.mean(axis=0))

    print(f"\nMean offset ||mean(sim) - mean(droid)||_2:")
    print(f"  Raw:           {raw_mean_dist:.4f}")
    print(f"  Mean-aligned:  {mean_aligned_dist:.4f}")
    print(f"  CORAL-aligned: {coral_aligned_dist:.4f}")

    # Covariance Frobenius norm error
    cov_droid = np.cov(droid_embeddings.T)
    cov_sim_raw = np.cov(sim_embeddings.T)
    cov_sim_mean = np.cov(sim_mean_aligned.T)
    cov_sim_coral = np.cov(sim_coral_aligned.T)

    cov_norm = np.linalg.norm(cov_droid, 'fro')
    cov_err_raw = np.linalg.norm(cov_sim_raw - cov_droid, 'fro') / cov_norm
    cov_err_mean = np.linalg.norm(cov_sim_mean - cov_droid, 'fro') / cov_norm
    cov_err_coral = np.linalg.norm(cov_sim_coral - cov_droid, 'fro') / cov_norm

    print(f"\nCovariance error ||cov(sim) - cov(droid)||_F / ||cov(droid)||_F:")
    print(f"  Raw:           {cov_err_raw:.4f}")
    print(f"  Mean-aligned:  {cov_err_mean:.4f}")
    print(f"  CORAL-aligned: {cov_err_coral:.4f}")

    # Sample pairwise distances
    n_pairs = 1000
    rng = np.random.default_rng(42)
    idx_d = rng.choice(len(droid_embeddings), size=n_pairs, replace=True)
    idx_s = rng.choice(len(sim_embeddings), size=n_pairs, replace=True)

    dist_raw = np.mean([np.linalg.norm(sim_embeddings[idx_s[i]] - droid_embeddings[idx_d[i]])
                        for i in range(n_pairs)])
    dist_mean = np.mean([np.linalg.norm(sim_mean_aligned[idx_s[i]] - droid_embeddings[idx_d[i]])
                         for i in range(n_pairs)])
    dist_coral = np.mean([np.linalg.norm(sim_coral_aligned[idx_s[i]] - droid_embeddings[idx_d[i]])
                          for i in range(n_pairs)])

    print(f"\nMean pairwise distance (sim→droid, n={n_pairs} pairs):")
    print(f"  Raw:           {dist_raw:.4f}")
    print(f"  Mean-aligned:  {dist_mean:.4f}")
    print(f"  CORAL-aligned: {dist_coral:.4f}")


if __name__ == "__main__":
    main()
