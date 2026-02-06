"""
Generate a figure showing one sampled frame pair from each lab.

Creates a 6-row grid (one per lab) showing before/after frames.
"""

import json
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from collections import defaultdict


# Known DROID labs
KNOWN_LABS = {"AUTOLab", "CLVR", "GuptaLab", "ILIAD", "IPRL", "IRIS"}


def extract_lab(trajectory_id: str) -> str | None:
    """Extract lab name from trajectory ID (first part before '+')."""
    parts = trajectory_id.split("+")
    if len(parts) >= 2 and parts[0] in KNOWN_LABS:
        return parts[0]
    return None


def select_diverse_pairs(pairs_data: list[dict], seed: int = 42) -> dict[str, dict]:
    """
    Select one high-quality pair from each lab.

    Prioritizes Tier A pairs with significant visual change (gripper or position delta).
    """
    random.seed(seed)

    # Group pairs by lab (only known labs, skip pairs with missing paths)
    lab_pairs = defaultdict(list)
    for pair in pairs_data:
        # Skip pairs with missing frame paths
        if pair.get("frame_k_path") is None or pair.get("frame_k_d_path") is None:
            continue

        lab = extract_lab(pair["trajectory_id"])
        if lab is not None:
            lab_pairs[lab].append(pair)

    selected = {}
    for lab, pairs in lab_pairs.items():
        # Prefer Tier A pairs
        tier_a = [p for p in pairs if p["salience_tier"] == "A"]
        candidates = tier_a if tier_a else pairs

        # Prefer pairs with gripper change (grasping actions are visually interesting)
        gripper_pairs = [p for p in candidates if abs(p.get("gripper_delta", 0)) > 0.3]
        if gripper_pairs:
            candidates = gripper_pairs

        # Randomly select from candidates
        selected[lab] = random.choice(candidates)

    return selected


def create_lab_samples_figure(
    pairs_by_lab: dict[str, dict],
    base_dir: Path,
    output_path: str,
    figsize: tuple = (8, 10)
):
    """
    Create a grid showing one frame pair per lab.

    Layout: 6 rows (one per lab), 3 columns (Before | After | Lab name)
    """
    # Sort labs alphabetically for consistent ordering
    labs = sorted(pairs_by_lab.keys())
    n_labs = len(labs)

    # Create figure
    fig, axes = plt.subplots(n_labs, 3, figsize=figsize,
                              gridspec_kw={'width_ratios': [1, 1, 0.4], 'wspace': 0.02, 'hspace': 0.08})

    # Add column headers
    axes[0, 0].set_title('Before ($x_k$)', fontsize=11, fontweight='bold', pad=5)
    axes[0, 1].set_title('After ($x_{k+d}$)', fontsize=11, fontweight='bold', pad=5)
    axes[0, 2].set_title('Lab', fontsize=11, fontweight='bold', pad=5)

    for row, lab in enumerate(labs):
        pair = pairs_by_lab[lab]

        # Get frame paths
        frame_k_path = base_dir / pair["frame_k_path"]
        frame_kd_path = base_dir / pair["frame_k_d_path"]

        # Load images
        img_before = mpimg.imread(frame_k_path)
        img_after = mpimg.imread(frame_kd_path)

        # Plot before frame
        axes[row, 0].imshow(img_before)
        axes[row, 0].axis('off')

        # Plot after frame
        axes[row, 1].imshow(img_after)
        axes[row, 1].axis('off')

        # Lab name with details
        axes[row, 2].axis('off')

        # Format info text
        d = pair["d"]
        tier = pair["salience_tier"]
        info_text = f"{lab}\n\n$d$={d}\nTier {tier}"

        axes[row, 2].text(0.5, 0.5, info_text,
                          transform=axes[row, 2].transAxes,
                          fontsize=10, va='center', ha='center',
                          bbox=dict(boxstyle='round,pad=0.4',
                                   facecolor='#f5f5f5',
                                   edgecolor='#cccccc',
                                   alpha=0.9))

    plt.subplots_adjust(top=0.95, bottom=0.02, left=0.01, right=0.99)

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved lab samples figure to: {output_path}")


def main():
    # Paths
    data_dir = Path("/data/s185927/droid_raw/sampled_pairs")
    data_path = data_dir / "frame_pairs.json"
    output_dir = Path("/home/s185927/thesis/experiments/04_language_grounding")

    # Load frame pairs
    print("Loading frame pairs...")
    with open(data_path) as f:
        pairs_data = json.load(f)
    print(f"Loaded {len(pairs_data)} pairs")

    # Select one pair per lab
    print("\nSelecting diverse pairs (one per lab)...")
    pairs_by_lab = select_diverse_pairs(pairs_data)

    for lab, pair in sorted(pairs_by_lab.items()):
        print(f"  {lab}: trajectory {pair['trajectory_id'][:30]}... "
              f"(frames {pair['frame_k']}->{pair['frame_k_d']}, tier {pair['salience_tier']})")

    # Generate figure
    print("\nGenerating figure...")
    create_lab_samples_figure(
        pairs_by_lab,
        base_dir=data_dir,
        output_path=str(output_dir / "lab_samples.png")
    )

    # Also save as PDF
    create_lab_samples_figure(
        pairs_by_lab,
        base_dir=data_dir,
        output_path=str(output_dir / "lab_samples.pdf")
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
