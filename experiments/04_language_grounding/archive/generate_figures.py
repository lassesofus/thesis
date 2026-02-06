"""
Generate figures for the language grounding methods section.

Figures:
1. Frame pair grid - 2xN grid showing before/after frames with instructions
2. Diverse instructions - Single frame pair with 5 different instruction variants
"""

import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import textwrap


def wrap_text(text, width=40):
    """Wrap text to specified width."""
    return "\n".join(textwrap.wrap(text, width=width))


def create_frame_pair_grid(
    pairs_data: list[dict],
    output_path: str,
    figsize: tuple = (8, 5)
):
    """
    Create a compact vertical grid showing before/after frame pairs with instructions.

    Each row: Before frame | After frame | Instruction text
    """
    # Select diverse pairs (different actions in the manipulation sequence)
    selected_indices = [0, 3, 4, 5]  # Lower/grasp, grasp cup, lift, pour
    selected_pairs = [pairs_data[i] for i in selected_indices if i < len(pairs_data)]

    n_pairs = len(selected_pairs)

    # Create figure with tight layout - wider instruction column
    fig, axes = plt.subplots(n_pairs, 3, figsize=figsize,
                              gridspec_kw={'width_ratios': [1, 1, 0.65], 'wspace': 0.02, 'hspace': 0.08})

    # Add column headers
    axes[0, 0].set_title('Before ($x_k$)', fontsize=10, fontweight='bold', pad=3)
    axes[0, 1].set_title('After ($x_{k+d}$)', fontsize=10, fontweight='bold', pad=3)
    axes[0, 2].set_title('VLM-generated instruction', fontsize=10, fontweight='bold', pad=3)

    for row, pair in enumerate(selected_pairs):
        # Get frame paths
        base_dir = Path("/home/s185927/thesis/experiments/04_language_grounding")
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

        # Instruction text - fill the column
        axes[row, 2].axis('off')
        instruction = pair["instruction"]
        wrapped = wrap_text(instruction, width=20)
        axes[row, 2].text(0.05, 0.5, f'"{wrapped}"',
                          transform=axes[row, 2].transAxes,
                          fontsize=8, va='center', ha='left',
                          style='italic',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#cccccc', alpha=0.8))

    plt.subplots_adjust(top=0.92, bottom=0.02, left=0.01, right=0.99)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved frame pair grid to: {output_path}")


def create_diverse_instructions_figure(
    pair: dict,
    diverse_instructions: list[str],
    output_path: str,
    figsize: tuple = (12, 5)
):
    """
    Create figure showing a single frame pair with 5 diverse instructions.

    Layout: Two images side by side, with numbered instruction list below.
    """
    fig = plt.figure(figsize=figsize)

    # Create grid: 2 columns for images, spanning row for instructions
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.3, wspace=0.1)

    # Get frame paths
    base_dir = Path("/home/s185927/thesis/experiments/04_language_grounding")
    frame_k_path = base_dir / pair["frame_k_path"]
    frame_kd_path = base_dir / pair["frame_k_d_path"]

    # Load images
    img_before = mpimg.imread(frame_k_path)
    img_after = mpimg.imread(frame_kd_path)

    # Plot before frame
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_before)
    ax1.set_title(f'Frame {pair["frame_k"]} (Before)', fontsize=11, fontweight='bold')
    ax1.axis('off')

    # Plot after frame
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_after)
    ax2.set_title(f'Frame {pair["frame_k_d"]} (After, d={pair["d"]})', fontsize=11, fontweight='bold')
    ax2.axis('off')

    # Add arrow between images
    fig.text(0.5, 0.62, '→', fontsize=30, ha='center', va='center',
             transform=fig.transFigure)

    # Create text area for instructions
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    # Format instructions as numbered list
    instruction_text = "Diverse Instructions (all describing the same action):\n\n"
    categories = [
        "(brief)",
        "(detailed)",
        "(directional)",
        "(different verb)",
        "(spatial reference)"
    ]

    for i, (inst, cat) in enumerate(zip(diverse_instructions, categories), 1):
        instruction_text += f"  {i}. \"{inst}\"  {cat}\n"

    ax3.text(0.5, 0.9, instruction_text, transform=ax3.transAxes,
             fontsize=10, va='top', ha='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    # Add what changed description
    what_changed = pair.get("what_changed", "")
    if what_changed:
        ax3.text(0.5, -0.05, f'What changed: "{what_changed}"',
                 transform=ax3.transAxes, fontsize=9, va='top', ha='center',
                 style='italic', color='gray')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved diverse instructions figure to: {output_path}")


def main():
    # Load frame pairs data
    data_path = Path("/home/s185927/thesis/experiments/04_language_grounding/sampled_pairs_v4/frame_pairs_v2.json")
    with open(data_path) as f:
        pairs_data = json.load(f)

    output_dir = Path("/home/s185927/thesis/Report/experiments/03_language_grounding/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Frame pair grid (vertical layout for A4)
    print("\nGenerating frame pair grid...")
    create_frame_pair_grid(
        pairs_data,
        output_path=str(output_dir / "frame_pair_grid.pdf")
    )

    # Also save as PNG for quick viewing
    create_frame_pair_grid(
        pairs_data,
        output_path=str(output_dir / "frame_pair_grid.png")
    )

    # Figure 2: Diverse instructions
    # Use the "lift the yellow cup" pair as it's a clear action
    lift_pair = pairs_data[4]  # frame 195 -> 225, "Lift the yellow cup upward"

    # Example diverse instructions (these would come from v3 generator in practice)
    diverse_instructions = [
        "Lift the cup.",
        "Raise the yellow cup upward from the table surface.",
        "Move the gripper vertically upward while holding the cup.",
        "Elevate the grasped object away from the workspace.",
        "Pull the cup up toward the robot's shoulder."
    ]

    print("\nGenerating diverse instructions figure...")
    create_diverse_instructions_figure(
        lift_pair,
        diverse_instructions,
        output_path=str(output_dir / "diverse_instructions.pdf")
    )

    # Also save as PNG
    create_diverse_instructions_figure(
        lift_pair,
        diverse_instructions,
        output_path=str(output_dir / "diverse_instructions.png")
    )

    print("\nDone! Figures saved to:", output_dir)


if __name__ == "__main__":
    main()
