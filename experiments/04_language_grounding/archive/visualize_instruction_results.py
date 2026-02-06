"""
Visualize frame pairs with their GPT-5 generated instructions.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import textwrap


def wrap_text(text, width=35):
    """Wrap text to specified width."""
    return "\n".join(textwrap.wrap(text, width=width))


def create_instruction_results_figure(
    results: list[dict],
    base_dir: Path,
    output_path: str,
    figsize: tuple = None
):
    """
    Create a figure showing frame pairs with their generated instructions.

    Layout: Each row shows Before | After | Instructions (all 5)
    """
    # Filter to only pairs with instructions
    valid_results = [r for r in results if r.get("instructions")]
    n_pairs = len(valid_results)

    if n_pairs == 0:
        print("No valid results to visualize")
        return

    # Auto-size figure based on number of pairs
    if figsize is None:
        figsize = (14, 3.2 * n_pairs)

    fig, axes = plt.subplots(n_pairs, 3, figsize=figsize,
                              gridspec_kw={'width_ratios': [1, 1, 1.5], 'wspace': 0.03, 'hspace': 0.15})

    # Handle single row case
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    # Add column headers
    axes[0, 0].set_title('Before ($x_k$)', fontsize=12, fontweight='bold', pad=8)
    axes[0, 1].set_title('After ($x_{k+d}$)', fontsize=12, fontweight='bold', pad=8)
    axes[0, 2].set_title('GPT-5 Generated Instructions', fontsize=12, fontweight='bold', pad=8)

    for row, result in enumerate(valid_results):
        # Get lab name
        lab = result["trajectory_id"].split("+")[0]

        # Load images
        frame_k_path = base_dir / result["frame_k_path"]
        frame_kd_path = base_dir / result["frame_k_d_path"]

        img_before = mpimg.imread(frame_k_path)
        img_after = mpimg.imread(frame_kd_path)

        # Plot before frame
        axes[row, 0].imshow(img_before)
        axes[row, 0].axis('off')
        # Add lab label
        axes[row, 0].text(0.02, 0.98, lab, transform=axes[row, 0].transAxes,
                          fontsize=9, fontweight='bold', va='top',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Plot after frame
        axes[row, 1].imshow(img_after)
        axes[row, 1].axis('off')
        # Add d value
        axes[row, 1].text(0.98, 0.98, f'd={result["d"]}', transform=axes[row, 1].transAxes,
                          fontsize=9, va='top', ha='right',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Instructions panel
        axes[row, 2].axis('off')

        # Format instructions
        instructions = result.get("instructions", [])
        what_changed = result.get("what_changed", "")

        # Build text block
        text_lines = []
        if what_changed:
            text_lines.append(f"Changed: {wrap_text(what_changed, 45)}")
            text_lines.append("")

        for i, inst in enumerate(instructions[:5], 1):
            text_lines.append(f"{i}. {inst}")

        instruction_text = "\n".join(text_lines)

        axes[row, 2].text(0.02, 0.95, instruction_text,
                          transform=axes[row, 2].transAxes,
                          fontsize=9, va='top', ha='left',
                          family='sans-serif',
                          bbox=dict(boxstyle='round,pad=0.4',
                                   facecolor='#f8f8f8',
                                   edgecolor='#cccccc',
                                   alpha=0.95))

    plt.subplots_adjust(top=0.95, bottom=0.02, left=0.01, right=0.99)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved figure to: {output_path}")


def main():
    # Load results
    results_path = Path("/home/s185927/thesis/experiments/04_language_grounding/sampled_pairs_multi_gpt5/diverse_test_results.json")
    base_dir = Path("/data/s185927/droid_raw/sampled_pairs")
    output_dir = Path("/home/s185927/thesis/experiments/04_language_grounding/sampled_pairs_multi_gpt5")

    print(f"Loading results from {results_path}...")
    with open(results_path) as f:
        results = json.load(f)

    # Filter to valid results
    valid = [r for r in results if r.get("instructions")]
    print(f"Found {len(valid)} pairs with instructions out of {len(results)} total")

    # Show summary
    print("\nPairs with instructions:")
    for r in valid:
        lab = r["trajectory_id"].split("+")[0]
        print(f"  {lab}: frames {r['frame_k']}->{r['frame_k_d']} (d={r['d']})")

    # Generate figure
    print("\nGenerating figure...")
    create_instruction_results_figure(
        results,
        base_dir=base_dir,
        output_path=str(output_dir / "gpt5_instruction_samples.png")
    )

    # Also save as PDF
    create_instruction_results_figure(
        results,
        base_dir=base_dir,
        output_path=str(output_dir / "gpt5_instruction_samples.pdf")
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
