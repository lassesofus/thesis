"""
Create PDF showing atomic instructions for 10 frame pairs from GPT-5.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import textwrap
from matplotlib.backends.backend_pdf import PdfPages


def wrap_text(text, width=35):
    """Wrap text to specified width."""
    return "\n".join(textwrap.wrap(text, width=width))


def create_atomic_instructions_pdf(results_path: str, output_path: str):
    """Create multi-page PDF showing frame pairs with atomic instructions."""

    with open(results_path) as f:
        results = json.load(f)

    base_dir = Path("/home/s185927/thesis/experiments/04_language_grounding")

    # Filter to pairs with instructions
    valid_results = [r for r in results if r["instructions"]]

    with PdfPages(output_path) as pdf:
        # Create pages, 2 pairs per page
        for page_idx in range(0, len(valid_results), 2):
            fig, axes = plt.subplots(2, 3, figsize=(11, 8),
                                     gridspec_kw={'width_ratios': [1, 1, 1.2],
                                                  'wspace': 0.05, 'hspace': 0.15})

            for row in range(2):
                result_idx = page_idx + row
                if result_idx >= len(valid_results):
                    # Hide unused row
                    for col in range(3):
                        axes[row, col].axis('off')
                    continue

                result = valid_results[result_idx]

                # Load images
                frame_k_path = base_dir / result["frame_k_path"]
                frame_kd_path = base_dir / result["frame_k_d_path"]

                img_before = mpimg.imread(frame_k_path)
                img_after = mpimg.imread(frame_kd_path)

                # Before frame
                axes[row, 0].imshow(img_before)
                axes[row, 0].set_title(f'Frame {result["frame_k"]} (Before)', fontsize=9)
                axes[row, 0].axis('off')

                # After frame
                axes[row, 1].imshow(img_after)
                axes[row, 1].set_title(f'Frame {result["frame_k_d"]} (After, d={result["d"]})', fontsize=9)
                axes[row, 1].axis('off')

                # Instructions
                axes[row, 2].axis('off')

                # Build instruction text
                instr_text = f"What changed:\n{wrap_text(result['what_changed'], 40)}\n\n"
                instr_text += "Atomic instructions:\n"
                for i, inst in enumerate(result["instructions"], 1):
                    instr_text += f"  {i}. {inst}\n"

                axes[row, 2].text(0.02, 0.95, instr_text,
                                 transform=axes[row, 2].transAxes,
                                 fontsize=8, va='top', ha='left',
                                 family='monospace',
                                 bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='#f5f5f5',
                                          edgecolor='#cccccc'))

            fig.suptitle('GPT-5 Atomic Instructions (v3 Generator)', fontsize=12, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=150)
            plt.close()

    print(f"Saved PDF to: {output_path}")

    # Also create PNG overview of first 4 pairs
    fig, axes = plt.subplots(4, 3, figsize=(12, 12),
                             gridspec_kw={'width_ratios': [1, 1, 1.3],
                                          'wspace': 0.05, 'hspace': 0.12})

    for row in range(min(4, len(valid_results))):
        result = valid_results[row]

        frame_k_path = base_dir / result["frame_k_path"]
        frame_kd_path = base_dir / result["frame_k_d_path"]

        img_before = mpimg.imread(frame_k_path)
        img_after = mpimg.imread(frame_kd_path)

        axes[row, 0].imshow(img_before)
        axes[row, 0].set_title(f'Frame {result["frame_k"]}', fontsize=9)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(img_after)
        axes[row, 1].set_title(f'Frame {result["frame_k_d"]} (d={result["d"]})', fontsize=9)
        axes[row, 1].axis('off')

        axes[row, 2].axis('off')

        instr_text = f"{result['what_changed'][:50]}...\n\n"
        for i, inst in enumerate(result["instructions"], 1):
            instr_text += f"{i}. {inst}\n"

        axes[row, 2].text(0.02, 0.95, instr_text,
                         transform=axes[row, 2].transAxes,
                         fontsize=8, va='top', ha='left',
                         bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='#f5f5f5',
                                  edgecolor='#cccccc'))

    # Add column headers
    axes[0, 0].set_title('Before ($x_k$)', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('After ($x_{k+d}$)', fontsize=10, fontweight='bold')
    axes[0, 2].set_title('GPT-5 Atomic Instructions', fontsize=10, fontweight='bold')

    fig.suptitle('Atomic Instruction Generation with GPT-5', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved PNG to: {png_path}")


if __name__ == "__main__":
    results_path = "/home/s185927/thesis/experiments/04_language_grounding/sampled_pairs_v4/results_v3_gpt5_atomic.json"
    output_path = "/home/s185927/thesis/experiments/04_language_grounding/sampled_pairs_v4/gpt5_atomic_instructions.pdf"

    create_atomic_instructions_pdf(results_path, output_path)
