"""
Create instruction examples figure for thesis.
Layout: Images side-by-side on top, VLM answer below.
"""

import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import textwrap
from pathlib import Path


def create_instruction_figure(
    results: list,
    base_dir: Path,
    output_path: str,
    n_examples: int = None,
    examples_per_page: int = 3
):
    """
    Create figure with images on top, answer below.
    For many examples, creates a multi-page PDF.

    Layout per example:
    +------------------+------------------+
    |    Frame 1       |     Frame 2      |
    |    (Before)      |     (After)      |
    +------------------+------------------+
    |         VLM Answer                  |
    | Frame 1: ...                        |
    | Frame 2: ...                        |
    | Changed: ...                        |
    | 1. instruction...                   |
    | ...                                 |
    +-------------------------------------+
    """
    # Filter to valid results and limit
    # Handle both nested format {"pair": ..., "result": ...} and flat format
    valid_results = []
    for r in results:
        if "result" in r:
            # Nested format
            if r.get("result", {}).get("instructions"):
                valid_results.append(r)
        else:
            # Flat format
            if r.get("instructions"):
                valid_results.append({"pair": r, "result": r})

    if n_examples is not None:
        valid_results = valid_results[:n_examples]
    n_pairs = len(valid_results)

    if n_pairs == 0:
        print("No valid results to visualize")
        return

    # For PDF output with many examples, use multi-page
    is_pdf = output_path.endswith('.pdf')
    n_pages = (n_pairs + examples_per_page - 1) // examples_per_page if is_pdf else 1

    if is_pdf and n_pairs > examples_per_page:
        _create_multipage_pdf(valid_results, base_dir, output_path, examples_per_page)
    else:
        _create_single_figure(valid_results, base_dir, output_path)


def _create_single_figure(valid_results: list, base_dir: Path, output_path: str):
    """Create a single figure with all examples."""
    n_pairs = len(valid_results)

    # Figure size: each example needs ~5.5 inches height
    fig_height = 5.5 * n_pairs
    fig = plt.figure(figsize=(12, fig_height))

    for i, item in enumerate(valid_results):
        _add_example_to_figure(fig, item, i, n_pairs, base_dir)

    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def _create_multipage_pdf(valid_results: list, base_dir: Path, output_path: str, examples_per_page: int):
    """Create a multi-page PDF with examples."""
    n_pairs = len(valid_results)
    n_pages = (n_pairs + examples_per_page - 1) // examples_per_page

    with PdfPages(output_path) as pdf:
        for page in range(n_pages):
            start_idx = page * examples_per_page
            end_idx = min(start_idx + examples_per_page, n_pairs)
            page_results = valid_results[start_idx:end_idx]

            n_on_page = len(page_results)
            fig_height = 5.5 * n_on_page
            fig = plt.figure(figsize=(12, fig_height))

            for i, item in enumerate(page_results):
                _add_example_to_figure(fig, item, i, n_on_page, base_dir)

            pdf.savefig(fig, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    print(f"Saved: {output_path} ({n_pages} pages, {n_pairs} examples)")


def _add_example_to_figure(fig, item: dict, idx: int, total: int, base_dir: Path):
    """Add a single example to a figure."""
    pair = item["pair"]
    result = item["result"]

    # Load images
    frame_k = Image.open(base_dir / pair["frame_k_path"])
    frame_kd = Image.open(base_dir / pair["frame_k_d_path"])

    # Calculate vertical positions for this example
    top = 1 - (idx / total)
    bottom = 1 - ((idx + 1) / total)
    height = top - bottom

    # Images take upper 40%, text takes lower 60%
    img_top = top - 0.02
    img_bottom = bottom + height * 0.58
    text_top = img_bottom - 0.01
    text_bottom = bottom + 0.02

    # Create grid for two images side by side (no gap)
    gs_img = GridSpec(1, 2, figure=fig,
                      left=0.05, right=0.95,
                      top=img_top, bottom=img_bottom,
                      wspace=0.005)

    # Frame K (Before)
    ax_k = fig.add_subplot(gs_img[0, 0])
    ax_k.imshow(frame_k, interpolation='lanczos')
    ax_k.axis("off")
    lab = pair["trajectory_id"].split("+")[0]
    ax_k.text(0.02, 0.98, lab, transform=ax_k.transAxes,
              fontsize=9, fontweight="bold", va="top",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    # Frame K+D (After)
    ax_kd = fig.add_subplot(gs_img[0, 1])
    ax_kd.imshow(frame_kd, interpolation='lanczos')
    ax_kd.axis("off")
    ax_kd.text(0.98, 0.98, f"d={pair['d']}", transform=ax_kd.transAxes,
               fontsize=9, fontweight="bold", va="top", ha="right",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    # Text area below images
    gs_text = GridSpec(1, 1, figure=fig,
                       left=0.05, right=0.95,
                       top=text_top, bottom=text_bottom)
    ax_text = fig.add_subplot(gs_text[0, 0])
    ax_text.axis("off")

    # Build text content
    wrap_width = 100
    fontsize = 8
    label_width = 17  # Characters for label column
    indent = " " * label_width

    # Frame descriptions (full, with wrapping)
    f1_desc = result.get('frame1_description', 'N/A')
    f2_desc = result.get('frame2_description', 'N/A')
    changed = result.get('what_changed', 'N/A')

    # Build content text (without labels, just indented content)
    content_lines = []
    label_info = []  # (line_index, label_text)

    # FRAME 1
    label_info.append((len(content_lines), "FRAME 1:"))
    wrapped_f1 = textwrap.fill(f1_desc, width=wrap_width - label_width,
                                initial_indent="", subsequent_indent="")
    for line in wrapped_f1.split('\n'):
        content_lines.append(indent + line)
    content_lines.append("")

    # FRAME 2
    label_info.append((len(content_lines), "FRAME 2:"))
    wrapped_f2 = textwrap.fill(f2_desc, width=wrap_width - label_width,
                                initial_indent="", subsequent_indent="")
    for line in wrapped_f2.split('\n'):
        content_lines.append(indent + line)
    content_lines.append("")

    # CHANGED
    label_info.append((len(content_lines), "CHANGED:"))
    wrapped_changed = textwrap.fill(changed, width=wrap_width - label_width,
                                     initial_indent="", subsequent_indent="")
    for line in wrapped_changed.split('\n'):
        content_lines.append(indent + line)
    content_lines.append("")

    # INSTRUCTIONS (same indentation as others)
    for j, inst in enumerate(result.get("instructions", [])[:5], 1):
        label_info.append((len(content_lines), f"INSTRUCTION {j}:"))
        content_lines.append(indent + inst)

    full_text = "\n".join(content_lines)

    # Draw each line separately so we can control bold vs regular
    y_pos = 0.97
    line_height = 0.041  # Approximate line height in axes coords for fontsize=8

    # First, draw the background box by computing total height and width
    total_height = len(content_lines) * line_height + 0.04
    # Calculate box width based on longest line (monospace: ~0.0062 per char at fontsize 8)
    max_line_len = max(len(line) for line in content_lines) if content_lines else 50
    box_width = min(0.98, max_line_len * 0.0062 + 0.04)  # Add padding, cap at 0.98
    ax_text.add_patch(plt.Rectangle((0.005, y_pos - total_height),
                                     box_width, total_height + 0.02,
                                     transform=ax_text.transAxes,
                                     facecolor="lightyellow",
                                     edgecolor="#cccccc",
                                     linewidth=1,
                                     clip_on=False))

    # Create a set of line indices that should have bold labels
    label_lines = {line_idx: label for line_idx, label in label_info}

    # Draw each line
    for i, line in enumerate(content_lines):
        if i in label_lines:
            # Draw bold label
            ax_text.text(0.02, y_pos, label_lines[i], transform=ax_text.transAxes,
                        fontsize=fontsize, fontweight="bold", va="top", ha="left",
                        family="monospace")
        # Draw content (always, including indented part)
        ax_text.text(0.02, y_pos, line, transform=ax_text.transAxes,
                    fontsize=fontsize, va="top", ha="left",
                    family="monospace")
        y_pos -= line_height


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create instruction examples figure")
    parser.add_argument("--results_file",
                        default="sampled_pairs_multi_gpt5/gpt5mini_cot_results.json",
                        help="Path to results JSON file")
    parser.add_argument("--base_dir",
                        default="/data/s185927/droid_raw/sampled_pairs",
                        help="Base directory for frame images")
    parser.add_argument("--output",
                        default="sampled_pairs_multi_gpt5/instruction_examples.png",
                        help="Output path (supports .png and .pdf)")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Number of examples to include (default: all)")
    args = parser.parse_args()

    # Load results
    results_path = Path(__file__).parent / args.results_file
    print(f"Loading results from {results_path}...")
    with open(results_path) as f:
        results = json.load(f)

    base_dir = Path(args.base_dir)
    output_path = Path(__file__).parent / args.output

    # Create PNG
    create_instruction_figure(
        results,
        base_dir=base_dir,
        output_path=str(output_path),
        n_examples=args.n_examples
    )

    # Also create PDF version
    pdf_path = str(output_path).replace(".png", ".pdf")
    if pdf_path != str(output_path):
        create_instruction_figure(
            results,
            base_dir=base_dir,
            output_path=pdf_path,
            n_examples=args.n_examples
        )


if __name__ == "__main__":
    main()
