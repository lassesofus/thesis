"""
Run gpt-5-mini with chain-of-thought reasoning prompt.
Model describes each frame, reasons about changes, then generates instructions.
"""

import json
import base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import textwrap
import re
from tqdm import tqdm

# Chain-of-thought prompt with step-by-step reasoning
PROMPT_COT = '''You are shown two frames from a robot manipulation video. Analyze them step by step.

FRAME 1 (BEFORE): The first image
FRAME 2 (AFTER): The second image

Please reason through this carefully:

STEP 1 - DESCRIBE FRAME 1:
Look at the first image. Describe:
- Where is the robot gripper? (position, orientation, open/closed)
- What objects are visible and where are they?

STEP 2 - DESCRIBE FRAME 2:
Look at the second image. Describe:
- Where is the robot gripper now? Is it holding something?
- Where are the objects now? What changed position?

STEP 3 - WHAT CHANGED:
Compare the two frames. What action did the robot COMPLETE between frame 1 and frame 2?

CRITICAL: Describe ONLY what has ALREADY happened, NOT what will happen next!
- If gripper is still holding an object in frame 2 → action is "moved X to Y" NOT "placed X"
- If object is released and on surface in frame 2 → action is "placed X" or "put down X"
- If gripper just closed on object → action is "grabbed X" NOT "picked up and moved"

STEP 4 - INSTRUCTIONS:
Write 5 ways a human might command the EXACT action shown (frame 1 → frame 2 ONLY).
- Instruction 1 MUST be SHORT (3-6 words)
- Do NOT anticipate future steps - describe only what happened
- Stay casual and natural

Format (do NOT include brackets in your response):
FRAME 1: description here
FRAME 2: description here
CHANGED: what happened
INSTRUCTION 1: short instruction
INSTRUCTION 2: natural instruction
INSTRUCTION 3: natural instruction
INSTRUCTION 4: natural instruction
INSTRUCTION 5: natural instruction'''


def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_response(text: str) -> dict:
    """Parse the structured response."""
    result = {
        "frame1_description": "",
        "frame2_description": "",
        "what_changed": "",
        "instructions": [],
        "raw_response": text
    }

    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.upper().startswith("FRAME 1:"):
            result["frame1_description"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("FRAME 2:"):
            result["frame2_description"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CHANGED:"):
            result["what_changed"] = line.split(":", 1)[1].strip()
        elif re.match(r"INSTRUCTION\s*\d+:", line, re.IGNORECASE):
            inst = line.split(":", 1)[1].strip()
            if inst:
                result["instructions"].append(inst)

    return result


def generate_instructions(client, frame_k_path: str, frame_kd_path: str) -> dict:
    """Generate instructions for a frame pair using chain-of-thought."""
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_COT},
                {"type": "text", "text": "\n\nFRAME 1 (BEFORE):"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_base64(frame_k_path)}",
                        "detail": "high"
                    }
                },
                {"type": "text", "text": "\n\nFRAME 2 (AFTER):"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_base64(frame_kd_path)}",
                        "detail": "high"
                    }
                }
            ]
        }],
        max_completion_tokens=3000
    )

    result = parse_response(response.choices[0].message.content)
    result["input_tokens"] = response.usage.prompt_tokens
    result["output_tokens"] = response.usage.completion_tokens
    return result


def create_pdf(results: list, output_path: str, base_dir: Path):
    """Create PDF visualization with 3 pairs per page."""
    pairs_per_page = 3
    n_pages = (len(results) + pairs_per_page - 1) // pairs_per_page

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        for page in range(n_pages):
            start_idx = page * pairs_per_page
            end_idx = min(start_idx + pairs_per_page, len(results))
            page_results = results[start_idx:end_idx]

            fig = plt.figure(figsize=(18, 6 * len(page_results)))

            for i, item in enumerate(page_results):
                pair = item["pair"]
                result = item["result"]

                # Load images
                frame_k = Image.open(base_dir / pair["frame_k_path"])
                frame_kd = Image.open(base_dir / pair["frame_k_d_path"])

                # Create subplots: [frame_k | frame_kd | reasoning + instructions]
                gs = GridSpec(1, 3, figure=fig,
                              left=0.01, right=0.99,
                              top=1 - (i / len(page_results)) - 0.01,
                              bottom=1 - ((i + 1) / len(page_results)) + 0.01,
                              wspace=0.02, width_ratios=[1, 1, 2])

                # Frame K
                ax_k = fig.add_subplot(gs[0, 0])
                ax_k.imshow(frame_k)
                ax_k.axis("off")
                lab = pair["trajectory_id"].split("+")[0]
                ax_k.text(0.02, 0.98, lab, transform=ax_k.transAxes,
                          fontsize=10, fontweight="bold", va="top",
                          bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

                # Frame K+D
                ax_kd = fig.add_subplot(gs[0, 1])
                ax_kd.imshow(frame_kd)
                ax_kd.axis("off")
                ax_kd.text(0.98, 0.98, f"d={pair['d']}", transform=ax_kd.transAxes,
                           fontsize=10, fontweight="bold", va="top", ha="right",
                           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

                # Reasoning + Instructions
                ax_text = fig.add_subplot(gs[0, 2])
                ax_text.axis("off")

                text_lines = []

                # Frame descriptions
                f1_desc = result.get('frame1_description', 'N/A')[:80]
                f2_desc = result.get('frame2_description', 'N/A')[:80]
                text_lines.append(f"Frame 1: {f1_desc}...")
                text_lines.append(f"Frame 2: {f2_desc}...")
                text_lines.append("")
                text_lines.append(f"Changed: {result.get('what_changed', 'N/A')}")
                text_lines.append("")

                for j, inst in enumerate(result.get("instructions", [])[:5], 1):
                    wrapped = textwrap.fill(f"{j}. {inst}", width=60)
                    text_lines.append(wrapped)

                text = "\n".join(text_lines)
                ax_text.text(0.02, 0.95, text, transform=ax_text.transAxes,
                            fontsize=8, va="top", ha="left",
                            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # Also save preview PNG
    png_path = output_path.replace(".pdf", "_preview.png")

    fig = plt.figure(figsize=(18, 6 * min(3, len(results))))
    for i, item in enumerate(results[:3]):
        pair = item["pair"]
        result = item["result"]

        frame_k = Image.open(base_dir / pair["frame_k_path"])
        frame_kd = Image.open(base_dir / pair["frame_k_d_path"])

        gs = GridSpec(1, 3, figure=fig,
                      left=0.01, right=0.99,
                      top=1 - (i / 3) - 0.01,
                      bottom=1 - ((i + 1) / 3) + 0.01,
                      wspace=0.02, width_ratios=[1, 1, 2])

        ax_k = fig.add_subplot(gs[0, 0])
        ax_k.imshow(frame_k)
        ax_k.axis("off")
        lab = pair["trajectory_id"].split("+")[0]
        ax_k.text(0.02, 0.98, lab, transform=ax_k.transAxes,
                  fontsize=10, fontweight="bold", va="top",
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax_kd = fig.add_subplot(gs[0, 1])
        ax_kd.imshow(frame_kd)
        ax_kd.axis("off")
        ax_kd.text(0.98, 0.98, f"d={pair['d']}", transform=ax_kd.transAxes,
                   fontsize=10, fontweight="bold", va="top", ha="right",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax_text = fig.add_subplot(gs[0, 2])
        ax_text.axis("off")

        text_lines = []
        f1_desc = result.get('frame1_description', 'N/A')[:80]
        f2_desc = result.get('frame2_description', 'N/A')[:80]
        text_lines.append(f"Frame 1: {f1_desc}...")
        text_lines.append(f"Frame 2: {f2_desc}...")
        text_lines.append("")
        text_lines.append(f"Changed: {result.get('what_changed', 'N/A')}")
        text_lines.append("")
        for j, inst in enumerate(result.get("instructions", [])[:5], 1):
            wrapped = textwrap.fill(f"{j}. {inst}", width=60)
            text_lines.append(wrapped)

        ax_text.text(0.02, 0.95, "\n".join(text_lines), transform=ax_text.transAxes,
                    fontsize=8, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved PDF: {output_path}")
    print(f"Saved preview: {png_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_file", default="sampled_pairs_multi_gpt5/diverse_test_pairs.json")
    parser.add_argument("--base_dir", default="/data/s185927/droid_raw/sampled_pairs")
    parser.add_argument("--output_dir", default="sampled_pairs_multi_gpt5")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs")
    args = parser.parse_args()

    # Load pairs
    with open(args.pairs_file) as f:
        pairs = json.load(f)

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Processing {len(pairs)} pairs with gpt-5-mini (chain-of-thought prompt)...")

    base_dir = Path(args.base_dir)
    client = OpenAI()

    results = []
    total_input = 0
    total_output = 0

    for pair in tqdm(pairs, desc="Generating instructions"):
        frame_k_path = str(base_dir / pair["frame_k_path"])
        frame_kd_path = str(base_dir / pair["frame_k_d_path"])

        try:
            result = generate_instructions(client, frame_k_path, frame_kd_path)
            total_input += result["input_tokens"]
            total_output += result["output_tokens"]

            results.append({
                "pair": pair,
                "result": result
            })
        except Exception as e:
            print(f"\nError on pair {pair['frame_k']} -> {pair['frame_k_d']}: {e}")
            results.append({
                "pair": pair,
                "result": {"error": str(e), "instructions": []}
            })

    # Save JSON results
    output_dir = Path(args.output_dir)
    json_path = output_dir / "gpt5mini_cot_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Print token stats
    print(f"\nToken usage:")
    print(f"  Total input:  {total_input:,}")
    print(f"  Total output: {total_output:,}")

    # Cost estimate (gpt-5-mini: $0.125/M input, $1.00/M output)
    cost = (total_input * 0.125 / 1_000_000) + (total_output * 1.00 / 1_000_000)
    print(f"  Estimated cost: ${cost:.4f}")

    # Create PDF
    pdf_path = str(output_dir / "gpt5mini_cot_instructions.pdf")
    create_pdf(results, pdf_path, base_dir)


if __name__ == "__main__":
    main()
