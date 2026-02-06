"""
Compare instruction generation quality across GPT-5 model variants.

Tests gpt-5, gpt-5-mini, and gpt-5-nano on the same frame pairs
and generates a visual comparison PDF.
"""

import json
import base64
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import textwrap

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from openai import OpenAI


PROMPT = '''You are shown two frames from a robot manipulation video.

FRAME 1 (BEFORE): The first image
FRAME 2 (AFTER): The second image

Please provide:

1. WHAT CHANGED: Brief description of what changed.

2. INSTRUCTIONS: Write 5 DIFFERENT ways a human might naturally command this action.

IMPORTANT - Use NATURAL, CASUAL language:
   - GOOD: "Grab the cup", "Put it in the bowl", "Pick that up and move it over"
   - BAD: "Position the end-effector to grasp the cylindrical object"
   - Instruction 1 MUST be SHORT (3-6 words only)
   - Others can be longer if needed, but stay natural and casual
   - If multiple things happened, describe naturally: "grab it and put it in the bowl"

Format:
WHAT CHANGED: [brief]
INSTRUCTION 1: [3-6 words MAX]
INSTRUCTION 2: [natural, casual]
INSTRUCTION 3: [describes the motion]
INSTRUCTION 4: [different verb]
INSTRUCTION 5: [slightly more specific]'''


def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_model(client, model: str, frame_k_path: str, frame_kd_path: str) -> dict:
    """Call a model and return parsed results."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
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
        max_completion_tokens=8000
    )

    usage = response.usage
    text = response.choices[0].message.content

    # Parse response
    result = {
        "model": model,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "what_changed": "",
        "instructions": [],
        "raw_response": text
    }

    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.upper().startswith("WHAT CHANGED:"):
            result["what_changed"] = line.split(":", 1)[1].strip()
        elif "INSTRUCTION" in line.upper() and ":" in line:
            inst = line.split(":", 1)[1].strip()
            if inst:
                result["instructions"].append(inst)

    return result


def create_comparison_figure(pairs_results: list, output_path: str, base_dir: Path):
    """Create a figure comparing model outputs for multiple pairs."""
    n_pairs = len(pairs_results)
    models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

    # Figure with one row per pair
    fig = plt.figure(figsize=(20, 8 * n_pairs))

    for pair_idx, pair_data in enumerate(pairs_results):
        pair = pair_data["pair"]
        results = pair_data["results"]

        # Load images
        frame_k = Image.open(base_dir / pair["frame_k_path"])
        frame_kd = Image.open(base_dir / pair["frame_k_d_path"])

        # Create subplot grid for this pair
        # Layout: [frame_k | frame_kd | gpt-5 | gpt-5-mini | gpt-5-nano]
        gs = GridSpec(1, 5, figure=fig,
                      left=0.02, right=0.98,
                      top=1 - (pair_idx / n_pairs) - 0.02,
                      bottom=1 - ((pair_idx + 1) / n_pairs) + 0.02,
                      wspace=0.05)

        # Frame K
        ax_k = fig.add_subplot(gs[0, 0])
        ax_k.imshow(frame_k)
        ax_k.set_title(f"Frame {pair['frame_k']}", fontsize=10)
        ax_k.axis("off")

        # Add lab label
        lab = pair["trajectory_id"].split("+")[0]
        ax_k.text(0.02, 0.98, lab, transform=ax_k.transAxes,
                  fontsize=9, fontweight="bold", va="top",
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Frame K+D
        ax_kd = fig.add_subplot(gs[0, 1])
        ax_kd.imshow(frame_kd)
        ax_kd.set_title(f"Frame {pair['frame_k_d']} (d={pair['d']})", fontsize=10)
        ax_kd.axis("off")

        # Model outputs
        for model_idx, model in enumerate(models):
            ax = fig.add_subplot(gs[0, 2 + model_idx])
            ax.axis("off")

            result = results.get(model, {})

            # Format text
            text_lines = [f"**{model}**", ""]

            what_changed = result.get("what_changed", "N/A")
            wrapped = textwrap.fill(f"Changed: {what_changed}", width=40)
            text_lines.append(wrapped)
            text_lines.append("")

            instructions = result.get("instructions", [])
            for i, inst in enumerate(instructions[:5], 1):
                wrapped = textwrap.fill(f"{i}. {inst}", width=40)
                text_lines.append(wrapped)

            # Token info
            in_tok = result.get("input_tokens", 0)
            out_tok = result.get("output_tokens", 0)
            text_lines.append("")
            text_lines.append(f"Tokens: {in_tok} in, {out_tok} out")

            text = "\n".join(text_lines)
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                    fontsize=7, va="top", ha="left", family="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
            ax.set_title(model, fontsize=10, fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved comparison to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_file", type=str,
                        default="/home/s185927/thesis/experiments/04_language_grounding/sampled_pairs_multi_gpt5/diverse_test_pairs.json")
    parser.add_argument("--base_dir", type=str,
                        default="/data/s185927/droid_raw/sampled_pairs")
    parser.add_argument("--output_dir", type=str,
                        default="/home/s185927/thesis/experiments/04_language_grounding/sampled_pairs_multi_gpt5")
    parser.add_argument("--num_pairs", type=int, default=3)
    args = parser.parse_args()

    # Load pairs
    with open(args.pairs_file) as f:
        pairs = json.load(f)

    # Select diverse pairs (different d values)
    selected = []
    d_values_seen = set()
    for p in pairs:
        d = p.get("d", 0)
        if d not in d_values_seen and len(selected) < args.num_pairs:
            selected.append(p)
            d_values_seen.add(d)

    print(f"Testing {len(selected)} pairs with d values: {sorted(d_values_seen)}")

    base_dir = Path(args.base_dir)
    client = OpenAI()
    models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

    all_results = []

    for pair_idx, pair in enumerate(selected):
        print(f"\n{'='*60}")
        print(f"Pair {pair_idx+1}/{len(selected)}: d={pair['d']}, frames {pair['frame_k']} -> {pair['frame_k_d']}")
        print(f"{'='*60}")

        frame_k_path = str(base_dir / pair["frame_k_path"])
        frame_kd_path = str(base_dir / pair["frame_k_d_path"])

        pair_results = {"pair": pair, "results": {}}

        for model in models:
            print(f"\n  Testing {model}...")
            try:
                result = call_model(client, model, frame_k_path, frame_kd_path)
                pair_results["results"][model] = result
                print(f"    Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
                print(f"    Instructions: {len(result['instructions'])}")
                for i, inst in enumerate(result["instructions"][:3], 1):
                    print(f"      {i}. {inst[:60]}...")
            except Exception as e:
                print(f"    Error: {e}")
                pair_results["results"][model] = {"error": str(e)}

        all_results.append(pair_results)

    # Save JSON results
    output_dir = Path(args.output_dir)
    json_path = output_dir / "model_comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")

    # Create comparison figure
    pdf_path = str(output_dir / "model_comparison.pdf")
    create_comparison_figure(all_results, pdf_path, base_dir)


if __name__ == "__main__":
    main()
