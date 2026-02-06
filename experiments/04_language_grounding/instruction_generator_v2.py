"""
VLM-based Instruction Generation for Frame Pairs - Version 2

Key change: Show BOTH frames to the VLM in a single call, ask it to:
1. Describe what changed between the frames
2. Generate an instruction that could have guided the action

This avoids the inconsistency problem of describing frames separately.
"""

import base64
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import time

from frame_pair_sampler import FramePair


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================================
# Prompts - Version 2
# ============================================================================

TWO_FRAME_PROMPT = """You are shown two frames from a robot manipulation video.

FRAME 1 (BEFORE): The first image
FRAME 2 (AFTER): The second image

These frames are from the same trajectory. The robot arm may have moved, the gripper may have opened/closed, or objects may have been manipulated.

Please provide:

1. WHAT CHANGED: Describe specifically what is different between Frame 1 and Frame 2. Focus on:
   - Robot arm/gripper position changes
   - Gripper state (open/closed/holding)
   - Object positions or states

2. INSTRUCTION: Write ONE natural language instruction that could have guided this robot action.
   - Use action verbs: move, reach, lower, lift, grasp, release, pour, push, rotate
   - Be specific about the target (e.g., "toward the cup", "above the bowl")
   - Keep it concise (one sentence)

Format your response exactly as:
WHAT CHANGED: [your description]
INSTRUCTION: [your instruction]"""


# ============================================================================
# VLM Clients - Two-Frame Version
# ============================================================================

class OpenAITwoFrameGenerator:
    """OpenAI GPT-4o based generator using two frames."""

    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def generate_instruction(self, frame_k_path: str, frame_k_d_path: str) -> dict:
        """Generate instruction from two frames shown together."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": TWO_FRAME_PROMPT},
                    {"type": "text", "text": "\n\nFRAME 1 (BEFORE):"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_base64(frame_k_path)}"
                        }
                    },
                    {"type": "text", "text": "\n\nFRAME 2 (AFTER):"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_base64(frame_k_d_path)}"
                        }
                    }
                ]
            }],
            max_tokens=300
        )

        return self._parse_response(response.choices[0].message.content)

    def _parse_response(self, text: str) -> dict:
        """Parse the structured response."""
        result = {
            "what_changed": "",
            "instruction": "",
            "raw_response": text
        }

        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith("WHAT CHANGED:"):
                result["what_changed"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("INSTRUCTION:"):
                result["instruction"] = line.split(":", 1)[1].strip()

        # If parsing failed, try to extract instruction from raw text
        if not result["instruction"] and text:
            # Take the last sentence as instruction
            sentences = text.replace('\n', ' ').split('.')
            for s in reversed(sentences):
                s = s.strip()
                if len(s) > 10 and any(verb in s.lower() for verb in ['move', 'reach', 'grasp', 'lift', 'lower', 'pour', 'release', 'push', 'pull']):
                    result["instruction"] = s
                    break

        return result


class AnthropicTwoFrameGenerator:
    """Anthropic Claude based generator using two frames."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def generate_instruction(self, frame_k_path: str, frame_k_d_path: str) -> dict:
        """Generate instruction from two frames shown together."""

        def get_media_type(path):
            ext = path.lower().split('.')[-1]
            return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": TWO_FRAME_PROMPT + "\n\nFRAME 1 (BEFORE):"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": get_media_type(frame_k_path),
                            "data": encode_image_base64(frame_k_path)
                        }
                    },
                    {"type": "text", "text": "\n\nFRAME 2 (AFTER):"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": get_media_type(frame_k_d_path),
                            "data": encode_image_base64(frame_k_d_path)
                        }
                    }
                ]
            }]
        )

        return self._parse_response(response.content[0].text)

    def _parse_response(self, text: str) -> dict:
        """Parse the structured response."""
        result = {
            "what_changed": "",
            "instruction": "",
            "raw_response": text
        }

        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith("WHAT CHANGED:"):
                result["what_changed"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("INSTRUCTION:"):
                result["instruction"] = line.split(":", 1)[1].strip()

        return result


class GeminiTwoFrameGenerator:
    """Google Gemini based generator using two frames."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        self.model = genai.GenerativeModel(model)

    def generate_instruction(self, frame_k_path: str, frame_k_d_path: str) -> dict:
        """Generate instruction from two frames shown together."""
        import PIL.Image

        image1 = PIL.Image.open(frame_k_path)
        image2 = PIL.Image.open(frame_k_d_path)

        response = self.model.generate_content([
            TWO_FRAME_PROMPT,
            "\n\nFRAME 1 (BEFORE):",
            image1,
            "\n\nFRAME 2 (AFTER):",
            image2
        ])

        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> dict:
        """Parse the structured response."""
        result = {
            "what_changed": "",
            "instruction": "",
            "raw_response": text
        }

        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith("WHAT CHANGED:"):
                result["what_changed"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("INSTRUCTION:"):
                result["instruction"] = line.split(":", 1)[1].strip()

        return result


def get_generator(provider: str, model: str = None):
    """Factory function to create instruction generator."""

    if provider == "openai":
        return OpenAITwoFrameGenerator(model=model or "gpt-4o")
    elif provider == "anthropic":
        return AnthropicTwoFrameGenerator(model=model or "claude-sonnet-4-20250514")
    elif provider == "gemini":
        return GeminiTwoFrameGenerator(model=model or "gemini-2.0-flash")
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# Batch Processing
# ============================================================================

def process_frame_pairs(
    pairs: list[FramePair],
    provider: str = "openai",
    model: str = None,
    output_path: str = None,
    verbose: bool = True,
    rate_limit_delay: float = 1.0,
) -> list[dict]:
    """
    Process multiple frame pairs to generate instructions.
    """
    generator = get_generator(provider, model)

    results = []
    failed = []

    for i, pair in enumerate(pairs):
        if verbose:
            print(f"\nProcessing pair {i+1}/{len(pairs)}: frame {pair.frame_k} -> {pair.frame_k_d} (d={pair.d})")

        if not pair.frame_k_path or not pair.frame_k_d_path:
            print(f"    ✗ Missing frame paths")
            failed.append((pair, "Missing frame paths"))
            continue

        try:
            result = generator.generate_instruction(pair.frame_k_path, pair.frame_k_d_path)

            # Combine pair metadata with VLM result
            output = {
                "trajectory_id": pair.trajectory_id,
                "frame_k": pair.frame_k,
                "frame_k_d": pair.frame_k_d,
                "d": pair.d,
                "frame_k_path": pair.frame_k_path,
                "frame_k_d_path": pair.frame_k_d_path,
                "position_delta": pair.position_delta,
                "gripper_delta": pair.gripper_delta,
                "z_delta": pair.z_delta,
                "what_changed": result["what_changed"],
                "instruction": result["instruction"],
                "raw_response": result["raw_response"]
            }
            results.append(output)

            if verbose:
                print(f"    Changed: {result['what_changed'][:80]}...")
                print(f"    ✓ Instruction: {result['instruction']}")

            # Save incrementally
            if output_path:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            failed.append((pair, str(e)))

        # Rate limiting
        if rate_limit_delay > 0 and i < len(pairs) - 1:
            time.sleep(rate_limit_delay)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processed: {len(results)}/{len(pairs)}")
        print(f"Failed: {len(failed)}")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate instructions for frame pairs (v2 - two-frame)")
    parser.add_argument("--pairs_file", type=str, required=True,
                        help="Path to frame_pairs.json from sampler")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results")
    parser.add_argument("--provider", type=str, required=True,
                        choices=["openai", "anthropic", "gemini"])
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to use")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of pairs to process")
    parser.add_argument("--rate_limit", type=float, default=1.0,
                        help="Delay between API calls (seconds)")
    args = parser.parse_args()

    # Load frame pairs
    with open(args.pairs_file) as f:
        pairs_data = json.load(f)

    pairs = [FramePair(**{k: v for k, v in p.items()
                          if k in FramePair.__dataclass_fields__}) for p in pairs_data]
    print(f"Loaded {len(pairs)} frame pairs from {args.pairs_file}")

    if args.limit:
        pairs = pairs[:args.limit]
        print(f"Limited to {len(pairs)} pairs")

    # Output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.pairs_file).parent / "frame_pairs_v2.json"

    print(f"Provider: {args.provider}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Process
    results = process_frame_pairs(
        pairs,
        provider=args.provider,
        model=args.model,
        output_path=str(output_path),
        verbose=True,
        rate_limit_delay=args.rate_limit
    )

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print sample
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS")
    print("=" * 60)
    for r in results[:5]:
        print(f"\nd={r['d']:3d} | frame {r['frame_k']:3d} -> {r['frame_k_d']:3d}")
        print(f"  Changed: {r['what_changed'][:100]}...")
        print(f"  Instruction: {r['instruction']}")


if __name__ == "__main__":
    main()
