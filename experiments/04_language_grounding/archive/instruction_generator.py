"""
VLM-based Instruction Generation for Frame Pairs

Given frame pairs sampled by frame_pair_sampler.py, generate natural language
instructions describing the action/transition between frames.

Approach (based on Gemini 2.5 capabilities):
1. Describe frame_k in detail (robot position, objects, gripper state)
2. Describe frame_k+d in detail
3. Generate instruction that would guide the robot from state k to state k+d
"""

import base64
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import time

from frame_pair_sampler import FramePair


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================================
# Prompts
# ============================================================================

SCENE_DESCRIPTION_PROMPT = """Describe this robot manipulation scene in detail.

Focus on:
1. What objects are visible on the workspace? (names, colors, positions)
2. Where is the robot arm/gripper? Describe its position relative to objects.
3. Is the gripper open, closed, or holding something?
4. What is the overall spatial arrangement?

Be specific and factual. Describe only what you can see.
Write 2-4 sentences."""


INSTRUCTION_GENERATION_PROMPT = """A robot arm transitioned from Scene A to Scene B.

Scene A (before):
{scene_a}

Scene B (after):
{scene_b}

What single natural language instruction could have guided this robot action?

Requirements:
- Write ONE clear instruction (one sentence)
- Use action verbs: move, reach, grasp, pick up, lift, lower, place, pour, release, rotate, push, pull
- Be specific about objects and directions
- Focus on the main action, not small details

Examples of good instructions:
- "Move the gripper above the white mug"
- "Grasp the yellow cup"
- "Pour the contents into the bowl"
- "Lower the arm toward the table"
- "Release the object and retract"

Instruction:"""


# ============================================================================
# VLM Clients
# ============================================================================

class BaseInstructionGenerator:
    """Base class for instruction generators."""

    def describe_scene(self, image_path: str) -> str:
        raise NotImplementedError

    def generate_instruction(self, scene_a: str, scene_b: str) -> str:
        raise NotImplementedError

    def process_pair(self, pair: FramePair, verbose: bool = False) -> FramePair:
        """Process a single frame pair to generate instruction."""

        if not pair.frame_k_path or not pair.frame_k_d_path:
            raise ValueError(f"Frame paths not set for pair {pair.frame_k} -> {pair.frame_k_d}")

        # Step 1: Describe scene at frame k
        if verbose:
            print(f"    Describing frame {pair.frame_k}...")
        scene_k = self.describe_scene(pair.frame_k_path)
        pair.scene_k_description = scene_k

        # Step 2: Describe scene at frame k+d
        if verbose:
            print(f"    Describing frame {pair.frame_k_d}...")
        scene_k_d = self.describe_scene(pair.frame_k_d_path)
        pair.scene_k_d_description = scene_k_d

        # Step 3: Generate instruction from descriptions
        if verbose:
            print(f"    Generating instruction...")
        instruction = self.generate_instruction(scene_k, scene_k_d)
        pair.instruction = instruction

        return pair


class OpenAIInstructionGenerator(BaseInstructionGenerator):
    """OpenAI GPT-4o based generator."""

    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def describe_scene(self, image_path: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": SCENE_DESCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_base64(image_path)}"
                        }
                    }
                ]
            }],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    def generate_instruction(self, scene_a: str, scene_b: str) -> str:
        prompt = INSTRUCTION_GENERATION_PROMPT.format(scene_a=scene_a, scene_b=scene_b)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()


class AnthropicInstructionGenerator(BaseInstructionGenerator):
    """Anthropic Claude based generator."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def describe_scene(self, image_path: str) -> str:
        ext = image_path.lower().split('.')[-1]
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encode_image_base64(image_path)
                        }
                    },
                    {"type": "text", "text": SCENE_DESCRIPTION_PROMPT}
                ]
            }]
        )
        return response.content[0].text.strip()

    def generate_instruction(self, scene_a: str, scene_b: str) -> str:
        prompt = INSTRUCTION_GENERATION_PROMPT.format(scene_a=scene_a, scene_b=scene_b)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


class GeminiInstructionGenerator(BaseInstructionGenerator):
    """Google Gemini based generator."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        self.model = genai.GenerativeModel(model)

    def describe_scene(self, image_path: str) -> str:
        import PIL.Image
        image = PIL.Image.open(image_path)
        response = self.model.generate_content([SCENE_DESCRIPTION_PROMPT, image])
        return response.text.strip()

    def generate_instruction(self, scene_a: str, scene_b: str) -> str:
        prompt = INSTRUCTION_GENERATION_PROMPT.format(scene_a=scene_a, scene_b=scene_b)
        response = self.model.generate_content(prompt)
        return response.text.strip()


def get_generator(provider: str, model: str = None) -> BaseInstructionGenerator:
    """Factory function to create instruction generator."""

    if provider == "openai":
        return OpenAIInstructionGenerator(model=model or "gpt-4o")
    elif provider == "anthropic":
        return AnthropicInstructionGenerator(model=model or "claude-sonnet-4-20250514")
    elif provider == "gemini":
        return GeminiInstructionGenerator(model=model or "gemini-2.0-flash")
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
    rate_limit_delay: float = 0.5,  # seconds between API calls
) -> list[FramePair]:
    """
    Process multiple frame pairs to generate instructions.

    Args:
        pairs: List of FramePair objects (with frame paths set)
        provider: VLM provider (openai, anthropic, gemini)
        model: Specific model to use
        output_path: Path to save results (incremental saving)
        verbose: Print progress
        rate_limit_delay: Delay between API calls to avoid rate limits
    """
    generator = get_generator(provider, model)

    processed = []
    failed = []

    for i, pair in enumerate(pairs):
        if verbose:
            print(f"\nProcessing pair {i+1}/{len(pairs)}: frame {pair.frame_k} -> {pair.frame_k_d} (d={pair.d})")

        try:
            generator.process_pair(pair, verbose=verbose)
            processed.append(pair)

            if verbose:
                print(f"    ✓ Instruction: {pair.instruction}")

            # Save incrementally
            if output_path:
                with open(output_path, "w") as f:
                    json.dump([asdict(p) for p in processed], f, indent=2)

        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            failed.append((pair, str(e)))

        # Rate limiting
        if rate_limit_delay > 0 and i < len(pairs) - 1:
            time.sleep(rate_limit_delay)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processed: {len(processed)}/{len(pairs)}")
        print(f"Failed: {len(failed)}")

    return processed


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate instructions for frame pairs")
    parser.add_argument("--pairs_file", type=str, required=True,
                        help="Path to frame_pairs.json from sampler")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results (default: same dir as pairs_file)")
    parser.add_argument("--provider", type=str, required=True,
                        choices=["openai", "anthropic", "gemini"])
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to use")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of pairs to process")
    parser.add_argument("--rate_limit", type=float, default=0.5,
                        help="Delay between API calls (seconds)")
    args = parser.parse_args()

    # Load frame pairs
    with open(args.pairs_file) as f:
        pairs_data = json.load(f)

    pairs = [FramePair(**p) for p in pairs_data]
    print(f"Loaded {len(pairs)} frame pairs from {args.pairs_file}")

    if args.limit:
        pairs = pairs[:args.limit]
        print(f"Limited to {len(pairs)} pairs")

    # Output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.pairs_file).parent / "frame_pairs_with_instructions.json"

    print(f"Provider: {args.provider}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Process
    processed = process_frame_pairs(
        pairs,
        provider=args.provider,
        model=args.model,
        output_path=str(output_path),
        verbose=True,
        rate_limit_delay=args.rate_limit
    )

    # Final save
    with open(output_path, "w") as f:
        json.dump([asdict(p) for p in processed], f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print sample
    print("\n" + "=" * 60)
    print("SAMPLE INSTRUCTIONS")
    print("=" * 60)
    for p in processed[:5]:
        print(f"\nd={p.d:3d} | frame {p.frame_k:3d} -> {p.frame_k_d:3d}")
        print(f"  {p.instruction}")


if __name__ == "__main__":
    main()
