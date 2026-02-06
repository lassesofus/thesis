"""
VLM-based instruction generation using scene descriptions.

Workflow:
1. VLM describes Scene A (from image)
2. VLM describes Scene B (from image)
3. VLM generates instruction from (description A, description B) - text only
"""

import base64
import json
import os
from pathlib import Path


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================================
# Prompts
# ============================================================================

SCENE_DESCRIPTION_PROMPT = """Describe this robot manipulation scene in detail.

Focus on:
1. What objects are on the table? (names, colors, positions, contents if containers)
2. Where is the robot gripper? (position relative to objects)
3. Is the gripper open, closed, or holding something?
4. Any other relevant spatial relationships

Be specific and factual. Describe only what you can see.
Write 2-4 sentences."""


INSTRUCTION_GENERATION_PROMPT = """A robot arm performed an action to transition from Scene A to Scene B.

Scene A (before):
{scene_a}

Scene B (after):
{scene_b}

What natural language instruction could have guided this robot action?

Requirements:
- Write a single, clear instruction (one sentence)
- Use action verbs like: reach, grasp, pick up, lift, move, place, pour, release, rotate
- Be specific about objects involved
- Do NOT describe both scenes, just give the instruction

Example format: "Pick up the red cup from the table"

Instruction:"""


# ============================================================================
# VLM Clients
# ============================================================================

class OpenAIInstructionGenerator:
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def describe_scene(self, image_path: str) -> str:
        """Get scene description from image."""
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

    def generate_instruction(self, scene_a_desc: str, scene_b_desc: str) -> str:
        """Generate instruction from two scene descriptions (text only)."""
        prompt = INSTRUCTION_GENERATION_PROMPT.format(
            scene_a=scene_a_desc,
            scene_b=scene_b_desc
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()


class AnthropicInstructionGenerator:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def describe_scene(self, image_path: str) -> str:
        """Get scene description from image."""
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

    def generate_instruction(self, scene_a_desc: str, scene_b_desc: str) -> str:
        """Generate instruction from two scene descriptions (text only)."""
        prompt = INSTRUCTION_GENERATION_PROMPT.format(
            scene_a=scene_a_desc,
            scene_b=scene_b_desc
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


# ============================================================================
# Pipeline
# ============================================================================

def process_keyframe_pair(
    image_a_path: str,
    image_b_path: str,
    generator,
    verbose: bool = True
) -> dict:
    """
    Process a pair of keyframes to generate an instruction.

    Returns dict with scene descriptions and generated instruction.
    """
    # Step 1: Describe scene A
    if verbose:
        print(f"  Describing scene A: {Path(image_a_path).name}")
    scene_a_desc = generator.describe_scene(image_a_path)

    # Step 2: Describe scene B
    if verbose:
        print(f"  Describing scene B: {Path(image_b_path).name}")
    scene_b_desc = generator.describe_scene(image_b_path)

    # Step 3: Generate instruction from descriptions (text only, no images)
    if verbose:
        print(f"  Generating instruction...")
    instruction = generator.generate_instruction(scene_a_desc, scene_b_desc)

    return {
        "image_a": image_a_path,
        "image_b": image_b_path,
        "scene_a_description": scene_a_desc,
        "scene_b_description": scene_b_desc,
        "instruction": instruction
    }


def process_all_keyframes(
    keyframe_paths: list[str],
    provider: str = "openai",
    model: str = None,
    verbose: bool = True
) -> list[dict]:
    """
    Process all consecutive keyframe pairs.
    """
    # Initialize generator
    if provider == "openai":
        generator = OpenAIInstructionGenerator(model=model or "gpt-4o")
    elif provider == "anthropic":
        generator = AnthropicInstructionGenerator(model=model or "claude-sonnet-4-20250514")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    results = []
    for i in range(len(keyframe_paths) - 1):
        if verbose:
            print(f"\nSegment {i + 1}:")

        result = process_keyframe_pair(
            keyframe_paths[i],
            keyframe_paths[i + 1],
            generator,
            verbose=verbose
        )
        result["segment"] = i + 1
        results.append(result)

        if verbose:
            print(f"  → {result['instruction']}")

    return results


# ============================================================================
# Demo / CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe_dir", type=str,
                        default="/home/s185927/thesis/experiments/04_language_grounding/output/keyframes")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Find keyframe images
    keyframe_dir = Path(args.keyframe_dir)
    keyframe_paths = sorted([
        str(p) for p in keyframe_dir.glob("keyframe_*.jpg")
    ])

    print(f"Found {len(keyframe_paths)} keyframes in {keyframe_dir}")
    print(f"Using provider: {args.provider}")
    print("=" * 60)

    # Process
    results = process_all_keyframes(
        keyframe_paths,
        provider=args.provider,
        model=args.model
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for r in results:
        print(f"\nSegment {r['segment']}:")
        print(f"  Scene A: {r['scene_a_description'][:100]}...")
        print(f"  Scene B: {r['scene_b_description'][:100]}...")
        print(f"  Instruction: {r['instruction']}")

    # Full narrative
    instructions = [r["instruction"] for r in results]
    print("\n" + "-" * 60)
    print("Full task narration:")
    print(" → ".join(instructions))

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = keyframe_dir.parent / "vlm_instructions.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
