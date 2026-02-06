"""
VLM-based Instruction Generation for Frame Pairs - Version 3

Key change: Generate MULTIPLE diverse instructions per frame pair for contrastive training.
This enables robustness to language ambiguity - the same action can be described many ways.

Supports both synchronous API calls and OpenAI Batch API for cost-effective large-scale processing.
Batch API provides 50% cost reduction.

Output format includes a list of instructions that all describe the same transition.
"""

import base64
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import time
import re
import tempfile

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from frame_pair_sampler import FramePair


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================================
# Prompts - Version 3 (Multiple Diverse Instructions)
# ============================================================================

DIVERSE_INSTRUCTION_PROMPT = """You are shown two frames from a robot manipulation video.

FRAME 1 (BEFORE): The first image
FRAME 2 (AFTER): The second image

These frames are from the same trajectory. The robot arm may have moved, the gripper may have opened/closed, or objects may have been manipulated.

Please provide:

1. WHAT CHANGED: Describe specifically what is different between Frame 1 and Frame 2.

2. INSTRUCTIONS: Write 5 DIFFERENT natural language instructions that could ALL describe this robot action. Vary them by:
   - Level of detail (brief vs specific)
   - Reference frame (relative to objects vs spatial directions)
   - Action verb choice (move/reach/lower/lift/grasp/approach/shift/position/pick/place)
   - What aspect is emphasized (motion, target, or both)

CRITICAL REQUIREMENTS:
   - Each instruction must FULLY describe the transition from Frame 1 to Frame 2
   - If multiple things happened (e.g., grasp + move), you MUST include both
   - Compound actions are allowed and encouraged when accurate (e.g., "Pick up the cup and move it over the bowl")
   - DO NOT omit important actions just to keep instructions short
   - Use natural language a human would use to command this action
   - Action verbs: move, reach, lower, lift, grasp, release, approach, shift, position, rotate, push, pull, pick up, place, transfer

Format your response exactly as:
WHAT CHANGED: [your description]
INSTRUCTION 1: [brief but complete instruction]
INSTRUCTION 2: [instruction with specific object details]
INSTRUCTION 3: [instruction emphasizing the motion/direction]
INSTRUCTION 4: [instruction using different verbs]
INSTRUCTION 5: [instruction with spatial references]"""


# Chain-of-thought prompt - more accurate for complex actions
COT_INSTRUCTION_PROMPT = '''You are shown two frames from a robot manipulation video. Analyze them step by step.

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


# ============================================================================
# VLM Clients - Multi-Instruction Version
# ============================================================================

class OpenAIDiverseGenerator:
    """OpenAI GPT-4o based generator for multiple diverse instructions."""

    def __init__(self, model: str = "gpt-4o", detail: str = "high", use_cot_prompt: bool = True):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.detail = detail
        self.use_cot_prompt = use_cot_prompt
        self.prompt = COT_INSTRUCTION_PROMPT if use_cot_prompt else DIVERSE_INSTRUCTION_PROMPT

    def generate_instructions(self, frame_k_path: str, frame_k_d_path: str) -> dict:
        """Generate multiple diverse instructions from two frames."""
        # GPT-5+ models use max_completion_tokens instead of max_tokens
        # GPT-5 uses reasoning tokens internally, so we need much higher limits
        is_gpt5 = "gpt-5" in self.model or "gpt-4.1" in self.model
        token_param = "max_completion_tokens" if is_gpt5 else "max_tokens"
        token_limit = 3000 if is_gpt5 else 500

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "text", "text": "\n\nFRAME 1 (BEFORE):"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_base64(frame_k_path)}",
                            "detail": self.detail
                        }
                    },
                    {"type": "text", "text": "\n\nFRAME 2 (AFTER):"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_base64(frame_k_d_path)}",
                            "detail": self.detail
                        }
                    }
                ]
            }],
            **{token_param: token_limit}
        )

        return self._parse_response(response.choices[0].message.content)

    def _parse_response(self, text: str) -> dict:
        """Parse the structured response with multiple instructions."""
        result = {
            "what_changed": "",
            "instructions": [],
            "raw_response": text
        }

        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith("WHAT CHANGED:") or line.upper().startswith("CHANGED:"):
                result["what_changed"] = line.split(":", 1)[1].strip()
            elif re.match(r"INSTRUCTION\s*\d+:", line, re.IGNORECASE):
                instruction = line.split(":", 1)[1].strip()
                if instruction:
                    result["instructions"].append(instruction)

        return result


# ============================================================================
# OpenAI Batch API Support
# ============================================================================

class OpenAIBatchGenerator:
    """OpenAI Batch API generator for cost-effective large-scale processing.

    Batch API provides 50% cost reduction compared to standard API.
    Process: Create JSONL -> Upload -> Submit Batch -> Poll -> Download Results
    """

    def __init__(self, model: str = "gpt-5-mini", detail: str = "high", use_cot_prompt: bool = True):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.detail = detail
        self.use_cot_prompt = use_cot_prompt
        self.prompt = COT_INSTRUCTION_PROMPT if use_cot_prompt else DIVERSE_INSTRUCTION_PROMPT

    def create_batch_requests(self, pairs: list, base_dir: str, output_jsonl: str) -> str:
        """Create JSONL file with batch requests for all pairs.

        Args:
            pairs: List of FramePair objects or dicts
            base_dir: Base directory for resolving frame paths
            output_jsonl: Path to write the JSONL file

        Returns:
            Path to the created JSONL file
        """
        base_path = Path(base_dir)

        with open(output_jsonl, 'w') as f:
            for i, pair in enumerate(pairs):
                # Handle both FramePair objects and dicts
                if hasattr(pair, 'frame_k_path'):
                    frame_k_path = str(base_path / pair.frame_k_path)
                    frame_kd_path = str(base_path / pair.frame_k_d_path)
                    custom_id = f"pair_{i}_{pair.trajectory_id}_{pair.frame_k}_{pair.frame_k_d}"
                else:
                    frame_k_path = str(base_path / pair['frame_k_path'])
                    frame_kd_path = str(base_path / pair['frame_k_d_path'])
                    custom_id = f"pair_{i}_{pair['trajectory_id']}_{pair['frame_k']}_{pair['frame_k_d']}"

                # Build the request
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "max_completion_tokens": 3000,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "text", "text": "\n\nFRAME 1 (BEFORE):"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encode_image_base64(frame_k_path)}",
                                        "detail": self.detail
                                    }
                                },
                                {"type": "text", "text": "\n\nFRAME 2 (AFTER):"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encode_image_base64(frame_kd_path)}",
                                        "detail": self.detail
                                    }
                                }
                            ]
                        }]
                    }
                }
                f.write(json.dumps(request) + '\n')

        print(f"Created batch request file: {output_jsonl}")
        print(f"Total requests: {len(pairs)}")
        return output_jsonl

    def submit_batch(self, jsonl_path: str, description: str = "Frame pair instruction generation") -> str:
        """Upload JSONL and submit batch job.

        Returns:
            Batch ID for tracking
        """
        # Upload the file
        print(f"Uploading batch file: {jsonl_path}")
        with open(jsonl_path, 'rb') as f:
            batch_file = self.client.files.create(file=f, purpose="batch")
        print(f"Uploaded file ID: {batch_file.id}")

        # Create the batch
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description}
        )
        print(f"Submitted batch ID: {batch.id}")
        print(f"Status: {batch.status}")

        return batch.id

    def check_batch_status(self, batch_id: str) -> dict:
        """Check the status of a batch job."""
        batch = self.client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "failed_at": batch.failed_at,
            "request_counts": {
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed
            },
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id
        }

    def wait_for_batch(self, batch_id: str, poll_interval: int = 60, verbose: bool = True) -> dict:
        """Wait for batch to complete, polling at regular intervals."""
        while True:
            status = self.check_batch_status(batch_id)

            if verbose:
                completed = status['request_counts']['completed']
                total = status['request_counts']['total']
                print(f"[{time.strftime('%H:%M:%S')}] Status: {status['status']} - {completed}/{total} completed")

            if status['status'] in ['completed', 'failed', 'cancelled', 'expired']:
                return status

            time.sleep(poll_interval)

    def download_results(self, batch_id: str, output_path: str = None) -> list:
        """Download and parse batch results.

        Returns:
            List of parsed results with instructions
        """
        status = self.check_batch_status(batch_id)

        if status['status'] != 'completed':
            raise ValueError(f"Batch not completed. Status: {status['status']}")

        if not status['output_file_id']:
            raise ValueError("No output file available")

        # Download the output file
        content = self.client.files.content(status['output_file_id'])

        # Parse results
        results = []
        for line in content.text.strip().split('\n'):
            if not line:
                continue
            result = json.loads(line)

            # Parse the custom_id to get pair info
            custom_id = result['custom_id']
            parts = custom_id.split('_')
            pair_idx = int(parts[1])

            # Parse the response
            if result.get('response') and result['response'].get('body'):
                body = result['response']['body']
                if body.get('choices') and len(body['choices']) > 0:
                    content_text = body['choices'][0]['message']['content']
                    parsed = self._parse_response(content_text)
                    parsed['custom_id'] = custom_id
                    parsed['pair_index'] = pair_idx
                    if body.get('usage'):
                        parsed['input_tokens'] = body['usage'].get('prompt_tokens', 0)
                        parsed['output_tokens'] = body['usage'].get('completion_tokens', 0)
                    results.append(parsed)
                else:
                    results.append({
                        'custom_id': custom_id,
                        'pair_index': pair_idx,
                        'error': 'No choices in response',
                        'instructions': []
                    })
            else:
                results.append({
                    'custom_id': custom_id,
                    'pair_index': pair_idx,
                    'error': result.get('error', 'Unknown error'),
                    'instructions': []
                })

        # Sort by pair index
        results.sort(key=lambda x: x['pair_index'])

        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to: {output_path}")

        return results

    def _parse_response(self, text: str) -> dict:
        """Parse the structured response with multiple instructions."""
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
            elif line.upper().startswith("WHAT CHANGED:") or line.upper().startswith("CHANGED:"):
                result["what_changed"] = line.split(":", 1)[1].strip()
            elif re.match(r"INSTRUCTION\s*\d+:", line, re.IGNORECASE):
                instruction = line.split(":", 1)[1].strip()
                if instruction:
                    result["instructions"].append(instruction)

        return result


class AnthropicDiverseGenerator:
    """Anthropic Claude based generator for multiple diverse instructions."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def generate_instructions(self, frame_k_path: str, frame_k_d_path: str) -> dict:
        """Generate multiple diverse instructions from two frames."""

        def get_media_type(path):
            ext = path.lower().split('.')[-1]
            return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": DIVERSE_INSTRUCTION_PROMPT + "\n\nFRAME 1 (BEFORE):"},
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
        """Parse the structured response with multiple instructions."""
        result = {
            "what_changed": "",
            "instructions": [],
            "raw_response": text
        }

        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith("WHAT CHANGED:"):
                result["what_changed"] = line.split(":", 1)[1].strip()
            elif re.match(r"INSTRUCTION\s*\d+:", line, re.IGNORECASE):
                instruction = line.split(":", 1)[1].strip()
                if instruction:
                    result["instructions"].append(instruction)

        return result


class GeminiDiverseGenerator:
    """Google Gemini based generator for multiple diverse instructions."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        self.model = genai.GenerativeModel(model)

    def generate_instructions(self, frame_k_path: str, frame_k_d_path: str) -> dict:
        """Generate multiple diverse instructions from two frames."""
        import PIL.Image

        image1 = PIL.Image.open(frame_k_path)
        image2 = PIL.Image.open(frame_k_d_path)

        response = self.model.generate_content([
            DIVERSE_INSTRUCTION_PROMPT,
            "\n\nFRAME 1 (BEFORE):",
            image1,
            "\n\nFRAME 2 (AFTER):",
            image2
        ])

        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> dict:
        """Parse the structured response with multiple instructions."""
        result = {
            "what_changed": "",
            "instructions": [],
            "raw_response": text
        }

        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith("WHAT CHANGED:"):
                result["what_changed"] = line.split(":", 1)[1].strip()
            elif re.match(r"INSTRUCTION\s*\d+:", line, re.IGNORECASE):
                instruction = line.split(":", 1)[1].strip()
                if instruction:
                    result["instructions"].append(instruction)

        return result


def get_diverse_generator(provider: str, model: str = None, detail: str = "high"):
    """Factory function to create diverse instruction generator."""

    if provider == "openai":
        return OpenAIDiverseGenerator(model=model or "gpt-4o", detail=detail)
    elif provider == "anthropic":
        return AnthropicDiverseGenerator(model=model or "claude-sonnet-4-20250514")
    elif provider == "gemini":
        return GeminiDiverseGenerator(model=model or "gemini-2.0-flash")
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# Batch Processing
# ============================================================================

def process_frame_pairs_diverse(
    pairs: list[FramePair],
    provider: str = "openai",
    model: str = None,
    output_path: str = None,
    verbose: bool = True,
    rate_limit_delay: float = 1.0,
    base_dir: str = None,
) -> list[dict]:
    """
    Process multiple frame pairs to generate diverse instructions.

    Output format per pair:
    {
        "trajectory_id": ...,
        "frame_k": ...,
        "frame_k_d": ...,
        "instructions": ["inst1", "inst2", "inst3", "inst4", "inst5"],
        ...
    }
    """
    generator = get_diverse_generator(provider, model)

    results = []
    failed = []

    for i, pair in enumerate(pairs):
        if verbose:
            print(f"\nProcessing pair {i+1}/{len(pairs)}: frame {pair.frame_k} -> {pair.frame_k_d} (d={pair.d})")

        if not pair.frame_k_path or not pair.frame_k_d_path:
            print(f"    Missing frame paths")
            failed.append((pair, "Missing frame paths"))
            continue

        # Resolve frame paths with base directory
        frame_k_path = pair.frame_k_path
        frame_k_d_path = pair.frame_k_d_path
        if base_dir:
            frame_k_path = str(Path(base_dir) / pair.frame_k_path)
            frame_k_d_path = str(Path(base_dir) / pair.frame_k_d_path)

        try:
            # Retry logic for API failures
            max_retries = 3
            result = None
            for attempt in range(max_retries):
                result = generator.generate_instructions(frame_k_path, frame_k_d_path)
                if result.get("instructions"):
                    break
                if verbose and attempt < max_retries - 1:
                    print(f"    Retry {attempt + 1}/{max_retries} - no instructions parsed")
                    time.sleep(2)  # Wait before retry

            if not result.get("instructions"):
                if verbose:
                    print(f"    Failed after {max_retries} attempts. Raw response: {result.get('raw_response', '')[:100]}...")
                failed.append((pair, "No instructions parsed"))
                continue

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
                "instructions": result["instructions"],
                "raw_response": result["raw_response"]
            }
            results.append(output)

            if verbose:
                print(f"    Changed: {result['what_changed'][:60]}...")
                print(f"    Generated {len(result['instructions'])} instructions:")
                for j, inst in enumerate(result['instructions'][:3]):
                    print(f"      {j+1}. {inst[:50]}...")
                if len(result['instructions']) > 3:
                    print(f"      ... and {len(result['instructions'])-3} more")

            # Save incrementally
            if output_path:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            failed.append((pair, str(e)))

        # Rate limiting
        if rate_limit_delay > 0 and i < len(pairs) - 1:
            time.sleep(rate_limit_delay)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processed: {len(results)}/{len(pairs)}")
        print(f"Failed: {len(failed)}")

        # Stats on instruction diversity
        total_instructions = sum(len(r["instructions"]) for r in results)
        avg_instructions = total_instructions / len(results) if results else 0
        print(f"Total instructions: {total_instructions}")
        print(f"Avg per pair: {avg_instructions:.1f}")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate diverse instructions for frame pairs (v3)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Synchronous processing command
    sync_parser = subparsers.add_parser("sync", help="Process pairs synchronously (real-time)")
    sync_parser.add_argument("--pairs_file", type=str, required=True,
                             help="Path to frame_pairs.json from sampler")
    sync_parser.add_argument("--output", type=str, default=None,
                             help="Output path for results")
    sync_parser.add_argument("--provider", type=str, required=True,
                             choices=["openai", "anthropic", "gemini"])
    sync_parser.add_argument("--model", type=str, default=None,
                             help="Specific model to use")
    sync_parser.add_argument("--limit", type=int, default=None,
                             help="Limit number of pairs to process")
    sync_parser.add_argument("--rate_limit", type=float, default=1.0,
                             help="Delay between API calls (seconds)")
    sync_parser.add_argument("--base_dir", type=str, default=None,
                             help="Base directory for resolving relative frame paths")

    # Batch API commands
    batch_create_parser = subparsers.add_parser("batch-create", help="Create batch request JSONL file")
    batch_create_parser.add_argument("--pairs_file", type=str, required=True,
                                     help="Path to frame_pairs.json")
    batch_create_parser.add_argument("--base_dir", type=str, required=True,
                                     help="Base directory for frame images")
    batch_create_parser.add_argument("--output", type=str, required=True,
                                     help="Output JSONL path (will add _partN suffix if splitting)")
    batch_create_parser.add_argument("--model", type=str, default="gpt-5-mini",
                                     help="Model to use (default: gpt-5-mini)")
    batch_create_parser.add_argument("--limit", type=int, default=None,
                                     help="Limit number of pairs")
    batch_create_parser.add_argument("--batch_size", type=int, default=10000,
                                     help="Max pairs per batch file (default: 10000, ~40M tokens)")
    batch_create_parser.add_argument("--start_idx", type=int, default=0,
                                     help="Start index for pairs (for resuming)")

    batch_submit_parser = subparsers.add_parser("batch-submit", help="Submit batch job")
    batch_submit_parser.add_argument("--jsonl", type=str, required=True,
                                     help="Path to batch request JSONL file")
    batch_submit_parser.add_argument("--description", type=str,
                                     default="Frame pair instruction generation",
                                     help="Batch job description")

    batch_status_parser = subparsers.add_parser("batch-status", help="Check batch job status")
    batch_status_parser.add_argument("--batch_id", type=str, required=True,
                                     help="Batch ID to check")

    batch_wait_parser = subparsers.add_parser("batch-wait", help="Wait for batch to complete")
    batch_wait_parser.add_argument("--batch_id", type=str, required=True,
                                   help="Batch ID to wait for")
    batch_wait_parser.add_argument("--poll_interval", type=int, default=60,
                                   help="Polling interval in seconds (default: 60)")

    batch_download_parser = subparsers.add_parser("batch-download", help="Download batch results")
    batch_download_parser.add_argument("--batch_id", type=str, required=True,
                                       help="Batch ID to download")
    batch_download_parser.add_argument("--output", type=str, required=True,
                                       help="Output JSON path for results")
    batch_download_parser.add_argument("--pairs_file", type=str, default=None,
                                       help="Original pairs file to merge metadata")

    # Quick batch pipeline command
    batch_run_parser = subparsers.add_parser("batch-run", help="Run full batch pipeline (create, submit, wait, download)")
    batch_run_parser.add_argument("--pairs_file", type=str, required=True,
                                  help="Path to frame_pairs.json")
    batch_run_parser.add_argument("--base_dir", type=str, required=True,
                                  help="Base directory for frame images")
    batch_run_parser.add_argument("--output_dir", type=str, required=True,
                                  help="Output directory for results")
    batch_run_parser.add_argument("--model", type=str, default="gpt-5-mini",
                                  help="Model to use (default: gpt-5-mini)")
    batch_run_parser.add_argument("--limit", type=int, default=None,
                                  help="Limit number of pairs")
    batch_run_parser.add_argument("--poll_interval", type=int, default=60,
                                  help="Polling interval in seconds (default: 60)")

    args = parser.parse_args()

    if args.command == "sync":
        # Synchronous processing (original behavior)
        with open(args.pairs_file) as f:
            pairs_data = json.load(f)

        pairs = [FramePair(**{k: v for k, v in p.items()
                              if k in FramePair.__dataclass_fields__}) for p in pairs_data]
        print(f"Loaded {len(pairs)} frame pairs from {args.pairs_file}")

        if args.limit:
            pairs = pairs[:args.limit]
            print(f"Limited to {len(pairs)} pairs")

        output_path = args.output or str(Path(args.pairs_file).parent / "frame_pairs_v3_diverse.json")

        print(f"Provider: {args.provider}")
        print(f"Output: {output_path}")
        print("=" * 60)

        results = process_frame_pairs_diverse(
            pairs,
            provider=args.provider,
            model=args.model,
            output_path=str(output_path),
            verbose=True,
            rate_limit_delay=args.rate_limit,
            base_dir=args.base_dir
        )

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    elif args.command == "batch-create":
        # Create batch request file(s)
        with open(args.pairs_file) as f:
            pairs = json.load(f)

        # Apply start index and limit
        pairs = pairs[args.start_idx:]
        if args.limit:
            pairs = pairs[:args.limit]

        print(f"Loaded {len(pairs)} pairs (starting from index {args.start_idx})")

        generator = OpenAIBatchGenerator(model=args.model, use_cot_prompt=True)

        # Split into batches if needed
        batch_size = args.batch_size
        n_batches = (len(pairs) + batch_size - 1) // batch_size

        if n_batches == 1:
            # Single batch
            generator.create_batch_requests(pairs, args.base_dir, args.output)
            created_files = [args.output]
        else:
            # Multiple batches - add part numbers
            output_base = args.output.replace('.jsonl', '')
            created_files = []
            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, len(pairs))
                batch_pairs = pairs[start:end]
                output_path = f"{output_base}_part{i+1}.jsonl"
                generator.create_batch_requests(batch_pairs, args.base_dir, output_path)
                created_files.append(output_path)
            print(f"\nCreated {n_batches} batch files (rate limit: 40M TPD)")

        # Estimate cost
        est_input = len(pairs) * 2617  # ~2617 tokens/pair
        est_output = len(pairs) * 1302  # ~1302 tokens/pair

        if "nano" in args.model:
            input_cost = est_input * 0.025 / 1_000_000
            output_cost = est_output * 0.20 / 1_000_000
        elif "mini" in args.model:
            input_cost = est_input * 0.125 / 1_000_000
            output_cost = est_output * 1.00 / 1_000_000
        else:  # gpt-5 or gpt-5.1
            input_cost = est_input * 0.625 / 1_000_000
            output_cost = est_output * 5.00 / 1_000_000

        print(f"\nEstimated cost ({args.model} batch):")
        print(f"  Input:  {est_input:,} tokens @ ${input_cost:.2f}")
        print(f"  Output: {est_output:,} tokens @ ${output_cost:.2f}")
        print(f"  Total:  ${input_cost + output_cost:.2f}")

        if n_batches > 1:
            print(f"\n⚠️  Rate limit: 40M tokens/day")
            print(f"   Submit ONE batch file per day:")
            for i, f in enumerate(created_files):
                print(f"   Day {i+1}: {f}")

    elif args.command == "batch-submit":
        generator = OpenAIBatchGenerator()
        batch_id = generator.submit_batch(args.jsonl, args.description)
        print(f"\nBatch submitted! ID: {batch_id}")
        print(f"Use this ID to check status and download results")

    elif args.command == "batch-status":
        generator = OpenAIBatchGenerator()
        status = generator.check_batch_status(args.batch_id)
        print(f"\nBatch Status: {status['status']}")
        print(f"Progress: {status['request_counts']['completed']}/{status['request_counts']['total']}")
        print(f"Failed: {status['request_counts']['failed']}")
        if status['output_file_id']:
            print(f"Output file ready: {status['output_file_id']}")

    elif args.command == "batch-wait":
        generator = OpenAIBatchGenerator()
        print(f"Waiting for batch {args.batch_id} to complete...")
        status = generator.wait_for_batch(args.batch_id, poll_interval=args.poll_interval)
        print(f"\nFinal status: {status['status']}")
        print(f"Completed: {status['request_counts']['completed']}/{status['request_counts']['total']}")

    elif args.command == "batch-download":
        generator = OpenAIBatchGenerator()
        results = generator.download_results(args.batch_id, args.output)

        # Merge with original pairs if provided
        if args.pairs_file:
            with open(args.pairs_file) as f:
                pairs = json.load(f)

            merged = []
            for i, result in enumerate(results):
                if i < len(pairs):
                    merged.append({
                        "pair": pairs[i],
                        "result": result
                    })
                else:
                    merged.append({"result": result})

            with open(args.output, 'w') as f:
                json.dump(merged, f, indent=2)

        print(f"\nDownloaded {len(results)} results")
        successful = sum(1 for r in results if r.get('instructions'))
        print(f"Successful: {successful}/{len(results)}")

    elif args.command == "batch-run":
        # Full pipeline
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(args.pairs_file) as f:
            pairs = json.load(f)

        if args.limit:
            pairs = pairs[:args.limit]

        print(f"=== BATCH PIPELINE ===")
        print(f"Pairs: {len(pairs)}")
        print(f"Model: {args.model}")
        print(f"Output: {output_dir}")
        print()

        generator = OpenAIBatchGenerator(model=args.model, use_cot_prompt=True)

        # Step 1: Create JSONL
        jsonl_path = str(output_dir / "batch_requests.jsonl")
        print("Step 1: Creating batch request file...")
        generator.create_batch_requests(pairs, args.base_dir, jsonl_path)

        # Step 2: Submit batch
        print("\nStep 2: Submitting batch...")
        batch_id = generator.submit_batch(jsonl_path, f"Instructions for {len(pairs)} pairs")

        # Save batch ID
        with open(output_dir / "batch_id.txt", 'w') as f:
            f.write(batch_id)

        # Step 3: Wait for completion
        print(f"\nStep 3: Waiting for batch completion (ID: {batch_id})...")
        status = generator.wait_for_batch(batch_id, poll_interval=args.poll_interval)

        if status['status'] != 'completed':
            print(f"\nBatch failed with status: {status['status']}")
            return

        # Step 4: Download results
        print("\nStep 4: Downloading results...")
        results_path = str(output_dir / "batch_results.json")
        results = generator.download_results(batch_id, results_path)

        # Merge with pairs
        merged = []
        for i, result in enumerate(results):
            if i < len(pairs):
                merged.append({
                    "pair": pairs[i],
                    "result": result
                })
            else:
                merged.append({"result": result})

        final_path = str(output_dir / "instructions_final.json")
        with open(final_path, 'w') as f:
            json.dump(merged, f, indent=2)

        # Summary
        successful = sum(1 for r in results if r.get('instructions'))
        total_input = sum(r.get('input_tokens', 0) for r in results)
        total_output = sum(r.get('output_tokens', 0) for r in results)

        print(f"\n=== COMPLETE ===")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Total input tokens: {total_input:,}")
        print(f"Total output tokens: {total_output:,}")
        print(f"Results saved to: {final_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
