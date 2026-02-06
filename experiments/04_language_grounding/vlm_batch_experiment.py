"""
VLM Batch Experiment - Testing multiple VLM APIs on sampled frame pairs.

Supports:
- OpenAI gpt-4o-mini (with detail=high for images)
- OpenAI Batch API (50% cost reduction, async processing)
- Google Gemini 2.5/2.0 Flash (with batch processing)

Usage:
    # Run gpt-4o-mini (detail=high) synchronously
    python vlm_batch_experiment.py --provider openai --model gpt-4o-mini --sync

    # Create OpenAI batch job (async, 50% cheaper)
    python vlm_batch_experiment.py --provider openai --model gpt-4o-mini --batch

    # Check batch job status
    python vlm_batch_experiment.py --check-batch <batch_id>

    # Download batch results
    python vlm_batch_experiment.py --download-batch <batch_id>

    # Run Gemini Flash
    python vlm_batch_experiment.py --provider gemini --model gemini-2.0-flash

    # Run Gemini 2.5 Flash
    python vlm_batch_experiment.py --provider gemini --model gemini-2.5-flash-preview-05-20
"""

import base64
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime
import tempfile

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Try loading from current directory and experiment directory
    load_dotenv()
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def check_api_key(provider: str):
    """Check if API key is set and provide helpful message if not."""
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set!")
            print("\nTo set it, either:")
            print("  1. Export in terminal: export OPENAI_API_KEY='your-key'")
            print("  2. Create a .env file with: OPENAI_API_KEY=your-key")
            print("  3. Add --api-key argument (for testing only)")
            return False
    elif provider == "gemini":
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            print("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not set!")
            print("\nTo set it, either:")
            print("  1. Export in terminal: export GOOGLE_API_KEY='your-key'")
            print("  2. Create a .env file with: GOOGLE_API_KEY=your-key")
            return False
    return True


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================================
# Prompt
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
# OpenAI Generator with detail=high support
# ============================================================================

class OpenAIGenerator:
    """OpenAI GPT-4o/4o-mini generator with configurable image detail level."""

    def __init__(self, model: str = "gpt-4o-mini", detail: str = "high"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.detail = detail  # "low", "high", or "auto"

    def generate_instruction(self, frame_k_path: str, frame_k_d_path: str) -> dict:
        """Generate instruction from two frames shown together."""
        # GPT-5+ models use max_completion_tokens instead of max_tokens, and need more tokens
        is_gpt5 = "gpt-5" in self.model or "gpt-4.1" in self.model
        token_param = "max_completion_tokens" if is_gpt5 else "max_tokens"
        token_limit = 1000 if is_gpt5 else 300  # gpt-5 needs more tokens

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


# ============================================================================
# OpenAI Batch API Support
# ============================================================================

class OpenAIBatchProcessor:
    """OpenAI Batch API processor for 50% cost reduction."""

    def __init__(self, model: str = "gpt-4o-mini", detail: str = "high"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.detail = detail

    def create_batch_file(self, pairs: list[dict], output_path: str) -> str:
        """Create JSONL file for batch processing."""
        with open(output_path, "w") as f:
            for i, pair in enumerate(pairs):
                request = {
                    "custom_id": f"pair_{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "max_tokens": 300,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": TWO_FRAME_PROMPT},
                                {"type": "text", "text": "\n\nFRAME 1 (BEFORE):"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encode_image_base64(pair['frame_k_path'])}",
                                        "detail": self.detail
                                    }
                                },
                                {"type": "text", "text": "\n\nFRAME 2 (AFTER):"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encode_image_base64(pair['frame_k_d_path'])}",
                                        "detail": self.detail
                                    }
                                }
                            ]
                        }]
                    }
                }
                f.write(json.dumps(request) + "\n")
        return output_path

    def submit_batch(self, jsonl_path: str, metadata: dict = None) -> dict:
        """Submit batch job to OpenAI."""
        # Upload the file
        with open(jsonl_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="batch")

        # Create batch
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata or {}
        )

        return {
            "batch_id": batch.id,
            "input_file_id": file_obj.id,
            "status": batch.status,
            "created_at": datetime.now().isoformat()
        }

    def check_batch_status(self, batch_id: str) -> dict:
        """Check status of a batch job."""
        batch = self.client.batches.retrieve(batch_id)
        return {
            "batch_id": batch.id,
            "status": batch.status,
            "request_counts": {
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed
            } if batch.request_counts else None,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at
        }

    def download_results(self, batch_id: str, output_path: str) -> list[dict]:
        """Download and parse batch results."""
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise ValueError(f"Batch not completed. Status: {batch.status}")

        if not batch.output_file_id:
            raise ValueError("No output file available")

        # Download results
        content = self.client.files.content(batch.output_file_id)
        raw_results = content.text.strip().split("\n")

        results = []
        for line in raw_results:
            result = json.loads(line)
            custom_id = result["custom_id"]
            response = result["response"]

            if response["status_code"] == 200:
                text = response["body"]["choices"][0]["message"]["content"]
                parsed = self._parse_response(text)
                results.append({
                    "custom_id": custom_id,
                    "success": True,
                    **parsed
                })
            else:
                results.append({
                    "custom_id": custom_id,
                    "success": False,
                    "error": response.get("error", "Unknown error")
                })

        # Save to file
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _parse_response(self, text: str) -> dict:
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


# ============================================================================
# Gemini Generator (using new google-genai SDK)
# ============================================================================

class GeminiGenerator:
    """Google Gemini generator with Flash models using new SDK."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        from google import genai
        from google.genai import types
        self.client = genai.Client()
        self.model_name = model
        self.types = types

    def generate_instruction(self, frame_k_path: str, frame_k_d_path: str) -> dict:
        """Generate instruction from two frames shown together."""
        import PIL.Image

        image1 = PIL.Image.open(frame_k_path)
        image2 = PIL.Image.open(frame_k_d_path)

        # Build multimodal content using new SDK
        contents = [
            self.types.Part.from_text(TWO_FRAME_PROMPT),
            self.types.Part.from_text("\n\nFRAME 1 (BEFORE):"),
            self.types.Part.from_image(image1),
            self.types.Part.from_text("\n\nFRAME 2 (AFTER):"),
            self.types.Part.from_image(image2),
        ]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )

        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> dict:
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


# ============================================================================
# Gemini Batch API Support
# ============================================================================

class GeminiBatchProcessor:
    """Gemini Batch Prediction API processor using new SDK."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        from google import genai
        from google.genai import types
        self.client = genai.Client()
        self.model_name = model
        self.types = types

    def process_batch_sync(self, pairs: list[dict], output_path: str,
                           rate_limit_delay: float = 0.5,
                           verbose: bool = True) -> list[dict]:
        """Process pairs synchronously with rate limiting.

        Note: Gemini's true batch API requires GCP setup. This provides
        rate-limited sequential processing.
        """
        import PIL.Image

        results = []

        for i, pair in enumerate(pairs):
            if verbose:
                print(f"  [{i+1}/{len(pairs)}] Processing frame {pair['frame_k']} -> {pair['frame_k_d']}...")

            try:
                image1 = PIL.Image.open(pair['frame_k_path'])
                image2 = PIL.Image.open(pair['frame_k_d_path'])

                contents = [
                    self.types.Part.from_text(TWO_FRAME_PROMPT),
                    self.types.Part.from_text("\n\nFRAME 1 (BEFORE):"),
                    self.types.Part.from_image(image1),
                    self.types.Part.from_text("\n\nFRAME 2 (AFTER):"),
                    self.types.Part.from_image(image2),
                ]

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )

                parsed = self._parse_response(response.text)

                result = {
                    **pair,
                    "what_changed": parsed["what_changed"],
                    "instruction": parsed["instruction"],
                    "raw_response": parsed["raw_response"],
                    "model": self.model_name
                }
                results.append(result)

                if verbose:
                    print(f"    ✓ {parsed['instruction'][:60]}...")

                # Save incrementally
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                if verbose:
                    print(f"    ✗ Error: {e}")
                results.append({
                    **pair,
                    "error": str(e),
                    "model": self.model_name
                })

            if rate_limit_delay > 0 and i < len(pairs) - 1:
                time.sleep(rate_limit_delay)

        return results

    def _parse_response(self, text: str) -> dict:
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


# ============================================================================
# Main Processing Functions
# ============================================================================

def load_frame_pairs(pairs_file: str, base_dir: str = None) -> list[dict]:
    """Load frame pairs and resolve paths."""
    with open(pairs_file) as f:
        pairs = json.load(f)

    if base_dir:
        base_dir = Path(base_dir)
        for pair in pairs:
            if not Path(pair['frame_k_path']).is_absolute():
                pair['frame_k_path'] = str(base_dir / pair['frame_k_path'])
                pair['frame_k_d_path'] = str(base_dir / pair['frame_k_d_path'])

    return pairs


def run_sync_experiment(
    pairs: list[dict],
    provider: str,
    model: str,
    output_path: str,
    detail: str = "high",
    rate_limit_delay: float = 1.0,
    limit: int = None,
    verbose: bool = True
) -> list[dict]:
    """Run synchronous experiment with any provider."""

    if limit:
        pairs = pairs[:limit]

    print(f"\nRunning {provider} with model {model}")
    print(f"Processing {len(pairs)} pairs")
    print(f"Output: {output_path}")
    print("=" * 60)

    if provider == "openai":
        generator = OpenAIGenerator(model=model, detail=detail)
    elif provider == "gemini":
        generator = GeminiGenerator(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    results = []

    for i, pair in enumerate(pairs):
        if verbose:
            print(f"  [{i+1}/{len(pairs)}] frame {pair['frame_k']} -> {pair['frame_k_d']}...")

        try:
            vlm_result = generator.generate_instruction(
                pair['frame_k_path'],
                pair['frame_k_d_path']
            )

            result = {
                **pair,
                "what_changed": vlm_result["what_changed"],
                "instruction": vlm_result["instruction"],
                "raw_response": vlm_result["raw_response"],
                "model": model,
                "detail": detail if provider == "openai" else None
            }
            results.append(result)

            if verbose:
                print(f"    ✓ {vlm_result['instruction'][:60]}...")

            # Save incrementally
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            results.append({
                **pair,
                "error": str(e),
                "model": model
            })

        if rate_limit_delay > 0 and i < len(pairs) - 1:
            time.sleep(rate_limit_delay)

    print(f"\n{'='*60}")
    print(f"Completed: {len([r for r in results if 'error' not in r])}/{len(pairs)}")
    print(f"Saved to: {output_path}")

    return results


def run_openai_batch(
    pairs: list[dict],
    model: str,
    output_dir: str,
    detail: str = "high",
    limit: int = None
) -> dict:
    """Create and submit OpenAI batch job."""

    if limit:
        pairs = pairs[:limit]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = OpenAIBatchProcessor(model=model, detail=detail)

    # Create batch file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"batch_input_{model}_{timestamp}.jsonl"

    print(f"Creating batch file with {len(pairs)} requests...")
    processor.create_batch_file(pairs, str(jsonl_path))
    print(f"Batch file: {jsonl_path}")

    # Submit batch
    print("Submitting batch to OpenAI...")
    batch_info = processor.submit_batch(
        str(jsonl_path),
        metadata={"model": model, "detail": detail, "n_pairs": str(len(pairs))}
    )

    # Save batch info
    batch_info_path = output_dir / f"batch_info_{batch_info['batch_id']}.json"
    batch_info["pairs_file"] = str(jsonl_path)
    batch_info["n_pairs"] = len(pairs)
    batch_info["model"] = model
    batch_info["detail"] = detail

    with open(batch_info_path, "w") as f:
        json.dump(batch_info, f, indent=2)

    print(f"\nBatch submitted!")
    print(f"  Batch ID: {batch_info['batch_id']}")
    print(f"  Status: {batch_info['status']}")
    print(f"  Info saved to: {batch_info_path}")
    print(f"\nTo check status:")
    print(f"  python vlm_batch_experiment.py --check-batch {batch_info['batch_id']}")

    return batch_info


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="VLM Batch Experiment")

    # Data arguments
    parser.add_argument("--pairs_file", type=str,
                        default="/home/s185927/thesis/experiments/04_language_grounding/sampled_pairs_v4/frame_pairs.json")
    parser.add_argument("--base_dir", type=str,
                        default="/home/s185927/thesis/experiments/04_language_grounding")
    parser.add_argument("--output_dir", type=str,
                        default="/home/s185927/thesis/experiments/04_language_grounding/sampled_pairs_v4")

    # Provider/model arguments
    parser.add_argument("--provider", type=str, choices=["openai", "gemini"])
    parser.add_argument("--model", type=str, help="Model name (e.g., gpt-4o-mini, gemini-2.0-flash)")
    parser.add_argument("--detail", type=str, default="high",
                        choices=["low", "high", "auto"],
                        help="OpenAI image detail level")

    # Mode arguments
    parser.add_argument("--sync", action="store_true", help="Run synchronously (default)")
    parser.add_argument("--batch", action="store_true", help="Use OpenAI Batch API")
    parser.add_argument("--check-batch", type=str, metavar="BATCH_ID",
                        help="Check batch job status")
    parser.add_argument("--download-batch", type=str, metavar="BATCH_ID",
                        help="Download batch results")

    # Processing arguments
    parser.add_argument("--limit", type=int, help="Limit number of pairs to process")
    parser.add_argument("--rate_limit", type=float, default=0.5,
                        help="Delay between API calls (seconds)")

    # API key (optional - prefer env var)
    parser.add_argument("--api-key", type=str, help="API key (prefer using env var instead)")

    args = parser.parse_args()

    # Set API key from argument if provided
    if args.api_key:
        if args.provider == "openai" or args.check_batch or args.download_batch:
            os.environ["OPENAI_API_KEY"] = args.api_key
        elif args.provider == "gemini":
            os.environ["GOOGLE_API_KEY"] = args.api_key

    # Handle batch status check
    if args.check_batch:
        if not check_api_key("openai"):
            return
        processor = OpenAIBatchProcessor()
        status = processor.check_batch_status(args.check_batch)
        print(json.dumps(status, indent=2, default=str))
        return

    # Handle batch download
    if args.download_batch:
        if not check_api_key("openai"):
            return
        processor = OpenAIBatchProcessor()
        output_path = Path(args.output_dir) / f"batch_results_{args.download_batch}.json"
        results = processor.download_results(args.download_batch, str(output_path))
        print(f"Downloaded {len(results)} results to {output_path}")
        return

    # Check API key before loading data
    if args.provider and not check_api_key(args.provider):
        return

    # Load pairs
    pairs = load_frame_pairs(args.pairs_file, args.base_dir)
    print(f"Loaded {len(pairs)} frame pairs")

    if args.batch and args.provider == "openai":
        # OpenAI Batch API mode
        run_openai_batch(
            pairs,
            model=args.model or "gpt-4o-mini",
            output_dir=args.output_dir,
            detail=args.detail,
            limit=args.limit
        )
    else:
        # Synchronous mode
        if not args.provider:
            parser.error("--provider required for sync mode")

        model = args.model
        if not model:
            model = "gpt-4o-mini" if args.provider == "openai" else "gemini-2.0-flash"

        output_path = Path(args.output_dir) / f"results_{args.provider}_{model.replace('/', '_')}.json"

        run_sync_experiment(
            pairs,
            provider=args.provider,
            model=model,
            output_path=str(output_path),
            detail=args.detail,
            rate_limit_delay=args.rate_limit,
            limit=args.limit
        )


if __name__ == "__main__":
    main()
