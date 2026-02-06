"""
Batch Scheduler for Automated Daily OpenAI Batch API Processing

Handles the 40M tokens/day rate limit by:
1. Splitting frame pairs into daily batches
2. Submitting one batch per day at midnight UTC
3. Downloading results as batches complete
4. Merging all results with full traceability

Usage:
    python batch_scheduler.py init --pairs_file ... --base_dir ... --output_dir ...
    python batch_scheduler.py create-batches
    python batch_scheduler.py tick
    python batch_scheduler.py status
    python batch_scheduler.py merge
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Import lazily to avoid cv2 dependency at startup
OpenAIBatchGenerator = None

def get_batch_generator(model: str = "gpt-5-mini"):
    """Lazily import and instantiate the batch generator."""
    global OpenAIBatchGenerator
    if OpenAIBatchGenerator is None:
        from instruction_generator_v3 import OpenAIBatchGenerator as _OpenAIBatchGenerator
        OpenAIBatchGenerator = _OpenAIBatchGenerator
    return OpenAIBatchGenerator(model=model, use_cot_prompt=True)

# ============================================================================
# Constants
# ============================================================================

TOKENS_PER_PAIR = 4000  # ~2617 input + ~1302 output, rounded up for safety
DEFAULT_DAILY_LIMIT = 40_000_000
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_PAIRS_PER_BATCH = 1000  # Keep files under ~500MB for upload limit

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: Path, verbose: bool = True):
    """Configure logging to both file and stdout."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "scheduler.log"

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S UTC'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================================
# State Management
# ============================================================================

def load_state(state_file: Path) -> dict:
    """Load state from JSON file."""
    if not state_file.exists():
        raise FileNotFoundError(f"State file not found: {state_file}\nRun 'init' first.")
    with open(state_file) as f:
        return json.load(f)


def save_state(state: dict, state_file: Path):
    """Save state to JSON file with atomic write."""
    temp_file = state_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(state, f, indent=2)
    temp_file.rename(state_file)


# ============================================================================
# Commands
# ============================================================================

def cmd_init(args):
    """Initialize state file and calculate batch splits."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(output_dir / "logs")

    # Load pairs file
    logging.info(f"Loading pairs from: {args.pairs_file}")
    with open(args.pairs_file) as f:
        pairs = json.load(f)
    total_pairs = len(pairs)
    logging.info(f"Total pairs: {total_pairs}")

    # Calculate batch sizes
    # Use fixed batch size for file upload limit (~500MB per file)
    # But we can submit multiple batches per day up to the token limit
    daily_limit = args.daily_limit
    pairs_per_batch = args.pairs_per_batch
    batches_per_day = daily_limit // (pairs_per_batch * TOKENS_PER_PAIR)
    num_batches = (total_pairs + pairs_per_batch - 1) // pairs_per_batch
    estimated_days = (num_batches + batches_per_day - 1) // batches_per_day

    logging.info(f"Daily token limit: {daily_limit:,}")
    logging.info(f"Tokens per pair: ~{TOKENS_PER_PAIR}")
    logging.info(f"Pairs per batch: {pairs_per_batch} (for file size limit)")
    logging.info(f"Batches per day: {batches_per_day} (token limit)")
    logging.info(f"Total batches: {num_batches}")
    logging.info(f"Estimated days: {estimated_days}")

    # Create batch entries
    batches = []
    for i in range(num_batches):
        start_idx = i * pairs_per_batch
        end_idx = min(start_idx + pairs_per_batch, total_pairs)
        batches.append({
            "part": i + 1,
            "jsonl_file": f"batches/batch_part_{i+1:03d}.jsonl",
            "pair_indices": [start_idx, end_idx - 1],
            "status": "pending"
        })

    # Create state
    state = {
        "config": {
            "pairs_file": str(Path(args.pairs_file).resolve()),
            "base_dir": str(Path(args.base_dir).resolve()),
            "output_dir": str(output_dir.resolve()),
            "model": args.model,
            "daily_token_limit": daily_limit,
            "pairs_per_batch": pairs_per_batch,
            "batches_per_day": batches_per_day
        },
        "batches": batches,
        "last_submission_date": None,
        "tokens_submitted_today": 0,
        "total_pairs": total_pairs,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    # Create directories
    (output_dir / "batches").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    (output_dir / "final").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # Save state
    state_file = output_dir / "batch_state.json"
    save_state(state, state_file)

    logging.info(f"State initialized: {state_file}")
    logging.info(f"Will process {total_pairs} pairs in {num_batches} batches over ~{estimated_days} days")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Initialization complete!")
    print(f"{'='*60}")
    print(f"Total pairs:      {total_pairs:,}")
    print(f"Batch files:      {num_batches}")
    print(f"Pairs per batch:  {pairs_per_batch:,}")
    print(f"Batches per day:  {batches_per_day}")
    print(f"Estimated days:   {estimated_days}")
    print(f"\nNext step: python batch_scheduler.py create-batches")


def cmd_create_batches(args):
    """Create all JSONL batch files."""
    output_dir = Path(args.output_dir)
    state_file = output_dir / "batch_state.json"
    state = load_state(state_file)

    setup_logging(output_dir / "logs")

    # Load pairs
    logging.info(f"Loading pairs from: {state['config']['pairs_file']}")
    with open(state['config']['pairs_file']) as f:
        pairs = json.load(f)

    # Create generator (lazy import to avoid cv2 dependency)
    generator = get_batch_generator(model=state['config']['model'])

    base_dir = state['config']['base_dir']
    batches_dir = output_dir / "batches"

    for batch in state['batches']:
        if batch['status'] != 'pending':
            logging.info(f"Skipping batch {batch['part']} (status: {batch['status']})")
            continue

        jsonl_path = output_dir / batch['jsonl_file']
        if jsonl_path.exists():
            logging.info(f"Batch {batch['part']} JSONL already exists: {jsonl_path}")
            continue

        start_idx, end_idx = batch['pair_indices']
        batch_pairs_raw = pairs[start_idx:end_idx + 1]

        # Filter out pairs with missing frame paths
        batch_pairs = [p for p in batch_pairs_raw if p.get('frame_k_path') and p.get('frame_k_d_path')]
        skipped = len(batch_pairs_raw) - len(batch_pairs)
        if skipped > 0:
            logging.warning(f"Skipping {skipped} pairs with missing frame paths")

        logging.info(f"Creating batch {batch['part']}: pairs {start_idx}-{end_idx} ({len(batch_pairs)} valid pairs)")

        generator.create_batch_requests(
            pairs=batch_pairs,
            base_dir=base_dir,
            output_jsonl=str(jsonl_path)
        )

        logging.info(f"Created: {jsonl_path}")

    save_state(state, state_file)

    print(f"\n{'='*60}")
    print(f"Batch files created!")
    print(f"{'='*60}")
    print(f"Location: {batches_dir}")
    print(f"\nNext step: python batch_scheduler.py tick")
    print(f"Or set up cron: crontab -e")


def cmd_tick(args):
    """Run one scheduler tick: check status, download results, submit next batch."""
    output_dir = Path(args.output_dir)
    state_file = output_dir / "batch_state.json"
    state = load_state(state_file)

    setup_logging(output_dir / "logs")
    logging.info("=== Scheduler tick started ===")

    generator = get_batch_generator(model=state['config']['model'])
    today_utc = datetime.now(timezone.utc).date().isoformat()

    # Step 1: Check in-progress batches
    for batch in state['batches']:
        if batch.get('status') == 'submitted':
            batch_id = batch.get('batch_id')
            if not batch_id:
                continue

            logging.info(f"Checking batch {batch['part']} (ID: {batch_id})")
            try:
                status = generator.check_batch_status(batch_id)
                api_status = status['status']
                completed = status['request_counts']['completed']
                total = status['request_counts']['total']
                failed = status['request_counts']['failed']

                logging.info(f"  Status: {api_status} ({completed}/{total} completed, {failed} failed)")

                if api_status == 'completed':
                    # Download results
                    results_file = f"results/results_part_{batch['part']:03d}.json"
                    results_path = output_dir / results_file

                    logging.info(f"  Downloading results to: {results_path}")
                    results = generator.download_results(batch_id, str(results_path))

                    successful = sum(1 for r in results if r.get('instructions'))
                    batch['status'] = 'completed'
                    batch['completed_at'] = datetime.now(timezone.utc).isoformat()
                    batch['results_file'] = results_file
                    batch['stats'] = {
                        'successful': successful,
                        'failed': len(results) - successful,
                        'total': len(results)
                    }

                    logging.info(f"  Downloaded: {successful}/{len(results)} successful")

                elif api_status in ['failed', 'cancelled', 'expired']:
                    batch['status'] = 'failed'
                    batch['failed_at'] = datetime.now(timezone.utc).isoformat()
                    batch['error'] = api_status
                    logging.error(f"  Batch failed with status: {api_status}")

            except Exception as e:
                logging.error(f"  Error checking batch {batch['part']}: {e}")

    # Step 2: Submit batches up to daily token limit
    daily_limit = state['config']['daily_token_limit']
    pairs_per_batch = state['config']['pairs_per_batch']
    tokens_per_batch = pairs_per_batch * TOKENS_PER_PAIR

    # Reset token counter if new day
    if state.get('last_submission_date') != today_utc:
        state['tokens_submitted_today'] = 0
        state['last_submission_date'] = today_utc

    tokens_today = state.get('tokens_submitted_today', 0)
    batches_submitted_this_tick = 0

    while tokens_today + tokens_per_batch <= daily_limit or args.force:
        # Find next pending batch with JSONL file ready
        pending = None
        for batch in state['batches']:
            if batch['status'] == 'pending':
                jsonl_path = output_dir / batch['jsonl_file']
                if jsonl_path.exists():
                    pending = batch
                    break
                else:
                    logging.warning(f"Batch {batch['part']} pending but JSONL not found: {jsonl_path}")

        if not pending:
            logging.info("No more pending batches to submit")
            break

        jsonl_path = output_dir / pending['jsonl_file']
        logging.info(f"Submitting batch {pending['part']}: {jsonl_path}")

        try:
            batch_id = generator.submit_batch(
                str(jsonl_path),
                description=f"Frame pair instructions batch {pending['part']}"
            )

            pending['status'] = 'submitted'
            pending['batch_id'] = batch_id
            pending['submitted_at'] = datetime.now(timezone.utc).isoformat()
            tokens_today += tokens_per_batch
            state['tokens_submitted_today'] = tokens_today
            batches_submitted_this_tick += 1

            logging.info(f"Submitted batch {pending['part']} with ID: {batch_id}")
            logging.info(f"Tokens submitted today: {tokens_today:,} / {daily_limit:,}")

            # Save state after each successful submission
            save_state(state, state_file)

        except Exception as e:
            logging.error(f"Failed to submit batch {pending['part']}: {e}")
            break  # Stop trying if we hit an error

        if args.force:
            break  # Only submit one batch when forcing

    if batches_submitted_this_tick == 0 and tokens_today >= daily_limit:
        logging.info(f"Daily token limit reached ({tokens_today:,} / {daily_limit:,})")

    # Save state
    save_state(state, state_file)
    logging.info("=== Scheduler tick completed ===")

    # Print summary
    cmd_status(args)


def cmd_status(args):
    """Print human-readable status."""
    output_dir = Path(args.output_dir)
    state_file = output_dir / "batch_state.json"
    state = load_state(state_file)

    # Count by status
    counts = {'pending': 0, 'submitted': 0, 'completed': 0, 'failed': 0}
    for batch in state['batches']:
        status = batch.get('status', 'pending')
        counts[status] = counts.get(status, 0) + 1

    total_successful = sum(
        batch.get('stats', {}).get('successful', 0)
        for batch in state['batches']
    )

    print(f"\n{'='*60}")
    print(f"Batch Scheduler Status")
    print(f"{'='*60}")
    print(f"Total pairs:     {state['total_pairs']:,}")
    print(f"Total batches:   {len(state['batches'])}")
    print(f"")
    print(f"  Pending:       {counts['pending']}")
    print(f"  Submitted:     {counts['submitted']}")
    print(f"  Completed:     {counts['completed']}")
    print(f"  Failed:        {counts['failed']}")
    print(f"")
    print(f"Instructions generated: {total_successful:,}")
    print(f"Last submission: {state.get('last_submission_date', 'Never')}")

    # Show in-progress batches
    in_progress = [b for b in state['batches'] if b.get('status') == 'submitted']
    if in_progress:
        print(f"\nIn-progress batches:")
        for batch in in_progress:
            print(f"  Part {batch['part']}: {batch.get('batch_id', 'unknown')} (submitted {batch.get('submitted_at', '?')})")

    # Estimate completion
    remaining = counts['pending'] + counts['submitted']
    if remaining > 0:
        print(f"\nEstimated days remaining: {remaining}")
    else:
        print(f"\nAll batches complete! Run 'merge' to combine results.")


def cmd_merge(args):
    """Merge all completed results into final output."""
    output_dir = Path(args.output_dir)
    state_file = output_dir / "batch_state.json"
    state = load_state(state_file)

    setup_logging(output_dir / "logs")
    logging.info("=== Merging results ===")

    # Load original pairs for metadata
    with open(state['config']['pairs_file']) as f:
        pairs = json.load(f)

    all_results = []
    total_successful = 0
    total_failed = 0

    for batch in state['batches']:
        if batch.get('status') != 'completed':
            logging.warning(f"Skipping batch {batch['part']} (status: {batch.get('status')})")
            continue

        results_file = batch.get('results_file')
        if not results_file:
            logging.warning(f"Batch {batch['part']} has no results file")
            continue

        results_path = output_dir / results_file
        if not results_path.exists():
            logging.warning(f"Results file not found: {results_path}")
            continue

        logging.info(f"Loading results from batch {batch['part']}: {results_path}")
        with open(results_path) as f:
            results = json.load(f)

        # Merge with original pair metadata
        start_idx = batch['pair_indices'][0]
        for i, result in enumerate(results):
            pair_idx = start_idx + i
            if pair_idx < len(pairs):
                pair = pairs[pair_idx]
                merged = {
                    "pair_index": pair_idx,
                    **pair,  # Original pair metadata
                    "instructions": result.get('instructions', []),
                    "what_changed": result.get('what_changed', ''),
                    "frame1_description": result.get('frame1_description', ''),
                    "frame2_description": result.get('frame2_description', ''),
                }
                if result.get('instructions'):
                    total_successful += 1
                else:
                    total_failed += 1
                    merged['error'] = result.get('error', 'No instructions parsed')
                all_results.append(merged)

    # Sort by pair index
    all_results.sort(key=lambda x: x['pair_index'])

    # Save merged results
    final_path = output_dir / "final" / "all_instructions.json"
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logging.info(f"Merged {len(all_results)} results to: {final_path}")

    print(f"\n{'='*60}")
    print(f"Merge complete!")
    print(f"{'='*60}")
    print(f"Total results:   {len(all_results):,}")
    print(f"Successful:      {total_successful:,}")
    print(f"Failed:          {total_failed:,}")
    print(f"Output:          {final_path}")


def cmd_retry_failed(args):
    """Mark failed batches as pending for retry."""
    output_dir = Path(args.output_dir)
    state_file = output_dir / "batch_state.json"
    state = load_state(state_file)

    setup_logging(output_dir / "logs")

    retried = 0
    for batch in state['batches']:
        if batch.get('status') == 'failed':
            logging.info(f"Marking batch {batch['part']} for retry")
            batch['status'] = 'pending'
            batch.pop('batch_id', None)
            batch.pop('submitted_at', None)
            batch.pop('failed_at', None)
            batch.pop('error', None)
            retried += 1

    save_state(state, state_file)
    print(f"Marked {retried} failed batches for retry")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch Scheduler for automated daily OpenAI Batch API processing"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize state and calculate batch splits")
    init_parser.add_argument("--pairs_file", type=str, required=True,
                             help="Path to frame_pairs.json")
    init_parser.add_argument("--base_dir", type=str, required=True,
                             help="Base directory for frame images")
    init_parser.add_argument("--output_dir", type=str, required=True,
                             help="Output directory for batches, results, logs")
    init_parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                             help=f"Model to use (default: {DEFAULT_MODEL})")
    init_parser.add_argument("--daily_limit", type=int, default=DEFAULT_DAILY_LIMIT,
                             help=f"Daily token limit (default: {DEFAULT_DAILY_LIMIT:,})")
    init_parser.add_argument("--pairs_per_batch", type=int, default=DEFAULT_PAIRS_PER_BATCH,
                             help=f"Pairs per batch file (default: {DEFAULT_PAIRS_PER_BATCH}, keeps files under upload limit)")

    # create-batches
    create_parser = subparsers.add_parser("create-batches", help="Create JSONL batch files")
    create_parser.add_argument("--output_dir", type=str, default="./batch_run",
                               help="Output directory (default: ./batch_run)")

    # tick
    tick_parser = subparsers.add_parser("tick", help="Run scheduler tick (check, download, submit)")
    tick_parser.add_argument("--output_dir", type=str, default="./batch_run",
                             help="Output directory (default: ./batch_run)")
    tick_parser.add_argument("--force", action="store_true",
                             help="Force submission even if already submitted today")

    # status
    status_parser = subparsers.add_parser("status", help="Show current status")
    status_parser.add_argument("--output_dir", type=str, default="./batch_run",
                               help="Output directory (default: ./batch_run)")

    # merge
    merge_parser = subparsers.add_parser("merge", help="Merge all completed results")
    merge_parser.add_argument("--output_dir", type=str, default="./batch_run",
                              help="Output directory (default: ./batch_run)")

    # retry-failed
    retry_parser = subparsers.add_parser("retry-failed", help="Mark failed batches for retry")
    retry_parser.add_argument("--output_dir", type=str, default="./batch_run",
                              help="Output directory (default: ./batch_run)")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "create-batches":
        cmd_create_batches(args)
    elif args.command == "tick":
        cmd_tick(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "retry-failed":
        cmd_retry_failed(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
