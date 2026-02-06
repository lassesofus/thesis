"""
Sample diverse frame pairs for testing instruction generation.
Selects 5 pairs from different trajectories per lab.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

KNOWN_LABS = {"AUTOLab", "CLVR", "GuptaLab", "ILIAD", "IPRL", "IRIS"}


def extract_lab(trajectory_id: str) -> str | None:
    """Extract lab name from trajectory ID."""
    parts = trajectory_id.split("+")
    if len(parts) >= 2 and parts[0] in KNOWN_LABS:
        return parts[0]
    return None


def sample_diverse_pairs(pairs_data: list[dict], n_per_lab: int = 5, seed: int = 42) -> list[dict]:
    """
    Sample n pairs from different trajectories per lab.
    Prioritizes Tier A pairs.
    """
    random.seed(seed)

    # Group pairs by lab and trajectory
    lab_traj_pairs = defaultdict(lambda: defaultdict(list))
    for pair in pairs_data:
        if pair.get("frame_k_path") is None:
            continue
        lab = extract_lab(pair["trajectory_id"])
        if lab:
            lab_traj_pairs[lab][pair["trajectory_id"]].append(pair)

    selected = []
    for lab in sorted(KNOWN_LABS):
        traj_pairs = lab_traj_pairs[lab]
        trajectories = list(traj_pairs.keys())
        random.shuffle(trajectories)

        lab_selected = []
        for traj_id in trajectories:
            if len(lab_selected) >= n_per_lab:
                break

            pairs = traj_pairs[traj_id]
            # Prefer Tier A
            tier_a = [p for p in pairs if p.get("salience_tier") == "A"]
            candidates = tier_a if tier_a else pairs

            # Pick one random pair from this trajectory
            pair = random.choice(candidates)
            lab_selected.append(pair)

        selected.extend(lab_selected)
        print(f"{lab}: selected {len(lab_selected)} pairs from {len(lab_selected)} trajectories")

    return selected


def main():
    input_path = Path("/data/s185927/droid_raw/sampled_pairs/frame_pairs.json")
    output_path = Path("/home/s185927/thesis/experiments/04_language_grounding/sampled_pairs_multi_gpt5/diverse_test_pairs.json")

    print(f"Loading pairs from {input_path}...")
    with open(input_path) as f:
        pairs_data = json.load(f)
    print(f"Loaded {len(pairs_data)} pairs")

    print("\nSampling diverse pairs (5 per lab from different trajectories)...")
    selected = sample_diverse_pairs(pairs_data, n_per_lab=5)

    print(f"\nTotal selected: {len(selected)} pairs")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
