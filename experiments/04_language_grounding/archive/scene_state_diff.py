"""
Scene-State Diffing Pipeline for Generating Robot Instructions

Approach:
1. Extract keyframes at trajectory segment boundaries
2. Query VLM to describe scene STATE (not action) at each keyframe
3. Diff consecutive states to generate action instructions
"""

import json
import os
from dataclasses import dataclass
from typing import Optional
import base64

# ============================================================================
# Scene State Schema
# ============================================================================

@dataclass
class ObjectState:
    name: str
    location: str  # e.g., "on table", "in bowl", "in gripper"
    state: Optional[str] = None  # e.g., "empty", "contains balls"

@dataclass
class SceneState:
    objects: list[ObjectState]
    gripper_state: str  # "open", "closed", "holding <object>"
    gripper_position: str  # "above table", "near cup", "retracted"

# ============================================================================
# VLM Prompting
# ============================================================================

SCENE_STATE_PROMPT = """Analyze this robot manipulation scene and describe the current state.

Focus ONLY on:
1. Objects on the table (name, location, contents if container)
2. Robot gripper state (open/closed, holding anything?)
3. Gripper position relative to objects

Respond in this exact JSON format:
{
    "objects": [
        {"name": "pink bowl", "location": "center of table", "state": "empty"},
        {"name": "yellow cup", "location": "right side of table", "state": "contains small balls"}
    ],
    "gripper_state": "open",
    "gripper_position": "above and behind the objects"
}

Be precise and factual. Only describe what you can see."""


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for API calls."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_vlm_for_scene_state(image_path: str, vlm_client=None) -> dict:
    """
    Query a VLM to describe the scene state.

    Replace this with your actual VLM client (OpenAI, Anthropic, local model, etc.)
    """
    # Example for OpenAI API:
    #
    # from openai import OpenAI
    # client = OpenAI()
    #
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": SCENE_STATE_PROMPT},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{encode_image_base64(image_path)}"
    #                     }
    #                 }
    #             ]
    #         }
    #     ],
    #     response_format={"type": "json_object"}
    # )
    # return json.loads(response.choices[0].message.content)

    # Placeholder - replace with actual VLM call
    raise NotImplementedError("Implement VLM client call here")


# ============================================================================
# State Diffing
# ============================================================================

def diff_scene_states(state_before: dict, state_after: dict) -> list[str]:
    """
    Compare two scene states and generate action descriptions.

    Returns list of changes detected.
    """
    changes = []

    # Build object lookup by name
    objects_before = {obj["name"]: obj for obj in state_before.get("objects", [])}
    objects_after = {obj["name"]: obj for obj in state_after.get("objects", [])}

    # Check gripper state changes
    gripper_before = state_before.get("gripper_state", "")
    gripper_after = state_after.get("gripper_state", "")

    if "holding" in gripper_after and "holding" not in gripper_before:
        # Robot picked something up
        held_object = gripper_after.replace("holding ", "")
        changes.append(f"grasp the {held_object}")
    elif "holding" in gripper_before and "holding" not in gripper_after:
        # Robot released something
        released_object = gripper_before.replace("holding ", "")
        changes.append(f"release the {released_object}")
    elif gripper_before == "closed" and gripper_after == "open":
        changes.append("open the gripper")
    elif gripper_before == "open" and gripper_after == "closed":
        changes.append("close the gripper")

    # Check object state changes
    for name, obj_after in objects_after.items():
        obj_before = objects_before.get(name)

        if obj_before:
            # Location changed
            if obj_before.get("location") != obj_after.get("location"):
                changes.append(f"move {name} from {obj_before.get('location')} to {obj_after.get('location')}")

            # State changed (e.g., empty -> contains items)
            state_b = obj_before.get("state", "")
            state_a = obj_after.get("state", "")
            if state_b != state_a:
                if "empty" in str(state_b) and "contains" in str(state_a):
                    changes.append(f"pour contents into {name}")
                elif "contains" in str(state_b) and "empty" in str(state_a):
                    changes.append(f"empty the {name}")

    # Check gripper position for approach/retract
    pos_before = state_before.get("gripper_position", "")
    pos_after = state_after.get("gripper_position", "")

    if pos_before != pos_after and not changes:
        # Only mention position if no other action detected
        if "near" in pos_after or "above" in pos_after:
            for obj_name in objects_after.keys():
                if obj_name in pos_after:
                    changes.append(f"move gripper toward {obj_name}")
                    break
        elif "retracted" in pos_after or "away" in pos_after:
            changes.append("retract gripper")

    return changes if changes else ["hold position"]


def generate_instruction(changes: list[str], task_context: str = "") -> str:
    """
    Convert list of changes into a natural language instruction.
    """
    if len(changes) == 1:
        return changes[0].capitalize()
    else:
        # Combine multiple changes
        return ", then ".join(changes).capitalize()


# ============================================================================
# Full Pipeline
# ============================================================================

def process_trajectory_keyframes(
    keyframe_dir: str,
    trajectory_metadata: dict,
    vlm_client=None
) -> list[dict]:
    """
    Process all keyframes and generate instructions for each segment.
    """
    # Get sorted keyframe files
    keyframe_files = sorted([
        f for f in os.listdir(keyframe_dir)
        if f.startswith("keyframe_") and f.endswith(".jpg")
    ])

    task = trajectory_metadata.get("current_task", "manipulation task")
    results = []

    # Get scene state for each keyframe
    scene_states = []
    for kf_file in keyframe_files:
        kf_path = os.path.join(keyframe_dir, kf_file)
        try:
            state = query_vlm_for_scene_state(kf_path, vlm_client)
            scene_states.append({"file": kf_file, "state": state})
        except NotImplementedError:
            # Use mock states for demonstration
            scene_states.append({"file": kf_file, "state": None})

    # Generate instructions by diffing consecutive states
    for i in range(len(scene_states) - 1):
        state_before = scene_states[i]["state"]
        state_after = scene_states[i + 1]["state"]

        if state_before and state_after:
            changes = diff_scene_states(state_before, state_after)
            instruction = generate_instruction(changes, task)
        else:
            instruction = "[VLM states not available]"

        results.append({
            "segment": i + 1,
            "from_keyframe": scene_states[i]["file"],
            "to_keyframe": scene_states[i + 1]["file"],
            "instruction": instruction
        })

    return results


# ============================================================================
# Demo with Mock Data
# ============================================================================

def demo_with_mock_states():
    """
    Demonstrate the diffing logic with manually created states
    matching the actual keyframes we extracted.
    """
    print("=" * 60)
    print("Scene-State Diffing Demo")
    print("Task: Use cup to pour something granular")
    print("=" * 60)

    # Mock scene states based on what we observed in the keyframes
    mock_states = [
        {  # Keyframe 0: Initial state
            "objects": [
                {"name": "pink bowl", "location": "left side of table", "state": "empty"},
                {"name": "yellow cup", "location": "right side of table", "state": "contains small balls"}
            ],
            "gripper_state": "open",
            "gripper_position": "retracted, not visible"
        },
        {  # Keyframe 1: Approaching cup
            "objects": [
                {"name": "pink bowl", "location": "left side of table", "state": "empty"},
                {"name": "yellow cup", "location": "right side of table", "state": "contains small balls"}
            ],
            "gripper_state": "open",
            "gripper_position": "above yellow cup"
        },
        {  # Keyframe 2: Grasped cup
            "objects": [
                {"name": "pink bowl", "location": "left side of table", "state": "empty"},
                {"name": "yellow cup", "location": "in gripper", "state": "contains small balls"}
            ],
            "gripper_state": "holding yellow cup",
            "gripper_position": "above table"
        },
        {  # Keyframe 3: Repositioning
            "objects": [
                {"name": "pink bowl", "location": "left side of table", "state": "empty"},
                {"name": "yellow cup", "location": "in gripper", "state": "contains small balls"}
            ],
            "gripper_state": "holding yellow cup",
            "gripper_position": "above pink bowl"
        },
        {  # Keyframe 4: After pouring
            "objects": [
                {"name": "pink bowl", "location": "left side of table", "state": "contains small balls"},
                {"name": "yellow cup", "location": "in gripper", "state": "empty"}
            ],
            "gripper_state": "holding yellow cup",
            "gripper_position": "above pink bowl"
        },
        {  # Keyframe 5: Released cup, task complete
            "objects": [
                {"name": "pink bowl", "location": "left side of table", "state": "contains small balls"},
                {"name": "yellow cup", "location": "right side of table", "state": "empty"}
            ],
            "gripper_state": "open",
            "gripper_position": "retracted"
        }
    ]

    keyframe_files = [
        "keyframe_00_trajidx000.jpg",
        "keyframe_01_trajidx148.jpg",
        "keyframe_02_trajidx168.jpg",
        "keyframe_03_trajidx188.jpg",
        "keyframe_04_trajidx422.jpg",
        "keyframe_05_trajidx471.jpg"
    ]

    print("\n--- Generated Instructions ---\n")

    for i in range(len(mock_states) - 1):
        changes = diff_scene_states(mock_states[i], mock_states[i + 1])
        instruction = generate_instruction(changes)

        print(f"Segment {i + 1}: {keyframe_files[i]} → {keyframe_files[i + 1]}")
        print(f"  Instruction: {instruction}")
        print()

    print("--- Full Task Narration ---\n")
    all_instructions = []
    for i in range(len(mock_states) - 1):
        changes = diff_scene_states(mock_states[i], mock_states[i + 1])
        all_instructions.extend(changes)

    # Create flowing narrative
    narrative = ". ".join([instr.capitalize() for instr in all_instructions]) + "."
    print(f"'{narrative}'")


if __name__ == "__main__":
    demo_with_mock_states()
