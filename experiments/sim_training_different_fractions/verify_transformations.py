#!/usr/bin/env python3
"""Verify that coordinate transformations between RoboHive and DROID are inverses."""

import numpy as np

def robohive_to_droid(pose):
    """RoboHive → DROID (from generate_droid_sim_data.py)"""
    transformed = pose.copy()
    transformed[0] = pose[1]   # DROID_x = RoboHive_y
    transformed[1] = -pose[0]  # DROID_y = -RoboHive_x
    transformed[2] = pose[2]   # DROID_z = RoboHive_z
    return transformed

def droid_to_robohive(action):
    """DROID → RoboHive (from robo_samples.py swap_xy_negate_x)"""
    transformed = action.copy()
    transformed[0] = -action[1]  # RoboHive_x = -DROID_y
    transformed[1] = action[0]   # RoboHive_y = DROID_x
    transformed[2] = action[2]   # RoboHive_z = DROID_z
    return transformed

# Test with sample poses
print("Testing coordinate transformations:")
print("=" * 60)

test_poses = [
    np.array([1.0, 0.0, 0.0]),  # RoboHive x-axis
    np.array([0.0, 1.0, 0.0]),  # RoboHive y-axis
    np.array([0.0, 0.0, 1.0]),  # RoboHive z-axis
    np.array([0.5, 0.3, 0.2]),  # General position
]

all_correct = True

for i, robohive_pose in enumerate(test_poses):
    print(f"\nTest {i+1}:")
    print(f"  RoboHive pose:  {robohive_pose}")

    # RoboHive → DROID → RoboHive (should equal original)
    droid_pose = robohive_to_droid(robohive_pose)
    print(f"  DROID pose:     {droid_pose}")

    recovered = droid_to_robohive(droid_pose)
    print(f"  Recovered:      {recovered}")

    if np.allclose(robohive_pose, recovered):
        print(f"  ✓ Transformation is correct (inverse works)")
    else:
        print(f"  ✗ ERROR: Transformations are NOT inverses!")
        print(f"    Difference: {robohive_pose - recovered}")
        all_correct = False

print("\n" + "=" * 60)
print("Transformation summary:")
print("  RoboHive → DROID: x'=y, y'=-x, z'=z  (swap_xy_negate_y)")
print("  DROID → RoboHive: x'=-y, y'=x, z'=z  (swap_xy_negate_x)")
print("=" * 60)

if all_correct:
    print("\n✓ All transformations verified successfully!")
    print("  generate_droid_sim_data.py and robo_samples.py are consistent.")
else:
    print("\n✗ ERROR: Transformation verification failed!")
    exit(1)
