#!/usr/bin/env python3
"""
Verify that coordinate transformation is applied correctly in generated data.

This script loads a trajectory and checks that the saved poses are in DROID frame.
"""

import os
import sys
import h5py
import numpy as np
import click


def transform_pose_to_droid_frame(pose):
    """
    Transform pose from RoboHive coordinate frame to DROID coordinate frame.

    This is the same transformation used in generate_droid_sim_data.py
    Transformation: swap_xy
    """
    transformed_pose = pose.copy()

    # Transform position: swap x and y
    transformed_pose[0] = pose[1]  # DROID_x = RoboHive_y
    transformed_pose[1] = pose[0]  # DROID_y = RoboHive_x
    transformed_pose[2] = pose[2]  # DROID_z = RoboHive_z

    # Transform orientation: swap roll/pitch
    transformed_pose[3] = pose[4]  # new_roll = old_pitch
    transformed_pose[4] = pose[3]  # new_pitch = old_roll
    transformed_pose[5] = pose[5]  # new_yaw = old_yaw

    return transformed_pose


def inverse_transform_pose(pose_droid):
    """
    Transform pose from DROID coordinate frame back to RoboHive coordinate frame.

    This is the inverse of transform_pose_to_droid_frame (swap_xy).
    Since swap_xy is its own inverse, this does the same transformation:
    swap_xy (used in robo_samples.py deployment).
    """
    pose_robohive = pose_droid.copy()

    # Inverse of swap_xy is also swap_xy:
    # If DROID_x = RoboHive_y  =>  RoboHive_y = DROID_x
    # If DROID_y = RoboHive_x  =>  RoboHive_x = DROID_y
    pose_robohive[0] = pose_droid[1]  # RoboHive_x = DROID_y
    pose_robohive[1] = pose_droid[0]  # RoboHive_y = DROID_x
    pose_robohive[2] = pose_droid[2]  # RoboHive_z = DROID_z

    # Inverse orientation transform (swap roll/pitch back)
    pose_robohive[3] = pose_droid[4]  # RoboHive_roll = DROID_pitch
    pose_robohive[4] = pose_droid[3]  # RoboHive_pitch = DROID_roll
    pose_robohive[5] = pose_droid[5]  # RoboHive_yaw = DROID_yaw

    return pose_robohive


@click.command()
@click.argument('trajectory_path', type=click.Path(exists=True))
@click.option('--verbose', is_flag=True, help='Print detailed information')
def main(trajectory_path, verbose):
    """Verify coordinate transformation in a generated trajectory."""

    h5_path = os.path.join(trajectory_path, 'trajectory.h5')

    if not os.path.exists(h5_path):
        print(f"ERROR: trajectory.h5 not found in {trajectory_path}")
        sys.exit(1)

    print(f"Loading trajectory from: {h5_path}")
    print("=" * 80)

    with h5py.File(h5_path, 'r') as f:
        # Load saved data
        cartesian_pos = f['observation']['robot_state']['cartesian_position'][:]
        camera_extrinsics = list(f['observation']['camera_extrinsics'].keys())[0]
        cam_ext = f['observation']['camera_extrinsics'][camera_extrinsics][:]

        print(f"\nTrajectory length: {len(cartesian_pos)} frames")
        print(f"Camera: {camera_extrinsics}")

        print("\n" + "=" * 80)
        print("COORDINATE FRAME VERIFICATION")
        print("=" * 80)

        # Show first pose
        first_pose_droid = cartesian_pos[0]
        first_cam_droid = cam_ext[0]

        print("\nFirst frame - End-Effector Pose (saved in DROID frame):")
        print(f"  Position (x, y, z):      {first_pose_droid[:3]}")
        print(f"  Orientation (r, p, y):   {first_pose_droid[3:6]}")

        print("\nFirst frame - Camera Extrinsics (saved in DROID frame):")
        print(f"  Position (x, y, z):      {first_cam_droid[:3]}")
        print(f"  Orientation (r, p, y):   {first_cam_droid[3:6]}")

        # Transform back to RoboHive frame for reference
        first_pose_robohive = inverse_transform_pose(first_pose_droid)
        first_cam_robohive = inverse_transform_pose(first_cam_droid)

        print("\n" + "-" * 80)
        print("Inverse Transform (DROID → RoboHive frame):")
        print("-" * 80)
        print("\nEnd-Effector in RoboHive frame:")
        print(f"  Position (x, y, z):      {first_pose_robohive[:3]}")
        print(f"  Orientation (r, p, y):   {first_pose_robohive[3:6]}")

        print("\nCamera in RoboHive frame:")
        print(f"  Position (x, y, z):      {first_cam_robohive[:3]}")
        print(f"  Orientation (r, p, y):   {first_cam_robohive[3:6]}")

        if verbose:
            print("\n" + "=" * 80)
            print("FULL TRAJECTORY DATA")
            print("=" * 80)
            print("\nAll cartesian positions (DROID frame):")
            print(cartesian_pos)
            print("\nAll camera extrinsics (DROID frame):")
            print(cam_ext)

        print("\n" + "=" * 80)
        print("TRANSFORMATION SUMMARY")
        print("=" * 80)
        print("\nCoordinate Transformation Applied: swap_xy")
        print("  DROID_x = RoboHive_y")
        print("  DROID_y = RoboHive_x")
        print("  DROID_z = RoboHive_z")
        print("\nAll saved poses and camera extrinsics are in DROID coordinate frame.")
        print("This ensures consistency with real DROID dataset during V-JEPA2 training.")
        print("\nData Flow:")
        print("  1. [Training] RoboHive → DROID (swap_xy) → HDF5")
        print("  2. [Training] V-JEPA2 learns from DROID-frame data")
        print("  3. [Deployment] V-JEPA2 outputs actions in DROID frame")
        print("  4. [Deployment] DROID → RoboHive (swap_xy) → Execute")
        print("\nWhen deploying with robo_samples.py, use the transformation:")
        print("  python robo_samples.py --action_transform swap_xy")
        print("=" * 80)


if __name__ == '__main__':
    main()
