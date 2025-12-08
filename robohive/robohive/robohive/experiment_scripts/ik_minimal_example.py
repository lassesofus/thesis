#!/usr/bin/env python3
"""
Minimal IK Example

This script demonstrates the basic IK workflow in RoboHive with detailed output
to help understand how the solver works.

USAGE:
    python ik_minimal_example.py
"""

import os
import numpy as np

# Enable headless rendering
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose


def main():
    print("=" * 80)
    print("MINIMAL IK EXAMPLE")
    print("=" * 80)

    # Load simulation
    sim_path = '/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_reach_v0.xml'
    print(f"\n1. Loading simulation: {sim_path}")
    sim = SimScene.get_sim(model_handle=sim_path)

    # Setup
    EE_SITE = "end_effector"
    ARM_nJnt = 7
    ARM_JNT0 = np.array([
        -0.0321842, -0.394346, 0.00932319, -2.77917,
        -0.011826, 0.713889, 1.53183
    ])

    ee_sid = sim.model.site_name2id(EE_SITE)

    # Reset to starting configuration
    print("\n2. Setting starting configuration")
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt)
    sim.forward()

    start_ee_pos = sim.data.site_xpos[ee_sid].copy()
    print(f"   Starting joint angles: {ARM_JNT0}")
    print(f"   Starting EE position:  [{start_ee_pos[0]:.6f}, {start_ee_pos[1]:.6f}, {start_ee_pos[2]:.6f}]")

    # Define target
    print("\n3. Defining target position")
    target_offset = np.array([0.1, 0.0, 0.1])  # Move 10cm in x and z
    target_pos = start_ee_pos + target_offset
    target_distance = np.linalg.norm(target_offset)
    print(f"   Target position:       [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}]")
    print(f"   Target distance:       {target_distance:.6f}m")

    # Run IK
    print("\n4. Running IK solver")
    print("   Parameters:")
    print("     - tol: 1e-4")
    print("     - max_steps: 2000")
    print("     - regularization_strength: 1.0")

    ik_result = qpos_from_site_pose(
        physics=sim,
        site_name=EE_SITE,
        target_pos=target_pos,
        target_quat=None,
        inplace=False,
        tol=1e-4,
        max_steps=2000,
        regularization_strength=1.0
    )

    print(f"\n5. IK solver results:")
    print(f"   Success:               {ik_result.success}")
    print(f"   Iterations:            {ik_result.steps}")
    print(f"   Final error norm:      {ik_result.err_norm:.6e}")
    print(f"   Solution joint angles: {ik_result.qpos[:ARM_nJnt]}")

    # Apply solution and measure actual error
    print("\n6. Applying IK solution and measuring error")
    sim.data.qpos[:ARM_nJnt] = ik_result.qpos[:ARM_nJnt]
    sim.forward()
    final_ee_pos = sim.data.site_xpos[ee_sid].copy()

    absolute_error = np.linalg.norm(final_ee_pos - target_pos)
    relative_error = (absolute_error / target_distance) * 100

    print(f"   Final EE position:     [{final_ee_pos[0]:.6f}, {final_ee_pos[1]:.6f}, {final_ee_pos[2]:.6f}]")
    print(f"   Target position:       [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}]")
    print(f"   Error vector:          [{final_ee_pos[0]-target_pos[0]:.6f}, "
          f"{final_ee_pos[1]-target_pos[1]:.6f}, {final_ee_pos[2]-target_pos[2]:.6f}]")

    print(f"\n7. Error metrics:")
    print(f"   Absolute error:        {absolute_error:.6f}m = {absolute_error*1000:.3f}mm")
    print(f"   Relative error:        {relative_error:.3f}%")

    # Joint displacement
    joint_displacement = np.linalg.norm(ik_result.qpos[:ARM_nJnt] - ARM_JNT0)
    print(f"   Joint displacement:    {joint_displacement:.6f} rad")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(f"\nThe IK solver {'SUCCEEDED' if ik_result.success else 'FAILED'} in {ik_result.steps} iterations.")
    print(f"It achieved an absolute error of {absolute_error*1000:.3f}mm,")
    print(f"which is {relative_error:.3f}% of the target distance.")

    if absolute_error < 0.001:  # < 1mm
        print("\n✓ EXCELLENT: Sub-millimeter accuracy!")
    elif absolute_error < 0.005:  # < 5mm
        print("\n✓ GOOD: Acceptable error for most manipulation tasks.")
    elif absolute_error < 0.010:  # < 10mm
        print("\n⚠ FAIR: Marginal accuracy, may affect task performance.")
    else:
        print("\n✗ POOR: High error - check if target is reachable.")

    print("\nThe solver used damped least squares with regularization to iteratively")
    print("minimize the distance between the current and target end-effector positions.")
    print(f"Each iteration updated the joint angles based on the Jacobian matrix,")
    print(f"taking {ik_result.steps} steps to converge (or reach max iterations).")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
