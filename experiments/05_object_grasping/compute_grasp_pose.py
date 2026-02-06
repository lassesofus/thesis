#!/usr/bin/env python3
"""
Compute grasp pose for the sugar box and generate goal image.

This script:
1. Loads the grasp scene with robotiq gripper
2. Moves from home position to pre-grasp (above box)
3. Lowers to grasp position
4. Closes gripper
5. Records video of the full motion
"""

import os
import sys

# Set up rendering BEFORE importing mujoco
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
from PIL import Image
import skvideo.io

# Add robohive to path
sys.path.insert(0, '/home/s185927/thesis/robohive/robohive/robohive')

from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.xml_utils import reassign_parent
from robohive.utils.min_jerk import generate_joint_space_min_jerk

# Constants
ARM_nJnt = 7
EE_SITE = "end_effector"

# Scene paths
SIM_PATH = '/home/s185927/thesis/robohive/robohive/robohive/envs/arms/franka/assets/franka_grasp_robotiq_v0.xml'
OUT_DIR = '/home/s185927/thesis/experiments/05_object_grasping'

# Tomato soup can geometry (from scene XML)
# Cylinder: size=".033 .05" means radius=0.033m, half-height=0.05m
# pos="0 0 .05" means center is 5cm above body origin
CAN_BODY_POS = np.array([0, 0.475, 0.78])  # Body position in world
CAN_COL_OFFSET = np.array([0, 0, 0.05])    # Collision cylinder center offset
CAN_RADIUS = 0.033  # 3.3cm radius = 6.6cm diameter
CAN_HALF_HEIGHT = 0.05  # 5cm half-height = 10cm tall

# Can center in world coordinates
CAN_CENTER = CAN_BODY_POS + CAN_COL_OFFSET
print(f"Can center: {CAN_CENTER}")
print(f"Can dimensions: diameter={CAN_RADIUS*2:.3f}m, height={CAN_HALF_HEIGHT*2:.3f}m")
print(f"Can top: {CAN_CENTER[2] + CAN_HALF_HEIGHT:.3f} m")
print(f"Can bottom: {CAN_CENTER[2] - CAN_HALF_HEIGHT:.3f} m")

# Grasp parameters
GRASP_HEIGHT_OFFSET = 0.02  # EE height above can center for grasp (lowered for better grip)
PRE_GRASP_HEIGHT = 0.20     # EE height above can center for pre-grasp

GRASP_TARGET_POS = np.array([
    CAN_CENTER[0],
    CAN_CENTER[1],
    CAN_CENTER[2] + GRASP_HEIGHT_OFFSET
])

PRE_GRASP_POS = np.array([
    CAN_CENTER[0],
    CAN_CENTER[1],
    CAN_CENTER[2] + PRE_GRASP_HEIGHT
])

LIFT_HEIGHT = 0.30  # How high to lift above can center
LIFT_POS = np.array([
    CAN_CENTER[0],
    CAN_CENTER[1],
    CAN_CENTER[2] + LIFT_HEIGHT
])

print(f"\nPre-grasp position (EE): {PRE_GRASP_POS}")
print(f"Grasp target position (EE): {GRASP_TARGET_POS}")
print(f"Lift position (EE): {LIFT_POS}")


def main():
    # Load simulation
    print("\nLoading simulation...")
    sim = SimScene.get_sim(model_handle=SIM_PATH)

    # Reparent robotiq gripper to Franka arm
    print("Attaching RobotiQ gripper...")
    raw_xml = sim.model.get_xml()
    processed_xml = reassign_parent(
        xml_str=raw_xml,
        receiver_node="panda0_link7",
        donor_node="ee_mount"
    )
    processed_path = os.path.join(os.path.dirname(SIM_PATH), '_grasp_processed.xml')
    with open(processed_path, 'w') as f:
        f.write(processed_xml)
    sim = SimScene.get_sim(model_handle=processed_path)
    os.remove(processed_path)
    print("RobotiQ gripper attached.")

    # Get site IDs
    ee_sid = sim.model.site_name2id(EE_SITE)
    step_dt = sim.model.opt.timestep

    # Initial arm configuration (home position)
    ARM_JNT0 = np.array([
        -0.0321842, -0.394346, 0.00932319, -2.77917,
        -0.011826, 0.713889, 0.74663
    ])

    # Set initial position with gripper open
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.data.qvel[:] = 0
    sim.forward()

    initial_ee_pos = sim.data.site_xpos[ee_sid].copy()
    print(f"\nInitial EE position (home): {initial_ee_pos}")

    # Find gripper joint indices
    gripper_joint_names = [
        'robotiq_2f_85_right_driver_joint',
        'robotiq_2f_85_left_driver_joint',
    ]
    follower_joint_names = [
        'robotiq_2f_85_right_follower_joint',
        'robotiq_2f_85_left_follower_joint',
        'robotiq_2f_85_right_spring_link_joint',
        'robotiq_2f_85_left_spring_link_joint',
    ]

    gripper_qpos_indices = []
    for jname in gripper_joint_names:
        try:
            jid = sim.model.joint_name2id(jname)
            gripper_qpos_indices.append(sim.model.jnt_qposadr[jid])
        except:
            pass

    follower_qpos_indices = []
    for jname in follower_joint_names:
        try:
            jid = sim.model.joint_name2id(jname)
            follower_qpos_indices.append(sim.model.jnt_qposadr[jid])
        except:
            pass

    # Gripper values for tomato soup can (6.6cm diameter)
    # Grip harder to prevent slipping
    open_val = 0.0
    closed_val = 0.38  # Tighter grip
    open_follower = 0.0
    closed_follower = -0.28

    # Set gripper to open
    for idx in gripper_qpos_indices:
        sim.data.qpos[idx] = open_val
    for idx in follower_qpos_indices:
        sim.data.qpos[idx] = open_follower
    sim.forward()

    # Compute IK for pre-grasp position
    print(f"\nComputing IK for pre-grasp position...")
    ik_pre = qpos_from_site_pose(
        physics=sim,
        site_name=EE_SITE,
        target_pos=PRE_GRASP_POS,
        target_quat=None,
        inplace=False,
        regularization_strength=1.0,
        max_steps=2000,
        tol=1e-4
    )
    pre_grasp_qpos = ik_pre.qpos[:ARM_nJnt]

    # Compute IK for grasp position
    print(f"Computing IK for grasp position...")
    sim.data.qpos[:ARM_nJnt] = pre_grasp_qpos  # Start from pre-grasp for better IK
    sim.forward()
    ik_grasp = qpos_from_site_pose(
        physics=sim,
        site_name=EE_SITE,
        target_pos=GRASP_TARGET_POS,
        target_quat=None,
        inplace=False,
        regularization_strength=1.0,
        max_steps=2000,
        tol=1e-4
    )
    grasp_qpos = ik_grasp.qpos[:ARM_nJnt]

    # Compute IK for lift position
    print(f"Computing IK for lift position...")
    ik_lift = qpos_from_site_pose(
        physics=sim,
        site_name=EE_SITE,
        target_pos=LIFT_POS,
        target_quat=None,
        inplace=False,
        regularization_strength=1.0,
        max_steps=2000,
        tol=1e-4
    )
    lift_qpos = ik_lift.qpos[:ARM_nJnt]

    # Reset to home
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    for idx in gripper_qpos_indices:
        sim.data.qpos[idx] = open_val
    for idx in follower_qpos_indices:
        sim.data.qpos[idx] = open_follower
    sim.forward()

    # Video recording setup
    frames = []
    cam_name = 'front_cam'

    def capture_frame():
        rgb = sim.renderer.render_offscreen(width=640, height=480, camera_id=cam_name, device_id=0)
        if rgb is not None:
            frames.append(rgb)

    def set_gripper(driver_val, follower_val):
        for idx in gripper_qpos_indices:
            sim.data.qpos[idx] = driver_val
        for idx in follower_qpos_indices:
            sim.data.qpos[idx] = follower_val

    # Generate trajectories (slower movements)
    print("\nGenerating trajectories...")
    traj_home_to_pre = generate_joint_space_min_jerk(
        start=ARM_JNT0,
        goal=pre_grasp_qpos,
        time_to_go=4.0,  # Slower
        dt=step_dt
    )
    traj_pre_to_grasp = generate_joint_space_min_jerk(
        start=pre_grasp_qpos,
        goal=grasp_qpos,
        time_to_go=3.0,  # Slower
        dt=step_dt
    )
    traj_grasp_to_lift = generate_joint_space_min_jerk(
        start=grasp_qpos,
        goal=lift_qpos,
        time_to_go=3.0,  # Slow lift
        dt=step_dt
    )

    print(f"  Home to pre-grasp: {len(traj_home_to_pre)} steps")
    print(f"  Pre-grasp to grasp: {len(traj_pre_to_grasp)} steps")
    print(f"  Grasp to lift: {len(traj_grasp_to_lift)} steps")

    # Phase 1: Move from home to pre-grasp (gripper open)
    print("\nPhase 1: Moving to pre-grasp position...")
    capture_frame()
    for i, wp in enumerate(traj_home_to_pre):
        sim.data.ctrl[:ARM_nJnt] = wp['position']
        set_gripper(open_val, open_follower)
        sim.advance(render=False)
        if i % 10 == 0:  # Capture every 10th step for smooth video
            capture_frame()

    # Hold at pre-grasp briefly
    for i in range(100):
        sim.data.ctrl[:ARM_nJnt] = pre_grasp_qpos
        set_gripper(open_val, open_follower)
        sim.advance(render=False)
        if i % 10 == 0:
            capture_frame()

    # Phase 2: Lower to grasp position (gripper open)
    print("Phase 2: Lowering to grasp position...")
    for i, wp in enumerate(traj_pre_to_grasp):
        sim.data.ctrl[:ARM_nJnt] = wp['position']
        set_gripper(open_val, open_follower)
        sim.advance(render=False)
        if i % 10 == 0:
            capture_frame()

    # Hold at grasp position briefly
    for i in range(100):
        sim.data.ctrl[:ARM_nJnt] = grasp_qpos
        set_gripper(open_val, open_follower)
        sim.advance(render=False)
        if i % 10 == 0:
            capture_frame()

    # Phase 3: Close gripper very gradually to avoid ejecting box
    print("Phase 3: Closing gripper...")
    num_close_steps = 500  # Very slow closing to prevent clipping
    for i in range(num_close_steps):
        t = i / num_close_steps
        current_driver = open_val + t * (closed_val - open_val)
        current_follower = open_follower + t * (closed_follower - open_follower)

        sim.data.ctrl[:ARM_nJnt] = grasp_qpos
        set_gripper(current_driver, current_follower)
        sim.advance(render=False)
        if i % 5 == 0:
            capture_frame()

    # Phase 4: Hold grasp briefly before lift
    print("Phase 4: Holding grasp...")
    for i in range(50):
        sim.data.ctrl[:ARM_nJnt] = grasp_qpos
        set_gripper(closed_val, closed_follower)
        sim.advance(render=False)
        if i % 5 == 0:
            capture_frame()

    # Phase 5: Lift the box
    print("Phase 5: Lifting box...")
    for i, wp in enumerate(traj_grasp_to_lift):
        sim.data.ctrl[:ARM_nJnt] = wp['position']
        set_gripper(closed_val, closed_follower)
        sim.advance(render=False)
        if i % 10 == 0:
            capture_frame()

    # Hold at lifted position
    print("Phase 6: Holding lifted position...")
    for i in range(150):
        sim.data.ctrl[:ARM_nJnt] = lift_qpos
        set_gripper(closed_val, closed_follower)
        sim.advance(render=False)
        if i % 5 == 0:
            capture_frame()

    # Save video
    if frames:
        video_path = os.path.join(OUT_DIR, 'grasp_simulation.mp4')
        frames_array = np.array(frames, dtype=np.uint8)
        skvideo.io.vwrite(video_path, frames_array, outputdict={"-pix_fmt": "yuv420p", "-r": "30"})
        print(f"\nSaved video: {video_path} ({len(frames)} frames)")

    # Verify final position
    sim.forward()
    final_ee_pos = sim.data.site_xpos[ee_sid].copy()
    print(f"\nFinal EE position: {final_ee_pos}")

    # Save grasp configuration
    grasp_config = {
        'arm_qpos': sim.data.qpos[:ARM_nJnt].copy(),
        'full_qpos': sim.data.qpos.copy(),
        'ee_position': final_ee_pos.copy(),
        'target_position': GRASP_TARGET_POS.copy(),
        'can_center': CAN_CENTER.copy(),
        'gripper_closed_val': closed_val,
        'gripper_follower_val': closed_follower,
    }

    config_path = os.path.join(OUT_DIR, 'grasp_config.npz')
    np.savez(config_path, **grasp_config)
    print(f"Saved grasp configuration to: {config_path}")

    # Render goal images
    print("\nRendering goal images...")
    cameras = ['left_cam', 'front_cam', 'top_cam', 'right_cam']

    for cam in cameras:
        try:
            rgb = sim.renderer.render_offscreen(width=640, height=480, camera_id=cam, device_id=0)
            if rgb is not None:
                img_path = os.path.join(OUT_DIR, f'goal_grasp_{cam}.png')
                Image.fromarray(rgb).save(img_path)
                print(f"  Saved: {img_path}")

                rgb_256 = np.array(Image.fromarray(rgb).resize((256, 256), Image.LANCZOS))
                img_path_256 = os.path.join(OUT_DIR, f'goal_grasp_{cam}_256.png')
                Image.fromarray(rgb_256).save(img_path_256)
        except Exception as e:
            print(f"  Could not render {cam}: {e}")

    print("\n" + "="*60)
    print("GRASP CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Arm joint positions: {list(grasp_config['arm_qpos'])}")
    print(f"EE target: {GRASP_TARGET_POS}")
    print(f"EE actual: {final_ee_pos}")
    print(f"Gripper: driver={closed_val}, follower={closed_follower}")
    print("="*60)

    return grasp_config


if __name__ == '__main__':
    main()
