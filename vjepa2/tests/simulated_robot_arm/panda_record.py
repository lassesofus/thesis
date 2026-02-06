import math, argparse
import numpy as np
import imageio.v2 as imageio
import pybullet as p
import pybullet_data as pd

EE_LINK = 11
FINGER_L, FINGER_R = 9, 10

# -------------------- Sim init --------------------
def init_sim(use_egl=False):
    p.connect(p.DIRECT)  # headless
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0,0,-9.81)
    p.setTimeStep(1/240.)  # 240 Hz

    renderer = p.ER_TINY_RENDERER
    egl_plugin = -1
    if use_egl:
        try:
            egl_plugin = p.loadPlugin(p.getPluginPath("eglRendererPlugin"))
            renderer = p.ER_BULLET_HARDWARE_OPENGL
            print("[info] Using EGL hardware renderer.")
        except Exception as e:
            print("[warn] EGL not available, falling back to TinyRenderer:", e)

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.5,0,-0.625], p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)
    panda = p.loadURDF("franka_panda/panda.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)

    # Hold the arm in a safe "home" pose so it doesn't sag into the table/cube
    home = [0.0, -0.6, 0.0, -2.4, 0.0, 1.8, 0.7]
    for j in range(7):
        p.resetJointState(panda, j, home[j])
        p.setJointMotorControl2(panda, j, p.POSITION_CONTROL, targetPosition=home[j], force=1000)
    # open gripper wide
    p.setJointMotorControl2(panda, FINGER_L, p.POSITION_CONTROL, targetPosition=0.04, force=160)
    p.setJointMotorControl2(panda, FINGER_R, p.POSITION_CONTROL, targetPosition=0.04, force=160)
    # Let motors engage for a short settle time
    for _ in range(120):
        p.stepSimulation()

    # Cube sized to fit gripper; placed on table (spawn after the arm is stable)
    cube_pos = [0.60, 0.00, 0.02 + 0.62]
    cube = p.loadURDF("cube_small.urdf", cube_pos, useFixedBase=False, globalScaling=1.5)

    # Friction tweaks help stable contacts
    for link in [-1, FINGER_L, FINGER_R]:
        p.changeDynamics(panda, link, lateralFriction=2.0, rollingFriction=0.001, spinningFriction=0.001, restitution=0.0)
    p.changeDynamics(cube, -1, lateralFriction=1.5, restitution=0.0)

    return panda, cube, cube_pos, renderer, egl_plugin

# -------------------- Control helpers --------------------
def set_gripper(panda, width=0.08, force=160):
    half = float(np.clip(width/2, 0.0, 0.04))
    p.setJointMotorControl2(panda, FINGER_L, p.POSITION_CONTROL, targetPosition=half, force=force)
    p.setJointMotorControl2(panda, FINGER_R, p.POSITION_CONTROL, targetPosition=half, force=force)

def set_arm_target(panda, xyz, rpy):
    orn = p.getQuaternionFromEuler(rpy)
    q = p.calculateInverseKinematics(
        panda, EE_LINK, xyz, orn,
        lowerLimits=[-2.9]*7, upperLimits=[2.9]*7, jointRanges=[5.8]*7,
        restPoses=[0]*7, residualThreshold=1e-4, maxNumIterations=200
    )
    for j in range(7):
        p.setJointMotorControl2(panda, j, p.POSITION_CONTROL, q[j], force=800)

def contact_both_fingers_with(panda, cube):
    c1 = p.getContactPoints(bodyA=panda, bodyB=cube, linkIndexA=FINGER_L)
    c2 = p.getContactPoints(bodyA=panda, bodyB=cube, linkIndexA=FINGER_R)
    return (len(c1) > 0) and (len(c2) > 0)

def weld_to_hand(panda, cube):
    hand = p.getLinkState(panda, EE_LINK, computeForwardKinematics=True)
    hand_pos, hand_orn = hand[4], hand[5]
    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube)
    inv_hand_pos, inv_hand_orn = p.invertTransform(hand_pos, hand_orn)
    parent_pos, parent_orn = p.multiplyTransforms(inv_hand_pos, inv_hand_orn, cube_pos, cube_orn)
    cid = p.createConstraint(
        parentBodyUniqueId=panda, parentLinkIndex=EE_LINK,
        childBodyUniqueId=cube, childLinkIndex=-1,
        jointType=p.JOINT_FIXED, jointAxis=[0,0,0],
        parentFramePosition=parent_pos, parentFrameOrientation=parent_orn,
        childFramePosition=[0,0,0], childFrameOrientation=[0,0,0,1]
    )
    p.changeConstraint(cid, maxForce=250)
    return cid

# -------------------- Rendering --------------------

def sim_steps(n):
    for _ in range(n):
        p.stepSimulation()

# Smoothly move to target while recording
def move_and_record(
    writer,
    panda,
    target_xyz,
    target_rpy,
    secs,
    fps,
    substeps_per_frame,
    view,
    proj,
    renderer,
    w,
    h,
    grip=None,
    cam_state=None,
    cam_step=None,
):
    if grip is not None:
        set_gripper(panda, grip)
    set_arm_target(panda, target_xyz, target_rpy)
    # pass camera args through so you can animate the camera during motion
    play_and_record(
        writer,
        secs,
        fps,
        substeps_per_frame,
        view,
        proj,
        renderer,
        w,
        h,
        cam_state=cam_state,
        cam_step=cam_step,
    )

# Spin 360° at a fixed position (yaw sweep), recording continuously
def spin_in_place(
    writer,
    panda,
    xyz,
    base_rpy,
    secs,
    fps,
    substeps_per_frame,
    view,
    proj,
    renderer,
    w,
    h,
    revolutions=1,
    cam_state=None,
    cam_step=None,
):
    frames = int(secs * fps)
    for i in range(frames):
        t = i / max(1, frames-1)
        yaw_offset = 2*math.pi*revolutions * t
        rpy = (base_rpy[0], base_rpy[1], base_rpy[2] + yaw_offset)
        set_arm_target(panda, xyz, rpy)
        for _ in range(substeps_per_frame):
            p.stepSimulation()
        # optional camera update per frame
        if callable(cam_step):
            cam_step(cam_state, i)
            view, proj = get_cam(w, h, **cam_state)
        writer.append_data(render_frame(view, proj, renderer, w, h))
    

def get_cam(w, h, target=(0.5, 0.0, 0.2), distance=1.2, yaw=45.0, pitch=-30.0, roll=0.0, fov=60.0, near=0.01, far=3.0):
    """Build view/projection for offscreen rendering."""
    aspect = float(w) / float(h)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,            # degrees
        pitch=pitch,        # degrees
        roll=roll,          # degrees
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=fov, aspect=aspect, nearVal=near, farVal=far
    )
    return view, proj

def render_frame(view, proj, renderer, w, h):
    """Render RGB frame."""
    _, _, px, _, _ = p.getCameraImage(
        width=w, height=h,
        viewMatrix=view, projectionMatrix=proj,
        renderer=renderer,
        shadow=1,
        lightDirection=[-1, -1, -1],
        lightAmbientCoeff=0.6,
        lightDiffuseCoeff=0.8,
        lightSpecularCoeff=0.1,
    )
    return px[:, :, :3]  # drop alpha

def play_and_record(writer, secs, fps, substeps_per_frame, view, proj, renderer, w, h, cam_state=None, cam_step=None):
    """
    Advance physics and record frames. Optionally update camera each frame.
    cam_state: dict of camera params (target, distance, yaw, pitch, roll, fov, near, far)
    cam_step: callable(state, frame_idx) -> None to mutate cam_state per frame
    """
    n_frames = int(secs * fps)
    # Default camera state if not provided
    if cam_state is None:
        cam_state = dict(target=(0.5, 0.0, 0.2), distance=1.2, yaw=45.0, pitch=-30.0, roll=0.0, fov=60.0, near=0.01, far=3.0)

    for i in range(n_frames):
        # physics substeps
        for _ in range(substeps_per_frame):
            p.stepSimulation()
        # optional camera update
        if callable(cam_step):
            cam_step(cam_state, i)
            view, proj = get_cam(w, h, **cam_state)
        # render and write
        frame = render_frame(view, proj, renderer, w, h)
        writer.append_data(frame)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="panda_spin_pick.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--egl", action="store_true")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--crf", default="23")
    args = ap.parse_args()

    panda, cube, cube_pos, renderer, egl_plugin = init_sim(use_egl=args.egl)
    view, proj = get_cam(args.width, args.height)
    substeps_per_frame = max(1, int((1/args.fps) / (1/240.)))  # ~8 for 30 FPS

    writer = imageio.get_writer(
        args.out, fps=args.fps, codec="libx264",
        ffmpeg_params=["-pix_fmt","yuv420p","-preset",args.preset,"-crf",str(args.crf)]
    )

    # Baseline orientation (pointing down)
    down_rpy = (math.pi, 0.0, 0.0)

    # 1) Home & open
    move_and_record(writer, panda, [0.50, 0.00, 0.50], down_rpy, secs=2.0, fps=args.fps,
                    substeps_per_frame=substeps_per_frame, view=view, proj=proj, renderer=renderer, w=args.width, h=args.height, grip=0.08)

    # 2) Go above cube (pre-grasp)
    cube_world_pos, _ = p.getBasePositionAndOrientation(cube)
    # Approximate cube size (cube_small ~0.05m; scaled 1.5 => ~0.075m)
    cube_side = 0.05 * 1.5
    half_side = cube_side / 2.0
    pre_grasp_z = cube_world_pos[2] + half_side + 0.045  # ~4.5cm above top
    grasp_z = cube_world_pos[2] + half_side + 0.010      # ~1cm above top to contact sides

    above = [cube_world_pos[0], cube_world_pos[1], pre_grasp_z]
    move_and_record(
        writer,
        panda,
        above,
        down_rpy,
        secs=1.6,
        fps=args.fps,
        substeps_per_frame=substeps_per_frame,
        view=view,
        proj=proj,
        renderer=renderer,
        w=args.width,
        h=args.height,
        grip=0.06,  # approach with wider opening than cube
    )

    # 3) (Removed camera revolve for static camera before grasp)

    # 4) Descend to grasp height (just above cube top)
    at = [cube_world_pos[0], cube_world_pos[1], grasp_z]
    move_and_record(
        writer,
        panda,
        at,
        down_rpy,
        secs=1.2,
        fps=args.fps,
        substeps_per_frame=substeps_per_frame,
        view=view,
        proj=proj,
        renderer=renderer,
        w=args.width,
        h=args.height,
        grip=0.06,
    )

    # 5) Close gripper to grasp and wait; then weld if both fingers touch
    set_gripper(panda, 0.03)  # close to slightly smaller than cube
    play_and_record(
        writer,
        secs=0.8,
        fps=args.fps,
        substeps_per_frame=substeps_per_frame,
        view=view,
        proj=proj,
        renderer=renderer,
        w=args.width,
        h=args.height,
    )

    weld_id = None
    if contact_both_fingers_with(panda, cube):
        weld_id = weld_to_hand(panda, cube)

    # 6) Lift and move to place
    move_and_record(writer, panda, [cube_pos[0], cube_pos[1], 0.30], down_rpy, secs=1.5, fps=args.fps,
                    substeps_per_frame=substeps_per_frame, view=view, proj=proj, renderer=renderer, w=args.width, h=args.height, grip=0.028)
    move_and_record(writer, panda, [0.55, -0.15, 0.30], down_rpy, secs=2.0, fps=args.fps,
                    substeps_per_frame=substeps_per_frame, view=view, proj=proj, renderer=renderer, w=args.width, h=args.height, grip=0.028)

    # 7) Release
    if weld_id is not None:
        p.removeConstraint(weld_id)
    set_gripper(panda, 0.08)
    play_and_record(writer, secs=1.2, fps=args.fps, substeps_per_frame=substeps_per_frame, view=view, proj=proj, renderer=renderer, w=args.width, h=args.height)

    writer.close()
    if egl_plugin >= 0:
        p.unloadPlugin(egl_plugin)
    p.disconnect()
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()
