import threading
import time
from typing import Optional, Callable

import numpy as np
import ipywidgets as W
from IPython.display import display

import pybullet as p
import pybullet_data as pd
import cv2
import torch


class FrankaSim:
    def __init__(
        self,
        world_model,
        forward_target: Callable,           # function taking clips->[B, T*N, D]
        transform: Callable,                # image transform
        device,
        tokens_per_frame: int,
        use_egl: bool = True,
        auto_start: bool = True,
        sim_fps: int = 300,
        target_fps: int = 60,
        width: int = 720,
        height: int = 480,
    ):
        self.world_model = world_model
        self.forward_target = forward_target
        self.transform = transform
        self.device = device
        self.tokens_per_frame = tokens_per_frame

        # Config
        self.SIM_FPS = sim_fps
        self.TARGET_FPS = target_fps
        self.WIDTH, self.HEIGHT = width, height
        self.USE_EGL = use_egl
        self.AUTO_START_SIM = auto_start

        # State
        self.player = {"run": False, "thread": None}
        self.last_q = None
        self.base_pos = None
        self.base_rpy = None
        self.goal_rgb = None
        self.goal_pos = None
        self.goal_rpy = None
        self.goal_grip = None
        self.renderer = None

        # UI widgets
        self.w = {}

        # Constants
        self.EE_LINK, self.FINGER_L, self.FINGER_R = 11, 9, 10
        self.NUM_JOINTS = None
        self.ARM_JOINTS = list(range(7))
        self.IK_LL = None
        self.IK_UL = None
        self.IK_JR = None
        self.IK_DAMP = None
        self.TABLE_TOP_Z = None
        self.Z_SAFETY = 0.03
        self.Z_MIN = None
        self.Z_MAX = 1.8
        self.CUBE_SCALE = 1.3
        self.GRIPPER_ACTION_IS_DELTA = True

        # Bodies
        self.table_id = None
        self.panda = None
        self.cube = None

    # ---------- Public API ----------
    def launch(self):
        self._setup_sim()
        self._setup_widgets()
        self._wire_ui()
        self._first_frame()
        if self.AUTO_START_SIM:
            self.run_7d(None)

    def dispose(self):
        """Stop threads & leave sim in a clean state so a re-run starts fresh."""
        try:
            self.stop_7d(None)
            t = self.player.get("thread")
            if t and t.is_alive():
                t.join(timeout=0.5)
        except Exception:
            pass

    # ---------- Logging ----------
    def _log(self, msg: str):
        out = self.w.get("log_out")
        if not out:
            return
        with out:
            print(msg)

    # ---------- Setup ----------
    def _setup_sim(self):
        try:
            p.disconnect()
        except Exception:
            pass

        if self.USE_EGL:
            p.connect(p.DIRECT, options="--egl")
        else:
            p.connect(p.DIRECT)

        p.resetSimulation()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.SIM_FPS)

        # World
        p.loadURDF("plane.urdf", useFixedBase=True)
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], useFixedBase=True)

        # Robot
        self.panda = p.loadURDF("franka_panda/panda.urdf", [0.5, 0, 0.625], useFixedBase=True)

        # IK config
        self.NUM_JOINTS = p.getNumJoints(self.panda)
        self.IK_LL = [-2.9] * self.NUM_JOINTS
        self.IK_UL = [2.9] * self.NUM_JOINTS
        self.IK_JR = [5.8] * self.NUM_JOINTS
        self.IK_DAMP = [0.1] * self.NUM_JOINTS

        table_aabb_min, table_aabb_max = p.getAABB(self.table_id)
        self.TABLE_TOP_Z = float(table_aabb_max[2])
        self.Z_MIN = self.TABLE_TOP_Z + self.Z_SAFETY

        # Cube
        cube_xyz = [0.9, 0.00, 0.625 + 0.12]
        self.cube = p.loadURDF("cube_small.urdf", cube_xyz, globalScaling=self.CUBE_SCALE, useFixedBase=False)

        # Renderer
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL if self.USE_EGL else p.ER_TINY_RENDERER

        # Arm init
        q_home = np.deg2rad([0, 0, 0, 0, 0, 0, 0]).astype(float)
        for j in range(7):
            p.resetJointState(self.panda, j, float(q_home[j]))
            p.setJointMotorControl2(self.panda, j, p.POSITION_CONTROL, targetPosition=float(q_home[j]), force=800)

        self.last_q = self._get_arm_q()
        self.base_pos, self.base_rpy, _ = self._current_ee_pose()

    def _setup_widgets(self):
        # Camera
        cam_yaw   = W.FloatSlider(description="Yaw",   min=-180, max=180, step=0.2, value=0)
        cam_pitch = W.FloatSlider(description="Pitch", min=-89,  max=0,   step=0.2, value=0)
        cam_dist  = W.FloatSlider(description="Dist",  min=0.4,  max=3.0, step=0.05, value=0.8)
        cam_tx    = W.FloatSlider(description="TargetX", min=0.3, max=0.8, step=0.01, value=0.76)
        cam_ty    = W.FloatSlider(description="TargetY", min=-0.3, max=0.3, step=0.01, value=0.00)
        cam_tz    = W.FloatSlider(description="TargetZ", min=0.05, max=1.0, step=0.01, value=0.70)

        # Base-relative deltas (live)
        dx = W.FloatText(description="Δx (m)", value=0.00, step=0.005)
        dy = W.FloatText(description="Δy (m)", value=0.00, step=0.005)
        dz = W.FloatText(description="Δz (m)", value=0.00, step=0.005)
        droll  = W.FloatText(description="Δroll°",  value=0.0, step=1.0)
        dpitch = W.FloatText(description="Δpitch°", value=0.0, step=1.0)
        dyaw   = W.FloatText(description="Δyaw°",   value=0.0, step=1.0)
        grip_target = W.FloatSlider(description="Grip (m)", min=0.0, max=0.08, step=0.001, value=0.08)

        # Buttons
        set_goal_btn = W.Button(description="Set Goal Pose", button_style="primary")
        reset_btn    = W.Button(description="↺ Reset animation", button_style="info")
        snap_btn     = W.Button(description="Set Base = Current")
        clear_btn    = W.Button(description="Clear Δ", button_style="warning")
        respawn_btn  = W.Button(description="Respawn Cube", icon='refresh')
        remove_btn   = W.Button(description="Remove Cube", icon='trash')
        start_planning_btn = W.Button(description="Start Action Planning", button_style="primary")

        # Readouts
        goal_delta = W.HTML(value="<b>Δ to Goal:</b> —")
        goal_distance = W.HTML(value="<b>Distance to Goal:</b> —")

        # Images and log
        img = W.Image(format='jpeg')
        snap_img = W.Image(format='jpeg')
        img.layout = W.Layout(border='1px solid #444', width='100%')
        snap_img.layout = W.Layout(border='1px solid #444', width='100%')
        live_col = W.VBox([W.HTML("<b>Live</b>"), img], layout=W.Layout(width='50%'))
        snap_col = W.VBox([W.HTML("<b>Goal Snapshot</b>"), snap_img], layout=W.Layout(width='50%'))
        two_up = W.HBox([live_col, snap_col], layout=W.Layout(width='100%', gap='12px', align_items='flex-start'))
        log_out = W.Output(layout=W.Layout(border='1px solid #444', width='100%', max_height='140px', overflow='auto'))

        ui = W.VBox([
            W.HTML("<b>Camera</b>"),
            W.HBox([cam_yaw, cam_pitch, cam_dist]),
            W.HBox([cam_tx, cam_ty, cam_tz]),
            W.HTML("<b>Change end-effector state relative to origin</b>"),
            W.HBox([dx, dy, dz]),
            W.HBox([droll, dpitch, dyaw]),
            grip_target,
            W.HBox([set_goal_btn, reset_btn, snap_btn, clear_btn, respawn_btn, remove_btn, start_planning_btn]),
            goal_delta,
            goal_distance,
            W.HTML("<b>Log</b>"),
            log_out,
            two_up,
        ])
        display(ui)

        # Store
        self.w.update(dict(
            cam_yaw=cam_yaw, cam_pitch=cam_pitch, cam_dist=cam_dist, cam_tx=cam_tx, cam_ty=cam_ty, cam_tz=cam_tz,
            dx=dx, dy=dy, dz=dz, droll=droll, dpitch=dpitch, dyaw=dyaw, grip_target=grip_target,
            set_goal_btn=set_goal_btn, reset_btn=reset_btn, snap_btn=snap_btn, clear_btn=clear_btn,
            respawn_btn=respawn_btn, remove_btn=remove_btn, start_planning_btn=start_planning_btn,
            goal_delta=goal_delta, goal_distance=goal_distance, img=img, snap_img=snap_img, log_out=log_out
        ))

    def _wire_ui(self):
        w = self.w
        w["respawn_btn"].on_click(self._respawn_cube)
        w["remove_btn"].on_click(self._remove_cube)
        w["set_goal_btn"].on_click(self._goal_snapshot)
        w["reset_btn"].on_click(self._reset)
        w["snap_btn"].on_click(self._snap_base)
        w["clear_btn"].on_click(self._clear_delta)
        w["start_planning_btn"].on_click(self._start_planning)

        for k in ("cam_yaw", "cam_pitch", "cam_dist", "cam_tx", "cam_ty", "cam_tz"):
            w[k].observe(self._refresh_img, 'value')

    def _first_frame(self):
        self._snap_base()
        self._set_gripper(self.w["grip_target"].value)
        self._refresh_img()
        self._update_goal_delta()
        self._update_euclidean_distance_to_goal()

    # ---------- Helpers ----------
    def _get_arm_q(self):
        return np.array([p.getJointState(self.panda, j)[0] for j in range(7)], dtype=float)

    def _current_ee_pose(self):
        st = p.getLinkState(self.panda, self.EE_LINK, computeForwardKinematics=True)
        pos = np.array(st[4], dtype=float)
        rpy = np.array(p.getEulerFromQuaternion(st[5]), dtype=float)
        qL = p.getJointState(self.panda, self.FINGER_L)[0]
        qR = p.getJointState(self.panda, self.FINGER_R)[0]
        gripper_width = float(np.clip(qL + qR, 0.0, 0.08))
        return pos, rpy, gripper_width

    def _execute_action_relative(self, a, settle_steps=45):
        dx_a, dy_a, dz_a, droll_a, dpitch_a, dyaw_a, w_a = [float(x) for x in a[:7]]
        pos, rpy, gripper_width = self._current_ee_pose()

        tgt_pos = pos + np.array([dx_a, dy_a, dz_a], dtype=float)
        tgt_pos[2] = float(np.clip(tgt_pos[2], self.Z_MIN, self.Z_MAX))
        tgt_rpy = rpy + np.array([droll_a, dpitch_a, dyaw_a], dtype=float)
        self._set_arm_target(tgt_pos.tolist(), tuple(tgt_rpy))

        if self.GRIPPER_ACTION_IS_DELTA:
            tgt_width = float(np.clip(gripper_width + w_a, 0.0, 0.08))
        else:
            tgt_width = float(np.clip(w_a, 0.0, 0.08))
        self._set_gripper(tgt_width)
        self.w["grip_target"].value = tgt_width

        for _ in range(settle_steps):
            p.stepSimulation()

        self.base_pos, self.base_rpy, _ = self._current_ee_pose()
        self._update_goal_delta()
        self._update_euclidean_distance_to_goal()

    def _grab_frame_bytes(self, return_rgb=False):
        w = self.w
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[float(w["cam_tx"].value), float(w["cam_ty"].value), float(w["cam_tz"].value)],
            distance=float(w["cam_dist"].value),
            yaw=float(w["cam_yaw"].value),
            pitch=float(w["cam_pitch"].value),
            roll=0.0,
            upAxisIndex=2
        )
        proj = p.computeProjectionMatrixFOV(fov=60.0, aspect=float(self.WIDTH/self.HEIGHT), nearVal=0.01, farVal=10.0)
        img = p.getCameraImage(int(self.WIDTH), int(self.HEIGHT), view, proj, renderer=self.renderer, flags=p.ER_NO_SEGMENTATION_MASK)
        w_, h_, rgba = img[0], img[1], img[2]
        rgba = np.reshape(rgba, (h_, w_, 4))
        bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])

        if return_rgb:
            return buf.tobytes() if ok else None, rgba[:, :, 3::-1][:, :, :3] if False else rgba[:, :, :3]
        return buf.tobytes() if ok else None

    def _refresh_img(self, *_):
        frame = self._grab_frame_bytes()
        if frame is not None:
            self.w["img"].value = frame

    def _set_gripper(self, width, force=220):
        half = float(np.clip(width * 0.5, 0.0, 0.04))
        p.setJointMotorControl2(self.panda, self.FINGER_L, p.POSITION_CONTROL, targetPosition=half, force=force)
        p.setJointMotorControl2(self.panda, self.FINGER_R, p.POSITION_CONTROL, targetPosition=half, force=force)

    def _set_arm_target(self, xyz, rpy):
        orn = p.getQuaternionFromEuler(rpy)
        rest_all = [0.0] * self.NUM_JOINTS
        for idx, qv in zip(self.ARM_JOINTS, self.last_q):
            rest_all[idx] = float(qv)

        q = p.calculateInverseKinematics(
            self.panda, self.EE_LINK, xyz, orn,
            lowerLimits=self.IK_LL, upperLimits=self.IK_UL, jointRanges=self.IK_JR,
            restPoses=rest_all, jointDamping=self.IK_DAMP,
            residualThreshold=1e-4, maxNumIterations=200
        )

        self.last_q = np.array([q[j] for j in self.ARM_JOINTS], dtype=float)
        for j, qj in zip(self.ARM_JOINTS, self.last_q):
            p.setJointMotorControl2(self.panda, j, p.POSITION_CONTROL, targetPosition=float(qj), force=800)

    def _set_goal(self, _=None):
        frame, rgb = self._grab_frame_bytes(return_rgb=True)
        pos, rpy, gripper_width = self._current_ee_pose()
        self._log(f"Goal EE pos = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), rpy = ({np.rad2deg(rpy[0]):.1f}, {np.rad2deg(rpy[1]):.1f}, {np.rad2deg(rpy[2]):.1f}), grip = {gripper_width:.3f} m")
        self.goal_rgb = rgb
        self.goal_pos, self.goal_rpy, self.goal_grip = pos.copy(), rpy.copy(), float(gripper_width)
        self._update_goal_delta()
        self._update_euclidean_distance_to_goal()
        if frame is not None:
            self.w["snap_img"].value = frame
        return frame

    @staticmethod
    def _wrap_to_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def _update_goal_delta(self):
        if self.goal_pos is None:
            self.w["goal_delta"].value = "<b>Δ to Goal:</b> —"
            return
        pos, rpy, grip = self._current_ee_pose()
        dpos = self.goal_pos - pos
        drpy = self._wrap_to_pi(self.goal_rpy - rpy)
        dgrip = self.goal_grip - grip
        self.w["goal_delta"].value = (
            f"<b>Δ to Goal</b> "
            f"dx={dpos[0]:+0.3f} dy={dpos[1]:+0.3f} dz={dpos[2]:+0.3f} m | "
            f"droll={np.rad2deg(drpy[0]):+0.1f}° dpitch={np.rad2deg(drpy[1]):+0.1f}° dyaw={np.rad2deg(drpy[2]):+0.1f}° | "
            f"dgrip={dgrip:+0.3f} m"
        )

    def _update_euclidean_distance_to_goal(self):
        if self.goal_pos is None:
            self.w["goal_distance"].value = "<b>Distance to Goal:</b> —"
            return
        pos, _, _ = self._current_ee_pose()
        dpos = self.goal_pos - pos
        distance = np.linalg.norm(dpos)
        self.w["goal_distance"].value = f"<b>Distance to Goal:</b> {distance:.3f} m"

    # ---------- Cube ops ----------
    def _respawn_cube(self, _=None):
        try:
            p.removeBody(self.cube)
        except Exception:
            pass
        x = float(np.clip(0.60 + np.random.uniform(-0.05, 0.05), 0.45, 0.75))
        y = float(np.clip(0.00 + np.random.uniform(-0.05, 0.05), -0.20, 0.20))
        cube_xyz = [x + 0.3, y, self.TABLE_TOP_Z + 0.02]
        self.cube = p.loadURDF("cube_small.urdf", cube_xyz, globalScaling=self.CUBE_SCALE, useFixedBase=False)
        self._refresh_img()

    def _remove_cube(self, _=None):
        try:
            p.removeBody(self.cube)
        except Exception:
            pass
        self.cube = None
        self._refresh_img()

    # ---------- Live player ----------
    def _snap_base(self, _=None):
        self.base_pos, self.base_rpy, _ = self._current_ee_pose()
        self._update_goal_delta()
        self._update_euclidean_distance_to_goal()

    def _clear_delta(self, _=None):
        self.w["dx"].value = self.w["dy"].value = self.w["dz"].value = 0.0
        self.w["droll"].value = self.w["dpitch"].value = self.w["dyaw"].value = 0.0

    def _player_loop(self):
        i = 0
        steps_per_frame = max(1, int(self.SIM_FPS / self.TARGET_FPS))
        while self.player["run"]:
            dpos = np.array([self.w["dx"].value, self.w["dy"].value, self.w["dz"].value], dtype=float)
            drpy = np.deg2rad([self.w["droll"].value, self.w["dpitch"].value, self.w["dyaw"].value]).astype(float)

            tgt_pos = self.base_pos + dpos
            tgt_pos[2] = float(np.clip(tgt_pos[2], self.Z_MIN, self.Z_MAX))
            tgt_rpy = self.base_rpy + drpy

            self._set_arm_target(tgt_pos.tolist(), tuple(tgt_rpy))
            self._set_gripper(float(np.clip(self.w["grip_target"].value, 0.0, 0.08)))

            p.stepSimulation()
            if (i % steps_per_frame) == 0:
                self._refresh_img()
                self._update_goal_delta()
                self._update_euclidean_distance_to_goal()
            i += 1

    def run_7d(self, _=None):
        if self.player["run"]:
            return
        self._snap_base()
        self.last_q = self._get_arm_q()
        self.player["run"] = True
        t = threading.Thread(target=self._player_loop, daemon=True)
        self.player["thread"] = t
        t.start()

    def stop_7d(self, _=None):
        self.player["run"] = False
        self._refresh_img()

    def _reset(self, _=None):
        self.player["run"] = False
        t = self.player.get("thread")
        if t and t.is_alive():
            t.join(timeout=0.5)

        self._set_arm_target(self.base_pos.tolist(), tuple(self.base_rpy))
        for _ in range(30):
            p.stepSimulation()

        self._clear_delta()
        self._set_gripper(0.08)
        self._snap_base()
        self._refresh_img()
        self._update_goal_delta()
        self.run_7d(None)

    def _goal_snapshot(self, _=None):
        self._set_goal()

    # ---------- Planning ----------
    def _free_cuda_cache(self):
        for name in [
            'z_hat','s_hat','a_hat','loss',
            'plot_data','delta_x','delta_z','energy',
            'heatmap','xedges','yedges'
        ]:
            if name in globals():
                try:
                    del globals()[name]
                except Exception:
                    pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _start_planning(self, _=None):
        if self.goal_rgb is None:
            self._log("Error: No snapshot taken yet. Please take a snapshot first.")
            return

        self._free_cuda_cache()
        with torch.no_grad():
            _, rgb = self._grab_frame_bytes(return_rgb=True)
            combined_rgb = np.stack([rgb, self.goal_rgb], axis=0)  # [2, H, W, 3]
            clips = self.transform(combined_rgb).unsqueeze(0).to(self.device)

            pos, rpy, gripper_width = self._current_ee_pose()
            self._log(
                "Starting action planning from current pose: "
                + ", ".join([f"{v:.3f}" for v in pos]) + " m, "
                + ", ".join([f"{np.rad2deg(v):.1f}" for v in rpy]) + " deg, "
                + f"grip {gripper_width:.3f} m"
            )

            current_state = np.concatenate([pos, rpy, [gripper_width]], axis=0)
            states = torch.tensor(current_state, device=self.device).unsqueeze(0).unsqueeze(0)

            h = self.forward_target(clips)
            z_n   = h[:, :self.tokens_per_frame].contiguous().clone()
            z_goal = h[:, -self.tokens_per_frame:].contiguous().clone()
            self._free_cuda_cache()
            s_n = states[:, :1].to(dtype=z_n.dtype)

            start_time = time.time()
            actions = self.world_model.infer_next_action(z_n, s_n, z_goal).cpu().numpy()
            end_time = time.time()
            self._log(f"Planning time: {end_time - start_time:.3f} seconds")
            self._log(f"Planned action (x,y,z) = ({actions[0, 0]:.3f},{actions[0, 1]:.3f} {actions[0, 2]:.3f})")
            self._log(f"Planned action (pitch,yaw,roll,open) = ({actions[0, 3]:.2f},{actions[0, 4]:.2f} {actions[0, 5]:.2f} {actions[0, 6]:.2f})")

            self._execute_action_relative(actions[0])
            self._log("Action executed.")
            self._log("New position: " + ", ".join([f"{v:.3f}" for v in self._current_ee_pose()[0]]))