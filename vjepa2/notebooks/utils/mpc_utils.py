# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import pdb

from src.utils.logging import get_logger

logger = get_logger(__name__, force=True)

# Optimization flags
USE_GPU_POSE = os.environ.get('OPT_GPU_POSE', '0') == '1' or os.environ.get('VJEPA_OPTIMIZE', '0') == '1'


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X", "Y", or "Z"
        angle: Tensor of shape (...) with Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3)
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R = torch.stack([
            torch.stack([one, zero, zero], dim=-1),
            torch.stack([zero, cos, -sin], dim=-1),
            torch.stack([zero, sin, cos], dim=-1),
        ], dim=-2)
    elif axis == "Y":
        R = torch.stack([
            torch.stack([cos, zero, sin], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-sin, zero, cos], dim=-1),
        ], dim=-2)
    elif axis == "Z":
        R = torch.stack([
            torch.stack([cos, -sin, zero], dim=-1),
            torch.stack([sin, cos, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ], dim=-2)
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return R


def euler_angles_to_matrix_gpu(euler_angles: torch.Tensor, convention: str = "xyz") -> torch.Tensor:
    """
    Convert Euler angles to rotation matrices.

    Args:
        euler_angles: Tensor of shape (..., 3) with Euler angles in radians
        convention: String of 3 characters specifying the convention.
                   Lowercase ("xyz") = extrinsic rotations (scipy default)
                   Uppercase ("XYZ") = intrinsic rotations

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3)
    """
    if len(convention) != 3:
        raise ValueError(f"Convention must have 3 characters, got {len(convention)}")

    # Check if extrinsic (lowercase) or intrinsic (uppercase)
    is_extrinsic = convention[0].islower()

    matrices = [
        _axis_angle_rotation(c.upper(), euler_angles[..., i])
        for i, c in enumerate(convention)
    ]

    if is_extrinsic:
        # For extrinsic rotations (lowercase), we multiply left to right: R = R_2 @ R_1 @ R_0
        return torch.matmul(torch.matmul(matrices[2], matrices[1]), matrices[0])
    else:
        # For intrinsic rotations (uppercase), we multiply right to left: R = R_0 @ R_1 @ R_2
        return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def matrix_to_euler_angles_gpu(R: torch.Tensor, convention: str = "xyz") -> torch.Tensor:
    """
    Convert rotation matrices to Euler angles.

    Args:
        R: Rotation matrices as tensor of shape (..., 3, 3)
        convention: String of 3 characters specifying the convention.
                   Lowercase ("xyz") = extrinsic rotations (scipy default)
                   Uppercase ("XYZ") = intrinsic rotations

    Returns:
        Euler angles as tensor of shape (..., 3) in radians
    """
    is_extrinsic = convention[0].islower()

    if is_extrinsic and convention.lower() == "xyz":
        # For extrinsic XYZ (scipy default):
        # R = Rz @ Ry @ Rx
        # R[0,0] = cz*cy,  R[1,0] = sz*cy,  R[2,0] = -sy
        # R[2,1] = cy*sx,  R[2,2] = cy*cx
        cy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
        singular = cy < 1e-6

        # Non-singular case
        x = torch.atan2(R[..., 2, 1], R[..., 2, 2])
        y = torch.atan2(-R[..., 2, 0], cy)
        z = torch.atan2(R[..., 1, 0], R[..., 0, 0])

        # Singular case (gimbal lock at y = ±90°)
        # When cy ≈ 0, we can only determine x+z or x-z
        x_singular = torch.atan2(-R[..., 1, 2], R[..., 1, 1])
        y_singular = torch.atan2(-R[..., 2, 0], cy)
        z_singular = torch.zeros_like(x)

        x = torch.where(singular, x_singular, x)
        y = torch.where(singular, y_singular, y)
        z = torch.where(singular, z_singular, z)

        return torch.stack([x, y, z], dim=-1)

    elif not is_extrinsic and convention.upper() == "XYZ":
        # For intrinsic XYZ:
        # R = Rx @ Ry @ Rz
        sy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
        singular = sy < 1e-6

        # Non-singular case
        x = torch.atan2(R[..., 2, 1], R[..., 2, 2])
        y = torch.atan2(-R[..., 2, 0], sy)
        z = torch.atan2(R[..., 1, 0], R[..., 0, 0])

        # Singular case (gimbal lock)
        x_singular = torch.atan2(-R[..., 1, 2], R[..., 1, 1])
        y_singular = torch.atan2(-R[..., 2, 0], sy)
        z_singular = torch.zeros_like(x)

        x = torch.where(singular, x_singular, x)
        y = torch.where(singular, y_singular, y)
        z = torch.where(singular, z_singular, z)

        return torch.stack([x, y, z], dim=-1)

    else:
        raise NotImplementedError(f"Convention {convention} is not currently supported")


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)


def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor


def cem(
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    rollout=1,
    cem_steps=100,
    momentum_mean=0.25,
    momentum_std=0.95,
    momentum_mean_gripper=0.15,
    momentum_std_gripper=0.15,
    samples=100,
    topk=10,
    verbose=False,
    maxnorm=0.05,
    axis={},
    objective=l1,
    close_gripper=None,
):
    """
    :param context_frame: [B=1, T=1, HW, D]
    :param goal_frame: [B=1, T=1, HW, D]
    :param world_model: f(context_frame, action) -> next_frame [B, 1, HW, D]
    :return: [B=1, rollout, 7] an action trajectory over rollout horizon

    Cross-Entropy Method
    -----------------------
    1. for rollout horizon:
    1.1. sample several actions
    1.2. compute next states using WM
    3. compute similarity of final states to goal_frames
    4. select topk samples and update mean and std using topk action trajs
    5. choose final action to be mean of distribution
    """
    context_frame = context_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    goal_frame = goal_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    context_pose = context_pose.repeat(samples, 1, 1)  # Reshape to [S, 1, 7]

    # Current estimate of the mean/std of distribution over action trajectories
    mean = torch.cat(
        [
            torch.zeros((rollout, 3), device=context_frame.device),
            torch.zeros((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    std = torch.cat(
        [
            torch.ones((rollout, 3), device=context_frame.device) * maxnorm,
            torch.ones((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    def sample_action_traj():
        """Sample several action trajectories"""
        action_traj, frame_traj, pose_traj = None, context_frame, context_pose

        for h in range(rollout):

            # -- sample new action
            action_samples = torch.randn(samples, mean.size(1), device=mean.device) * std[h] + mean[h]
            action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm)
            action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=-0.75, max=0.75)
            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]
            action_samples = torch.cat(
                [
                    action_samples[:, :3],
                    torch.zeros((len(action_samples), 3), device=mean.device),
                    action_samples[:, -1:],
                ],
                dim=-1,
            )[:, None]
            if close_gripper is not None and h >= close_gripper:
                action_samples[:, :, -1] = 1.0

            action_traj = (
                torch.cat([action_traj, action_samples], dim=1) if action_traj is not None else action_samples
            )

            # -- compute next state
            next_frame, next_pose = world_model(frame_traj, action_traj, pose_traj)
            frame_traj = torch.cat([frame_traj, next_frame], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)

        return action_traj, frame_traj

    def select_topk_action_traj(final_state, goal_state, actions):
        """Get the topk action trajectories that bring us closest to goal"""
        sims = objective(final_state.flatten(1), goal_state.flatten(1))
        indices = sims.topk(topk, largest=False).indices
        selected_actions = actions[indices]
        return selected_actions

    for step in tqdm(range(cem_steps), disable=True):
        action_traj, frame_traj = sample_action_traj()
        selected_actions = select_topk_action_traj(
            final_state=frame_traj[:, -1], goal_state=goal_frame, actions=action_traj
        )
        mean_selected_actions = selected_actions.mean(dim=0)
        std_selected_actions = selected_actions.std(dim=0)

        # -- Update new sampling mean and std based on the top-k samples
        mean = torch.cat(
            [
                mean_selected_actions[..., :3] * (1.0 - momentum_mean) + mean[..., :3] * momentum_mean,
                mean_selected_actions[..., -1:] * (1.0 - momentum_mean_gripper)
                + mean[..., -1:] * momentum_mean_gripper,
            ],
            dim=-1,
        )
        std = torch.cat(
            [
                std_selected_actions[..., :3] * (1.0 - momentum_std) + std[..., :3] * momentum_std,
                std_selected_actions[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
            ],
            dim=-1,
        )

        logger.info(f"new mean: {mean.sum(dim=0)} {std.sum(dim=0)}")

    new_action = torch.cat(
        [
            mean[..., :3],
            torch.zeros((rollout, 3), device=mean.device),
            round_small_elements(mean[..., -1:], 0.25),
        ],
        dim=-1,
    )[None, :]

    return new_action


def compute_new_pose(pose, action):
    """
    :param pose: [B, T=1, 7]
    :param action: [B, T=1, 7]
    :returns: [B, T=1, 7]
    """
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()
    # -- compute delta xyz
    new_xyz = pose[:, :3] + action[:, :3]
    # -- compute delta theta
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]
    # -- compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    # -- new pose
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]


def compute_new_pose_gpu(pose, action):
    """
    GPU-only pose computation - no CPU transfers.
    Uses pure PyTorch for batched rotation matrix operations.

    :param pose: [B, T=1, 7] on GPU
    :param action: [B, T=1, 7] on GPU
    :returns: [B, T=1, 7] on GPU
    """
    # Remove temporal dimension: [B, 1, 7] -> [B, 7]
    pose = pose[:, 0]
    action = action[:, 0]

    # Position: simple addition
    new_xyz = pose[:, :3] + action[:, :3]

    # Rotation: compose using rotation matrices on GPU
    # Using lowercase "xyz" for extrinsic rotations (matching scipy's default)
    R_current = euler_angles_to_matrix_gpu(pose[:, 3:6], "xyz")      # [B, 3, 3]
    R_delta = euler_angles_to_matrix_gpu(action[:, 3:6], "xyz")      # [B, 3, 3]
    R_new = torch.bmm(R_delta, R_current)                            # [B, 3, 3]
    new_angles = matrix_to_euler_angles_gpu(R_new, "xyz")            # [B, 3]

    # Gripper: clamp to [0, 1]
    new_gripper = torch.clamp(pose[:, -1:] + action[:, -1:], 0, 1)

    # Assemble new pose
    new_pose = torch.cat([new_xyz, new_angles, new_gripper], dim=-1)
    return new_pose[:, None]  # [B, 1, 7]


def poses_to_diff(start, end):
    """
    :param start: [7]
    :param end: [7]
    """
    try:
        start = start.numpy()
        end = end.numpy()
    except Exception:
        pass

    # --

    s_xyz = start[:3]
    e_xyz = end[:3]
    xyz_diff = e_xyz - s_xyz

    # --

    s_thetas = start[3:6]
    e_thetas = end[3:6]
    s_rotation = Rotation.from_euler("xyz", s_thetas, degrees=False).as_matrix()
    e_rotation = Rotation.from_euler("xyz", e_thetas, degrees=False).as_matrix()
    rotation_diff = e_rotation @ s_rotation.T
    theta_diff = Rotation.from_matrix(rotation_diff).as_euler("xyz", degrees=False)

    # --

    s_gripper = start[-1:]
    e_gripper = end[-1:]
    gripper_diff = e_gripper - s_gripper

    action = np.concatenate([xyz_diff, theta_diff, gripper_diff], axis=0)
    return torch.from_numpy(action)
