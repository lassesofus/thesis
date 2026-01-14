# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import torch.nn.functional as F

from .mpc_utils import cem, compute_new_pose, compute_new_pose_gpu, USE_GPU_POSE

# AMP optimization flag
USE_AMP = os.environ.get('OPT_AMP', '0') == '1' or os.environ.get('VJEPA_OPTIMIZE', '0') == '1'


class WorldModel(object):

    def __init__(
        self,
        encoder,
        predictor,
        tokens_per_frame,
        transform,
        mpc_args={
            "rollout": 2,
            "samples": 400,
            "topk": 10,
            "cem_steps": 10,
            "momentum_mean": 0.15,
            "momentum_std": 0.15,
            "maxnorm": 0.05,
            "verbose": True,
        },
        normalize_reps=True,
        device="cuda:0",
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalize_reps = normalize_reps
        self.transform = transform
        self.tokens_per_frame = tokens_per_frame
        self.device = device
        self.mpc_args = mpc_args

    def encode(self, image):
        clip = np.expand_dims(image, axis=0)
        clip = self.transform(clip)[None, :]
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        clip = clip.to(self.device, non_blocking=True)
        h = self.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    def infer_next_action(self, rep, pose, goal_rep, close_gripper=None):

        def step_predictor(reps, actions, poses):
            B, T, N_T, D = reps.size()
            reps = reps.flatten(1, 2)
            # Use AMP for predictor inference if enabled
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                next_rep = self.predictor(reps, actions, poses)[:, -self.tokens_per_frame :]
                if self.normalize_reps:
                    next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
            next_rep = next_rep.view(B, 1, N_T, D)
            # Pose computation outside autocast (small tensor, precision matters)
            if USE_GPU_POSE:
                next_pose = compute_new_pose_gpu(poses[:, -1:], actions[:, -1:])
            else:
                next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])
            return next_rep, next_pose

        mpc_action = cem(
            context_frame=rep,
            context_pose=pose,
            goal_frame=goal_rep,
            world_model=step_predictor,
            close_gripper=close_gripper,
            **self.mpc_args,
        )[0]

        return mpc_action
