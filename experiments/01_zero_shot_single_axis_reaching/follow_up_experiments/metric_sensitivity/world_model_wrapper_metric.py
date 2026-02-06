"""
World model wrapper with configurable energy metric for CEM planning.

This module extends the original WorldModel to support different distance metrics
for the CEM objective function:
- l1: Original L1 (mean absolute) distance
- cosine: Scale-invariant cosine distance
- norm_l1: L1 distance after per-dimension normalization using DROID statistics
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

# Import energy functions
import sys
_metric_path = os.path.dirname(os.path.abspath(__file__))
if _metric_path not in sys.path:
    sys.path.insert(0, _metric_path)

from energy_functions import EnergyFunction, l1_energy, cosine_energy, normalized_l1_energy

# Import pose computation utilities from mpc_utils
from notebooks.utils.mpc_utils import compute_new_pose, compute_new_pose_gpu, USE_GPU_POSE

# AMP optimization flag
USE_AMP = os.environ.get('OPT_AMP', '0') == '1' or os.environ.get('VJEPA_OPTIMIZE', '0') == '1'


def create_cem_objective(energy_fn):
    """
    Create a CEM objective function that works with flattened representations.

    The CEM in mpc_utils.py calls objective(final_state.flatten(1), goal_state.flatten(1))
    where final_state is [S, HW, D] and gets flattened to [S, HW*D].

    We need to reshape back to [S, HW, D] for our energy functions.
    """
    def objective(a_flat, b_flat):
        # a_flat and b_flat are [S, HW*D]
        # We need to figure out HW and D to reshape
        # V-JEPA-2 Giant: D=1408, typical HW=256 tokens
        # Total = 256 * 1408 = 360448
        S = a_flat.shape[0]
        total_dim = a_flat.shape[1]

        # Assuming D=1408 (V-JEPA-2 Giant)
        D = 1408
        HW = total_dim // D

        # Reshape to [S, HW, D]
        a = a_flat.view(S, HW, D)
        b = b_flat.view(S, HW, D)

        # Compute energy
        return energy_fn(a, b)

    return objective


class WorldModelWithMetric:
    """
    World model wrapper with configurable energy metric for CEM planning.

    This extends the base WorldModel to support different distance metrics.
    """

    def __init__(
        self,
        encoder,
        predictor,
        tokens_per_frame,
        transform,
        energy_metric='l1',
        energy_stats_path=None,
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
        """
        Initialize world model with energy metric.

        Args:
            encoder: V-JEPA encoder
            predictor: V-JEPA predictor
            tokens_per_frame: Number of tokens per frame
            transform: Image transform
            energy_metric: One of 'l1', 'cosine', 'norm_l1'
            energy_stats_path: Path to DROID statistics (required for norm_l1)
            mpc_args: CEM hyperparameters
            normalize_reps: Whether to layer-normalize representations
            device: Torch device
        """
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalize_reps = normalize_reps
        self.transform = transform
        self.tokens_per_frame = tokens_per_frame
        self.device = device
        self.mpc_args = mpc_args

        # Initialize energy function
        self.energy_metric = energy_metric
        self.energy_fn = EnergyFunction(
            metric=energy_metric,
            stats_path=energy_stats_path,
            device=device
        )

        # Create CEM objective that uses our energy function
        self.cem_objective = create_cem_objective(self.energy_fn)

        print(f"WorldModelWithMetric initialized with energy metric: {energy_metric}")

    def encode(self, image):
        """Encode single image to representation."""
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
        """
        Infer next action using CEM with configurable energy metric.

        Args:
            rep: Current representation [1, tokens, D]
            pose: Current pose [1, 1, 7]
            goal_rep: Goal representation [1, tokens, D]
            close_gripper: Step at which to close gripper

        Returns:
            action: [1, rollout, 7]
        """
        def step_predictor(reps, actions, poses):
            B, T, N_T, D = reps.size()
            reps = reps.flatten(1, 2)
            # Use AMP for predictor inference if enabled
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                next_rep = self.predictor(reps, actions, poses)[:, -self.tokens_per_frame:]
                if self.normalize_reps:
                    next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
            next_rep = next_rep.view(B, 1, N_T, D)

            # Use existing pose computation from mpc_utils
            if USE_GPU_POSE:
                next_pose = compute_new_pose_gpu(poses[:, -1:], actions[:, -1:])
            else:
                next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])

            return next_rep, next_pose

        # Import cem from mpc_utils to use with our custom objective
        from notebooks.utils.mpc_utils import cem

        mpc_action = cem(
            context_frame=rep,
            context_pose=pose,
            goal_frame=goal_rep,
            world_model=step_predictor,
            objective=self.cem_objective,
            close_gripper=close_gripper,
            **self.mpc_args,
        )[0]

        return mpc_action

    def compute_energy(self, z_pred, z_goal):
        """
        Compute energy between predicted and goal representations.

        Useful for logging and analysis.

        Args:
            z_pred: Predicted representation [T, D] or [B, T, D]
            z_goal: Goal representation [T, D] or [B, T, D]

        Returns:
            Energy value (scalar or [B])
        """
        return self.energy_fn(z_pred, z_goal)


if __name__ == "__main__":
    print("Testing WorldModelWithMetric...")

    # Create dummy data
    B, T, D = 10, 256, 1408
    z_pred = torch.randn(B, T, D)
    z_goal = torch.randn(B, T, D)

    # Test each metric
    for metric in ['l1', 'cosine']:
        energy_fn = EnergyFunction(metric=metric)
        energies = energy_fn(z_pred, z_goal)
        print(f"{metric}: shape={energies.shape}, range=[{energies.min():.4f}, {energies.max():.4f}]")

    # Test CEM objective wrapper
    print("\nTesting CEM objective wrapper...")
    for metric in ['l1', 'cosine']:
        energy_fn = EnergyFunction(metric=metric)
        cem_obj = create_cem_objective(energy_fn)

        # Flatten as CEM does
        a_flat = z_pred.flatten(1)  # [B, T*D]
        b_flat = z_goal.flatten(1)  # [B, T*D]

        energies = cem_obj(a_flat, b_flat)
        print(f"{metric} (via CEM objective): shape={energies.shape}, range=[{energies.min():.4f}, {energies.max():.4f}]")

    print("\nAll tests passed!")
