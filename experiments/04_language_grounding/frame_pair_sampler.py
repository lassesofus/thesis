"""
Frame Pair Sampling for Language-Conditioned Latent Goal Learning

Purpose:
    Generate (frame_k, frame_k+d, instruction) triplets for training a model that maps
    (current_latent, language_instruction) → latent_change_code

    This enables language-conditioned planning with V-JEPA-2 without modifying the world model.

Usage:
    python frame_pair_sampler.py --trajectory_dir /data/droid_raw/... --output_dir ./output
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import cv2
import h5py
import numpy as np


@dataclass
class FramePair:
    """A single frame pair with metadata."""
    trajectory_id: str
    frame_k: int
    frame_k_d: int
    d: int  # time horizon

    # Paths to extracted frames
    frame_k_path: Optional[str] = None
    frame_k_d_path: Optional[str] = None

    # Change metrics (for filtering)
    position_delta: Optional[float] = None
    gripper_delta: Optional[float] = None
    z_delta: Optional[float] = None
    rotation_delta: Optional[float] = None  # Total rotation change in radians

    # Salience tier: "A" (high salience) or "B" (moderate salience)
    salience_tier: Optional[str] = None

    # Generated instruction
    instruction: Optional[str] = None
    scene_k_description: Optional[str] = None
    scene_k_d_description: Optional[str] = None


@dataclass
class SamplingConfig:
    """Configuration for frame pair sampling."""

    # Time horizons to sample (in frames)
    # Drop very short horizons - they produce visually indistinguishable pairs
    # Medium: single actions, Long: compound actions
    time_horizons: tuple = (30, 50, 100, 200)

    # Stride for sampling start frames (to avoid too many similar pairs)
    # Set to match smallest horizon to avoid redundant overlapping pairs
    stride: int = 30

    # ==========================================================================
    # Tiered salience thresholds for meaningful change detection
    # ==========================================================================
    # Tier A (high salience): Caption-friendly, visually obvious changes
    tier_a_min_position_delta: float = 0.10  # 10cm movement
    tier_a_min_gripper_delta: float = 0.30   # clear open/close
    tier_a_min_z_delta: float = 0.05         # 5cm height change

    # Tier B (moderate salience): Broader coverage for contrastive learning
    tier_b_min_position_delta: float = 0.05  # 5cm movement
    tier_b_min_gripper_delta: float = 0.20   # moderate gripper change
    tier_b_min_z_delta: float = 0.03         # 3cm height change

    # Sampling quotas (must sum to 1.0)
    tier_a_quota: float = 0.60  # 60% from high-salience pairs
    tier_b_quota: float = 0.40  # 40% from moderate-salience pairs

    # Legacy thresholds (kept for backward compatibility, use Tier A values)
    min_position_delta: float = 0.10
    min_gripper_delta: float = 0.30
    min_z_delta: float = 0.05

    # Maximum rotation delta threshold (radians)
    # Limits total rotation change between frames. Set high to disable.
    # The Rx deviation filter below is more effective for blocking prevention.
    max_rotation_delta: float = 3.14  # ~180° (effectively disabled)

    # Maximum Rx deviation from vertical (radians)
    # This is the PRIMARY filter for camera occlusion prevention.
    # When Rx ≈ ±180°, gripper points down and arm is vertical (safe)
    # When Rx deviates significantly, arm rotates sideways and can block camera
    # Computed as: min(|Rx - 180°|, |Rx + 180°|)
    max_rx_deviation: float = 0.70  # ~40 degrees from vertical

    # Robot arm visibility filter (for ext1 camera)
    # Robot arm is likely in frame when it's moved forward and lowered into workspace
    # These thresholds are approximate and may need tuning per lab/camera setup
    min_x_for_visibility: float = 0.50  # arm must be forward enough to be in frame
    max_z_for_visibility: float = 0.42  # arm must be lowered enough to be in frame

    # Maximum pairs per trajectory per horizon (to balance dataset)
    max_pairs_per_horizon: int = 50

    # Maximum pairs per trajectory (across all horizons)
    # Limits total pairs from any single trajectory to encourage scene diversity
    max_pairs_per_trajectory: int = 12

    # Camera to use
    camera: str = "ext1"  # ext1, ext2, or wrist


class MeaningfulChangeDetector:
    """Detect if a frame pair represents a meaningful, describable change."""

    def __init__(self, config: SamplingConfig):
        self.config = config

    @staticmethod
    def _angle_diff(a1: float, a2: float) -> float:
        """Compute shortest angular difference, handling wrap-around."""
        diff = a2 - a1
        return np.arctan2(np.sin(diff), np.cos(diff))

    def compute_change_metrics(
        self,
        pos_k: np.ndarray,
        pos_k_d: np.ndarray,
        gripper_k: float,
        gripper_k_d: float
    ) -> dict:
        """Compute metrics describing the change between two frames."""

        pos_delta = np.linalg.norm(pos_k_d[:3] - pos_k[:3])
        z_delta = pos_k_d[2] - pos_k[2]
        gripper_delta = abs(gripper_k_d - gripper_k)

        # Compute rotation delta (handling wrap-around for Euler angles)
        # pos format: [x, y, z, rx, ry, rz]
        if len(pos_k) >= 6 and len(pos_k_d) >= 6:
            rot_delta = np.array([
                self._angle_diff(pos_k[i], pos_k_d[i])
                for i in range(3, 6)
            ])
            rotation_delta = float(np.linalg.norm(rot_delta))
        else:
            rotation_delta = 0.0

        return {
            "position_delta": float(pos_delta),
            "z_delta": float(z_delta),
            "gripper_delta": float(gripper_delta),
            "rotation_delta": rotation_delta,
        }

    def get_salience_tier(self, metrics: dict) -> Optional[str]:
        """Classify the change into a salience tier.

        Returns:
            "A" for high-salience (caption-friendly)
            "B" for moderate-salience (coverage)
            None if below minimum thresholds
        """
        pos_delta = metrics["position_delta"]
        grip_delta = metrics["gripper_delta"]
        z_delta = abs(metrics["z_delta"])

        # Tier A: High salience - visually obvious, caption-friendly
        tier_a = (
            pos_delta >= self.config.tier_a_min_position_delta or
            z_delta >= self.config.tier_a_min_z_delta or
            grip_delta >= self.config.tier_a_min_gripper_delta
        )
        if tier_a:
            return "A"

        # Tier B: Moderate salience - broader coverage for contrastive learning
        tier_b = (
            pos_delta >= self.config.tier_b_min_position_delta or
            z_delta >= self.config.tier_b_min_z_delta or
            grip_delta >= self.config.tier_b_min_gripper_delta
        )
        if tier_b:
            return "B"

        return None

    def is_meaningful(self, metrics: dict) -> bool:
        """Check if the change is meaningful enough (Tier A or B)."""
        return self.get_salience_tier(metrics) is not None

    def has_acceptable_rotation(self, metrics: dict) -> bool:
        """Check if rotation is within acceptable limits for VLM processing.

        Large rotations cause the robot arm to occlude manipulated objects,
        making it difficult for VLMs to describe the action.
        """
        rotation_delta = metrics.get("rotation_delta", 0.0)
        return rotation_delta <= self.config.max_rotation_delta

    @staticmethod
    def compute_rx_deviation(rx_rad: float) -> float:
        """Compute how far Rx deviates from ±180° (gripper pointing down).

        When Rx ≈ ±180°, the gripper points down and the arm is vertical.
        When Rx deviates, the arm rotates sideways and may block camera view.

        Returns deviation in radians.
        """
        rx_deg = np.rad2deg(rx_rad)
        dev_from_neg180 = abs(rx_deg - (-180))
        dev_from_pos180 = abs(rx_deg - 180)
        return np.deg2rad(min(dev_from_neg180, dev_from_pos180))

    def has_acceptable_rx_orientation(self, pos_k: np.ndarray, pos_k_d: np.ndarray) -> bool:
        """Check if arm orientation won't block camera view.

        Filters frames where the gripper has rotated away from pointing down,
        which causes the arm to sweep across the camera's field of view.
        """
        if len(pos_k) < 4 or len(pos_k_d) < 4:
            return True  # Can't check, assume OK

        rx_dev_k = self.compute_rx_deviation(pos_k[3])
        rx_dev_kd = self.compute_rx_deviation(pos_k_d[3])
        max_dev = max(rx_dev_k, rx_dev_kd)

        return max_dev <= self.config.max_rx_deviation

    def get_change_type(self, metrics: dict) -> str:
        """Categorize the type of change for diversity tracking."""

        types = []
        if metrics["position_delta"] >= self.config.min_position_delta:
            types.append("move")
        if metrics["gripper_delta"] >= self.config.min_gripper_delta:
            types.append("gripper")
        if metrics["z_delta"] > self.config.min_z_delta:
            types.append("lift")
        elif metrics["z_delta"] < -self.config.min_z_delta:
            types.append("lower")

        return "+".join(types) if types else "static"

    def is_arm_visible(self, pos: np.ndarray) -> bool:
        """
        Check if robot arm is likely visible in the camera frame.

        For ext1 camera (side view), the arm is visible when it has moved
        forward into the workspace (high X) and lowered (low Z).

        Note: These thresholds are approximate and may need tuning per setup.
        """
        x, y, z = pos[0], pos[1], pos[2]
        return x >= self.config.min_x_for_visibility and z <= self.config.max_z_for_visibility


class FramePairSampler:
    """Sample frame pairs from DROID trajectories."""

    def __init__(self, config: SamplingConfig = None):
        self.config = config or SamplingConfig()
        self.change_detector = MeaningfulChangeDetector(self.config)

    def _subsample_trajectory_pairs(self, pairs: list[FramePair]) -> list[FramePair]:
        """Subsample pairs to max_pairs_per_trajectory, preserving diversity.

        Stratifies by horizon and tier, then samples uniformly within each stratum
        to preserve both temporal and salience diversity.
        """
        max_total = self.config.max_pairs_per_trajectory

        # Group by (horizon, tier)
        strata = {}
        for p in pairs:
            key = (p.d, p.salience_tier)
            if key not in strata:
                strata[key] = []
            strata[key].append(p)

        # Allocate slots proportionally to each stratum
        n_strata = len(strata)
        base_per_stratum = max_total // n_strata if n_strata > 0 else 0
        remainder = max_total % n_strata if n_strata > 0 else 0

        result = []
        for i, (key, stratum_pairs) in enumerate(sorted(strata.items())):
            # Give extra slot to first few strata
            n_slots = base_per_stratum + (1 if i < remainder else 0)

            if len(stratum_pairs) <= n_slots:
                result.extend(stratum_pairs)
            else:
                # Uniformly sample by frame index (k) to preserve temporal diversity
                stratum_pairs.sort(key=lambda p: p.frame_k)
                indices = np.linspace(0, len(stratum_pairs) - 1, n_slots, dtype=int)
                result.extend([stratum_pairs[j] for j in indices])

        # Sort final result by frame index
        result.sort(key=lambda p: (p.d, p.frame_k))
        return result

    def _apply_tier_quotas(self, pairs: list[FramePair]) -> list[FramePair]:
        """Apply tiered quota sampling to balance high and moderate salience pairs.

        Samples according to configured quotas (e.g., 60% Tier A, 40% Tier B),
        respecting max_pairs_per_horizon limit.
        """
        if not pairs:
            return pairs

        # Split by tier
        tier_a = [p for p in pairs if p.salience_tier == "A"]
        tier_b = [p for p in pairs if p.salience_tier == "B"]

        max_total = self.config.max_pairs_per_horizon

        # Calculate target counts based on quotas
        target_a = int(max_total * self.config.tier_a_quota)
        target_b = max_total - target_a  # Remainder to B

        # Sample from each tier (or take all if fewer available)
        if len(tier_a) <= target_a:
            sampled_a = tier_a
            # Give extra slots to tier B if tier A underflows
            extra_for_b = target_a - len(tier_a)
            target_b += extra_for_b
        else:
            # Uniformly sample to preserve temporal diversity
            indices = np.linspace(0, len(tier_a) - 1, target_a, dtype=int)
            sampled_a = [tier_a[i] for i in indices]

        if len(tier_b) <= target_b:
            sampled_b = tier_b
        else:
            indices = np.linspace(0, len(tier_b) - 1, target_b, dtype=int)
            sampled_b = [tier_b[i] for i in indices]

        # Combine and sort by frame index to maintain temporal order
        result = sampled_a + sampled_b
        result.sort(key=lambda p: p.frame_k)

        return result

    def sample_from_trajectory(
        self,
        trajectory_dir: Path,
        output_dir: Path,
        extract_frames: bool = True
    ) -> list[FramePair]:
        """
        Sample frame pairs from a single trajectory.

        Returns list of FramePair objects with metadata.
        """
        trajectory_dir = Path(trajectory_dir)
        output_dir = Path(output_dir)

        # Load trajectory data
        h5_path = trajectory_dir / "trajectory.h5"
        with h5py.File(h5_path, "r") as f:
            positions = f["observation/robot_state/cartesian_position"][:]
            gripper = f["observation/robot_state/gripper_position"][:]

        n_frames = len(positions)

        # Load metadata for trajectory ID and video path
        metadata_files = list(trajectory_dir.glob("metadata_*.json"))
        if metadata_files:
            with open(metadata_files[0]) as f:
                metadata = json.load(f)
            trajectory_id = metadata.get("uuid", trajectory_dir.name)
            # Use left_mp4_path from metadata (extract filename)
            left_mp4_path = metadata.get("left_mp4_path", "")
            if left_mp4_path:
                video_filename = Path(left_mp4_path).name
            else:
                # Fallback to ext1 camera serial
                cam_serial = metadata.get("ext1_cam_serial", "22008760")
                video_filename = f"{cam_serial}.mp4"
        else:
            trajectory_id = trajectory_dir.name
            video_filename = "22008760.mp4"

        # Video for frame extraction (left camera)
        video_path = trajectory_dir / "recordings" / "MP4" / video_filename

        # Check if video exists
        if not video_path.exists():
            print(f"  WARNING: Video not found: {video_path}")

        frame_pairs = []

        for d in self.config.time_horizons:
            if d >= n_frames:
                continue

            horizon_pairs = []

            for k in range(0, n_frames - d, self.config.stride):
                k_d = k + d

                # Filter: Robot arm must be visible in BOTH frames
                # (otherwise VLM can't describe the transition)
                if not self.change_detector.is_arm_visible(positions[k]):
                    continue
                if not self.change_detector.is_arm_visible(positions[k_d]):
                    continue

                # Compute change metrics
                metrics = self.change_detector.compute_change_metrics(
                    positions[k], positions[k_d],
                    gripper[k], gripper[k_d]
                )

                # Classify salience tier (returns None if below minimum)
                salience_tier = self.change_detector.get_salience_tier(metrics)
                if salience_tier is None:
                    continue

                # Filter for acceptable rotation (avoid arm occluding objects)
                if not self.change_detector.has_acceptable_rotation(metrics):
                    continue

                # Filter for acceptable arm orientation (avoid blocking camera)
                if not self.change_detector.has_acceptable_rx_orientation(positions[k], positions[k_d]):
                    continue

                # Create frame pair
                pair = FramePair(
                    trajectory_id=trajectory_id,
                    frame_k=k,
                    frame_k_d=k_d,
                    d=d,
                    position_delta=metrics["position_delta"],
                    gripper_delta=metrics["gripper_delta"],
                    z_delta=metrics["z_delta"],
                    rotation_delta=metrics["rotation_delta"],
                    salience_tier=salience_tier,
                )

                horizon_pairs.append(pair)

            # Apply tiered quota sampling
            horizon_pairs = self._apply_tier_quotas(horizon_pairs)

            frame_pairs.extend(horizon_pairs)

        # Apply trajectory-level limit to encourage scene diversity
        if len(frame_pairs) > self.config.max_pairs_per_trajectory:
            frame_pairs = self._subsample_trajectory_pairs(frame_pairs)

        # Extract frames if requested
        if extract_frames and frame_pairs:
            self._extract_frames(video_path, frame_pairs, output_dir, n_frames)

        return frame_pairs

    def _extract_frames(
        self,
        video_path: Path,
        frame_pairs: list[FramePair],
        output_dir: Path,
        n_trajectory_frames: int
    ):
        """Extract and save frames for the sampled pairs."""

        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Collect all unique frame indices we need
        frame_indices = set()
        for pair in frame_pairs:
            frame_indices.add(pair.frame_k)
            frame_indices.add(pair.frame_k_d)

        # Open video and extract frames
        cap = cv2.VideoCapture(str(video_path))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        extracted = {}
        for idx in sorted(frame_indices):
            video_idx = int(idx * (total_video_frames / n_trajectory_frames))
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_idx)
            ret, frame = cap.read()

            if ret:
                frame_path = frames_dir / f"frame_{idx:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted[idx] = str(frame_path)

        cap.release()

        # Update frame pairs with paths
        for pair in frame_pairs:
            pair.frame_k_path = extracted.get(pair.frame_k)
            pair.frame_k_d_path = extracted.get(pair.frame_k_d)

    def get_sampling_stats(self, frame_pairs: list[FramePair]) -> dict:
        """Get statistics about the sampled pairs."""

        stats = {
            "total_pairs": len(frame_pairs),
            "by_horizon": {},
            "by_tier": {"A": 0, "B": 0},
            "by_change_type": {},
            "rotation_stats": {},
        }

        rotation_deltas = []
        for pair in frame_pairs:
            # By horizon
            d = pair.d
            stats["by_horizon"][d] = stats["by_horizon"].get(d, 0) + 1

            # By tier
            if pair.salience_tier:
                stats["by_tier"][pair.salience_tier] = stats["by_tier"].get(pair.salience_tier, 0) + 1

            # By change type
            metrics = {
                "position_delta": pair.position_delta,
                "z_delta": pair.z_delta,
                "gripper_delta": pair.gripper_delta,
            }
            change_type = self.change_detector.get_change_type(metrics)
            stats["by_change_type"][change_type] = stats["by_change_type"].get(change_type, 0) + 1

            # Rotation
            if pair.rotation_delta is not None:
                rotation_deltas.append(pair.rotation_delta)

        if rotation_deltas:
            rotation_deltas = np.array(rotation_deltas)
            stats["rotation_stats"] = {
                "mean_deg": float(np.rad2deg(rotation_deltas.mean())),
                "max_deg": float(np.rad2deg(rotation_deltas.max())),
                "min_deg": float(np.rad2deg(rotation_deltas.min())),
            }

        # Compute tier percentages
        total = stats["total_pairs"]
        if total > 0:
            stats["tier_percentages"] = {
                "A": 100 * stats["by_tier"]["A"] / total,
                "B": 100 * stats["by_tier"]["B"] / total,
            }

        return stats


def sample_trajectory(
    trajectory_dir: str,
    output_dir: str,
    config: SamplingConfig = None
) -> list[FramePair]:
    """
    Convenience function to sample frame pairs from a trajectory.
    """
    sampler = FramePairSampler(config)
    pairs = sampler.sample_from_trajectory(
        Path(trajectory_dir),
        Path(output_dir),
        extract_frames=True
    )

    # Save pairs metadata
    output_path = Path(output_dir) / "frame_pairs.json"
    with open(output_path, "w") as f:
        json.dump([asdict(p) for p in pairs], f, indent=2)

    return pairs


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sample frame pairs for language grounding")
    parser.add_argument("--trajectory_dir", type=str, required=True,
                        help="Path to DROID trajectory directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for frames and metadata")
    parser.add_argument("--horizons", type=int, nargs="+", default=[30, 50, 100, 200],
                        help="Time horizons to sample (default excludes d=10 as too subtle)")
    parser.add_argument("--stride", type=int, default=30,
                        help="Stride for sampling start frames (default matches smallest horizon)")
    parser.add_argument("--camera", type=str, default="ext1",
                        choices=["ext1", "ext2", "wrist"])
    parser.add_argument("--max_rotation", type=float, default=180.0,
                        help="Maximum rotation delta in degrees (default 180, effectively disabled)")
    parser.add_argument("--max_rx_deviation", type=float, default=40.0,
                        help="Maximum Rx deviation from vertical in degrees (default 40)")
    parser.add_argument("--tier_a_quota", type=float, default=0.60,
                        help="Quota for Tier A (high salience) pairs (default 0.60)")
    parser.add_argument("--tier_b_quota", type=float, default=0.40,
                        help="Quota for Tier B (moderate salience) pairs (default 0.40)")
    args = parser.parse_args()

    config = SamplingConfig(
        time_horizons=tuple(args.horizons),
        stride=args.stride,
        camera=args.camera,
        max_rotation_delta=np.deg2rad(args.max_rotation),
        max_rx_deviation=np.deg2rad(args.max_rx_deviation),
        tier_a_quota=args.tier_a_quota,
        tier_b_quota=args.tier_b_quota,
    )

    print(f"Sampling frame pairs from: {args.trajectory_dir}")
    print(f"Config: horizons={config.time_horizons}, stride={config.stride}")
    print(f"        max_rotation={args.max_rotation}°, max_rx_deviation={args.max_rx_deviation}°")
    print("=" * 60)

    sampler = FramePairSampler(config)
    pairs = sampler.sample_from_trajectory(
        Path(args.trajectory_dir),
        Path(args.output_dir),
        extract_frames=True
    )

    # Get stats
    stats = sampler.get_sampling_stats(pairs)

    print(f"\nSampled {stats['total_pairs']} frame pairs")
    print(f"\nBy horizon (d):")
    for d, count in sorted(stats["by_horizon"].items()):
        print(f"  d={d:3d}: {count} pairs")

    print(f"\nBy salience tier:")
    tier_pct = stats.get("tier_percentages", {})
    print(f"  Tier A (high salience):     {stats['by_tier']['A']:4d} ({tier_pct.get('A', 0):.1f}%)")
    print(f"  Tier B (moderate salience): {stats['by_tier']['B']:4d} ({tier_pct.get('B', 0):.1f}%)")

    print(f"\nBy change type:")
    for ctype, count in sorted(stats["by_change_type"].items(), key=lambda x: -x[1]):
        print(f"  {ctype}: {count}")

    if stats.get("rotation_stats"):
        rot = stats["rotation_stats"]
        print(f"\nRotation stats (filtered to <{args.max_rotation}°):")
        print(f"  Mean: {rot['mean_deg']:.1f}°, Max: {rot['max_deg']:.1f}°, Min: {rot['min_deg']:.1f}°")

    # Save
    output_path = Path(args.output_dir) / "frame_pairs.json"
    with open(output_path, "w") as f:
        json.dump([asdict(p) for p in pairs], f, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
