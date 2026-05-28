from __future__ import annotations

from pathlib import Path

import numpy as np

from lcicpflow.utils.geometry import invert_transform, transform_points


def load_pose(path: str | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float64)
    arr = np.loadtxt(Path(path), dtype=np.float64)
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError(f"Pose file {path} must contain a 4x4 matrix")


def ego_motion_transform(source_pose: np.ndarray | None, target_pose: np.ndarray | None) -> np.ndarray:
    """Return T_target_source for source points.

    Convention: poses are T_world_lidar. Compensation maps points from the
    source LiDAR frame to the target LiDAR frame:
        T_target_source = inv(T_world_target) @ T_world_source

    This is the T_ego term in ICP-Flow Eq. (1).
    """
    src = np.eye(4) if source_pose is None else source_pose
    dst = np.eye(4) if target_pose is None else target_pose
    return invert_transform(dst) @ src


def compensate_source(points: np.ndarray, transform_target_source: np.ndarray) -> np.ndarray:
    return transform_points(points, transform_target_source)
