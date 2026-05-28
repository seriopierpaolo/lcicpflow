from __future__ import annotations

import numpy as np


def as_xyz(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected point array with shape (N, >=3), got {arr.shape}")
    return np.ascontiguousarray(arr[:, :3])


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    xyz = as_xyz(points)
    T = np.asarray(transform, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {T.shape}")
    return xyz @ T[:3, :3].T + T[:3, 3]


def make_transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if rotation is not None:
        R = np.asarray(rotation, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError("rotation must be 3x3")
        T[:3, :3] = R
    if translation is not None:
        t = np.asarray(translation, dtype=np.float64).reshape(3)
        T[:3, 3] = t
    return T


def invert_transform(transform: np.ndarray) -> np.ndarray:
    T = np.asarray(transform, dtype=np.float64)
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = T[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ T[:3, 3]
    return inv
