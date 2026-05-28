from __future__ import annotations

import numpy as np

from lcicpflow.utils.geometry import transform_points


def recover_scene_flow(points_t: np.ndarray, cluster_ids: np.ndarray, transforms_by_cluster: dict[int, np.ndarray], ego_transform: np.ndarray) -> np.ndarray:
    """Recover per-point flow using ICP-Flow Eq. (1): F_k = T_k T_ego C_k - C_k."""
    flow = np.zeros((len(points_t), 3), dtype=np.float64)
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        T = transforms_by_cluster.get(int(cid), np.eye(4))
        total = T @ ego_transform
        moved = transform_points(points_t[mask], total)
        flow[mask] = moved - points_t[mask, :3]
    return flow
