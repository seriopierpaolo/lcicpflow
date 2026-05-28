from __future__ import annotations

import numpy as np


def cluster_centroids(points: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(i): points[labels == i, :3].mean(axis=0) for i in np.unique(labels) if i >= 0}


def pair_clusters(src_points: np.ndarray, dst_points: np.ndarray, src_labels: np.ndarray, dst_labels: np.ndarray, cfg: dict) -> list[tuple[int, int]]:
    """Pair clusters by local spatial gates, as in ICP-Flow Sec. 3.5."""
    src_c = cluster_centroids(src_points, src_labels)
    dst_c = cluster_centroids(dst_points, dst_labels)
    bounds = np.array([cfg["max_translation_x"], cfg["max_translation_y"], cfg["max_translation_z"]], dtype=float)
    pairs: list[tuple[int, int]] = []
    for sid, sc in src_c.items():
        for did, dc in dst_c.items():
            if np.all(np.abs(dc - sc) <= bounds):
                pairs.append((sid, did))
    return pairs
