from __future__ import annotations

import numpy as np


def cluster_points(points: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    backend = cfg.get("backend", "hdbscan")
    if len(points) == 0:
        return np.empty(0, dtype=np.int32), {}
    if backend == "hdbscan":
        labels = _hdbscan(points, cfg)
    elif backend == "dbscan":
        labels = _dbscan(points, cfg)
    else:
        raise ValueError(f"Unknown clustering backend: {backend}")
    labels = _keep_largest(labels, cfg["max_clusters"], cfg["min_cluster_size"])
    return labels, labels_to_indices(labels)


def labels_to_indices(labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(lbl): np.flatnonzero(labels == lbl) for lbl in np.unique(labels) if lbl >= 0}


def _hdbscan(points: np.ndarray, cfg: dict) -> np.ndarray:
    try:
        import hdbscan
    except ImportError:
        return _dbscan(points, cfg)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(cfg["min_cluster_size"]),
        min_samples=cfg.get("hdbscan_min_samples"),
        metric="euclidean",
        algorithm="best",
    )
    return clusterer.fit_predict(points[:, :3]).astype(np.int32)


def _dbscan(points: np.ndarray, cfg: dict) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=float(cfg["dbscan_eps"]), min_samples=int(cfg["min_cluster_size"])).fit_predict(points[:, :3]).astype(np.int32)


def _keep_largest(labels: np.ndarray, max_clusters: int, min_cluster_size: int) -> np.ndarray:
    out = np.full_like(labels, -1)
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    keep = [(i, c) for i, c in zip(ids, counts) if c >= min_cluster_size]
    keep.sort(key=lambda item: item[1], reverse=True)
    for new_id, (old_id, _) in enumerate(keep[:max_clusters]):
        out[labels == old_id] = new_id
    return out


def cluster_two_scans(src: np.ndarray, dst: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    if cfg.get("fuse_scans", True):
        fused = np.vstack([src[:, :3], dst[:, :3]])
        labels, _ = cluster_points(fused, cfg)
        return labels[: len(src)], labels[len(src) :]
    src_labels, _ = cluster_points(src, cfg)
    dst_labels, _ = cluster_points(dst, cfg)
    return src_labels, dst_labels
