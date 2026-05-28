from __future__ import annotations

import numpy as np

from lcicpflow.utils.geometry import make_transform


def histogram_translation_init(src: np.ndarray, dst: np.ndarray, cfg: dict, rng: np.random.Generator | None = None) -> np.ndarray:
    """Dominant pairwise-translation vote for ICP initialization.

    ICP-Flow Sec. 3.6 computes all vectors x_j^{t+dt} - x_i^t for a cluster
    pair, bins them in a 3D histogram, and initializes ICP with the most-voted
    translation bin. This function implements that directly with sampling and
    chunking controls to cap memory use.
    """
    rng = np.random.default_rng() if rng is None else rng
    src_xyz = _sample(src[:, :3], int(cfg["max_votes_per_cluster_points"]), rng)
    dst_xyz = _sample(dst[:, :3], int(cfg["max_votes_per_cluster_points"]), rng)
    if len(src_xyz) == 0 or len(dst_xyz) == 0:
        return np.eye(4)
    lo = np.array([cfg["bounds_x"][0], cfg["bounds_y"][0], cfg["bounds_z"][0]], dtype=float)
    hi = np.array([cfg["bounds_x"][1], cfg["bounds_y"][1], cfg["bounds_z"][1]], dtype=float)
    bin_size = float(cfg["bin_size"])
    bins = [np.arange(lo[i], hi[i] + bin_size, bin_size) for i in range(3)]
    hist = np.zeros(tuple(len(b) - 1 for b in bins), dtype=np.int32)
    chunk = int(cfg.get("chunk_size", 2_000_000))
    max_src = max(1, chunk // max(1, len(dst_xyz)))
    for start in range(0, len(src_xyz), max_src):
        votes = dst_xyz[None, :, :] - src_xyz[start : start + max_src, None, :]
        votes = votes.reshape(-1, 3)
        in_bounds = np.all((votes >= lo) & (votes <= hi), axis=1)
        if np.any(in_bounds):
            h, _ = np.histogramdd(votes[in_bounds], bins=bins)
            hist += h.astype(np.int32)
    if hist.max() == 0:
        return np.eye(4)
    idx = np.unravel_index(int(hist.argmax()), hist.shape)
    translation = np.array([(bins[i][idx[i]] + bins[i][idx[i] + 1]) * 0.5 for i in range(3)])
    return make_transform(translation=translation)


def _sample(points: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= limit:
        return points
    idx = rng.choice(len(points), size=limit, replace=False)
    return points[idx]
