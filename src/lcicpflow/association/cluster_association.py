from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from lcicpflow.icp.base import ICPResult


def associate(
    src_cluster_ids: list[int],
    dst_cluster_ids: list[int],
    pair_results: dict[tuple[int, int], ICPResult],
    cfg: dict,
) -> tuple[dict[int, int], dict[int, np.ndarray], dict[int, ICPResult]]:
    """Cluster association from ICP-Flow Sec. 3.7."""
    matches: dict[int, int] = {}
    transforms = {sid: np.eye(4) for sid in src_cluster_ids}
    metrics: dict[int, ICPResult] = {}
    if not src_cluster_ids or not dst_cluster_ids:
        return matches, transforms, metrics
    D = np.full((len(src_cluster_ids), len(dst_cluster_ids)), np.inf)
    for i, sid in enumerate(src_cluster_ids):
        for j, did in enumerate(dst_cluster_ids):
            res = pair_results.get((sid, did))
            if res and res.success:
                D[i, j] = res.average_distance
    if cfg.get("method", "argmin") == "hungarian":
        rows, cols = linear_sum_assignment(np.where(np.isfinite(D), D, 1e9))
        candidates = zip(rows, cols)
    else:
        candidates = ((i, int(np.argmin(D[i]))) for i in range(D.shape[0]) if np.isfinite(D[i]).any())
    for i, j in candidates:
        if not np.isfinite(D[i, j]):
            continue
        sid, did = src_cluster_ids[i], dst_cluster_ids[j]
        res = pair_results[(sid, did)]
        matches[sid] = did
        transforms[sid] = res.transform
        metrics[sid] = res
    return matches, transforms, metrics
