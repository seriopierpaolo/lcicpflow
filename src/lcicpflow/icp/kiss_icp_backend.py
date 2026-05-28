from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from lcicpflow.icp.base import ICPResult
from lcicpflow.utils.geometry import transform_points


class ICPBackend:
    """Small registration interface.

    The preferred local-cluster backend is Open3D point-to-point ICP. KISS-ICP
    is intentionally isolated here because its public API targets scan odometry,
    not arbitrary cluster-to-cluster registration. Deployments can replace this
    class with a KISS-ICP adapter without touching the pipeline.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def register(self, src: np.ndarray, dst: np.ndarray, init: np.ndarray) -> ICPResult:
        if len(src) < self.cfg["min_points"] or len(dst) < self.cfg["min_points"]:
            return ICPResult(init, float("inf"), 0.0, False, {"reason": "too_few_points"})
        try:
            result = self._open3d_icp(src, dst, init)
        except Exception as exc:
            result = self._numpy_icp(src, dst, init)
            result.diagnostics["fallback_reason"] = repr(exc)
        return result

    def _open3d_icp(self, src: np.ndarray, dst: np.ndarray, init: np.ndarray) -> ICPResult:
        import open3d as o3d

        p_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src[:, :3]))
        p_dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst[:, :3]))
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=float(self.cfg["tolerance"]),
            relative_rmse=float(self.cfg["tolerance"]),
            max_iteration=int(self.cfg["max_iterations"]),
        )
        reg = o3d.pipelines.registration.registration_icp(
            p_src,
            p_dst,
            float(self.cfg["max_correspondence_distance"]),
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria,
        )
        avg, ratio = correspondence_metrics(src, dst, reg.transformation, float(self.cfg["inlier_threshold"]))
        ok = avg <= self.cfg["max_average_distance"] and ratio >= self.cfg["min_inlier_ratio"]
        return ICPResult(reg.transformation, avg, ratio, ok, {"fitness": reg.fitness, "rmse": reg.inlier_rmse, "backend": "open3d"})

    def _numpy_icp(self, src: np.ndarray, dst: np.ndarray, init: np.ndarray) -> ICPResult:
        T = init.copy()
        prev = float("inf")
        tree = cKDTree(dst[:, :3])
        for _ in range(int(self.cfg["max_iterations"])):
            moved = transform_points(src, T)
            dist, idx = tree.query(moved, k=1)
            mask = dist <= float(self.cfg["max_correspondence_distance"])
            if mask.sum() < 3:
                break
            delta = _best_fit_transform(moved[mask], dst[idx[mask], :3])
            T = delta @ T
            mean = float(dist[mask].mean())
            if abs(prev - mean) < float(self.cfg["tolerance"]):
                break
            prev = mean
        avg, ratio = correspondence_metrics(src, dst, T, float(self.cfg["inlier_threshold"]))
        ok = avg <= self.cfg["max_average_distance"] and ratio >= self.cfg["min_inlier_ratio"]
        return ICPResult(T, avg, ratio, ok, {"backend": "numpy"})


def correspondence_metrics(src: np.ndarray, dst: np.ndarray, transform: np.ndarray, inlier_threshold: float) -> tuple[float, float]:
    """Paper Sec. 3.6: average NN distance d and inlier ratio r using Eq. (2)."""
    moved = transform_points(src, transform)
    dist, _ = cKDTree(dst[:, :3]).query(moved, k=1)
    inliers = dist <= inlier_threshold
    avg = float(dist.mean()) if len(dist) else float("inf")
    # r = sum 1(d_i) / (L_m + L_n - sum 1(d_i)), from Sec. 3.6.
    denom = len(src) + len(dst) - int(inliers.sum())
    ratio = float(inliers.sum() / denom) if denom > 0 else 0.0
    return avg, ratio


def _best_fit_transform(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
