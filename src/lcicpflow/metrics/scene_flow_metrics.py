from __future__ import annotations

import numpy as np


def scene_flow_metrics(pred: np.ndarray, gt: np.ndarray, strict: float = 0.05, relaxed: float = 0.10, eps: float = 1e-6) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shapes differ: {pred.shape} vs {gt.shape}")
    err = np.linalg.norm(pred - gt, axis=1)
    gt_norm = np.linalg.norm(gt, axis=1)
    rel = err / np.maximum(gt_norm, eps)
    return {
        "EPE3D": float(err.mean()) if len(err) else float("nan"),
        "AccS": float(np.mean((err <= strict) | (rel <= strict))) if len(err) else float("nan"),
        "AccR": float(np.mean((err <= relaxed) | (rel <= relaxed))) if len(err) else float("nan"),
    }
