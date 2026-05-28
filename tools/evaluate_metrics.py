#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np

from lcicpflow.metrics.scene_flow_metrics import scene_flow_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prediction_npz")
    p.add_argument("--gt-flow", default=None)
    args = p.parse_args()
    pred = np.load(args.prediction_npz)["flow_pred"]
    if args.gt_flow is None:
        print("Diagnostic mode: no ground-truth flow was provided, supervised EPE/Acc metrics cannot be computed.")
        print(f"Predicted flow norm mean: {np.linalg.norm(pred, axis=1).mean():.4f} m")
        return
    gt = np.load(args.gt_flow)
    if isinstance(gt, np.lib.npyio.NpzFile):
        gt = gt["flow_gt"]
    print(scene_flow_metrics(pred, gt))


if __name__ == "__main__":
    main()
