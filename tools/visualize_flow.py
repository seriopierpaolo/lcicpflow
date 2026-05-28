#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prediction_npz")
    p.add_argument("--output", default="visualization.png")
    p.add_argument("--max-points", type=int, default=50000)
    args = p.parse_args()
    d = np.load(args.prediction_npz)
    pts = d["points_t"]
    dst = d["points_t1"]
    flow = d["flow_pred"]
    step = max(1, len(pts) // args.max_points)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(dst[::step, 0], dst[::step, 1], s=0.2, c="lightgray", label="target")
    ax.scatter(pts[::step, 0], pts[::step, 1], s=0.2, c="tab:blue", label="source")
    qstep = max(1, len(pts) // 2000)
    ax.quiver(pts[::qstep, 0], pts[::qstep, 1], flow[::qstep, 0], flow[::qstep, 1], angles="xy", scale_units="xy", scale=1, width=0.001)
    ax.set_aspect("equal")
    ax.legend(markerscale=8)
    ax.set_xlim([-25,25])
    ax.set_ylim([-25,25])
    fig.tight_layout()
    fig.savefig(args.output, dpi=600)


if __name__ == "__main__":
    main()
