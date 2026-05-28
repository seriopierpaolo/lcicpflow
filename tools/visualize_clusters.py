#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prediction_npz")
    p.add_argument("--output", default="clusters.png")
    args = p.parse_args()
    d = np.load(args.prediction_npz)
    pts = d["points_t"]
    labels = d["cluster_ids_t"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.2, c=labels, cmap="tab20")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
