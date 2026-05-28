#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    args = p.parse_args()
    data = np.load(args.npz)
    summary = {k: {"shape": data[k].shape, "dtype": str(data[k].dtype)} for k in data.files}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
