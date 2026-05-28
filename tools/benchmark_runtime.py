#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runtime_json", nargs="+")
    args = p.parse_args()
    totals = {}
    for path in args.runtime_json:
        data = json.loads(Path(path).read_text())
        for k, v in data["stage_seconds"].items():
            totals[k] = totals.get(k, 0.0) + v
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
