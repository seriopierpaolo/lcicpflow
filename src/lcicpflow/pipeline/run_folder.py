from __future__ import annotations

import argparse

from lcicpflow.config.load_config import load_config
from lcicpflow.io.pcd_reader import consecutive_pairs
from lcicpflow.pipeline.run_pair import run_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ICP-Flow on consecutive PCD pairs in a folder.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-folder", default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    folder = args.input_folder or cfg["input"]["folder"]
    max_pairs = args.max_pairs if args.max_pairs is not None else cfg["input"].get("max_pairs")
    for source, target in consecutive_pairs(folder, max_pairs=max_pairs):
        run_pair(source, target, cfg)


if __name__ == "__main__":
    main()
