from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lcicpflow.association.cluster_association import associate
from lcicpflow.clustering.hdbscan_clusterer import cluster_two_scans
from lcicpflow.config.load_config import load_config, save_config
from lcicpflow.flow.scene_flow import recover_scene_flow
from lcicpflow.icp.kiss_icp_backend import ICPBackend
from lcicpflow.initialization.histogram_init import histogram_translation_init
from lcicpflow.io.pcd_reader import read_pcd
from lcicpflow.pairing.cluster_pairing import pair_clusters
from lcicpflow.preprocessing.ego_motion import ego_motion_transform, load_pose
from lcicpflow.preprocessing.ground_filter import filter_ground, crop_xy
from lcicpflow.utils.logging import setup_logger
from lcicpflow.utils.timing import StageTimer


def run_pair(source: str | Path, target: str | Path, cfg: dict, output_dir: str | Path | None = None) -> Path:
    source, target = Path(source), Path(target)
    sample = f"{source.stem}__{target.stem}"
    out = Path(output_dir or Path(cfg["output"]["root"]) / sample)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("lcicpflow", out / "log.txt", cfg["logging"]["level"])
    timer = StageTimer()
    rng = np.random.default_rng(int(cfg.get("random_seed", 7)))

    with timer.stage("load"):
        points_t = read_pcd(source)
        points_t1 = read_pcd(target)
        
    with timer.stage("ego_motion"):
        T_ego = ego_motion_transform(load_pose(cfg["poses"].get("source_pose")), load_pose(cfg["poses"].get("target_pose")))
        
    # 1. Ground Filtering (Saves full-sized global masks)
    with timer.stage("ground_filter"):
        src_ground, src_ground_mask = filter_ground(points_t, cfg["ground_filter"])
        dst_ground, dst_ground_mask = filter_ground(points_t1, cfg["ground_filter"])
    logger.info("Loaded %d/%d source points, %d/%d target points after ground filtering", len(src_ground), len(points_t), len(dst_ground), len(points_t1))
    
    # 2. XY Cropbox Filtering (Processes output of ground filter, saves sub-masks)
    with timer.stage("crop_xy"):
        src_ng, src_crop_mask = crop_xy(src_ground, cfg["xy_cropbox"])
        dst_ng, dst_crop_mask = crop_xy(dst_ground, cfg["xy_cropbox"])
    logger.info("Loaded %d/%d source points, %d/%d target points after XY cropbox filtering", len(src_ng), len(points_t), len(dst_ng), len(points_t1))

    # 3. Create Global Sequential Masks for index mapping
    src_ng_mask = src_ground_mask.copy()
    src_ng_mask[src_ground_mask] = src_crop_mask

    dst_ng_mask = dst_ground_mask.copy()
    dst_ng_mask[dst_ground_mask] = dst_crop_mask

    # 4. Clustering 
    with timer.stage("clustering"):
        src_labels_ng, dst_labels_ng = cluster_two_scans(src_ng, dst_ng, cfg["clustering"])
        
    src_labels = np.full(len(points_t), -1, dtype=np.int32)
    dst_labels = np.full(len(points_t1), -1, dtype=np.int32)
    src_labels[src_ng_mask] = src_labels_ng
    dst_labels[dst_ng_mask] = dst_labels_ng

    # 5. Pairing
    with timer.stage("pairing"):
        pairs = pair_clusters(src_ng, dst_ng, src_labels_ng, dst_labels_ng, cfg["pairing"])
    logger.info("Generated %d cluster candidate pairs", len(pairs))

    backend = ICPBackend(cfg["icp"])
    pair_results = {}
    with timer.stage("histogram_icp"):
        for sid, did in pairs:
            src_cluster = src_ng[src_labels_ng == sid]
            dst_cluster = dst_ng[dst_labels_ng == did]
            init = histogram_translation_init(src_cluster, dst_cluster, cfg["histogram_init"], rng)
            pair_results[(sid, did)] = backend.register(src_cluster, dst_cluster, init)

    src_ids = [int(i) for i in np.unique(src_labels_ng) if i >= 0]
    dst_ids = [int(i) for i in np.unique(dst_labels_ng) if i >= 0]
    with timer.stage("association"):
        matches, transforms_by_cluster, metrics_by_cluster = associate(src_ids, dst_ids, pair_results, cfg["association"])
    with timer.stage("scene_flow"):
        flow_pred = recover_scene_flow(points_t, src_labels, transforms_by_cluster, T_ego)

    matched_cluster_ids = np.full(len(src_ids), -1, dtype=np.int32)
    transforms = np.stack([transforms_by_cluster.get(cid, np.eye(4)) for cid in src_ids]) if src_ids else np.empty((0, 4, 4))
    icp_distances = np.array([metrics_by_cluster[cid].average_distance if cid in metrics_by_cluster else np.inf for cid in src_ids])
    icp_inlier_ratios = np.array([metrics_by_cluster[cid].inlier_ratio if cid in metrics_by_cluster else 0.0 for cid in src_ids])
    for k, cid in enumerate(src_ids):
        matched_cluster_ids[k] = matches.get(cid, -1)

    if cfg["output"]["save_npz"]:
        np.savez_compressed(
            out / "prediction.npz",
            points_t=points_t,
            points_t1=points_t1,
            flow_pred=flow_pred,
            cluster_ids_t=src_labels,
            cluster_ids_t1=dst_labels,
            source_cluster_ids=np.array(src_ids, dtype=np.int32),
            matched_cluster_ids=matched_cluster_ids,
            transforms=transforms,
            icp_distances=icp_distances,
            icp_inlier_ratios=icp_inlier_ratios,
        )
    runtime = {"stage_seconds": timer.times, "num_pairs": len(pairs), "num_matches": len(matches)}
    if cfg["output"]["save_runtime"]:
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    if cfg["output"]["save_config"]:
        save_config(cfg, out / "config_used.yaml")
    logger.info("Finished %s -> %s. Output: %s", source.name, target.name, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ICP-Flow on one PCD pair.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_pair(args.source, args.target, cfg, args.output_dir)


if __name__ == "__main__":
    main()
