# lcicpflow

Clean, modular implementation of **ICP-Flow: LiDAR Scene Flow Estimation with ICP**. The code follows the paper pipeline: PCD loading, optional ego-motion compensation, ground filtering, clustering, cluster pairing, histogram-based ICP initialization, ICP refinement, quality scoring, cluster association, scene-flow recovery, visualization, and evaluation.

This refactor is intentionally simpler than the original repository. Ground removal uses a configurable z-threshold instead of Patchwork++. ICP is isolated behind `src/lcicpflow/icp/kiss_icp_backend.py`; the default implementation uses Open3D point-to-point ICP with a NumPy fallback, and the wrapper is the only place that should know about a future KISS-ICP adapter.

## Install

```bash
cd lcicpflow
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e ".[dev,hdbscan]"
```

## Docker

```bash
cd lcicpflow
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm lcicpflow pytest
docker compose -f docker/docker-compose.yml run --rm lcicpflow \
  lcicpflow-folder --config configs/default.yaml --input-folder /workspace/project/data_from_rosbag --max-pairs 1
```

The compose file requests NVIDIA devices when available. For CPU-only systems, remove the `deploy.resources.reservations.devices` block or run plain Docker:

```bash
docker build -f docker/Dockerfile -t lcicpflow:dev .
docker run --rm -it -v "$PWD:/workspace/lcicpflow" lcicpflow:dev
```

## Inference

Run one pair:

```bash
cd lcicpflow
lcicpflow-pair \
  --config configs/default.yaml \
  --source ../data_from_rosbag/frame_1775029457945645216.pcd \
  --target ../data_from_rosbag/frame_1775029458350698656.pcd
```

Run automatically detected consecutive pairs:

```bash
lcicpflow-folder --config configs/default.yaml --input-folder ../data_from_rosbag --max-pairs 1
```

Outputs are saved under:

```text
outputs/inference/<source>__<target>/
  prediction.npz
  runtime.json
  config_used.yaml
  log.txt
```

`prediction.npz` contains `points_t`, `points_t1`, `flow_pred`, `cluster_ids_t`, `cluster_ids_t1`, `source_cluster_ids`, `matched_cluster_ids`, `transforms`, `icp_distances`, and `icp_inlier_ratios`.

## Tools

```bash
python tools/inspect_npz_output.py outputs/inference/<sample>/prediction.npz
python tools/visualize_flow.py outputs/inference/<sample>/prediction.npz --output outputs/inference/<sample>/visualization.png
python tools/visualize_clusters.py outputs/inference/<sample>/prediction.npz --output outputs/inference/<sample>/clusters.png
python tools/evaluate_metrics.py outputs/inference/<sample>/prediction.npz
python tools/benchmark_runtime.py outputs/inference/<sample>/runtime.json
```

When ground-truth flow is available:

```bash
python tools/evaluate_metrics.py outputs/inference/<sample>/prediction.npz --gt-flow path/to/flow_gt.npy
```

## Paper Mapping

- Sec. 3.3: `preprocessing/ego_motion.py`
- Sec. 3.4: `preprocessing/ground_filter.py`, `clustering/hdbscan_clusterer.py`
- Sec. 3.5: `pairing/cluster_pairing.py`
- Sec. 3.6: `initialization/histogram_init.py`, `icp/kiss_icp_backend.py`
- Eq. (2): inlier scoring in `icp/kiss_icp_backend.py`
- Sec. 3.7: `association/cluster_association.py`
- Eq. (1): scene-flow recovery in `flow/scene_flow.py`

All tunable parameters live in `configs/default.yaml`.
