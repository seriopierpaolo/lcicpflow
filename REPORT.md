# REPORT

## New Code Structure

The refactor creates a new package under `src/lcicpflow/`, with one small module per algorithmic phase. The pipeline entry points are `pipeline/run_pair.py` and `pipeline/run_folder.py`; tools and tests live outside the package.

## Mapping To ICP-Flow

- Input loading: `io/pcd_reader.py`
- Ego-motion compensation: `preprocessing/ego_motion.py`
- Ground removal: `preprocessing/ground_filter.py`
- Clustering: `clustering/hdbscan_clusterer.py`
- Cluster pairing: `pairing/cluster_pairing.py`
- Histogram initialization: `initialization/histogram_init.py`
- ICP and quality metrics: `icp/kiss_icp_backend.py`
- Association: `association/cluster_association.py`
- Scene-flow recovery: `flow/scene_flow.py`
- Evaluation: `metrics/scene_flow_metrics.py`

## Differences From The Paper

Patchwork++ is replaced by a simple z-threshold filter. This is easier to inspect and tune, but it may be less accurate on sloped roads or uneven terrain.

The paper uses PyTorch3D ICP. This refactor uses Open3D point-to-point ICP by default with a NumPy fallback. KISS-ICP is kept isolated behind the ICP backend because its common API is scan-odometry oriented rather than local cluster-pair registration oriented.

GPU support is configuration-visible but conservative. The current heavy operations are CPU-bound unless a future backend adds CuPy/KISS-ICP acceleration.

## Differences From The Original Repository

The original code is flat and mixes dataset logic, CUDA histogram code, matching, visualization, and debugging utilities. This refactor removes those couplings and keeps each algorithmic phase in a focused module. ROS support is only a placeholder adapter, not a core dependency.

## What Is Simpler

Configuration is centralized in `configs/default.yaml`. PCD loading is direct. Ground removal is a transparent z filter. ICP matching returns a small `ICPResult` with transform, distance, ratio, success flag, and diagnostics.

## What Is Faster

The histogram initialization samples large clusters and processes vote tensors in chunks to avoid excessive memory use. Runtime is recorded per stage in `runtime.json`.

## What May Be Less Accurate

The simplified ground removal can leave ground points in clusters or remove low objects. DBSCAN fallback may behave differently from HDBSCAN if `hdbscan` is not installed. The Open3D ICP backend may differ numerically from the paper's PyTorch3D implementation.

## Known Limitations

No hard ROS dependency is included. KISS-ICP is not wired as a mandatory local registration backend. Supervised metrics require external ground-truth flow.

## Recommended Future Work

Add a true local-cluster KISS-ICP adapter if its API is suitable in the target environment. Add pose-file discovery for ROS bag exports. Add CUDA histogram voting for large clusters. Reintroduce Patchwork++ as an optional ground filter plugin.
