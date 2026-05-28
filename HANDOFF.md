# HANDOFF

## Goal

Create a clean private repository named `lcicpflow` that preserves the ICP-Flow paper architecture while making the implementation easier to read, modify, test, and run on `.pcd` files from `data_from_rosbag/`.

## Current Progress

The repository structure, package, config, Docker files, tools, tests, and documentation have been created. The core pipeline supports single-pair and folder inference on `.pcd` files.

## What Worked

The PCD reader supports the binary `FIELDS x y z` files present in `../data_from_rosbag`. The pipeline is modular and saves `.npz`, `runtime.json`, `config_used.yaml`, and `log.txt` per sample.

## What Didn't Work

The host Python environment is missing scientific dependencies such as NumPy, SciPy, Open3D, HDBSCAN, Matplotlib, and pytest. Use Docker or install the package into a proper Python environment.

## Next Steps

1. Build the Docker image.
2. Run tests inside Docker.
3. Run one inference pair from `../data_from_rosbag`.
4. Inspect and visualize the output.

## Reproduction Commands

Install locally:

```bash
cd lcicpflow
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e ".[dev,hdbscan]"
```

Build Docker:

```bash
cd lcicpflow
docker compose -f docker/docker-compose.yml build
```

Run tests:

```bash
docker compose -f docker/docker-compose.yml run --rm lcicpflow pytest
```

Run inference:

```bash
docker compose -f docker/docker-compose.yml run --rm lcicpflow \
  lcicpflow-folder --config configs/default.yaml --input-folder /workspace/project/data_from_rosbag --max-pairs 1
```

Inspect output:

```bash
python tools/inspect_npz_output.py outputs/inference/<sample>/prediction.npz
python tools/visualize_flow.py outputs/inference/<sample>/prediction.npz --output outputs/inference/<sample>/visualization.png
python tools/evaluate_metrics.py outputs/inference/<sample>/prediction.npz
```

## Important Paths

- Package: `src/lcicpflow/`
- Config: `configs/default.yaml`
- Docker: `docker/Dockerfile`, `docker/docker-compose.yml`
- Outputs: `outputs/inference/`
