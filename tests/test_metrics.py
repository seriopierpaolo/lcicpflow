import numpy as np

from lcicpflow.metrics.scene_flow_metrics import scene_flow_metrics


def test_metrics_perfect():
    gt = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    m = scene_flow_metrics(gt, gt)
    assert m["EPE3D"] == 0.0
    assert m["AccS"] == 1.0
    assert m["AccR"] == 1.0
