import numpy as np

from lcicpflow.preprocessing.ground_filter import filter_ground


def test_z_threshold_ground_filter():
    pts = np.array([[0, 0, -2.0], [0, 0, -1.0], [0, 0, 0.5]])
    out, mask = filter_ground(pts, {"z_min": -3, "z_max": 3, "ground_z_threshold": -1.5})
    assert mask.tolist() == [False, True, True]
    assert out.shape == (2, 3)
