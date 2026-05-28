import numpy as np

from lcicpflow.flow.scene_flow import recover_scene_flow
from lcicpflow.utils.geometry import make_transform


def test_scene_flow_recovery():
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    labels = np.array([0, 0])
    T = make_transform(translation=np.array([1, 2, 3]))
    flow = recover_scene_flow(pts, labels, {0: T}, np.eye(4))
    assert np.allclose(flow, np.array([[1, 2, 3], [1, 2, 3]]))
