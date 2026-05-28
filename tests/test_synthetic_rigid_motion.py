import numpy as np

from lcicpflow.icp.kiss_icp_backend import ICPBackend
from lcicpflow.initialization.histogram_init import histogram_translation_init


def test_synthetic_rigid_translation_icp():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(60, 3))
    dst = src + np.array([0.3, 0.2, 0.0])
    hist_cfg = {"bin_size": 0.1, "bounds_x": [-1, 1], "bounds_y": [-1, 1], "bounds_z": [-1, 1], "max_votes_per_cluster_points": 60, "chunk_size": 100000}
    icp_cfg = {"min_points": 5, "max_iterations": 30, "tolerance": 1e-7, "max_correspondence_distance": 0.5, "inlier_threshold": 0.1, "max_average_distance": 0.05, "min_inlier_ratio": 0.5}
    init = histogram_translation_init(src, dst, hist_cfg, rng)
    result = ICPBackend(icp_cfg).register(src, dst, init)
    assert result.success
    assert np.allclose(result.transform[:3, 3], [0.3, 0.2, 0.0], atol=0.03)
