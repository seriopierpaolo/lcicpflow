import numpy as np

from lcicpflow.initialization.histogram_init import histogram_translation_init


def test_histogram_translation_init_recovers_translation():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(80, 3))
    t = np.array([0.4, -0.2, 0.1])
    dst = src + t
    cfg = {"bin_size": 0.1, "bounds_x": [-1, 1], "bounds_y": [-1, 1], "bounds_z": [-1, 1], "max_votes_per_cluster_points": 80, "chunk_size": 100000}
    T = histogram_translation_init(src, dst, cfg, rng)
    assert np.allclose(T[:3, 3], t, atol=0.06)
