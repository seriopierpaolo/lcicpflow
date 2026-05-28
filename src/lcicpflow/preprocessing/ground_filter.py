from __future__ import annotations

import numpy as np


def filter_ground(points: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Simple z-threshold ground removal.

    ICP-Flow Sec. 3.4 uses Patchwork++ before clustering. This implementation
    deliberately uses a transparent z filter so it is easy to tune or replace.
    """
    xyz = np.asarray(points)
    z = xyz[:, 2]
    threshold = cfg["ground_z_threshold"]
    if cfg.get("adaptive_percentile") is not None:
        threshold = np.percentile(z, float(cfg["adaptive_percentile"])) + float(cfg.get("adaptive_margin", 0.0))
    mask = (z >= cfg["z_min"]) & (z <= cfg["z_max"]) & (z > threshold)
    return xyz[mask], mask



import numpy as np

def crop_xy(points: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Simple CropBox Filter to process only meaningful points."""
    xyz = np.asarray(points)
    x = xyz[:, 0]
    y = xyz[:, 1]
    
    # Looking up the specific threshold key inside the configuration block
    threshold = cfg["xy_threshold"]
    
    # Override with adaptive percentile calculation based on both X and Y if requested
    if cfg.get("adaptive_percentile") is not None:
        pct = float(cfg["adaptive_percentile"])
        margin = float(cfg.get("adaptive_margin", 0.0))
        # Combines both X and Y points to find a single shared percentile threshold
        threshold = np.percentile(np.concatenate([x, y]), pct) + margin
        
    # Apply standard bounding box constraints alongside the shared threshold
    mask = (
        (x >= cfg["x_min"]) & (x <= cfg["x_max"]) & (x > threshold) & 
        (y >= cfg["y_min"]) & (y <= cfg["y_max"]) & (y > threshold)
    )
    
    return xyz[mask], mask