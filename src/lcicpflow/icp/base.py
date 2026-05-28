from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ICPResult:
    transform: np.ndarray
    average_distance: float
    inlier_ratio: float
    success: bool
    diagnostics: dict = field(default_factory=dict)
