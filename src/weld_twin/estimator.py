import numpy as np

from .scene import PLATE_T


def estimate_seam_offset(y_local, z):
    """Lateral seam offset relative to the torch (m), or None if no groove seen."""
    seen_floor = np.isfinite(z) & (z < -PLATE_T / 2)
    idx = np.flatnonzero(seen_floor)
    if idx.size < 2:
        return None
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(runs, key=len)
    if run.size < 2:
        return None
    return float((y_local[run[0]] + y_local[run[-1]]) / 2)
