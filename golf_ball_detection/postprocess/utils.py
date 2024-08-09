import numpy as np


def peak_mask_to_points(peak_mask):
    """Get the XY coordinates of the positive pixels."""
    return [(x, y) for (y, x) in zip(*np.where(peak_mask))]
