import numpy as np

def histogram2d_weighted(
    descriptors: np.ndarray,
    weights: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    density: bool = True,
    descriptor_indices: tuple = (0, 1),
) -> np.ndarray:
    """
    [MOVE_TO:rpe_mbar.hist] This is reusable across many analyses.
    """
    H, _, _ = np.histogram2d(
        descriptors[:, descriptor_indices[0]],
        descriptors[:, descriptor_indices[1]],
        bins=[xedges, yedges],
        weights=weights,
        density=density,
    )
    return H
