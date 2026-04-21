import numpy as np
from collections.abc import Iterable

def robust_flatten(input_data):
    """Flatten a nested structure while ensuring homogeneous types.
    Parameters:
        input_data (Iterable): Nested structure to flatten.
    Returns:
        np.ndarray: Flattened array with homogeneous types.     
    """
    if isinstance(input_data, np.ndarray) and input_data.ndim == 1 and input_data.dtype != object:
        return input_data  # Already flat and homogeneous
    return np.array([
        item
        for sublist in input_data
        for item in (
            sublist if isinstance(sublist, Iterable) and not isinstance(sublist, (str, bytes)) else [sublist]
        )
    ])

def find_last_crossed_interfaces(max_values, interface_positions):
    """    Find the last interface index crossed by each maximum value.
    Parameters:
        max_values (np.ndarray): Array of maximum values.
        interface_positions (np.ndarray): Array of interface positions.
    Returns:
        np.ndarray: Indices of the last crossed interfaces for each maximum value.
    """
    interface_positions = np.sort(interface_positions)
    indices = np.searchsorted(interface_positions, max_values, side='right') - 1
    indices = np.clip(indices, -1, len(interface_positions) - 1)
    return indices

