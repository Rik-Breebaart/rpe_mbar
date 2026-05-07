"""
Prepare data structures for multiset MBAR solver input.
Bridge between rpe_mbar.data (loaded dicts) and rpe_mbar.core (solvers).
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from rpe_mbar.utils.tools import robust_flatten


def dict_to_solver_inputs_for_multiset(
    data: dict,
    origin_sets_order: Optional[list[str]] = None,
    target_sets_order: Optional[list[str]] = None,
    direction: str = "forward",
    log: bool = False,
):
    """Build inputs for solve_multi_set_mbar_matrix from the loaded data dict.

    Parameters
    ----------
    data : dict[origin][iface_str][target] -> 1D np.ndarray
    origin_sets_order : order of origin sets; defaults to dict insertion order
    target_sets_order : order of target sets; defaults to keys of first interface
    direction : "forward" or "backward"
    log : print debug info

    Returns
    -------
    max_lambda_values_each_set : list[list[np.ndarray]]
    N_lambda_each_set : list[np.ndarray]
    interface_values_lambda_each_set : list[np.ndarray]
    max_lambda_labeled_per_set : dict[target] -> 1D np.ndarray
    """
    if origin_sets_order is None:
        origin_sets_order = list(data.keys())
    if target_sets_order is None:
        first_ifaces = data[origin_sets_order[0]]
        target_sets_order = list(first_ifaces[list(first_ifaces.keys())[0]].keys())

    sort_key = (lambda v: -float(v)) if direction == "backward" else float
    ordered_interface_values = {
        s: [float(v) for v in sorted(data[s].keys(), key=sort_key)]
        for s in origin_sets_order
    }
    if log:
        print(ordered_interface_values)

    max_lambda_values_each_set: list[list[np.ndarray]] = []
    max_lambda_labeled_per_set: dict[str, np.ndarray] = {}

    for target in target_sets_order:
        all_iface_samples = []
        for origin in origin_sets_order:
            for k in ordered_interface_values[origin]:
                arr = data.get(origin, {}).get(str(k), {}).get(target, None)
                if arr is not None and np.size(arr) > 0:
                    arr = -arr if direction == "backward" else arr
                    all_iface_samples.append(np.asarray(arr, dtype=float).ravel())
        if log:
            print(f"Target {target}: {len(all_iface_samples)} arrays collected.")
        if target in origin_sets_order:
            max_lambda_values_each_set.append(all_iface_samples)
        max_lambda_labeled_per_set[target] = robust_flatten(all_iface_samples)

    N_lambda_each_set: list[np.ndarray] = []
    interface_values_lambda_each_set: list[np.ndarray] = []
    for cv_label in origin_sets_order:
        N_single_set = []
        iface_vals = []
        for iface in ordered_interface_values[cv_label]:
            target0 = list(data[cv_label][str(iface)].keys())[0]
            N_single_set.append(len(data[cv_label][str(iface)][target0]))
            iface_vals.append(float(iface))
        interface_values_lambda_each_set.append(np.asarray(iface_vals, dtype=float))
        N_lambda_each_set.append(np.asarray(N_single_set, dtype=int))

    if direction == "backward":
        interface_values_lambda_each_set = [-iv for iv in interface_values_lambda_each_set]

    return (
        max_lambda_values_each_set,
        N_lambda_each_set,
        interface_values_lambda_each_set,
        max_lambda_labeled_per_set,
    )


def weights_split_into_sets(weights_combined, data_max):
    """Split a flat combined weights array back into per-origin-set/interface dicts."""
    weights_each_set = {}
    global_index = 0
    for set_key in data_max:
        weights_each_set[set_key] = {}
        for iface in data_max[set_key]:
            N = data_max[set_key][iface][set_key].shape[0]
            weights_each_set[set_key][iface] = weights_combined[global_index:global_index + N]
            global_index += N
    return weights_each_set
