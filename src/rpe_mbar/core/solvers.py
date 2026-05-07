import numpy as np
from typing import Dict, List, Tuple
from ..core.solver_prep import dict_to_solver_inputs_for_multiset
from .multiset import solve_multi_set_mbar_matrix

def solve_weights_multiset(
    data_red: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    direction: str,
    log: bool = False,
    log_details: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[List[float]]]:
    """
    [MOVE_TO:rpe_mbar.solvers]
    Solve multi-set MBAR for the *combined* dataset.
    Returns:
      - weights_flat (aligned with get_descriptors_and_weights order)
      - max_labeled_per_set
      - interface_values_list
    """
    max_lam, N_list, iface_list, max_labeled = dict_to_solver_inputs_for_multiset(
        data_red, direction=direction, log=log_details
    )
    _, w = solve_multi_set_mbar_matrix(
        max_lam,
        N_list,
        [iface_list[set] for set in range(len(iface_list))],
        max_iterations=10000,
        tolerance=1e-10,
        log=log,
    )
    return w, max_labeled, iface_list

def solve_weights_naive_concat(
    data_red: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    direction: str,
    log: bool = False,
    log_details: bool = False,
) -> np.ndarray:
    """
    Naive baseline:
      - Solve MBAR *per origin set separately* (single-set MBAR),
      - Normalize each set's weights by its minimum,
      - Concatenate in the iteration order of data_red.keys().

    This matches the spirit of your crossing-probability naive example.

    If you later want "reactive matching" scaling, add it here (see match_naive_weights()).
    """
    weights_chunks: List[np.ndarray] = []
    for setname in data_red.keys():
        data_one = {setname: data_red[setname]}
        max_lam, N_list, iface_list, _max_labeled = dict_to_solver_inputs_for_multiset(
            data_one, direction=direction, log=log_details
        )
        _, w = solve_multi_set_mbar_matrix(
            max_lam,
            N_list,
            iface_list,
            max_iterations=10000,
            tolerance=1e-10,
            log=log,
        )
        w = w / np.min(w)
        weights_chunks.append(w)

    w_concat = np.concatenate(weights_chunks) if weights_chunks else np.array([], dtype=float)
    return w_concat
