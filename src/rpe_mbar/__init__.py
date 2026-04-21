from ._version import __version__  # optional

from .core.multiset import solve_two_set_mbar_matrix, solve_multi_set_mbar_matrix
from .core.solvers import solve_weights_multiset, solve_weights_naive_concat
from .utils.tools import robust_flatten
from .diag.crossing import conditional_crossing_histogram, cumulative_crossing_histogram, bootstrap_crossing_probs, bootstrap_crossing_probs_naive_weighting
from .data.loaders import load_max_values, save_weights, load_weights, load_extrema_as_dict, determine_N_paths_per_interface, dict_to_solver_inputs_for_multiset, create_reduced_dataset, bootstrap_data, create_reduced_dataset_with_all_targets, show_keys, weights_split_into_sets

__all__ = [
    "solve_two_set_mbar_matrix",
    "solve_multi_set_mbar_matrix",
    "robust_flatten",
    "conditional_crossing_histogram",
    "cumulative_crossing_histogram",
    "load_max_values",
    "save_weights",
    "load_weights",
    "load_extrema_as_dict",
    "determine_N_paths_per_interface",
    "dict_to_solver_inputs_for_multiset",
    "__version__",
]