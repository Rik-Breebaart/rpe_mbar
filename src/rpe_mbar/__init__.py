from ._version import __version__  # optional

from .core.multiset import solve_two_set_mbar_matrix, solve_multi_set_mbar_matrix
from .core.solvers import solve_weights_multiset, solve_weights_naive_concat
from .core.solver_prep import dict_to_solver_inputs_for_multiset, weights_split_into_sets
from .utils.tools import robust_flatten, format_interface_value
from .diag.crossing import conditional_crossing_histogram, cumulative_crossing_histogram, bootstrap_crossing_probs, bootstrap_crossing_probs_naive_weighting, plot_bootstrap_cumulative_per_interface
from .data.specs import InterfaceFunctionSpec, coerce_interface_spec
from .data.extrema import (
    expected_extrema_columns, validate_extrema_df,
    save_extrema_csv, load_extrema_csv,
    compute_cv_extrema, compute_and_save_cv_extrema,
)
from .data.loaders import load_max_values, load_extrema_as_dict, load_extrema_as_dict_iterative
from .data.sampling import (
    determine_N_paths_per_interface,
    create_reduced_dataset, create_reduced_dataset_with_all_targets,
    bootstrap_TIS_set, bootstrap_data, show_keys,
)
from .data.weights import save_weights, load_weights
from .rpe.dataset import (
    TrajectoryInfo,
    InterfaceData,
    SetData,
    RpeDataset,
    build_rpe_dataset,
    convert_old_to_new_format,
)
from .rpe.stable import (
    compute_flux_compensation_stable_states_multiple_interfaces,
    compute_multi_interface_passed_flags,
    create_total_trainset,
)
from .rpe.assemble import (
    get_descriptors_and_weights,
    build_reweighted_rpe_trainset,
)

__all__ = [
    "__version__",
    # core solvers
    "solve_two_set_mbar_matrix",
    "solve_multi_set_mbar_matrix",
    "solve_weights_multiset",
    "solve_weights_naive_concat",
    # core prep
    "dict_to_solver_inputs_for_multiset",
    "weights_split_into_sets",
    # utils
    "robust_flatten",
    "format_interface_value",
    # diag
    "conditional_crossing_histogram",
    "cumulative_crossing_histogram",
    "bootstrap_crossing_probs",
    "bootstrap_crossing_probs_naive_weighting",
    "plot_bootstrap_cumulative_per_interface",
    # data specs
    "InterfaceFunctionSpec",
    "coerce_interface_spec",
    # data extrema
    "expected_extrema_columns",
    "validate_extrema_df",
    "save_extrema_csv",
    "load_extrema_csv",
    "compute_cv_extrema",
    "compute_and_save_cv_extrema",
    # data loaders
    "load_max_values",
    "load_extrema_as_dict",
    "load_extrema_as_dict_iterative",
    # data sampling
    "determine_N_paths_per_interface",
    "create_reduced_dataset",
    "create_reduced_dataset_with_all_targets",
    "bootstrap_TIS_set",
    "bootstrap_data",
    "show_keys",
    # data weights
    "save_weights",
    "load_weights",
    # rpe dataset
    "TrajectoryInfo",
    "InterfaceData",
    "SetData",
    "RpeDataset",
    "build_rpe_dataset",
    "convert_old_to_new_format",
    # rpe stable-state helpers
    "compute_flux_compensation_stable_states_multiple_interfaces",
    "compute_multi_interface_passed_flags",
    "create_total_trainset",
    # rpe assembly
    "get_descriptors_and_weights",
    "build_reweighted_rpe_trainset",
]