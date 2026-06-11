
from ensurepip import bootstrap
from tkinter import N
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..core.multiset import solve_multi_set_mbar_matrix
from ..utils.tools import robust_flatten
from ..core.solver_prep import dict_to_solver_inputs_for_multiset
from ..data.sampling import bootstrap_data, create_reduced_dataset_with_all_targets

def conditional_crossing_histogram(max_values, weights, bins=np.linspace(-1, 1, 500), interface_0=None):
    """
    Compute the conditional crossing probability histogram.

    Only includes paths that cross the first interface (i.e., have max_value >= bins[0]).
    Then calculates the cumulative crossing probability for higher interfaces.

    Parameters
    ----------
    max_values : np.ndarray
        Maximum values reached by each path.
    weights : np.ndarray
        Path weights (same shape as max_values).
    bins : np.ndarray
        Array of bin edges (interfaces) to evaluate.

    Returns
    -------
    crossing_probabilities : np.ndarray
        Weighted conditional crossing probabilities for each bin.
    bins : np.ndarray
        Bin values used (same as input).
    """

    # Filter paths that cross at least the first interface
    if interface_0 is None:
        interface_0 = bins[0]
    valid = max_values >= interface_0
    max_values = max_values[valid]
    weights = weights[valid]

    # Total weight of all valid paths (normalization)
    total_weight = np.sum(weights)
    if total_weight == 0:
        raise ValueError("No paths cross the first interface. Cannot compute conditional probabilities.")

    # Create cumulative contribution matrix
    contribution_matrix = (max_values[:, None] >= bins).astype(float)
    # Weight each row by the path's weight
    weighted_contributions = contribution_matrix * weights[:, None]

    # Normalize by total weight of valid paths
    crossing_probabilities = weighted_contributions.sum(axis=0) / total_weight

    return crossing_probabilities, bins

def cumulative_crossing_histogram(max_values, weights, bins=np.linspace(-1, 1, 100)):
    contribution_matrix = (max_values[:, None] >= bins).astype(float)
    weighted_contributions = contribution_matrix * weights[:, None]
    crossing_probabilities = weighted_contributions.sum(axis=0)
    return crossing_probabilities, bins

def bootstrap_crossing_probs(data, interface_0, n_bootstrap=100, block_size=10, set_analysis="rot_0", 
                             direction="forward",bins=np.linspace(-1, 1, 500),seed=None, n_interface_paths=None, return_weights=False, log=False):
    """
    Bootstrap the crossing probabilities for a given dataset using Multi-Set MBAR.
    Parameters:
        data: dict[origin_set][target_set][iface_val_str] -> 1D np.array
        interface_0: float, the first interface value to condition on in set "set_analysis"
        direction: str, "forward" or "backward"
        block_size: int, size of blocks for block bootstrapping 
        n_bootstrap: int, number of bootstrap samples
        set_analysis: int, index of the set to analyze
        bins: np.ndarray, bin edges for crossing histogram
        seed: int, random seed for reproducibility
        n_interface_paths: int, number of paths in the interface set to match during bootstrapping when provided 
    Returns:
        bins: np.ndarray, bin edges used
        bootstrap_array: np.ndarray, shape (n_bootstrap, len(bins)), crossing probabilities from each bootstrap sample
        bootstrap_mean: np.ndarray, mean crossing probabilities across bootstrap samples
        bootstrap_std: np.ndarray, std deviation of crossing probabilities across bootstrap samples 
    """


    if set_analysis not in data[list(data.keys())[0]][list(data[list(data.keys())[0]].keys())[0]].keys():
        raise ValueError(f"Set {set_analysis} not found in data keys: {list(data.keys())} add data_analysis to bootstrap_crossing_probs function call.")

    bootstrap_results = []
    bootstrap_weights_each_set = {set: [] for set in data.keys()}

    for i_boot in range(n_bootstrap):
        if log:
            print(f"Bootstrap sample {i_boot+1}/{n_bootstrap}")
        if seed is not None:
            np.random.seed(seed+i_boot)
        data_bootstrap = bootstrap_data(data, block_size, total_length=n_interface_paths)
        # Prepare inputs for Multi-Set MBAR
        max_lambda_combined_each_set, N_each_set_list, interface_values_list, max_labeled_per_set = dict_to_solver_inputs_for_multiset(data_bootstrap, direction=direction)

        # Solve Multi-Set MBAR
        Z_lambda_set, weights_combined_forward = solve_multi_set_mbar_matrix(
            max_lambda_combined_each_set,
            N_each_set_list,
            interface_values_list,
            max_iterations=10000,
            tolerance=1e-10,
            log=log
        )
        if return_weights:
            for i,set in enumerate(data.keys()):
                weights_set = weights_combined_forward[int(np.sum(N_each_set_list[:i])):int(np.sum(N_each_set_list[:i+1]))]
                bootstrap_weights_each_set[set].append(weights_set/np.min(weights_combined_forward))


        if len(max_labeled_per_set[set_analysis]) != len(weights_combined_forward):
            raise ValueError("Length mismatch between max_labeled_per_set and weights_combined_forward.")

        crossing_prob_lambda, bins = conditional_crossing_histogram(
                max_labeled_per_set[set_analysis],
                weights_combined_forward / np.min(weights_combined_forward), interface_0=interface_0,
                bins=bins
            )

        bootstrap_results.append(crossing_prob_lambda)
    
    bootstrap_array = np.array(bootstrap_results)
    bootstrap_mean = np.mean(bootstrap_array, axis=0)
    bootstrap_std = np.std(bootstrap_array, axis=0)
    bootstrap_log_mean = np.mean(np.log(bootstrap_array + 1e-300), axis=0)
    bootstrap_log_std = np.std(np.log(bootstrap_array + 1e-300), axis=0)



    return {"bins": bins, 
            "bootstrap_array": bootstrap_array, 
            "bootstrap_mean": bootstrap_mean, 
            "bootstrap_std": bootstrap_std, 
            "bootstrap_log_mean": bootstrap_log_mean, 
            "bootstrap_log_std": bootstrap_log_std,
            "bootstrap_weights_each_set": bootstrap_weights_each_set if return_weights else None}

def bootstrap_crossing_probs_naive_weighting(data, interface_0, n_bootstrap=100, block_size=10, set_analysis="rot_0", direction="forward",bins=np.linspace(-1, 1, 500),seed=None, n_interface_paths=None, return_weights=False):
    """
    Bootstrap the crossing probabilities for a given dataset using naive single-set MBAR weighting.
    Parameters:
        data: dict[origin_set][target_set][iface_val_str] -> 1D np.array
        interface_0: float, the first interface value to condition on in set "set"
        direction: str, "forward" or "backward"
        block_size: int, size of blocks for block bootstrapping 
        n_bootstrap: int, number of bootstrap samples
        set: int, index of the set to analyze
        bins: np.ndarray, bin edges for crossing histogram
        seed: int, random seed for reproducibility
        n_interface_paths: int, number of paths in the interface set to match during bootstrapping when provided 
    Returns:
        bins: np.ndarray, bin edges used
        bootstrap_array: np.ndarray, shape (n_bootstrap, len(bins)), crossing probabilities from each bootstrap sample
        bootstrap_mean: np.ndarray, mean crossing probabilities across bootstrap samples
        bootstrap_std: np.ndarray, std deviation of crossing probabilities across bootstrap samples 
    """


    bootstrap_results = []
    bootstrap_results_single_set = {set: [] for set in data.keys()}
    if return_weights:
        bootstrap_weights_single_set = {set: [] for set in data.keys()}

    for i_boot in range(n_bootstrap):
        if seed is not None:
            np.random.seed(seed+i_boot) #place seed here for reproducibility and to ensure that the adding a new set always has the same bootstrap samples when adding more M.
                                        # so that comparisons when adding more M sets are valid (it is really just adding more data, not changing the bootstrap samples)
        data_bootstrap = bootstrap_data(data, block_size, total_length=n_interface_paths)
        weights_single_set = {}
        for set in data.keys():
            # create reduced dataset
            interfaces_selected_set = {set: [float(i) for i in data[set].keys()]}
            # data_reduced = rpe_mbar.create_reduced_dataset(data_bootstrap,interfaces_selected_set)
            data_reduced = create_reduced_dataset_with_all_targets(data_bootstrap,interfaces_selected_set)

            # Prepare inputs for Multi-Set MBAR
            max_lambda_combined_each_set, N_each_set_list, interface_values_list, max_labeled_per_set = dict_to_solver_inputs_for_multiset(data_reduced)
            # Solve Multi-Set MBAR
            Z_lambda_set, weights_combined_forward = solve_multi_set_mbar_matrix(
                max_lambda_combined_each_set,
                N_each_set_list,
                interface_values_list,
                max_iterations=10000,
                tolerance=1e-10
            )
            weights_single_set[set] = weights_combined_forward
            if return_weights:
                bootstrap_weights_single_set[set].append(weights_combined_forward/np.min(weights_combined_forward))

            crossing_prob_naive, bins_naive = conditional_crossing_histogram(
                    max_labeled_per_set[set_analysis],
                    weights_combined_forward / np.min(weights_combined_forward), interface_0=interface_0,
                    bins=bins
            )
            bootstrap_results_single_set[set].append(crossing_prob_naive)

            

        # Prepare inputs for Multi-Set MBAR
        max_lambda_combined_each_set, N_each_set_list, interface_values_list, max_labeled_per_set = dict_to_solver_inputs_for_multiset(data_bootstrap)

        weights_naive = [weights_single_set[set]/np.min(weights_single_set[set]) for set in data.keys()]
        weights_naive = robust_flatten(weights_naive)

        crossing_prob_naive, bins_naive = conditional_crossing_histogram(
            max_labeled_per_set[set_analysis],
            weights_naive / np.min(weights_naive), interface_0=interface_0,
            bins=bins
        )

        bootstrap_results.append(crossing_prob_naive)

    bootstrap_array = np.array(bootstrap_results)
    bootstrap_mean = np.mean(bootstrap_array, axis=0)
    bootstrap_std = np.std(bootstrap_array, axis=0)
    bootstrap_log_mean = np.mean(np.log(bootstrap_array + 1e-300), axis=0)
    bootstrap_log_std = np.std(np.log(bootstrap_array + 1e-300), axis=0)



    return {"bins": bins_naive, "bootstrap_array": bootstrap_array, "bootstrap_mean": bootstrap_mean, "bootstrap_std": bootstrap_std, "bootstrap_log_mean": bootstrap_log_mean, "bootstrap_log_std": bootstrap_log_std, "bootstrap_single_set": bootstrap_results_single_set, "bootstrap_weights_single_set": bootstrap_weights_single_set if return_weights else None}

def plot_bootstrap_cumulative_per_interface(data,origin_set, interface_values, direction="forward", n_thermalization=30, fig = None, ax = None):
    N_bootstrap = 100
    block_size = 10
    n_bins = np.arange(-1, 1, 0.001)
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = plt.get_cmap("tab20b")(np.linspace(0, 1, len(interface_values)))

    for ind, iface in enumerate(interface_values):
        x = data[origin_set][iface][origin_set]
        print(x)
        if direction == "backward":
            x = -x
        x_len = len(x)
        all_y = []
        for _ in range(N_bootstrap):
            blocks = [x[i:i + block_size] for i in range(0, len(x) - block_size + 1)]
            n_blocks = int(np.ceil(x_len / block_size))
            boot_x = np.concatenate([blocks[np.random.randint(len(blocks))] for _ in range(n_blocks)])[:x_len]
            n, bins = np.histogram(boot_x, n_bins, density=True)
            dx = np.diff(bins)
            y = 1 - np.cumsum(n) * dx
            all_y.append(y)

        all_y = np.array(all_y)
        mean_y = all_y.mean(axis=0)
        std_y = all_y.std(axis=0)
        xdata = bins[1:]

        color = colors[ind]
        ax.plot(xdata, mean_y, "-", linewidth=0.5, color="black")
        ax.fill_between(xdata, mean_y - std_y, mean_y + std_y, color=color, alpha=0.3)
        # iface_plot = -iface if direction == "backward" else iface
        # ax.vlines(iface_plot, ymin=0, ymax=1, color=color)

    ax.set_yscale("log")
    ax.set_ylabel("Probability")
    ax.set_ylim([0.0001, 1.5])
    ax.grid(True)
    return fig, ax
