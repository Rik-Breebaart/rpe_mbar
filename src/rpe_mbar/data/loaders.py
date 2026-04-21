import numpy as np 
import pandas as pd
import json, time
from typing import Optional
from pathlib import Path

from rpe_mbar.rpe_mbar.utils.tools import robust_flatten

def load_max_values(csv_paths, interface_values, cv_column_base, direction="forward", n_thermalization=30):
    cv_col = f"max_{cv_column_base}" if direction == "forward" else f"min_{cv_column_base}"
    maxs = []
    for csv_file in csv_paths:
        df = pd.read_csv(csv_file)
        x = df[cv_col].values[n_thermalization:]
        maxs.append(x)
    return maxs

def save_weights(path: Path, weights: np.ndarray, *, meta: dict):
    payload = dict(meta)
    payload.update({
        "created": time.time(),
        "schema_version": 1,
    })
    np.savez_compressed(
        path,
        weights=weights.astype(float),
        metadata=json.dumps(payload, separators=(",", ":"))
    )

def load_weights(path: Path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(data["metadata"].item())
    return data["weights"], meta


def load_extrema_as_dict(
    run_folder: Path,
    system_name: str,
    direction: str,
    interface_functions,   # INTERFACE_FUNCTION_SETTINGS
    interfaces_forward,    # INTERFACE_VALUES_FORWARD_LAMBDA_M (list parallel to interface_functions) #TODO: change to interfaces_by_set
    n_thermalization: int
):
    """
    Returns:
      data: dict[origin_set][target_set][iface_val_str] -> 1D np.array
      interface_values_by_set: dict[target_set] -> list[str]
    """
    data = {}
    interface_values_by_set = {}

    # For each ORIGIN set, load the max values of *every* TARGET set CV at each ORIGIN interface
    for i_origin, origin_fset in enumerate(interface_functions):
        origin = origin_fset.filename
        data[origin] = {}

        # CSV paths are determined by ORIGIN set's CV label for the files being read
        origin_cv_label = origin_fset.filename
        origin_ifaces = interfaces_forward[i_origin]
        interface_values_by_set[origin] = [f"{v}" for v in origin_ifaces]
        data[origin] = {iface: {} for iface in interface_values_by_set[origin]}
        csv_paths_origin = [
            Path(run_folder) / direction / f"cv_extrema_{system_name}_cv_{origin_cv_label}_iface-{format_interface_value(v)}_{direction}.csv"
            for v in origin_ifaces
        ]

        # For each TARGET set, we want max values of its CV from the ORIGIN's runs
        for j_target, target_fset in enumerate(interface_functions):
            target = target_fset.filename
            target_cv_name = f"cv_{target_fset.name}"  # column name used by load_max_values

            # Load a list of arrays: one per interface in origin_ifaces (same length as csv_paths_origin)
            # Each array has shape (n_paths,) for that origin interface
            max_vals_list = load_max_values(
                csv_paths_origin,
                origin_ifaces,
                target_cv_name,
                direction,
                n_thermalization=n_thermalization
            )
            # max_vals_list: list[np.ndarray], aligned with origin_ifaces and cv values of TARGET interface_functions
            # Store in data dict
            # if target not in data[origin]:
            for idx, iface in enumerate(interface_values_by_set[origin]):
                data[origin][iface][target] = max_vals_list[idx]


    return data, interface_values_by_set


def determine_N_paths_per_interface(
        data: dict[str, dict[str, dict[str, np.ndarray]]],
        log: bool = False
) -> dict[str, dict[str, int]]:

    N_each_set = {}

    for cv_label in data.keys():
        N_each_set[cv_label] = {}
        for iface in data[cv_label].keys():
            target = list(data[cv_label][iface].keys())[0]
            if log:
                print(f"Origin {cv_label} iface {iface}: {len(data[cv_label][iface][target])} paths")
            N_each_set[cv_label][iface] = len(data[cv_label][iface][target])
    return N_each_set


def dict_to_solver_inputs_for_multiset(
    data: dict,                              # data[origin][iface_str][target] -> 1D np.array
    origin_sets_order: Optional[list[str]] = None,
    target_sets_order: Optional[list[str]] = None,
    direction: str = "forward",
    log: bool = False
):
    """
    Builds exactly the shapes/types expected by solve_multi_set_mbar_matrix.
    for a given data dict where it uses all origin sets as target sets. (so the sets simulated on are the once analyzed as it should be for Multi-Set MBAR).
    We output the max_lambda_values_each_set that is used for multisetMBAR and max_lambda_labeled_per_set the analysis max for all target sets that the simulation where analyzed on (this can be different from the origin sets)

    Assumes interface *indexing is aligned across sets* (kth origin iface aligns to kth target value).
    If grids differ, provide a mapping where marked below.
    """


    # Default orders
    if origin_sets_order is None:
        origin_sets_order = list(data.keys())
    if target_sets_order is None:
        target_sets_order = list(data[origin_sets_order[0]][list(data[origin_sets_order[0]].keys())[0]].keys())


    max_lambda_values_each_set: list[list[np.ndarray]] = []
    max_lambda_labeled_per_set = {}

    ordered_interface_values = {set: [float(v) for v in np.array(list(data[set].keys()))[np.argsort([-float(v) if direction == 'backward' else float(v) for v in data[set].keys()])]] for set in origin_sets_order}
    if log:
        print(ordered_interface_values)
    for target in target_sets_order:
        # Accumulate contributions from all origin sets
        all_iface_samples = []
        for origin in origin_sets_order:
            # origin_ifaces = origin_iface_list[origin]
            for k in ordered_interface_values[origin]:
                # origin_iface_str = origin_ifaces[k]
                arr = data.get(origin, {}).get(str(k), {}).get(target, None)
                if arr is not None and np.size(arr) > 0:
                    arr = -arr if direction == 'backward' else arr
                    # Force 1D float array
                    all_iface_samples.append(np.asarray(arr, dtype=float).ravel())
        if log:
            print(f"Target {target}: collected {len(all_iface_samples)} arrays from all origins.")
        # Counts per interface (np.ndarray[int])
        if target in origin_sets_order:
            max_lambda_values_each_set.append(all_iface_samples)                  # append to list a list of np.ndarray
        max_lambda_labeled_per_set[target] = robust_flatten(all_iface_samples)


    N_lambda_each_set: list[np.ndarray] = []
    interface_values_lambda_each_set: list[np.ndarray] = []
    for cv_label in origin_sets_order:
        N_single_set = []
        interfaces_values = []
        for iface in ordered_interface_values[cv_label]:
            target = list(data[cv_label][str(iface)].keys())[0]
            N_single_set.append(len(data[cv_label][str(iface)][target]))
            interfaces_values.append(float(iface))
        interface_values_lambda_each_set.append(np.asarray(interfaces_values, dtype=float))
        N_lambda_each_set.append(np.asarray(N_single_set, dtype=int))

    if direction == "backward":
        # set interface values to negative for backward direction
        interface_values_lambda_each_set = [-iv for iv in interface_values_lambda_each_set]

    return max_lambda_values_each_set, N_lambda_each_set, interface_values_lambda_each_set, max_lambda_labeled_per_set





def weights_split_into_sets(weights_combined, data_max):
    weights_each_set = {}
    global_index = 0
    for set_key in data_max.keys():
        weights_each_set[set_key] = {}
        for iface in data_max[set_key].keys():
            N_iface = data_max[set_key][iface][set_key].shape[0]
            weights_iface = weights_combined[global_index:global_index + N_iface]
            weights_each_set[set_key][iface] = weights_iface
            global_index += N_iface
    return weights_each_set


def format_interface_value(interface_value, scale=1000):
    """
    Convert a float interface value to an integer-style string for filenames.
    
    Example:
        0.45 -> '0450' if scale=1000
    """
    return f"{int(interface_value * scale):04d}"



def create_reduced_dataset(data,interface_values_by_set, N_each_set=None):
    """
    Create a reduced dataset containing only specified interfaces and number of paths.
    Parameters:
        data: dict[origin_set][target_set][iface_val_str] -> 1D np.array
        interface_values_by_set: dict[origin_set] -> list[str]
        N_each_set: Optional dict[origin_set][iface_val_str] -> int
    Returns:
        data_new: dict[origin_set][target_set][iface_val_str] -> 1D np.array
    """
    data_new = {set: {} for set in interface_values_by_set.keys()}
    for k in data.keys():
        if k in interface_values_by_set.keys():
            for interface in data[k].keys():
                if float(interface) in interface_values_by_set[k]:
                    if N_each_set is not None:
                        N_idx = N_each_set[k][interface]
                        data_new[k][interface] = {j: data[k][interface][j][:N_idx] for j in interface_values_by_set.keys() if j in data[k][interface].keys()}
                    else:
                        data_new[k][interface] = {j: data[k][interface][j] for j in interface_values_by_set.keys() if j in data[k][interface].keys()}
    return data_new

def create_reduced_dataset_with_all_targets(data,interface_values_by_set, N_each_set=None, start_idx=0):
    """
    Create a reduced dataset containing only specified interfaces but all target sets.
    Parameters:
        data: dict[origin_set][target_set][iface_val_str] -> 1D np.array
        interface_values_by_set: dict[origin_set] -> list[str]
        N_each_set: Optional dict[origin_set][iface_val_str] -> int
    Returns:
        data_new: dict[origin_set][target_set][iface_val_str] -> 1D np.array
    """
    data_new = {set: {} for set in interface_values_by_set.keys()}
    for k in data.keys():
        if k in interface_values_by_set.keys():
            for interface in data[k].keys():
                if float(interface) in interface_values_by_set[k]:
                    if N_each_set is not None:
                        N_idx = N_each_set[k][interface]
                        if start_idx+N_idx > len(data[k][interface][list(data[k][interface].keys())[0]]):
                            raise ValueError(f"start_idx {start_idx} + N_idx {N_idx} exceeds available data length {len(data[k][interface][list(data[k][interface].keys())[0]])} for origin set {k} interface {interface}.")
                        data_new[k][interface] = {j: data[k][interface][j][start_idx:start_idx+N_idx] for j in data[k][interface].keys()}
                    else:
                        data_new[k][interface] = {j: data[k][interface][j][start_idx:] for j in data[k][interface].keys()}
    return data_new

def bootstrap_TIS_set(data_tis_interface_set, block_size, total_length=None):
    """
    Bootstrap resampling for a TIS interface set with all its target sets changed accordingly.
    Parameters:
        data_tis_interface_set: dict[target_set] -> 1D np.array
        block_size: int, size of blocks for block bootstrapping
    Returns:
        data_new: dict[target_set] -> 1D np.array
    """
    data_new = {}
    raw_indices = np.arange(0, data_tis_interface_set[list(data_tis_interface_set.keys())[0]].shape[0], 1)
    if total_length is None:
        total_length = len(raw_indices)
    else: 
        total_length = min(total_length, len(raw_indices))
    n_blocks = len(raw_indices) // block_size
    n_blocks_sample = total_length // block_size
    blocks = np.array_split(raw_indices[:n_blocks * block_size], n_blocks)
    resampled_indices = [blocks[np.random.randint(len(blocks))] for _ in range(n_blocks_sample)]
    resampled_indices = np.concatenate(resampled_indices)[:total_length]  # trim to match original size
    data_new["origin_indices"] = resampled_indices
    for target_set in data_tis_interface_set.keys():
        resampled_data = data_tis_interface_set[target_set][resampled_indices]
        data_new[target_set] = resampled_data
    return data_new


def bootstrap_data(data, block_size, total_length=None):
    """
    Bootstrap resampling for the entire dataset across all origin sets and interfaces.
    Parameters:
        data: dict[origin_set][iface_str][target_set] -> 1D np.array
        block_size: int, size of blocks for block bootstrapping
    Returns:
        data_new: dict[origin_set][iface_str][target_set] -> 1D np.array
    """
    data_new = {set: {} for set in data.keys()}
    for k in data.keys():
        for interface in data[k].keys():
            bootstrap_tis_set = bootstrap_TIS_set(data[k][interface], block_size, total_length=total_length)
            data_new[k][interface] = {j: bootstrap_tis_set[j] for j in bootstrap_tis_set.keys()}
    return data_new


def show_keys(data):
    for origin in data.keys():
        print(f"Origin set: {origin}, with {len(data[origin].keys())} interfaces: {list(data[origin].keys())}")
        for interface in data[origin].keys():
            print(f"  Interface: {interface} of all {list(data[origin].keys())}")
            print(f"    Target sets: {list(data[origin][interface].keys())}, number of paths: {len(data[origin][interface][list(data[origin][interface].keys())[0]])}")