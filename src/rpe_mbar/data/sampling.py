"""
Dataset manipulation: reduce, bootstrap, inspect.
Operates on the loaded multiset dict structure:
  data[origin_set][iface_str][target_set] -> 1D np.ndarray
"""
from __future__ import annotations
import numpy as np


def determine_N_paths_per_interface(data, log=False):
    """Return the number of paths for each origin set and interface."""
    N_each_set = {}
    for cv_label in data:
        N_each_set[cv_label] = {}
        for iface in data[cv_label]:
            target = list(data[cv_label][iface].keys())[0]
            n = len(data[cv_label][iface][target])
            if log:
                print(f"Origin {cv_label} iface {iface}: {n} paths")
            N_each_set[cv_label][iface] = n
    return N_each_set


def create_reduced_dataset(data, interface_values_by_set, N_each_set=None):
    """Keep only the specified interfaces and optionally truncate path counts."""
    data_new = {s: {} for s in interface_values_by_set}
    for k in data:
        if k not in interface_values_by_set:
            continue
        for interface in data[k]:
            if float(interface) not in interface_values_by_set[k]:
                continue
            targets = {j for j in interface_values_by_set if j in data[k][interface]}
            if N_each_set is not None:
                N = N_each_set[k][interface]
                data_new[k][interface] = {j: data[k][interface][j][:N] for j in targets}
            else:
                data_new[k][interface] = {j: data[k][interface][j] for j in targets}
    return data_new


def create_reduced_dataset_with_all_targets(data, interface_values_by_set, N_each_set=None, start_idx=0):
    """Keep only the specified interfaces but retain all target sets."""
    data_new = {s: {} for s in interface_values_by_set}
    for k in data:
        if k not in interface_values_by_set:
            continue
        for interface in data[k]:
            if float(interface) not in interface_values_by_set[k]:
                continue
            if N_each_set is not None:
                N = N_each_set[k][interface]
                avail = len(data[k][interface][list(data[k][interface].keys())[0]])
                if start_idx + N > avail:
                    raise ValueError(
                        f"start_idx {start_idx} + N {N} exceeds available length {avail} "
                        f"for origin={k} iface={interface}."
                    )
                data_new[k][interface] = {j: data[k][interface][j][start_idx:start_idx + N] for j in data[k][interface]}
            else:
                data_new[k][interface] = {j: data[k][interface][j][start_idx:] for j in data[k][interface]}
    return data_new


def bootstrap_TIS_set(data_tis_interface_set, block_size, total_length=None):
    """Block bootstrap resampling for a single TIS interface set."""
    raw_indices = np.arange(len(data_tis_interface_set[list(data_tis_interface_set.keys())[0]]))
    total_length = min(total_length, len(raw_indices)) if total_length is not None else len(raw_indices)
    n_blocks = len(raw_indices) // block_size
    n_blocks_sample = total_length // block_size
    blocks = np.array_split(raw_indices[:n_blocks * block_size], n_blocks)
    resampled_indices = np.concatenate(
        [blocks[np.random.randint(len(blocks))] for _ in range(n_blocks_sample)]
    )[:total_length]

    data_new = {"origin_indices": resampled_indices}
    for target_set, arr in data_tis_interface_set.items():
        data_new[target_set] = arr[resampled_indices]
    return data_new


def bootstrap_data(data, block_size, total_length=None):
    """Block bootstrap the entire dataset across all origin sets and interfaces."""
    data_new = {s: {} for s in data}
    for k in data:
        for interface in data[k]:
            data_new[k][interface] = bootstrap_TIS_set(data[k][interface], block_size, total_length=total_length)
    return data_new


def show_keys(data):
    """Print a human-readable summary of the dataset structure."""
    for origin in data:
        print(f"Origin set: {origin}, interfaces: {list(data[origin].keys())}")
        for interface in data[origin]:
            targets = list(data[origin][interface].keys())
            n = len(data[origin][interface][targets[0]])
            print(f"  Interface {interface}: targets={targets}, n_paths={n}")
