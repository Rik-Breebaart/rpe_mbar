from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from ..core.solver_prep import weights_split_into_sets
from ..diag.interface_matching import compute_cv_all_sets, index_beyond_joined_interface
from .stable import (
    compute_flux_compensation_stable_states_multiple_interfaces,
    compute_multi_interface_passed_flags,
    create_total_trainset,
)


def _resolve_path_indices(block: Dict[str, Any], default_n: int) -> np.ndarray:
    if "origin_indices" in block:
        return np.asarray(block["origin_indices"], dtype=int)
    if "path_indices" in block:
        return np.asarray(block["path_indices"], dtype=int)
    return np.arange(default_n, dtype=int)


def get_descriptors_and_weights(
    data_max_values,
    weights_combined,
    dataset_descriptors,
    have_path_indices: bool = False,
    log: bool = False,
    compute_shot_result: bool = False,
    state_axis: int = 0,
    state_boundary_A: float = -0.9,
    state_boundary_B: float = 0.9,
):
    weights_each_set_iface = weights_split_into_sets(weights_combined, data_max_values)

    descriptors, weights, shot_results = [], [], []
    n_AA_calculated = 0
    n_AB_calculated = 0
    n_BB_calculated = 0

    for origin_set in data_max_values.keys():
        for iface in data_max_values[origin_set].keys():
            default_n = data_max_values[origin_set][iface][origin_set].shape[0]
            path_indices = _resolve_path_indices(data_max_values[origin_set][iface], default_n)
            notify = True
            for traj_idx in range(len(path_indices)):
                trajectory = dataset_descriptors.get_trajectory_descriptors(origin_set, iface, int(path_indices[traj_idx]))
                if len(trajectory) < 3:
                    if notify and log:
                        print(f"[WARN] Short trajectory (<3) in {origin_set}, iface {iface}; skipping.")
                        notify = False
                    continue
                if not compute_shot_result:
                    shot_result_traj = dataset_descriptors.get_trajectory_shot_results(origin_set, iface, int(path_indices[traj_idx]))
                    if shot_result_traj[0,0] == 2 and shot_result_traj[0,1] == 0:
                        n_AA_calculated += 1
                    elif shot_result_traj[0,0] == 1 and shot_result_traj[0,1] == 1:
                        n_AB_calculated += 1
                    elif shot_result_traj[0,0] == 0 and shot_result_traj[0,1] == 2:
                        n_BB_calculated += 1
                else: 
                    #TODO: this logic currently only works with 1d state functions, needs fix to extent to multi-dim state functions (still based on cv values in descriptor storage)
                    start_val = trajectory[0, state_axis]
                    end_val = trajectory[-1, state_axis]
                    if start_val < state_boundary_A and end_val < state_boundary_A:
                        n_AA_calculated += 1
                        shot_result = [2, 0]
                    elif (
                        (start_val < state_boundary_A and end_val > state_boundary_B)
                        or (start_val > state_boundary_B and end_val < state_boundary_A)
                    ):
                        n_AB_calculated += 1
                        shot_result = [1, 1]
                    elif start_val > state_boundary_B and end_val > state_boundary_B:
                        n_BB_calculated += 1
                        shot_result = [0, 2]
                    else:
                        shot_result = [0, 0]
                    shot_result_traj = np.array(shot_result) * np.ones((len(trajectory), 1))

                descriptors.append(trajectory)
                w = weights_each_set_iface[origin_set][iface][traj_idx]
                weights.append(w * np.ones((len(trajectory), 1)))
                shot_results.append(shot_result_traj)

    if len(descriptors) == 0:
        raise RuntimeError("No descriptor trajectories collected; check dataset paths / interface selections.")

    descriptors = np.concatenate(descriptors, axis=0)
    weights = np.concatenate(weights, axis=0).flatten()
    min_weight = np.min(weights[np.isfinite(weights)])
    if min_weight <= 0:
        raise ValueError(f"Non-positive minimum weight encountered: {min_weight}")
    weights = weights / min_weight
    shot_results = np.concatenate(shot_results, axis=0)
    return descriptors, weights, shot_results, (n_AA_calculated, n_AB_calculated, n_BB_calculated)


def build_reweighted_rpe_trainset(
    desc_f: np.ndarray,
    w_f: np.ndarray,
    shot_f: np.ndarray,
    counts_f,
    desc_b: np.ndarray,
    w_b: np.ndarray,
    shot_b: np.ndarray,
    counts_b,
    data_stable: Dict[str, Any],
    interface_functions: Sequence[Callable[[np.ndarray], np.ndarray]],
    first_interfaces_forward: Sequence[float],
    first_interfaces_backward: Sequence[float],
    state_function_A: Callable[[np.ndarray], np.ndarray],
    state_function_B: Callable[[np.ndarray], np.ndarray],
    state_threshold_A: float,
    state_threshold_B: float,
    cutoff_beyond_joined_interface: bool = True,
    name_lambda_functions: Optional[Sequence[str]] = None,
    c_AB: Optional[float] = None,
    interface_descriptor_keys: Optional[Sequence[str]] = None,
    descriptor_arrays_f: Optional[Dict[str, np.ndarray]] = None,
    descriptor_arrays_b: Optional[Dict[str, np.ndarray]] = None,
    training_descriptor_key: Optional[str] = None,
):
    if c_AB is None:
        if counts_b[1] == 0:
            raise ZeroDivisionError("Backward AB count is zero; cannot compute c_AB.")
        c_AB = float(counts_f[1] / counts_b[1])

    multi_flux = compute_flux_compensation_stable_states_multiple_interfaces(
        cv_functions=interface_functions,
        data_stable=data_stable,
        first_interfaces_forward=first_interfaces_forward,
        first_interfaces_backward=first_interfaces_backward,
        state_function_A=state_function_A,
        state_function_B=state_function_B,
        state_threshold_A=state_threshold_A,
        state_threshold_B=state_threshold_B,
        name_lambda_functions=name_lambda_functions,
        descriptor_keys=interface_descriptor_keys,
    )

    before_interface_set_all = compute_multi_interface_passed_flags(
        data_stable=data_stable,
        interface_function_sets=list(interface_functions),
        first_interfaces_forward=list(first_interfaces_forward),
        first_interfaces_backward=list(first_interfaces_backward),
        in_state_flags=multi_flux["in_state_flags"],
        descriptor_keys=interface_descriptor_keys,
    )

    if cutoff_beyond_joined_interface:
        cvs_all_forward = compute_cv_all_sets(
            list(interface_functions), desc_f,
            descriptor_arrays=descriptor_arrays_f,
            descriptor_keys=interface_descriptor_keys,
        )
        cvs_all_backward = compute_cv_all_sets(
            list(interface_functions), desc_b,
            descriptor_arrays=descriptor_arrays_b,
            descriptor_keys=interface_descriptor_keys,
        )

        idx_beyond_f = index_beyond_joined_interface(
            cvs_all_forward,
            list(first_interfaces_forward),
            state_function_values=state_function_A(desc_f),
            state_boundary=state_threshold_A,
            direction="forward",
        )
        idx_beyond_b = index_beyond_joined_interface(
            cvs_all_backward,
            list(first_interfaces_backward),
            state_function_values=state_function_B(desc_b),
            state_boundary=state_threshold_B,
            direction="backward",
        )
    else:
        idx_beyond_f = np.arange(len(desc_f))
        idx_beyond_b = np.arange(len(desc_b))

    weights_sum_forward = float(np.sum(w_f / np.min(w_f)))
    weights_sum_backward = float(np.sum(w_b / np.min(w_b)))
    n_eff_A = len(multi_flux["first_crossings"][0])
    n_eff_B = len(multi_flux["first_crossings"][1])
    print(f"Effective counts for A and B: n_eff_A={n_eff_A}, n_eff_B={n_eff_B}")
    if n_eff_A == 0 or n_eff_B == 0:
        raise ZeroDivisionError(
            f"n_eff_A={n_eff_A}, n_eff_B={n_eff_B} -> cannot compute weight_factors."
        )

    weight_factors = [weights_sum_forward/n_eff_A, c_AB*weights_sum_backward/n_eff_B]

    # Resolve training descriptors: when data_stable["descriptors"] is a per-key dict,
    # extract the training key; otherwise fall back to the plain list.
    stable_descriptors = data_stable.get("descriptors") if isinstance(data_stable, dict) else None
    if isinstance(stable_descriptors, dict):
        if training_descriptor_key is None:
            raise ValueError(
                "training_descriptor_key must be provided when data_stable['descriptors'] "
                "is a dict of {descriptor_key: [desc_A, desc_B]}."
            )
        stable_descriptors = stable_descriptors[training_descriptor_key]

    descriptors_total, weights_total, shot_results_total = create_total_trainset(
        desc_f[idx_beyond_f],
        w_f[idx_beyond_f],
        shot_f[idx_beyond_f],
        desc_b[idx_beyond_b],
        w_b[idx_beyond_b],
        shot_b[idx_beyond_b],
        c_AB,
        stable_descriptors,
        before_interface_set_all,
        weight_factors=weight_factors,
        load_stable=True,
    )
    return {
        "descriptors": descriptors_total,
        "weights": weights_total,
        "shot_results": shot_results_total,
        "c_AB": float(c_AB),
        "flux": multi_flux,
        "stable_flags": before_interface_set_all,
        "idx_beyond_forward": idx_beyond_f,
        "idx_beyond_backward": idx_beyond_b,
    }
