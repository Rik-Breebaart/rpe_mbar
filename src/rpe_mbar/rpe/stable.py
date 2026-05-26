from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _coerce_array_list(obj: Any) -> Optional[List[np.ndarray]]:
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [np.asarray(item) for item in obj.tolist()]
        if obj.ndim <= 2:
            return [np.asarray(obj)]
        return [np.asarray(obj[i]) for i in range(obj.shape[0])]
    if isinstance(obj, (list, tuple)):
        return [np.asarray(item) for item in obj]
    raise TypeError(f"Unsupported array container type: {type(obj)!r}")


def _unpack_stable_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], dict):
        merged = _unpack_stable_payload(payload[1])
        merged["meta"] = {**payload[0], **merged.get("meta", {})}
        return merged
    if isinstance(payload, dict):
        raw_desc = payload.get("descriptors")
        # Preserve dict format {key: [desc_A, desc_B]} used for per-set descriptor keys.
        descriptors_field = raw_desc if isinstance(raw_desc, dict) else _coerce_array_list(raw_desc)
        return {
            "meta": payload.get("meta", {}),
            "descriptors": descriptors_field,
            "weights": _coerce_array_list(payload.get("weights")),
            "shot_results": _coerce_array_list(payload.get("shot_results")),
            "in_state_flags": _coerce_array_list(payload.get("in_state_flags")) if len(payload.get("in_state_flags", [])) > 0 else None,
        }
    if isinstance(payload, (list, tuple)):
        if len(payload) >= 3 and not isinstance(payload[0], dict):
            return {
                "meta": {},
                "descriptors": _coerce_array_list(payload[0]),
                "weights": _coerce_array_list(payload[1]),
                "shot_results": _coerce_array_list(payload[2]),
                "in_state_flags": _coerce_array_list(payload[3]) if len(payload) > 3 else None,
            }
        if len(payload) == 2 and not isinstance(payload[0], dict):
            return {
                "meta": {},
                "descriptors": _coerce_array_list(payload[0]),
                "weights": _coerce_array_list(payload[1]),
                "shot_results": None,
                "in_state_flags": None,
            }
        if len(payload) == 1:
            return _unpack_stable_payload(payload[0])
    raise TypeError("Unsupported stable-state payload format")


def _infer_in_state_lists(
    descriptors: Sequence[np.ndarray],
    weights: Optional[Sequence[np.ndarray]] = None,
    explicit_flags: Optional[Sequence[np.ndarray]] = None,
    state_function_A: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    state_function_B: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    state_threshold_A: float = None,
    state_threshold_B: float = None,
) -> List[np.ndarray]:
    flags: List[np.ndarray] = []
    for idx, desc in enumerate(descriptors):
        if idx == 0 and state_function_A is not None:
            if state_threshold_A is None:
                raise ValueError("state_threshold_A must be provided if state_function_A is used.")
            cv_values = np.asarray(state_function_A(desc), dtype=float)
            flags.append(cv_values <= state_threshold_A)
            continue
        if idx == 1 and state_function_B is not None:
            if state_threshold_B is None:
                raise ValueError("state_threshold_B must be provided if state_function_B is used.")
            cv_values = np.asarray(state_function_B(desc), dtype=float)
            flags.append(cv_values >= state_threshold_B)
            continue

        n_points = desc.shape[0]
        if explicit_flags is not None and idx < len(explicit_flags):
            candidate = np.asarray(explicit_flags[idx], dtype=bool)
            if candidate.shape[0] != n_points:
                raise ValueError("Explicit in_state_flags length mismatch with descriptors.")
            flags.append(candidate)
            continue
        if weights is not None and idx < len(weights):
            w = np.asarray(weights[idx])
            if w.ndim > 1:
                w = np.squeeze(w)
            if w.shape[0] == n_points and np.all(np.isin(np.unique(w), [0, 1])):
                flags.append(w.astype(bool))
                continue
        flags.append(np.zeros(n_points, dtype=bool))
    return flags


def _count_state_leaves(in_state_flags: np.ndarray) -> int:
    flags = np.asarray(in_state_flags, dtype=bool)
    if flags.size <= 1:
        return 0
    return int(np.sum(flags[:-1] & (~flags[1:])))


def find_first_crossings_multi_segment(
    cv_series_list: List[np.ndarray],
    in_state: np.ndarray,
    interfaces: Sequence[float],
    direction: str,
    interface_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Tuple[int, str]], Dict[str, int]]:
    n = in_state.shape[0]
    m = len(cv_series_list)
    if interface_names is None:
        interface_names = [f"lambda_set_{i}" for i in range(m)]

    crossings: List[Tuple[int, str]] = []
    counts: Dict[str, int] = {name: 0 for name in interface_names}

    i = 0
    crossed_state = False
    while i < n:
        while i < n and in_state[i]:
            crossed_state = True
            i += 1
        if i >= n:
            break
        seg_start = i
        j = i
        while j < n and not in_state[j]:
            j += 1
        seg_end = j

        if not crossed_state:
            i = seg_end
            continue

        events = []
        for k, series in enumerate(cv_series_list):
            seg_vals = series[seg_start:seg_end]
            if seg_vals.size == 0:
                continue
            if direction == "forward":
                mask = seg_vals >= interfaces[k]
                overshoot_sign = 1.0
            else:
                mask = seg_vals <= interfaces[k]
                overshoot_sign = -1.0
            if mask.any():
                rel_idx = int(np.argmax(mask))
                t_cross = seg_start + rel_idx
                overshoot = (series[t_cross] - interfaces[k]) * overshoot_sign
                events.append((t_cross, k, overshoot))

        if events:
            events.sort(key=lambda x: (x[0], -x[2]))
            t_sel, k_sel, _ = events[0]
            name = interface_names[k_sel]
            crossings.append((t_sel, name))
            counts[name] += 1
        i = seg_end

    return crossings, counts


def compute_flux_compensation_stable_states_multiple_interfaces(
    cv_functions: Sequence[Callable[[np.ndarray], np.ndarray]],
    data_stable: Any,
    first_interfaces_forward: Sequence[float],
    first_interfaces_backward: Sequence[float],
    state_function_A: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    state_function_B: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    state_threshold_A: Optional[float] = None,
    state_threshold_B: Optional[float] = None,
    name_lambda_functions: Optional[Sequence[str]] = None,
    descriptor_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    payload = _unpack_stable_payload(data_stable)
    raw_descriptors = payload.get("descriptors")

    # Support dict format {desc_key: [desc_A, desc_B]} for per-function descriptor arrays.
    if isinstance(raw_descriptors, dict):
        descriptors_dict: Optional[Dict] = raw_descriptors
        # Use the first key's list as the reference for in-state inference and the loop.
        descriptors = next(iter(raw_descriptors.values()))
    else:
        descriptors_dict = None
        descriptors = raw_descriptors or []

    if len(descriptors) < 2:
        raise ValueError("Expected at least two stable-state descriptor arrays (A and B).")

    in_state_lists = _infer_in_state_lists(
        descriptors=descriptors,
        weights=payload.get("weights"),
        explicit_flags=payload.get("in_state_flags"),
        state_function_A=state_function_A,
        state_function_B=state_function_B,
        state_threshold_A=state_threshold_A,
        state_threshold_B=state_threshold_B,
    )

    results_flux = []
    results_p_state_lambda_joined = []
    results_crossings = []
    results_counts = []
    leaves_per_state = []

    for state_idx, desc in enumerate(descriptors):
        direction = "forward" if state_idx == 0 else "backward"
        interfaces = first_interfaces_forward if direction == "forward" else first_interfaces_backward
        if len(interfaces) != len(cv_functions):
            raise ValueError(
                f"Number of interfaces ({len(interfaces)}) must match number of cv_functions ({len(cv_functions)}) for state {state_idx}."
            )

        in_state = in_state_lists[state_idx]
        if descriptors_dict is not None and descriptor_keys is not None:
            cv_series_list = [
                np.asarray(f(descriptors_dict[key][state_idx]), dtype=float).ravel()
                for f, key in zip(cv_functions, descriptor_keys)
            ]
        else:
            cv_series_list = [np.asarray(f(desc), dtype=float).ravel() for f in cv_functions]
        leaves = _count_state_leaves(in_state)
        leaves_per_state.append(leaves)

        crossings, counts = find_first_crossings_multi_segment(
            cv_series_list=cv_series_list,
            in_state=in_state,
            interfaces=interfaces,
            direction=direction,
            interface_names=name_lambda_functions,
        )

        n_eff_joined = len(crossings)
        p_state_lambda_joined = (n_eff_joined / leaves) if leaves > 0 else 0.0
        results_flux.append(float(n_eff_joined))
        results_p_state_lambda_joined.append(p_state_lambda_joined)
        results_crossings.append(crossings)
        results_counts.append(counts)

    return {
        "flux_compensation": results_flux,
        "P_state_lambda_joined": results_p_state_lambda_joined,
        "first_crossings": results_crossings,
        "crossing_counts": results_counts,
        "leaves_per_state": leaves_per_state,
        "in_state_flags": in_state_lists,
    }


def compute_multi_interface_passed_flags(
    data_stable: Any,
    interface_function_sets: List[Callable[[np.ndarray], np.ndarray]],
    first_interfaces_forward: List[float],
    first_interfaces_backward: List[float],
    in_state_flags: Sequence[np.ndarray],
    descriptor_keys: Optional[Sequence[str]] = None,
) -> List[np.ndarray]:
    payload = _unpack_stable_payload(data_stable)
    raw_descriptors = payload.get("descriptors")

    if isinstance(raw_descriptors, dict):
        descriptors_dict: Optional[Dict] = raw_descriptors
        descriptors = next(iter(raw_descriptors.values()))
    else:
        descriptors_dict = None
        descriptors = raw_descriptors or []

    if len(descriptors) == 0:
        return []

    passed_flags: List[np.ndarray] = []
    for state_idx, desc in enumerate(descriptors):
        direction = "forward" if state_idx == 0 else "backward"
        current = np.ones(desc.shape[0], dtype=bool)
        for set_idx, interface_function in enumerate(interface_function_sets):
            threshold = first_interfaces_forward[set_idx] if direction == "forward" else first_interfaces_backward[set_idx]
            if descriptors_dict is not None and descriptor_keys is not None:
                eval_desc = descriptors_dict[descriptor_keys[set_idx]][state_idx]
            else:
                eval_desc = desc
            values = np.asarray(interface_function(eval_desc), dtype=float).ravel()
            if direction == "forward":
                current &= values <= threshold
            else:
                current &= values >= threshold
        current |= np.asarray(in_state_flags[state_idx], dtype=bool)
        passed_flags.append(current)
    return passed_flags


def create_total_trainset(
    X_forward,
    w_forward,
    results_forward,
    X_backward,
    w_backward,
    results_backward,
    c_AB,
    X_stable=None,
    in_stable_list=None,
    results_stable=None,
    load_stable=True,
    weight_factors=None,
):
    descriptors_total = np.concatenate([X_forward, X_backward], axis=0)
    weights_total = np.concatenate([w_forward, c_AB * w_backward], axis=0)
    shot_results_total = np.concatenate([results_forward, results_backward], axis=0)

    if load_stable and X_stable is not None:
        n_states = len(X_stable)
        if weight_factors is None:
            weight_factors = [1.0 for _ in range(n_states)]
        if in_stable_list is None or len(in_stable_list) < n_states:
            raise ValueError("Stable state flags missing or incomplete.")
        if results_stable is None or len(results_stable) < n_states:
            results_stable = [np.zeros((X_stable[state].shape[0], 2)) for state in range(n_states)]
            for i in range(len(results_stable)):
                results_stable[i][:, i] = 2.0

        descriptors_stable = []
        weights_stable = []
        shot_results_stable = []
        for state in range(n_states):
            X_st = X_stable[state]
            in_stable = np.asarray(in_stable_list[state], dtype=bool)
            results = results_stable[state]
            w_st = in_stable.astype(np.float32) * np.nan_to_num(weight_factors[state], posinf=0.0)
            descriptors_stable.append(X_st)
            weights_stable.append(w_st)
            shot_results_stable.append(results)

        descriptors_total = np.concatenate([descriptors_total, np.concatenate(descriptors_stable, axis=0)], axis=0)
        weights_total = np.concatenate([weights_total, np.concatenate(weights_stable, axis=0)], axis=0)
        shot_results_total = np.concatenate([shot_results_total, np.concatenate(shot_results_stable, axis=0)], axis=0)

    nonzero_mask = weights_total != 0
    descriptors_total = descriptors_total[nonzero_mask]
    weights_total = weights_total[nonzero_mask]
    shot_results_total = shot_results_total[nonzero_mask]

    min_weight = np.min(weights_total)
    if min_weight <= 0:
        raise ValueError(f"Non-positive minimum weight encountered: {min_weight}")
    weights_total = weights_total / min_weight
    return descriptors_total, weights_total, shot_results_total
