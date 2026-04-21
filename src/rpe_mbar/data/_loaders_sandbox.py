

def dict_to_solver_inputs_for_multiset(
    data: dict,                              # data[origin][iface_str][target] -> 1D np.array
    origin_sets_order: Optional[list[str]] = None,
    target_sets_order: Optional[list[str]] = None,
    direction: str = "forward",
    log: bool = False,
):
    """
    Build solver inputs for solve_multi_set_mbar_matrix in a way that is ORDER-SAFE.

    Key guarantees:
      1) The *set axis* ordering is identical for:
           - max_lambda_values_each_set
           - N_lambda_each_set
           - interface_values_lambda_each_set
         so solve_multi_set_mbar_matrix cannot be fed misaligned blocks.

      2) We do NOT infer target order from dict key order. By default:
           target_sets_order := origin_sets_order
         which is the correct convention for multi-set MBAR (sets simulated == sets targeted).

      3) We ignore non-target metadata keys like "origin_indices".

    Returns:
      max_lambda_values_each_set: list[list[np.ndarray]]  # per set: list over interfaces of 1D arrays
      N_lambda_each_set:         list[np.ndarray]         # per set: sample counts per interface
      interface_values_each_set: list[np.ndarray]         # per set: interface values
      max_lambda_labeled_per_set: dict[target_set] -> 1D np.array  # flattened samples for diagnostics
    """

    if origin_sets_order is None:
        origin_sets_order = list(data.keys())

    # Default: use exactly the same ordering for targets as for origins (order-safe).
    if target_sets_order is None:
        target_sets_order = list(origin_sets_order)

    # Helper: parse & sort interface keys numerically
    def sorted_iface_values(origin: str) -> list[float]:
        iface_vals = [float(k) for k in data.get(origin, {}).keys()]
        # forward: increasing; backward: decreasing in original code, then negated later
        if direction == "backward":
            iface_vals = sorted(iface_vals, reverse=True)
        else:
            iface_vals = sorted(iface_vals)
        return iface_vals

    # Collect interface values per origin set in a deterministic order
    ordered_interface_values = {origin: sorted_iface_values(origin) for origin in origin_sets_order}
    if log:
        print("origin_sets_order:", origin_sets_order)
        print("target_sets_order:", target_sets_order)
        print("ordered_interface_values:", ordered_interface_values)

    # Build max_lambda_values_each_set in the SAME ORDER as origin_sets_order
    max_lambda_values_each_set: list[list[np.ndarray]] = []
    max_lambda_labeled_per_set: dict[str, np.ndarray] = {}

    # For labeled diagnostics we still want to collect for all targets (in target_sets_order)
    for target in target_sets_order:
        all_iface_samples_for_target = []

        # For the solver, we only keep "diagonal targets" where target == origin set
        # but we build each set separately below in a fully order-safe way.
        for origin in origin_sets_order:
            for iface in ordered_interface_values[origin]:
                iface_str = str(iface)
                block = data.get(origin, {}).get(iface_str, {})
                if not isinstance(block, dict):
                    continue

                # ignore metadata like origin_indices
                if target == "origin_indices":
                    continue

                arr = block.get(target, None)
                if arr is None or np.size(arr) == 0:
                    continue

                arr = np.asarray(arr, dtype=float).ravel()
                if direction == "backward":
                    arr = -arr
                all_iface_samples_for_target.append(arr)

        # robust flatten (no dependency on your robust_flatten; do it here safely)
        if len(all_iface_samples_for_target) == 0:
            max_lambda_labeled_per_set[target] = np.array([], dtype=float)
        else:
            max_lambda_labeled_per_set[target] = np.concatenate(all_iface_samples_for_target, axis=0)

    # Now build the per-set solver structure *explicitly* in origin_sets_order
    for origin in origin_sets_order:
        per_iface_arrays = []
        for iface in ordered_interface_values[origin]:
            iface_str = str(iface)
            block = data.get(origin, {}).get(iface_str, {})
            if not isinstance(block, dict):
                per_iface_arrays.append(np.array([], dtype=float))
                continue

            # choose the diagonal target for multiset MBAR
            # (origin set analyzed as its own target set)
            if origin not in block:
                # If missing, fall back to any non-metadata target (prevents crashing, but logs)
                candidates = [k for k in block.keys() if k != "origin_indices"]
                if len(candidates) == 0:
                    arr = np.array([], dtype=float)
                else:
                    if log:
                        print(f"[WARN] Missing diagonal target '{origin}' in block {origin}/{iface_str}. Using '{candidates[0]}'")
                    arr = block[candidates[0]]
            else:
                arr = block[origin]

            arr = np.asarray(arr, dtype=float).ravel()
            if direction == "backward":
                arr = -arr
            per_iface_arrays.append(arr)

        max_lambda_values_each_set.append(per_iface_arrays)

    # Build N_lambda_each_set and interface_values_lambda_each_set in the SAME order
    N_lambda_each_set: list[np.ndarray] = []
    interface_values_lambda_each_set: list[np.ndarray] = []

    for origin in origin_sets_order:
        Ns = []
        ifaces = []
        for iface in ordered_interface_values[origin]:
            iface_str = str(iface)
            block = data.get(origin, {}).get(iface_str, {})
            if not isinstance(block, dict):
                Ns.append(0)
                ifaces.append(float(iface))
                continue

            # determine N consistently from the diagonal target if present,
            # otherwise any non-metadata target
            if origin in block:
                n = len(block[origin])
            else:
                candidates = [k for k in block.keys() if k != "origin_indices"]
                n = len(block[candidates[0]]) if candidates else 0

            Ns.append(int(n))
            ifaces.append(float(iface))

        interface_values_lambda_each_set.append(np.asarray(ifaces, dtype=float))
        N_lambda_each_set.append(np.asarray(Ns, dtype=int))

    if direction == "backward":
        interface_values_lambda_each_set = [-iv for iv in interface_values_lambda_each_set]

    return (
        max_lambda_values_each_set,
        N_lambda_each_set,
        interface_values_lambda_each_set,
        max_lambda_labeled_per_set,
    )