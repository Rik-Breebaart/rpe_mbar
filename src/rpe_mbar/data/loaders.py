"""
File loaders: reads extrema CSVs into the multiset dict format.

Two loaders:
  load_extrema_as_dict          - legacy benchmark naming scheme
  load_extrema_as_dict_iterative - AIMMD-TIS iterative naming scheme
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from rpe_mbar.data.specs import coerce_interface_spec
from rpe_mbar.data.path_resolver import resolve_existing_extrema_paths
from rpe_mbar.utils.tools import format_interface_value


def load_max_values(csv_paths, interface_values, cv_column_base, direction="forward", n_thermalization=30):
    """Read the relevant extrema column from each CSV and return a list of arrays."""
    cv_col = f"max_{cv_column_base}" if direction == "forward" else f"min_{cv_column_base}"
    result = []
    for csv_file in csv_paths:
        df = pd.read_csv(csv_file)
        result.append(df[cv_col].values[n_thermalization:])
    return result


def load_extrema_as_dict(
    run_folder: Path,
    system_name: str,
    direction: str,
    interface_functions,
    interfaces_forward,
    n_thermalization: int,
):
    """Legacy loader for benchmark file naming scheme.

    Returns
    -------
    data : dict[origin][iface_str][target] -> 1D np.ndarray
    interface_values_by_set : dict[origin] -> list[str]
    """
    data = {}
    interface_values_by_set = {}

    for i_origin, origin_fset in enumerate(interface_functions):
        origin = origin_fset.filename
        origin_ifaces = interfaces_forward[i_origin]
        interface_values_by_set[origin] = [f"{v}" for v in origin_ifaces]
        data[origin] = {iface: {} for iface in interface_values_by_set[origin]}

        csv_paths_origin = [
            Path(run_folder) / direction /
            f"cv_extrema_{system_name}_cv_{origin_fset.filename}_iface-{format_interface_value(v)}_{direction}.csv"
            for v in origin_ifaces
        ]

        for target_fset in interface_functions:
            target = target_fset.filename
            target_cv_name = f"cv_{target_fset.name}"
            max_vals_list = load_max_values(
                csv_paths_origin, origin_ifaces, target_cv_name,
                direction, n_thermalization=n_thermalization,
            )
            for idx, iface in enumerate(interface_values_by_set[origin]):
                data[origin][iface][target] = max_vals_list[idx]

    return data, interface_values_by_set


def load_extrema_as_dict_iterative(
    run_folder: Path,
    direction: str,
    interface_functions,
    interfaces_by_origin: dict,
    n_thermalization: int = 0,
    iterations=None,
    interface_scale: int = 100,
    strict: bool = True,
):
    """AIMMD-TIS iterative loader.

    Returns
    -------
    data : dict[origin][iface_str][target] -> 1D np.ndarray
    interface_values_by_set : dict[origin] -> list[str]
    file_index : dict[origin][iface_str] -> list[str]
    """
    specs = [coerce_interface_spec(x) for x in interface_functions]
    data = {}
    interface_values_by_set = {}
    file_index = {}

    for origin_spec in specs:
        origin = origin_spec.filename
        origin_interfaces = interfaces_by_origin.get(origin, [])
        interface_values_by_set[origin] = [f"{v}" for v in origin_interfaces]
        data[origin] = {iface: {} for iface in interface_values_by_set[origin]}
        file_index[origin] = {iface: [] for iface in interface_values_by_set[origin]}

        for iface_value, iface_str in zip(origin_interfaces, interface_values_by_set[origin]):
            csv_paths = resolve_existing_extrema_paths(
                run_folder=run_folder,
                direction=direction,
                origin_label=origin,
                interface_value=iface_value,
                iterations=iterations,
                interface_scale=interface_scale,
            )

            if not csv_paths:
                msg = (
                    f"No extrema CSV found for origin={origin}, iface={iface_value}, "
                    f"direction={direction}, iterations={iterations}."
                )
                if strict:
                    raise FileNotFoundError(msg)
                for target_spec in specs:
                    data[origin][iface_str][target_spec.filename] = np.array([], dtype=float)
                continue

            file_index[origin][iface_str] = [str(p) for p in csv_paths]

            for target_spec in specs:
                target = target_spec.filename
                values_by_file = load_max_values(
                    csv_paths=csv_paths,
                    interface_values=None,
                    cv_column_base=target_spec.target_cv_column_base(),
                    direction=direction,
                    n_thermalization=n_thermalization,
                )
                if len(values_by_file) == 0:
                    data[origin][iface_str][target] = np.array([], dtype=float)
                elif len(values_by_file) == 1:
                    data[origin][iface_str][target] = values_by_file[0]
                else:
                    data[origin][iface_str][target] = np.concatenate(values_by_file)

    return data, interface_values_by_set, file_index
