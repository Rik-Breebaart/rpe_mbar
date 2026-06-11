from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional


def _candidate_extrema_paths(run_folder, direction, origin_label, interface_indicator, iteration=None):
    run_folder = Path(run_folder)
    candidates = []

    if iteration is not None:
        candidates.extend([
            run_folder / f"cv_extrema_origin-{origin_label}_it{int(iteration)}_int{interface_indicator}_{direction}.csv",
            run_folder / f"cv_extrema_it{int(iteration)}_int{interface_indicator}_{direction}.csv",
            run_folder / direction / f"cv_extrema_origin-{origin_label}_it{int(iteration)}_int{interface_indicator}_{direction}.csv",
            run_folder / direction / f"cv_extrema_it{int(iteration)}_int{interface_indicator}_{direction}.csv",
        ])

    candidates.extend([
        run_folder / f"cv_extrema_origin-{origin_label}_int{interface_indicator}_{direction}.csv",
        run_folder / f"cv_extrema_int{interface_indicator}_{direction}.csv",
        run_folder / direction / f"cv_extrema_origin-{origin_label}_int{interface_indicator}_{direction}.csv",
        run_folder / direction / f"cv_extrema_int{interface_indicator}_{direction}.csv",
    ])

    seen = set()
    unique = []
    for p in candidates:
        s = str(p)
        if s not in seen:
            seen.add(s)
            unique.append(p)
    return unique


def resolve_existing_extrema_paths(
    run_folder,
    direction,
    origin_label,
    interface_value,
    iterations: Optional[list[int]] = None,
    interface_scale: int = 100,
):
    """Return existing CSV paths for the given origin/interface/direction combination.

    Tries every candidate naming pattern for each iteration (or once if no
    iterations are given) and returns the first match per iteration.
    """
    interface_indicator = int(np.round(float(interface_value) * interface_scale))
    iteration_list = list(iterations) if iterations is not None else [None]
    selected = []
    for iteration in iteration_list:
        for candidate in _candidate_extrema_paths(
            run_folder=run_folder,
            direction=direction,
            origin_label=origin_label,
            interface_indicator=interface_indicator,
            iteration=iteration,
        ):
            if candidate.exists():
                selected.append(candidate)
                break
    return selected
