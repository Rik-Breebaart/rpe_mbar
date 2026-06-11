import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

EXTREMA_FILE_RE = re.compile(r"^extrema_(forward|backward)_set(\d+)\.csv$")


def expected_extrema_columns(cv_names):
    cols = ["step"]
    for name in cv_names:
        cols.append(f"min_{name}")
        cols.append(f"max_{name}")
    return cols


def validate_extrema_df(df: pd.DataFrame, cv_names):
    required = expected_extrema_columns(cv_names)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Extrema CSV missing required columns: {missing}")
    if "step" in df.columns and not pd.api.types.is_integer_dtype(df["step"]):
        try:
            df["step"] = df["step"].astype(int)
        except Exception as exc:
            raise ValueError("Column 'step' must be integer-like.") from exc
    return df


def save_extrema_csv(
    df: pd.DataFrame,
    out_csv: Path,
    cv_names,
    *,
    direction: str,
    origin_set: int,
    n_thermalize: int = 0,
    system_name: Optional[str] = None,
):
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")

    out_csv = Path(out_csv)
    validate_extrema_df(df, cv_names)

    expected_name = f"extrema_{direction}_set{int(origin_set)}.csv"
    if out_csv.name != expected_name:
        out_csv = out_csv.with_name(expected_name)

    df = df[expected_extrema_columns(cv_names)].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    meta = {
        "schema": "rpe_mbar.extrema.v1",
        "direction": direction,
        "origin_set": int(origin_set),
        "n_thermalize": int(n_thermalize),
        "cv_names": list(cv_names),
        "system_name": system_name,
        "n_rows": int(len(df)),
    }
    meta_path = out_csv.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return out_csv, meta_path


def load_extrema_csv(path: Path, cv_names=None, validate=True):
    path = Path(path)
    df = pd.read_csv(path)

    meta_path = path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    if cv_names is None and meta.get("cv_names"):
        cv_names = meta["cv_names"]

    if validate and cv_names is not None:
        df = validate_extrema_df(df, cv_names)

    return df, meta


def compute_cv_extrema(storage, cvs_dict, n_thermalize=0):
    """
    Compute min and max of each CV for every trajectory in OPS storage.

    Parameters
    ----------
    storage : OPS Storage object
    cvs_dict : dict of {cv_name: cv_function}
    n_thermalize : number of steps to skip at the beginning

    Returns
    -------
    pandas.DataFrame with columns: step, min_{name}, max_{name} for each CV
    """
    data = {"step": []}
    for name in cvs_dict:
        data[f"min_{name}"] = []
        data[f"max_{name}"] = []

    for i, step in enumerate(storage.steps[n_thermalize:]):
        step_index = i + n_thermalize
        traj = step.active[0].trajectory
        if len(traj) == 0:
            continue
        data["step"].append(step_index)
        for name, cv in cvs_dict.items():
            values = [float(cv(snap)) for snap in traj]
            data[f"min_{name}"].append(float(np.min(values)))
            data[f"max_{name}"].append(float(np.max(values)))

    df = pd.DataFrame(data)
    validate_extrema_df(df, list(cvs_dict.keys()))
    return df


def compute_and_save_cv_extrema(
    storage,
    cvs_dict,
    out_dir: Path,
    *,
    direction: str,
    origin_set: int,
    n_thermalize: int = 0,
    system_name: Optional[str] = None,
):
    """
    Compute extrema and write to:
      extrema_{direction}_set{origin_set}.csv
      extrema_{direction}_set{origin_set}.meta.json
    """
    df = compute_cv_extrema(storage, cvs_dict, n_thermalize=n_thermalize)
    out_csv = Path(out_dir) / f"extrema_{direction}_set{int(origin_set)}.csv"
    save_extrema_csv(
        df, out_csv, list(cvs_dict.keys()),
        direction=direction, origin_set=origin_set,
        n_thermalize=n_thermalize, system_name=system_name,
    )
    return df
