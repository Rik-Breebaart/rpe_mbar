import json
import time
import numpy as np
from pathlib import Path


def save_weights(path: Path, weights: np.ndarray, *, meta: dict):
    payload = dict(meta)
    payload.update({"created": time.time(), "schema_version": 1})
    np.savez_compressed(
        path,
        weights=weights.astype(float),
        metadata=json.dumps(payload, separators=(",", ":")),
    )


def load_weights(path: Path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(data["metadata"].item())
    return data["weights"], meta
