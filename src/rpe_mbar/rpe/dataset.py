from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union
import json
import time

import numpy as np


@dataclass
class TrajectoryInfo:
    storage_index: int
    relative_index: int
    n_points: int
    descriptor_start: int
    descriptor_end: int


@dataclass
class InterfaceData:
    interface_value: float
    trajectories: List[TrajectoryInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interface_value": self.interface_value,
            "trajectories": [
                {
                    "storage_index": t.storage_index,
                    "relative_index": t.relative_index,
                    "n_points": t.n_points,
                    "descriptor_start": t.descriptor_start,
                    "descriptor_end": t.descriptor_end,
                }
                for t in self.trajectories
            ],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InterfaceData":
        return InterfaceData(
            interface_value=data["interface_value"],
            trajectories=[TrajectoryInfo(**t) for t in data["trajectories"]],
        )


@dataclass
class SetData:
    set_name: str
    interfaces: Dict[str, InterfaceData] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "set_name": self.set_name,
            "interfaces": {
                iface_key: iface_data.to_dict()
                for iface_key, iface_data in self.interfaces.items()
            },
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SetData":
        return SetData(
            set_name=data["set_name"],
            interfaces={
                iface_key: InterfaceData.from_dict(iface_data)
                for iface_key, iface_data in data["interfaces"].items()
            },
        )


@dataclass
class RpeDataset:
    sets: Dict[str, SetData] = field(default_factory=dict)
    descriptors: np.ndarray | None = None
    shot_results: np.ndarray | None = None
    trajectory_weights: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_trajectory_descriptors(self, set_name: str, interface_key: str, trajectory_index: int) -> np.ndarray:
        traj_info = self.sets[set_name].interfaces[interface_key].trajectories[trajectory_index]
        return self.descriptors[traj_info.descriptor_start:traj_info.descriptor_end]

    def get_trajectory_shot_results(self, set_name: str, interface_key: str, trajectory_index: int) -> np.ndarray:
        traj_info = self.sets[set_name].interfaces[interface_key].trajectories[trajectory_index]
        return self.shot_results[traj_info.descriptor_start:traj_info.descriptor_end]

    def set_trajectory_weights(self, set_name: str, interface_key: str, weights: np.ndarray) -> None:
        n_trajs = len(self.sets[set_name].interfaces[interface_key].trajectories)
        if len(weights) != n_trajs:
            raise ValueError(
                f"Weight array length {len(weights)} does not match number of trajectories {n_trajs}."
            )
        if set_name not in self.trajectory_weights:
            self.trajectory_weights[set_name] = {}
        self.trajectory_weights[set_name][interface_key] = weights.copy()

    def save(self, base_path: Union[str, Path], compress: bool = True) -> None:
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        structure = {set_name: set_data.to_dict() for set_name, set_data in self.sets.items()}
        with open(base_path.with_suffix(".structure.json"), "w") as f:
            json.dump(structure, f, indent=2)

        save_fn = np.savez_compressed if compress else np.savez
        save_fn(
            base_path.with_suffix(".descriptors.npz"),
            descriptors=self.descriptors,
            shot_results=self.shot_results,
        )

        if self.trajectory_weights:
            weights_dict = {}
            for set_name, interfaces in self.trajectory_weights.items():
                for interface_key, weights in interfaces.items():
                    weights_dict[f"{set_name}__{interface_key}"] = weights
            save_fn(base_path.with_suffix(".weights.npz"), **weights_dict)

        with open(base_path.with_suffix(".meta.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

    @staticmethod
    def load(base_path: Union[str, Path]) -> "RpeDataset":
        base_path = Path(base_path)
        with open(base_path.with_suffix(".structure.json"), "r") as f:
            structure = json.load(f)
        sets = {set_name: SetData.from_dict(set_data) for set_name, set_data in structure.items()}

        arrays = np.load(base_path.with_suffix(".descriptors.npz"))
        descriptors = arrays["descriptors"]
        shot_results = arrays["shot_results"]

        trajectory_weights = {}
        weights_path = base_path.with_suffix(".weights.npz")
        if weights_path.exists():
            weights_data = np.load(weights_path)
            for key in weights_data.files:
                set_name, interface_key = key.split("__", 1)
                trajectory_weights.setdefault(set_name, {})[interface_key] = weights_data[key]

        metadata = {}
        meta_path = base_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        return RpeDataset(
            sets=sets,
            descriptors=descriptors,
            shot_results=shot_results,
            trajectory_weights=trajectory_weights,
            metadata=metadata,
        )

    def summary(self) -> str:
        lines = ["RPE Dataset Summary", "=" * 50]
        total_trajs = 0
        total_points = 0
        for set_name, set_data in self.sets.items():
            lines.append(f"\nSet: {set_name}")
            for interface_key, interface_data in set_data.interfaces.items():
                n_trajs = len(interface_data.trajectories)
                n_pts = sum(t.n_points for t in interface_data.trajectories)
                total_trajs += n_trajs
                total_points += n_pts
                has_weights = (
                    set_name in self.trajectory_weights
                    and interface_key in self.trajectory_weights[set_name]
                )
                lines.append(
                    f"  Interface {interface_key}: {n_trajs} trajectories, {n_pts} points [weights: {'Y' if has_weights else 'N'}]"
                )
        lines.append(f"\nTotal: {total_trajs} trajectories, {total_points} points")
        lines.append("=" * 50)
        return "\n".join(lines)


def build_rpe_dataset(
    interface_sets: List[Any],
    interface_values_lambda_M: Dict[Any, List[float]],
    storages_each_set: List[List[Any]],
    descriptor_transform,
    descriptor_dim: int,
    states,
    N_each_set: Dict[Any, List[int]],
    n_thermalization: int,
    n_jump: int = 1,
    show_log: bool = True,
) -> RpeDataset:
    dataset = RpeDataset(
        metadata={
            "created": time.time(),
            "n_thermalization": n_thermalization,
            "n_jump": n_jump,
            "descriptor_dim": descriptor_dim,
        }
    )

    all_descriptors = []
    all_shot_results = []
    global_point_index = 0

    for set_idx, set_key in enumerate(interface_sets):
        set_name = str(set_key)
        set_data = SetData(set_name=set_name)
        interface_values = interface_values_lambda_M[set_key]
        
        for iface_idx, interface_value in enumerate(interface_values):
            interface_key = f"{interface_value}"
            storage = storages_each_set[set_idx][iface_idx]
            n_traj_total = int(N_each_set[set_key][iface_idx])
            n_steps = int(np.floor(n_traj_total / n_jump))
            interface_data = InterfaceData(interface_value=interface_value)

            for traj_local_idx in range(n_steps):
                storage_index = n_thermalization + traj_local_idx * n_jump
                relative_index = traj_local_idx * n_jump
                step = storage.steps[storage_index]
                traj = step.active[0].trajectory
                n_points = len(traj)
                desc = np.array(descriptor_transform(traj))
                if desc.ndim == 1:
                    desc = desc.reshape(-1, 1)
                shot_A = int(states[0](traj[-1])) + int(states[0](traj[0]))
                shot_B = int(states[1](traj[-1])) + int(states[1](traj[0]))

                interface_data.trajectories.append(
                    TrajectoryInfo(
                        storage_index=storage_index,
                        relative_index=relative_index,
                        n_points=n_points,
                        descriptor_start=global_point_index,
                        descriptor_end=global_point_index + n_points,
                    )
                )
                all_descriptors.append(desc)
                all_shot_results.append(
                    np.column_stack([np.full(n_points, shot_A), np.full(n_points, shot_B)])
                )
                global_point_index += n_points

            set_data.interfaces[interface_key] = interface_data
        dataset.sets[set_name] = set_data

    dataset.descriptors = np.concatenate(all_descriptors, axis=0)
    dataset.shot_results = np.concatenate(all_shot_results, axis=0)
    return dataset


def convert_old_to_new_format(old_dataset_path: Union[str, Path]) -> RpeDataset:
    old_path = Path(old_dataset_path)
    data = np.load(old_path, allow_pickle=False)

    metadata = {}
    meta_path = old_path.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    new_dataset = RpeDataset(metadata=metadata)
    new_dataset.descriptors = data["descriptors"]
    new_dataset.shot_results = data["shot_results"]
    trajectory_order = data["trajectory_order"]
    global_point_index = 0

    interface_values_meta = metadata.get("interface_values", None)
    n_thermalization = metadata.get("n_thermalization", 0)

    for traj_idx in range(trajectory_order.shape[0]):
        set_idx, interface_idx, storage_index, n_points = trajectory_order[traj_idx]
        set_name = f"set_{int(set_idx)}"
        if interface_values_meta is not None and int(set_idx) < len(interface_values_meta):
            interface_value = interface_values_meta[int(set_idx)][int(interface_idx)]
        else:
            interface_value = float(interface_idx)
        interface_key = f"{interface_value}"

        new_dataset.sets.setdefault(set_name, SetData(set_name=set_name))
        new_dataset.sets[set_name].interfaces.setdefault(
            interface_key, InterfaceData(interface_value=interface_value)
        )
        new_dataset.sets[set_name].interfaces[interface_key].trajectories.append(
            TrajectoryInfo(
                storage_index=int(storage_index),
                relative_index=int(storage_index - n_thermalization),
                n_points=int(n_points),
                descriptor_start=global_point_index,
                descriptor_end=global_point_index + int(n_points),
            )
        )
        global_point_index += int(n_points)

    return new_dataset
