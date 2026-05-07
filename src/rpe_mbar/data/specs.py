from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InterfaceFunctionSpec:
    """Bookkeeping descriptor for an interface/CV definition.

    Parameters
    ----------
    filename
        Stable identifier used as origin/target set key in multiset analysis.
    name
        Human-readable name. Defaults to ``filename``.
    cv_column_base
        Base column name in extrema CSVs (without min_/max_ prefix).
        Defaults to ``cv_<name>`` if omitted.
    model_label
        Optional model identifier for provenance.
    iteration
        Current AIMMD-TIS iteration index.
    iterations
        All known iterations for this function.
    meta
        Free-form metadata.
    """

    filename: str
    name: Optional[str] = None
    cv_column_base: Optional[str] = None
    model_label: Optional[str] = None
    iteration: Optional[int] = None
    iterations: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name is None:
            self.name = self.filename
        if self.iteration is not None and self.iteration not in self.iterations:
            self.iterations.append(int(self.iteration))

    def append_iteration(self, iteration: int, model_label: Optional[str] = None):
        iteration = int(iteration)
        if iteration not in self.iterations:
            self.iterations.append(iteration)
        self.iteration = iteration
        if model_label is not None:
            self.model_label = model_label

    def target_cv_column_base(self) -> str:
        if self.cv_column_base:
            return self.cv_column_base
        if self.name.startswith("cv_"):
            return self.name
        return f"cv_{self.name}"


def coerce_interface_spec(item) -> InterfaceFunctionSpec:
    """Coerce a string, dict, or duck-typed object into an InterfaceFunctionSpec."""
    if isinstance(item, InterfaceFunctionSpec):
        return item
    if isinstance(item, str):
        return InterfaceFunctionSpec(filename=item, name=item)
    if isinstance(item, dict):
        filename = item.get("filename")
        if filename is None:
            raise ValueError("Interface function dict must contain 'filename'.")
        return InterfaceFunctionSpec(
            filename=filename,
            name=item.get("name"),
            cv_column_base=item.get("cv_column_base"),
            model_label=item.get("model_label"),
            iteration=item.get("iteration"),
            iterations=item.get("iterations", []),
            meta=item.get("meta", {}),
        )
    filename = getattr(item, "filename", None)
    if filename is None:
        raise ValueError("Interface function object must provide a 'filename' attribute.")
    return InterfaceFunctionSpec(
        filename=filename,
        name=getattr(item, "name", filename),
        cv_column_base=getattr(item, "cv_column_base", None),
        model_label=getattr(item, "model_label", None),
        iteration=getattr(item, "iteration", None),
        iterations=getattr(item, "iterations", []),
        meta=getattr(item, "meta", {}),
    )
