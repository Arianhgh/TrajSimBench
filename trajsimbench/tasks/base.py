"""Shared immutable task schemas and duck-typed dataset utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from trajsimbench.perturbations.result import canonical_json, hash_payload


class TaskConstructionError(ValueError):
    """Invalid task inputs or an unmet explicit construction quality gate."""


def freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): freeze_value(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        return tuple(freeze_value(item) for item in value.tolist())
    if isinstance(value, list):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    return value


def _trajectory_id(view: Any, fallback: str) -> str:
    return str(getattr(view, "trajectory_id", getattr(view, "id", fallback)))


def dataset_ids(dataset: Any, split: str | None = None) -> tuple[str, ...]:
    if hasattr(dataset, "ids"):
        values = dataset.ids(split=split) if split is not None else dataset.ids()
        return tuple(sorted(str(value) for value in np.asarray(values).tolist()))
    if isinstance(dataset, Mapping):
        return tuple(sorted(str(key) for key in dataset))
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TaskConstructionError(
            "dataset must provide ids(), mapping access, or __len__/__getitem__"
        )
    return tuple(
        sorted(_trajectory_id(dataset[index], str(index)) for index in range(len(dataset)))
    )


def get_trajectory(dataset: Any, trajectory_id: str) -> Any:
    trajectory_id = str(trajectory_id)
    if hasattr(dataset, "by_id"):
        return dataset.by_id(trajectory_id)
    if isinstance(dataset, Mapping):
        return dataset[trajectory_id]
    for index in range(len(dataset)):
        view = dataset[index]
        if _trajectory_id(view, str(index)) == trajectory_id:
            return view
    raise KeyError(f"trajectory {trajectory_id!r} is not in the dataset")


@dataclass(frozen=True, slots=True)
class TaskQualityReport:
    attempted: int
    generated: int
    rejected: int
    rejection_reasons: Mapping[str, int] = field(default_factory=dict)
    required_count: int | None = None
    minimum_yield: float = 0.0
    quality_gate_passed: bool = True
    max_candidates_examined: int | None = None

    def __post_init__(self) -> None:
        if min(self.attempted, self.generated, self.rejected) < 0:
            raise TaskConstructionError("quality report counts must be non-negative")
        if self.generated + self.rejected > self.attempted:
            raise TaskConstructionError("generated+rejected cannot exceed attempted")
        if not 0 <= self.minimum_yield <= 1:
            raise TaskConstructionError("minimum_yield must be in [0, 1]")
        object.__setattr__(
            self, "rejection_reasons", MappingProxyType(dict(self.rejection_reasons))
        )

    @property
    def yield_rate(self) -> float:
        return self.generated / self.attempted if self.attempted else 0.0

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.attempted if self.attempted else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "generated": self.generated,
            "rejected": self.rejected,
            "yield_rate": self.yield_rate,
            "rejection_rate": self.rejection_rate,
            "rejection_reasons": dict(self.rejection_reasons),
            "required_count": self.required_count,
            "minimum_yield": self.minimum_yield,
            "quality_gate_passed": self.quality_gate_passed,
            "max_candidates_examined": self.max_candidates_examined,
        }


@dataclass(frozen=True, slots=True)
class TaskArtifact:
    task_type: str
    schema_version: str
    records: tuple[Mapping[str, Any], ...]
    generator: str
    generator_version: str
    seed: int
    config: Mapping[str, Any] = field(default_factory=dict)
    quality: TaskQualityReport = field(default_factory=lambda: TaskQualityReport(0, 0, 0))
    metadata: Mapping[str, Any] = field(default_factory=dict)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        frozen_records = tuple(freeze_value(record) for record in self.records)
        if not all(isinstance(record, Mapping) for record in frozen_records):
            raise TaskConstructionError("task records must be mappings")
        object.__setattr__(self, "records", frozen_records)
        object.__setattr__(self, "config", freeze_value(self.config))
        object.__setattr__(self, "metadata", freeze_value(self.metadata))
        self.validate()
        basis = {
            "task_type": self.task_type,
            "schema_version": self.schema_version,
            "records": [thaw_value(record) for record in self.records],
            "generator": self.generator,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "config": thaw_value(self.config),
            "quality": self.quality.to_dict(),
            "metadata": thaw_value(self.metadata),
        }
        object.__setattr__(self, "content_hash", hash_payload(basis))

    def validate(self) -> None:
        seen: set[str] = set()
        for index, record in enumerate(self.records):
            record_id = str(record.get("task_id", record.get("triplet_id", index)))
            if record_id in seen:
                raise TaskConstructionError(f"duplicate task record id: {record_id}")
            seen.add(record_id)
            query = record.get("query_id")
            candidate_ids = record.get("candidate_ids")
            if candidate_ids is not None:
                candidate_tuple = tuple(map(str, candidate_ids))
                if len(candidate_tuple) != len(set(candidate_tuple)):
                    raise TaskConstructionError(
                        f"duplicate candidate IDs in task record {record_id}"
                    )
                if query is not None and str(query) in candidate_tuple:
                    raise TaskConstructionError(
                        f"self match was not excluded in task record {record_id}"
                    )
            if self.task_type == "diagnostic" and record.get("expected_order") not in {
                "a_closer",
                "b_closer",
                "tie",
                "unspecified",
            }:
                raise TaskConstructionError(
                    f"invalid diagnostic expected_order in task record {record_id}"
                )
        if (
            self.quality.required_count is not None
            and self.quality.generated < self.quality.required_count
        ):
            if self.quality.quality_gate_passed:
                raise TaskConstructionError(
                    "quality report claims a passed gate below required_count"
                )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "task_type": self.task_type,
            "schema_version": self.schema_version,
            "records": [thaw_value(record) for record in self.records],
            "generator": self.generator,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "config": thaw_value(self.config),
            "quality": self.quality.to_dict(),
            "metadata": thaw_value(self.metadata),
        }
        if include_hash:
            result["content_hash"] = self.content_hash
        return result

    def to_json(self, path: str | Path | None = None) -> str:
        text = canonical_json(self.to_dict())
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


def make_quality(
    attempted: int,
    generated: int,
    reasons: Iterable[str] = (),
    *,
    required_count: int | None = None,
    minimum_yield: float = 0.0,
    max_candidates_examined: int | None = None,
) -> TaskQualityReport:
    counter = Counter(str(reason) for reason in reasons)
    rejected = max(0, attempted - generated)
    passed = (
        generated >= (required_count if required_count is not None else 0)
        and (generated / attempted if attempted else 0.0) >= minimum_yield
    )
    return TaskQualityReport(
        attempted=attempted,
        generated=generated,
        rejected=rejected,
        rejection_reasons=counter,
        required_count=required_count,
        minimum_yield=minimum_yield,
        quality_gate_passed=passed,
        max_candidates_examined=max_candidates_examined,
    )
