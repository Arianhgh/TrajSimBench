"""Immutable perturbation results and stable provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def hash_array(array: np.ndarray | None) -> str | None:
    if array is None:
        return None
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(canonical_json(list(arr.shape)).encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class PerturbationProvenance(Mapping[str, Any]):
    """The complete, serializable record needed to regenerate a variant."""

    variant_id: str
    source_trajectory_id: str
    transformation: str
    severity: Any
    units: str | None
    parameters: Mapping[str, Any]
    seed: int | None
    notion_expectations: Mapping[str, Any]
    generator_version: str
    input_hash: str
    output_hash: str | None
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", _freeze(self.severity))
        object.__setattr__(self, "parameters", _freeze(self.parameters))
        object.__setattr__(self, "notion_expectations", _freeze(self.notion_expectations))
        object.__setattr__(self, "quality_flags", tuple(self.quality_flags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "source_trajectory_id": self.source_trajectory_id,
            "source_id": self.source_trajectory_id,
            "transformation": self.transformation,
            "severity": self.severity,
            "units": self.units,
            "parameters": dict(self.parameters),
            "parameters_json": canonical_json(self.parameters),
            "seed": self.seed,
            "notion_expectations": dict(self.notion_expectations),
            "semantic_expectation": dict(self.notion_expectations),
            "generator_version": self.generator_version,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "quality_flags": list(self.quality_flags),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PerturbationProvenance:
        parameters = value.get("parameters")
        if parameters is None and value.get("parameters_json"):
            parameters = json.loads(str(value["parameters_json"]))
        expectations = value.get("notion_expectations", value.get("semantic_expectation", {}))
        return cls(
            variant_id=str(value["variant_id"]),
            source_trajectory_id=str(
                value.get("source_trajectory_id", value.get("source_id", "trajectory"))
            ),
            transformation=str(value["transformation"]),
            severity=value.get("severity"),
            units=value.get("units"),
            parameters=parameters or {},
            seed=value.get("seed"),
            notion_expectations=expectations or {},
            generator_version=str(value.get("generator_version", "1.0")),
            input_hash=str(value.get("input_hash", "")),
            output_hash=value.get("output_hash"),
            quality_flags=tuple(value.get("quality_flags", ())),
        )


@dataclass(frozen=True, slots=True)
class PerturbationResult:
    """Generated or explicitly rejected variant.

    Generated arrays are detached copies and marked read-only.  A rejected
    result has ``points is None`` and a machine-readable ``reason``.
    """

    status: str
    trajectory_id: str
    points: np.ndarray | None
    provenance: PerturbationProvenance
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"generated", "not_generated"}:
            raise ValueError("status must be 'generated' or 'not_generated'")
        if self.status == "generated" and self.points is None:
            raise ValueError("generated result requires points")
        if self.status == "not_generated" and not self.reason:
            raise ValueError("not_generated result requires a reason")
        if self.points is not None:
            points = np.array(self.points, copy=True)
            points.setflags(write=False)
            object.__setattr__(self, "points", points)

    @property
    def generated(self) -> bool:
        return self.status == "generated"

    @property
    def variant_id(self) -> str:
        return self.provenance.variant_id

    @property
    def output_hash(self) -> str | None:
        return self.provenance.output_hash

    def validate(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        if self.points is None:
            return
        from .base import validate_trajectory_points

        validate_trajectory_points(self.points, metadata or {})

    def to_dict(self, *, include_points: bool = False) -> dict[str, Any]:
        value = {
            "status": self.status,
            "trajectory_id": self.trajectory_id,
            "variant_id": self.variant_id,
            "provenance": self.provenance.to_dict(),
            "reason": self.reason,
        }
        if include_points and self.points is not None:
            value["points"] = self.points.tolist()
        return value
