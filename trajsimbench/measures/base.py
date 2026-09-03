"""Stable public contracts shared by trajectory similarity measures.

The measure layer deliberately depends only on NumPy and the Python standard
library. The rest of TrajSimBench can pass its canonical ``TrajectoryView``
objects directly; the small coercion helper also makes the classical measures
pleasant to use with plain ``(n, 2)`` arrays in tests and examples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import metadata
from time import perf_counter_ns
from typing import Any

import numpy as np

from .config import BaseMethodConfig


class MeasureCapabilityError(RuntimeError):
    """Raised when a caller requests an operation not advertised by a measure."""


@dataclass(frozen=True, slots=True)
class MeasureCapabilities:
    """Operations and data requirements supported by a measure."""

    learned: bool = False
    supports_batch: bool = False
    supports_encoding: bool = False
    supports_index: bool = False
    symmetric: bool = True
    requires_timestamps: bool = False


@dataclass(frozen=True, slots=True)
class DistanceResult:
    """Canonical distance, unmodified score, runtime, and method details."""

    distance: float
    raw_score: float
    runtime_ns: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        distance = float(self.distance)
        raw_score = float(self.raw_score)
        if not np.isfinite(distance):
            raise ValueError("distance must be finite")
        if distance < 0:
            raise ValueError("distance must be non-negative")
        if not np.isfinite(raw_score):
            raise ValueError("raw_score must be finite")
        if self.runtime_ns is not None:
            if isinstance(self.runtime_ns, bool) or int(self.runtime_ns) != self.runtime_ns:
                raise TypeError("runtime_ns must be a non-negative integer or None")
            if self.runtime_ns < 0:
                raise ValueError("runtime_ns must be non-negative")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "raw_score", raw_score)


@dataclass(frozen=True, slots=True)
class TrajectoryView:
    """Immutable metadata wrapper around one canonical trajectory point view."""

    trajectory_id: str
    points: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points)
        if points.ndim != 2:
            raise ValueError("trajectory points must be a two-dimensional array")
        if points.shape[1] < 2:
            raise ValueError("trajectory points must have at least two columns")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("trajectory metadata must be a mapping")
        object.__setattr__(self, "trajectory_id", str(self.trajectory_id))
        object.__setattr__(self, "points", points)


def as_trajectory_view(value: Any, *, role: str = "trajectory") -> TrajectoryView:
    """Coerce a foundation view, mapping, or NumPy path to ``TrajectoryView``."""

    if isinstance(value, TrajectoryView):
        return value
    if isinstance(value, Mapping) and "points" in value:
        return TrajectoryView(
            str(value.get("trajectory_id", value.get("id", role))),
            np.asarray(value["points"]),
            value.get("metadata", {}),
        )
    if hasattr(value, "points"):
        return TrajectoryView(
            str(getattr(value, "trajectory_id", getattr(value, "id", role))),
            np.asarray(value.points),
            getattr(value, "metadata", {}),
        )
    return TrajectoryView(role, np.asarray(value), {})


def projected_points(value: Any, *, allow_empty: bool = False) -> np.ndarray:
    """Compatibility helper returning projected coordinates for any accepted input."""

    view = as_trajectory_view(value)
    if allow_empty and view.points.shape[0] == 0:
        raw = np.asarray(view.points, dtype=np.float64)
        return np.ascontiguousarray(raw[:, 2:4] if raw.shape[1] >= 4 else raw[:, :2])
    from ._geometry import projected_points as _projected_points

    return _projected_points(view)


class TrajectoryMeasure(ABC):
    """Base class for deterministic lower-is-more-similar trajectory measures."""

    name: str = "measure"
    version: str = "1.0.0"
    capabilities = MeasureCapabilities()
    config: BaseMethodConfig

    def __init__(
        self,
        config: BaseMethodConfig | Mapping[str, Any] | None = None,
        **config_values: Any,
    ) -> None:
        if config is not None and config_values:
            raise TypeError("provide either config or keyword config fields, not both")
        source: Any = config_values if config_values else (config or {})
        self.config = self.config_model.model_validate(source)

    @property
    def config_model(self) -> type[BaseMethodConfig]:
        return BaseMethodConfig

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return serializable method provenance for run manifests."""

        try:
            package_version = metadata.version("numpy")
        except metadata.PackageNotFoundError:  # pragma: no cover - NumPy is required
            package_version = "unknown"
        return {
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "version": self.version,
            "config": self.config.model_dump(),
            "dependencies": {"numpy": package_version},
            "source": "TrajSimBench classical measures; definitions in docs/methods.md",
            "citation": None,
        }

    def fit(self, train_set: Any, val_set: Any | None = None) -> TrajectoryMeasure:
        """Fit a stateless classical measure and return ``self``."""

        del train_set, val_set
        return self

    def encode(self, trajectories: Iterable[Any]) -> np.ndarray:
        del trajectories
        if not self.capabilities.supports_encoding:
            raise MeasureCapabilityError(
                f"{self.name!r} does not support encoding; inspect capabilities first"
            )
        raise MeasureCapabilityError(f"{self.name!r} advertises encoding but has no implementation")

    def distance(self, a: Any, b: Any) -> DistanceResult:
        started = perf_counter_ns()
        result = self._distance_impl(
            as_trajectory_view(a, role="a"), as_trajectory_view(b, role="b")
        )
        if not isinstance(result, DistanceResult):
            raise TypeError("measure implementations must return DistanceResult")
        if result.runtime_ns is None:
            result = replace(result, runtime_ns=max(0, perf_counter_ns() - started))
        return result

    @abstractmethod
    def _distance_impl(self, a: TrajectoryView, b: TrajectoryView) -> DistanceResult:
        """Compute one pairwise result without timing concerns."""

    def pairwise(self, query: Any, candidates: Iterable[Any]) -> np.ndarray:
        """Score candidates in deterministic input order."""

        if not self.capabilities.supports_batch:
            raise MeasureCapabilityError(
                f"{self.name!r} does not support pairwise scoring; inspect capabilities first"
            )
        if isinstance(candidates, np.ndarray) and candidates.ndim == 3:
            values: Iterable[Any] = list(candidates)
        elif isinstance(candidates, np.ndarray) and candidates.ndim == 2:
            values = [candidates]
        else:
            values = candidates
        return np.asarray(
            [self.distance(query, candidate).distance for candidate in values], dtype=np.float64
        )

    def build_index(self, ids: Sequence[str], embeddings: np.ndarray | None = None) -> Any:
        del ids, embeddings
        if not self.capabilities.supports_index:
            raise MeasureCapabilityError(
                f"{self.name!r} does not support indexing; inspect capabilities first"
            )
        raise MeasureCapabilityError(f"{self.name!r} advertises indexing but has no implementation")

    def top_k(self, query: Any, k: int) -> list[tuple[str, float]]:
        del query, k
        if not self.capabilities.supports_index:
            raise MeasureCapabilityError(
                f"{self.name!r} does not support indexed top_k; inspect capabilities first"
            )
        raise MeasureCapabilityError(f"{self.name!r} advertises indexing but has no implementation")


__all__ = [
    "DistanceResult",
    "MeasureCapabilities",
    "MeasureCapabilityError",
    "TrajectoryMeasure",
    "TrajectoryView",
    "as_trajectory_view",
    "projected_points",
]
