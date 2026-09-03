"""Stable-name registry for classical and future learned measures."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .base import MeasureCapabilities, TrajectoryMeasure
from .classical import (
    DiscreteFrechetMeasure,
    DTWMeasure,
    EDRMeasure,
    ERPMeasure,
    EuclideanMeasure,
    HausdorffMeasure,
    LCSSMeasure,
)
from .config import BaseMethodConfig

MeasureFactory = Callable[[Mapping[str, Any] | BaseMethodConfig | None], TrajectoryMeasure]


@dataclass(frozen=True, slots=True)
class MeasureSpec:
    """Registry entry and provenance metadata for one stable method name."""

    name: str
    factory: type[TrajectoryMeasure] | MeasureFactory
    config_model: type[BaseMethodConfig]
    version: str
    capabilities: MeasureCapabilities
    source: str
    citation: str | None = None

    def create(
        self, config: Mapping[str, Any] | BaseMethodConfig | None = None
    ) -> TrajectoryMeasure:
        return self.factory(config)


class MeasureRegistry:
    """Ordered registry with strict names and deterministic default iteration."""

    def __init__(self) -> None:
        self._specs: dict[str, MeasureSpec] = {}

    def register(
        self,
        name: str,
        factory: type[TrajectoryMeasure] | MeasureFactory,
        *,
        config_model: type[BaseMethodConfig] | None = None,
        version: str | None = None,
        capabilities: MeasureCapabilities | None = None,
        source: str = "TrajSimBench classical measures",
        citation: str | None = None,
        replace: bool = False,
    ) -> MeasureSpec:
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("measure name must be a non-empty trimmed string")
        if name in self._specs and not replace:
            raise KeyError(f"measure {name!r} is already registered")
        prototype = factory(None)
        model = config_model or prototype.config_model
        spec = MeasureSpec(
            name=name,
            factory=factory,
            config_model=model,
            version=version or prototype.version,
            capabilities=capabilities or prototype.capabilities,
            source=source,
            citation=citation,
        )
        self._specs[name] = spec
        return spec

    def unregister(self, name: str) -> None:
        del self._specs[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def specs(self) -> tuple[MeasureSpec, ...]:
        return tuple(self._specs.values())

    def get_spec(self, name: str) -> MeasureSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            known = ", ".join(self._specs) or "(none)"
            raise KeyError(f"unknown measure {name!r}; registered names: {known}") from exc

    def create(
        self,
        name: str,
        config: Mapping[str, Any] | BaseMethodConfig | None = None,
        **config_values: Any,
    ) -> TrajectoryMeasure:
        spec = self.get_spec(name)
        if config is not None and config_values:
            raise TypeError("provide either config or keyword config fields, not both")
        if config_values:
            config = config_values
        # Validate against the registered model before construction.  The
        # constructor validates again, which protects custom factories too.
        validated = spec.config_model.model_validate(config or {})
        return spec.create(validated)

    def get(
        self,
        name: str,
        config: Mapping[str, Any] | BaseMethodConfig | None = None,
        **config_values: Any,
    ) -> TrajectoryMeasure:
        return self.create(name, config, **config_values)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> MeasureSpec:
        return self.get_spec(name)

    def __iter__(self) -> Iterator[TrajectoryMeasure]:
        # This deliberately supports the proposal's idiom:
        # ``for measure in registry: measure.distance(query, candidate)``.
        return iter(tuple(spec.create() for spec in self._specs.values()))

    def __len__(self) -> int:
        return len(self._specs)

    def metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "version": spec.version,
                "implementation": f"{spec.factory.__module__}.{spec.factory.__qualname__}",
                "capabilities": {
                    field: getattr(spec.capabilities, field)
                    for field in spec.capabilities.__dataclass_fields__
                },
                "config_model": f"{spec.config_model.__module__}.{spec.config_model.__qualname__}",
                "source": spec.source,
                "citation": spec.citation,
            }
            for spec in self._specs.values()
        ]


registry = MeasureRegistry()
registry.register("euclidean", EuclideanMeasure)
registry.register("dtw", DTWMeasure)
registry.register("hausdorff", HausdorffMeasure)
registry.register("discrete_frechet", DiscreteFrechetMeasure)
registry.register("lcss", LCSSMeasure)
registry.register("edr", EDRMeasure)
registry.register("erp", ERPMeasure)

# A descriptive alias is useful to callers that prefer an explicit constant.
MEASURE_REGISTRY = registry


def create_measure(
    name: str,
    config: Mapping[str, Any] | BaseMethodConfig | None = None,
    **config_values: Any,
) -> TrajectoryMeasure:
    """Construct a registered measure after strict config validation."""

    return registry.create(name, config, **config_values)


def get_measure(
    name: str,
    config: Mapping[str, Any] | BaseMethodConfig | None = None,
    **config_values: Any,
) -> TrajectoryMeasure:
    """Compatibility spelling for ``create_measure``."""

    return create_measure(name, config, **config_values)


__all__ = [
    "MEASURE_REGISTRY",
    "MeasureRegistry",
    "MeasureSpec",
    "create_measure",
    "get_measure",
    "registry",
]
