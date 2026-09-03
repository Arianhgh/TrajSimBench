"""Namespaced registry for negative generators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import NegativeConstructionError, NegativeGenerator
from .nearby_shape import NearbyShapeNegativeGenerator
from .partial_overlap import PartialOverlapNegativeGenerator
from .random import RandomNegativeGenerator
from .reversed import ReversedNegativeGenerator
from .same_od import SameODNegativeGenerator
from .temporal import SameRouteTemporalNegativeGenerator
from .translated import TranslatedShapeNegativeGenerator


class NegativeGeneratorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, type[NegativeGenerator]] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self, name: str, factory: type[NegativeGenerator], *, aliases: tuple[str, ...] = ()
    ) -> None:
        key = str(name).lower()
        if key in self._factories or key in self._aliases:
            raise NegativeConstructionError(f"duplicate negative generator: {name}")
        self._factories[key] = factory
        for alias in aliases:
            alias_key = str(alias).lower()
            if alias_key in self._factories or alias_key in self._aliases:
                raise NegativeConstructionError(f"duplicate negative generator alias: {alias}")
            self._aliases[alias_key] = key

    def get(self, name: str, **config: Any) -> NegativeGenerator:
        key = str(name).lower()
        key = self._aliases.get(key, key)
        if key not in self._factories:
            raise KeyError(
                f"unknown negative generator {name!r}; "
                f"available: {', '.join(sorted(self._factories))}"
            )
        return self._factories[key](**config)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def create(self, spec: str | Mapping[str, Any] | NegativeGenerator) -> NegativeGenerator:
        if isinstance(spec, NegativeGenerator):
            return spec
        if isinstance(spec, str):
            return self.get(spec)
        if isinstance(spec, Mapping):
            name = spec.get("name", spec.get("type"))
            if name is None:
                raise NegativeConstructionError("negative spec requires name or type")
            config = dict(spec.get("config", {}))
            config.update(
                {key: value for key, value in spec.items() if key not in {"name", "type", "config"}}
            )
            return self.get(str(name), **config)
        raise NegativeConstructionError(
            "negative spec must be a name, mapping, or NegativeGenerator"
        )


def _default_registry() -> NegativeGeneratorRegistry:
    registry = NegativeGeneratorRegistry()
    registry.register("random", RandomNegativeGenerator)
    registry.register("same_od", SameODNegativeGenerator, aliases=("same_origin_destination",))
    registry.register("nearby_shape", NearbyShapeNegativeGenerator, aliases=("spatially_nearby",))
    registry.register("translated_shape", TranslatedShapeNegativeGenerator, aliases=("translated",))
    registry.register("reversed", ReversedNegativeGenerator, aliases=("reversal",))
    registry.register(
        "partial_overlap", PartialOverlapNegativeGenerator, aliases=("route_overlap",)
    )
    registry.register(
        "same_route_temporal", SameRouteTemporalNegativeGenerator, aliases=("temporal",)
    )
    return registry


NEGATIVE_REGISTRY = _default_registry()


def get_negative_generator(
    spec: str | Mapping[str, Any] | NegativeGenerator, **config: Any
) -> NegativeGenerator:
    if config:
        if not isinstance(spec, str):
            raise NegativeConstructionError("extra config requires a generator name")
        return NEGATIVE_REGISTRY.get(spec, **config)
    return NEGATIVE_REGISTRY.create(spec)
