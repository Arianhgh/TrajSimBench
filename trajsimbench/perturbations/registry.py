"""Stable registry and regeneration helpers for perturbations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .base import Perturbation, PerturbationError
from .result import PerturbationProvenance
from .route import FreeSpaceDetourPerturbation, RoadNetworkDetourPerturbation
from .sampling import (
    ContiguousOutagePerturbation,
    RandomPointLossPerturbation,
    SamplingFrequencyReductionPerturbation,
    TruncationPerturbation,
)
from .spatial import (
    CorrelatedGPSDriftPerturbation,
    IndependentGPSNoisePerturbation,
    SpatialQuantizationPerturbation,
    SpatialTranslationPerturbation,
)
from .temporal import ReversalPerturbation, SpeedDistortionPerturbation, TemporalJitterPerturbation


class PerturbationRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Any] = {}
        self._aliases: dict[str, str] = {}

    def register(self, name: str, factory: Any, *, aliases: Iterable[str] = ()) -> None:
        key = str(name).strip().lower()
        if not key or key in self._factories:
            raise PerturbationError(f"invalid or duplicate perturbation name: {name!r}")
        self._factories[key] = factory
        for alias in aliases:
            alias_key = str(alias).strip().lower()
            if alias_key in self._factories or alias_key in self._aliases:
                raise PerturbationError(f"duplicate perturbation alias: {alias!r}")
            self._aliases[alias_key] = key

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def resolve_name(self, name: str) -> str:
        key = str(name).strip().lower()
        if key in self._factories:
            return key
        if key in self._aliases:
            return self._aliases[key]
        known = ", ".join(self.names())
        raise KeyError(f"unknown perturbation {name!r}; available: {known}")

    def get(self, name: str, **config: Any) -> Perturbation:
        factory = self._factories[self.resolve_name(name)]
        instance = factory(**config) if isinstance(factory, type) else factory(**config)
        if not isinstance(instance, Perturbation):
            raise PerturbationError(f"factory for {name!r} did not create a Perturbation")
        return instance

    def create(self, spec: str | Mapping[str, Any] | Perturbation) -> Perturbation:
        if isinstance(spec, Perturbation):
            return spec
        if isinstance(spec, str):
            return self.get(spec)
        if not isinstance(spec, Mapping):
            raise PerturbationError("perturbation spec must be a name, mapping, or Perturbation")
        name = spec.get("name", spec.get("type", spec.get("transformation")))
        if name is None:
            raise PerturbationError("perturbation spec requires name/type/transformation")
        config = dict(spec.get("config", {}))
        config.update(
            {
                k: v
                for k, v in spec.items()
                if k not in {"name", "type", "transformation", "config", "severity"}
            }
        )
        return self.get(str(name), **config)

    def apply(
        self,
        name_or_spec: str | Mapping[str, Any] | Perturbation,
        source: Any,
        *,
        severity: Any,
        seed: int,
    ):
        return self.create(name_or_spec).apply(source, severity=severity, seed=int(seed))

    def regenerate(self, source: Any, provenance: Any):
        """Regenerate a variant from its serialized provenance record."""

        if isinstance(provenance, Mapping):
            provenance = PerturbationProvenance.from_dict(provenance)
        transformation = (
            provenance.transformation
            if hasattr(provenance, "transformation")
            else provenance["transformation"]
        )
        parameters = (
            provenance.parameters
            if hasattr(provenance, "parameters")
            else provenance.get("parameters", {})
        )
        config: dict[str, Any] = {}
        if transformation == "gps_drift":
            config["rho"] = parameters.get("rho", 0.9)
        elif transformation in {"random_point_loss", "contiguous_outage"}:
            config["preserve_endpoints"] = parameters.get("preserve_endpoints", True)
        elif transformation == "sampling_reduction":
            config["mode"] = parameters.get("mode", "ratio")
            config["preserve_endpoints"] = parameters.get("preserve_endpoints", True)
        elif transformation == "spatial_quantization":
            config["origin"] = (
                tuple(parameters["grid_origin_m"]) if "grid_origin_m" in parameters else None
            )
        elif transformation == "temporal_jitter":
            config["distribution"] = parameters.get("distribution", "normal")
            config["repair"] = parameters.get("repair", "cumulative_max")
        elif transformation == "truncation":
            config["side"] = parameters.get("side", "end")
        elif transformation == "reversal":
            config["timestamp_policy"] = parameters.get("timestamp_policy", "rebase")
        elif transformation == "spatial_translation":
            config["bearing_rad"] = parameters.get("bearing_rad")
        elif transformation == "free_space_detour":
            if "anchor_fractions" in parameters:
                config["anchor_fraction"] = tuple(parameters["anchor_fractions"])
            if "max_amplitude_factor" in parameters:
                config["max_amplitude_factor"] = parameters["max_amplitude_factor"]
        perturbation = self.get(str(transformation), **config)
        return perturbation.regenerate(source, provenance)


def _default_registry() -> PerturbationRegistry:
    registry = PerturbationRegistry()
    registry.register(
        "gps_noise", IndependentGPSNoisePerturbation, aliases=("independent_gps_noise", "noise")
    )
    registry.register(
        "gps_drift", CorrelatedGPSDriftPerturbation, aliases=("correlated_gps_drift", "drift")
    )
    registry.register(
        "random_point_loss", RandomPointLossPerturbation, aliases=("point_loss", "random_loss")
    )
    registry.register(
        "contiguous_outage", ContiguousOutagePerturbation, aliases=("gps_outage", "outage")
    )
    registry.register(
        "sampling_reduction",
        SamplingFrequencyReductionPerturbation,
        aliases=("sampling_frequency_reduction", "downsampling"),
    )
    registry.register(
        "spatial_quantization", SpatialQuantizationPerturbation, aliases=("quantization",)
    )
    registry.register("temporal_jitter", TemporalJitterPerturbation, aliases=("time_jitter",))
    registry.register("speed_distortion", SpeedDistortionPerturbation, aliases=("speed",))
    registry.register("truncation", TruncationPerturbation, aliases=("truncate",))
    registry.register("reversal", ReversalPerturbation, aliases=("reverse",))
    registry.register(
        "spatial_translation", SpatialTranslationPerturbation, aliases=("translation",)
    )
    registry.register(
        "free_space_detour", FreeSpaceDetourPerturbation, aliases=("detour", "free_space")
    )
    registry.register(
        "road_network_detour", RoadNetworkDetourPerturbation, aliases=("road_detour",)
    )
    return registry


PERTURBATION_REGISTRY = _default_registry()


def get_perturbation(
    name_or_spec: str | Mapping[str, Any] | Perturbation, **config: Any
) -> Perturbation:
    if config:
        if isinstance(name_or_spec, str):
            return PERTURBATION_REGISTRY.get(name_or_spec, **config)
        raise PerturbationError("extra config is only accepted with a perturbation name")
    return PERTURBATION_REGISTRY.create(name_or_spec)
