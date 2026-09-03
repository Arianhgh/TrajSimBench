"""Notion-specific counterfactual diagnostic triplet generators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trajsimbench.notions import SimilarityNotion, default_notion_registry
from trajsimbench.perturbations import PERTURBATION_REGISTRY
from trajsimbench.perturbations.result import canonical_json, hash_payload

from .base import TaskArtifact, TaskConstructionError, dataset_ids, get_trajectory, make_quality

FAMILY_ALIASES = {
    "downsampled_source_vs_similar_distinct_candidate": "downsampled_vs_distinct",
    "downsampled_vs_distinct": "downsampled_vs_distinct",
    "low_noise_vs_high_noise": "noise_scale",
    "noise_scale": "noise_scale",
    "small_vs_large_detour": "detour_scale",
    "detour_scale": "detour_scale",
    "translated_shape_copy_vs_nearby_different_geometry": "translated_vs_nearby",
    "translated_vs_nearby": "translated_vs_nearby",
    "original_reversed": "original_vs_reversed",
    "original_vs_reversed": "original_vs_reversed",
    "spatially_fixed_time_warp": "time_warp",
    "time_warp": "time_warp",
    "same_origin_destination_different_route": "same_od_route",
    "same_od_route": "same_od_route",
    "partial_route_overlap": "partial_overlap",
    "partial_overlap": "partial_overlap",
}


def _seed(seed: int, *parts: Any) -> int:
    return int(hash_payload([seed, *parts])[:16], 16) % (2**63 - 1)


class DiagnosticTaskGenerator:
    name = "counterfactual_diagnostics"
    version = "1.0"

    def __init__(
        self, *, notion_registry=None, perturbation_registry=PERTURBATION_REGISTRY
    ) -> None:
        self.notion_registry = notion_registry or default_notion_registry()
        self.perturbation_registry = perturbation_registry

    def _resolve_notion(self, notion: str | SimilarityNotion) -> SimilarityNotion:
        if isinstance(notion, SimilarityNotion):
            return notion
        if "@" in str(notion):
            notion_id, version = str(notion).split("@", 1)
            return self.notion_registry.get(notion_id, version)
        return self.notion_registry.get(str(notion))

    def _variant(
        self,
        source: Any,
        name: str,
        severity: Any,
        seed: int,
        config: Mapping[str, Any] | None = None,
    ):
        spec = {"name": name, **dict(config or {})}
        perturbation = self.perturbation_registry.create(spec)
        return perturbation.apply(source, severity=severity, seed=seed)

    @staticmethod
    def _identity_variant(source: Any, seed: int, perturbation_registry=PERTURBATION_REGISTRY):
        return perturbation_registry.get("gps_noise").apply(source, severity=0.0, seed=seed)

    def _family_pair(
        self,
        family: str,
        source: Any,
        alternate: Any,
        *,
        seed: int,
        config: Mapping[str, Any],
    ) -> tuple[Any, Any, str, str, Mapping[str, Any]]:
        family = FAMILY_ALIASES.get(family, family)
        if family == "downsampled_vs_distinct":
            a = self._variant(
                source,
                "sampling_reduction",
                config.get("retention_ratio", 0.75),
                _seed(seed, "a"),
                {"mode": "ratio"},
            )
            b_source = alternate if alternate is not None else source
            b = self._variant(
                b_source, "gps_noise", config.get("distinct_noise_m", 50.0), _seed(seed, "b")
            )
            return (
                a,
                b,
                "sampling_reduction",
                "distinct_candidate",
                {"retention_ratio": config.get("retention_ratio", 0.75)},
            )
        if family == "noise_scale":
            a = self._variant(source, "gps_noise", config.get("low_sigma_m", 5.0), _seed(seed, "a"))
            b = self._variant(
                source, "gps_noise", config.get("high_sigma_m", 50.0), _seed(seed, "b")
            )
            return (
                a,
                b,
                "gps_noise_low",
                "gps_noise_high",
                {
                    "low_sigma_m": config.get("low_sigma_m", 5.0),
                    "high_sigma_m": config.get("high_sigma_m", 50.0),
                },
            )
        if family == "detour_scale":
            a = self._variant(
                source, "free_space_detour", config.get("small_ratio", 0.10), _seed(seed, "a")
            )
            b = self._variant(
                source, "free_space_detour", config.get("large_ratio", 0.50), _seed(seed, "b")
            )
            return (
                a,
                b,
                "free_space_detour_small",
                "free_space_detour_large",
                {
                    "small_ratio": config.get("small_ratio", 0.10),
                    "large_ratio": config.get("large_ratio", 0.50),
                },
            )
        if family == "translated_vs_nearby":
            a = self._variant(
                source,
                "spatial_translation",
                config.get("translation_m", 500.0),
                _seed(seed, "a"),
                {"bearing_rad": config.get("bearing_rad", 0.0)},
            )
            b_source = (
                alternate
                if alternate is not None
                else self._variant(
                    source,
                    "free_space_detour",
                    config.get("nearby_detour_ratio", 0.50),
                    _seed(seed, "nearby"),
                )
            )
            if alternate is not None:
                b = self._identity_variant(b_source, _seed(seed, "b"), self.perturbation_registry)
            else:
                b = b_source
            return (
                a,
                b,
                "spatial_translation",
                "nearby_different_geometry",
                {"translation_m": config.get("translation_m", 500.0)},
            )
        if family == "original_vs_reversed":
            a = self._identity_variant(source, _seed(seed, "a"), self.perturbation_registry)
            b = self._variant(
                source,
                "reversal",
                {"timestamp_policy": config.get("timestamp_policy", "rebase")},
                _seed(seed, "b"),
            )
            return (
                a,
                b,
                "identity",
                "reversal",
                {"timestamp_policy": config.get("timestamp_policy", "rebase")},
            )
        if family == "time_warp":
            a = self._identity_variant(source, _seed(seed, "a"), self.perturbation_registry)
            b = self._variant(
                source, "speed_distortion", config.get("speed_factor", 2.0), _seed(seed, "b")
            )
            return (
                a,
                b,
                "identity",
                "speed_distortion",
                {"speed_factor": config.get("speed_factor", 2.0), "spatially_fixed": True},
            )
        if family == "same_od_route":
            a = self._identity_variant(source, _seed(seed, "a"), self.perturbation_registry)
            b = self._variant(
                source,
                "free_space_detour",
                config.get("route_detour_ratio", 0.75),
                _seed(seed, "b"),
            )
            return (
                a,
                b,
                "identity",
                "free_space_detour",
                {
                    "same_origin_destination": True,
                    "route_detour_ratio": config.get("route_detour_ratio", 0.75),
                },
            )
        if family == "partial_overlap":
            a = self._variant(
                source, "free_space_detour", config.get("low_detour_ratio", 0.10), _seed(seed, "a")
            )
            b = self._variant(
                source, "free_space_detour", config.get("high_detour_ratio", 0.50), _seed(seed, "b")
            )
            return (
                a,
                b,
                "free_space_detour_low",
                "free_space_detour_high",
                {"overlap_definition": "resampled_path_coverage_v1"},
            )
        raise TaskConstructionError(f"unknown diagnostic family {family!r}")

    def _expected_order(
        self, family: str, notion: SimilarityNotion, a_name: str, b_name: str
    ) -> str:
        family = FAMILY_ALIASES.get(family, family)
        if family == "translated_vs_nearby":
            if notion.notion_id == "geometric_shape":
                return "a_closer"
            if notion.notion_id == "absolute_geographic_route":
                return "b_closer"
            return "unspecified"
        if family == "original_vs_reversed":
            if notion.notion_id in {"geometric_shape", "route_path_structure"}:
                return "tie" if notion.notion_id == "geometric_shape" else "unspecified"
            if notion.notion_id in {
                "direction_aware_movement",
                "direction_aware",
                "temporal_dynamics",
                "same_underlying_movement",
            }:
                return "a_closer"
            return "unspecified"
        if family == "time_warp":
            if notion.notion_id == "temporal_dynamics":
                return "a_closer"
            if notion.notion_id in {
                "geometric_shape",
                "absolute_geographic_route",
                "route_path_structure",
            }:
                return "tie"
            return "unspecified"
        if family in {"detour_scale", "partial_overlap"}:
            return (
                "a_closer"
                if notion.notion_id
                in {
                    "geometric_shape",
                    "absolute_geographic_route",
                    "route_path_structure",
                    "same_underlying_movement",
                }
                else "unspecified"
            )
        if family == "same_od_route":
            return (
                "a_closer"
                if notion.notion_id
                in {"absolute_geographic_route", "route_path_structure", "geometric_shape"}
                else "unspecified"
            )
        label = notion.triplet_label(a_name, b_name)
        return label

    def build(
        self,
        dataset: Any,
        *,
        family: str,
        notion: str | SimilarityNotion,
        count: int = 1,
        source_ids: Sequence[str] | None = None,
        seed: int = 0,
        config: Mapping[str, Any] | None = None,
    ) -> TaskArtifact:
        if count < 1:
            raise TaskConstructionError("diagnostic count must be positive")
        resolved_notion = self._resolve_notion(notion)
        family_key = FAMILY_ALIASES.get(family, family)
        sources = tuple(
            sorted(map(str, source_ids if source_ids is not None else dataset_ids(dataset)))
        )
        if not sources:
            raise TaskConstructionError("diagnostic task requires source trajectories")
        options = dict(config or {})
        records: list[dict[str, Any]] = []
        reasons: list[str] = []
        attempted = 0
        for index in range(count):
            source_id = sources[index % len(sources)]
            alternate_id = sources[(index + 1) % len(sources)] if len(sources) > 1 else None
            source = get_trajectory(dataset, source_id)
            alternate = get_trajectory(dataset, alternate_id) if alternate_id is not None else None
            attempted += 1
            try:
                a, b, a_name, b_name, constraints = self._family_pair(
                    family_key,
                    source,
                    alternate,
                    seed=_seed(seed, index, source_id),
                    config=options,
                )
            except (TaskConstructionError, ValueError) as exc:
                reasons.append(str(exc))
                continue
            if not a.generated or not b.generated or a.variant_id == b.variant_id:
                reasons.append("variant_not_generated_or_duplicate")
                continue
            expected = self._expected_order(family_key, resolved_notion, a_name, b_name)
            record_hash = hash_payload(
                [source_id, a.variant_id, b.variant_id, resolved_notion.key, family_key]
            )[:24]
            record_id = f"triplet:{record_hash}"
            records.append(
                {
                    "triplet_id": record_id,
                    "task_id": record_id,
                    "query_id": source_id,
                    "candidate_a_id": a.variant_id,
                    "candidate_b_id": b.variant_id,
                    "candidate_ids": (a.variant_id, b.variant_id),
                    "notion_id": resolved_notion.notion_id,
                    "notion_version": resolved_notion.version,
                    "expected_order": expected,
                    "generator": self.name,
                    "generator_version": self.version,
                    "parameters_json": canonical_json({"family": family_key, "config": options}),
                    "seed": int(_seed(seed, index, source_id)),
                    "quality_flags": (),
                    "constraint_values_json": constraints,
                    "candidate_a_provenance": a.provenance.to_dict(),
                    "candidate_b_provenance": b.provenance.to_dict(),
                }
            )
        quality = make_quality(
            attempted,
            len(records),
            reasons,
            required_count=count,
            minimum_yield=float(options.get("minimum_yield", 0.0)),
        )
        return TaskArtifact(
            task_type="diagnostic",
            schema_version="1.0",
            records=tuple(records),
            generator=self.name,
            generator_version=self.version,
            seed=int(seed),
            config={
                "family": family_key,
                "notion_id": resolved_notion.notion_id,
                "notion_version": resolved_notion.version,
                **options,
            },
            quality=quality,
            metadata={
                "notion": resolved_notion.to_dict(),
                "family_alias": family,
                "expectation_policy": (
                    "unspecified is excluded from triplet accuracy; ties use notion tolerance"
                ),
            },
        )


def generate_diagnostic_triplets(dataset: Any, **kwargs: Any) -> TaskArtifact:
    return DiagnosticTaskGenerator().build(dataset, **kwargs)
