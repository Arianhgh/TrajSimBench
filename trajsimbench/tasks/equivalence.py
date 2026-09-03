"""Underlying-trajectory equivalence task construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trajsimbench.perturbations import PERTURBATION_REGISTRY
from trajsimbench.perturbations.result import hash_payload

from .base import TaskArtifact, TaskConstructionError, dataset_ids, get_trajectory, make_quality


def _stable_seed(seed: int, *parts: Any) -> int:
    return int(hash_payload([int(seed), *parts])[:16], 16) % (2**63 - 1)


def _spec_name_severity(spec: Any) -> tuple[Any, Any, dict[str, Any]]:
    if isinstance(spec, Mapping):
        mapping = dict(spec)
        name = mapping.get("name", mapping.get("type", mapping.get("transformation")))
        if name is None:
            raise TaskConstructionError(
                "equivalence perturbation spec requires name/type/transformation"
            )
        if "severity" not in mapping:
            raise TaskConstructionError(f"equivalence perturbation {name!r} requires severity")
        return name, mapping["severity"], mapping
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return spec[0], spec[1], {"name": spec[0], "severity": spec[1]}
    raise TaskConstructionError(
        "equivalence perturbations must be mappings or (name, severity) pairs"
    )


class EquivalenceTaskGenerator:
    name = "underlying_trajectory_equivalence"
    version = "1.0"

    def __init__(self, *, registry=PERTURBATION_REGISTRY) -> None:
        self.registry = registry

    def build(
        self,
        dataset: Any,
        *,
        source_ids: Sequence[str] | None = None,
        perturbations: Sequence[Any] = (
            ("sampling_reduction", 0.75),
            ("gps_noise", 5.0),
            ("temporal_jitter", 1.0),
        ),
        negative_ids: Mapping[str, Sequence[str]] | None = None,
        seed: int = 0,
        config: Mapping[str, Any] | None = None,
    ) -> TaskArtifact:
        sources = tuple(
            sorted(map(str, source_ids if source_ids is not None else dataset_ids(dataset)))
        )
        if not sources:
            raise TaskConstructionError("equivalence task requires at least one source trajectory")
        specs = tuple(perturbations)
        records: list[dict[str, Any]] = []
        variant_provenance: dict[str, Any] = {}
        reasons: list[str] = []
        attempted = 0
        for source_id in sources:
            source = get_trajectory(dataset, source_id)
            relevant_ids: list[str] = []
            provenance: list[Mapping[str, Any]] = []
            for spec_index, spec in enumerate(specs):
                attempted += 1
                name, severity, spec_mapping = _spec_name_severity(spec)
                perturbation = self.registry.create(spec_mapping)
                result = perturbation.apply(
                    source,
                    severity=severity,
                    seed=_stable_seed(seed, source_id, spec_index, name, severity),
                )
                if not result.generated:
                    reasons.append(result.reason or "not_generated")
                    continue
                relevant_ids.append(result.variant_id)
                provenance.append(result.provenance.to_dict())
                variant_provenance[result.variant_id] = result.provenance.to_dict()
            if relevant_ids:
                negatives = tuple(sorted(map(str, (negative_ids or {}).get(source_id, ()))))
                candidates = tuple(relevant_ids) + tuple(
                    candidate
                    for candidate in negatives
                    if candidate not in relevant_ids and candidate != source_id
                )
                records.append(
                    {
                        "task_id": f"equivalence:{source_id}",
                        "query_id": source_id,
                        "source_id": source_id,
                        "candidate_ids": candidates,
                        "relevant_ids": tuple(relevant_ids),
                        "negative_ids": tuple(
                            candidate for candidate in candidates if candidate not in relevant_ids
                        ),
                        "variant_provenance": tuple(provenance),
                    }
                )
            else:
                reasons.append("source_has_no_generated_variants")
        artifact_config = dict(config or {})
        artifact_config.update(
            {
                "source_ids": sources,
                "perturbations": tuple(specs),
                "negative_ids": negative_ids or {},
            }
        )
        return TaskArtifact(
            task_type="equivalence",
            schema_version="1.0",
            records=tuple(records),
            generator=self.name,
            generator_version=self.version,
            seed=int(seed),
            config=artifact_config,
            quality=make_quality(
                attempted, len(records), reasons, required_count=len(sources), minimum_yield=0.0
            ),
            metadata={
                "source_id_hash": hash_payload(sources),
                "variant_provenance": variant_provenance,
                "relevance_policy": "all generated variants from the same source are relevant",
                "source_partition_policy": (
                    "caller must provide disjoint source IDs for evaluation partitions"
                ),
            },
        )


def generate_equivalence_tasks(dataset: Any, **kwargs: Any) -> TaskArtifact:
    return EquivalenceTaskGenerator().build(dataset, **kwargs)
