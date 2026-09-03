"""Turn bounded negative-generator searches into immutable task artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trajsimbench.negatives import NEGATIVE_REGISTRY, NegativeGenerator
from trajsimbench.perturbations.result import hash_payload

from .base import TaskArtifact, TaskConstructionError, dataset_ids, make_quality


class NegativeTaskGenerator:
    name = "negative_retrieval"
    version = "1.0"

    def __init__(self, *, registry=NEGATIVE_REGISTRY) -> None:
        self.registry = registry

    def build(
        self,
        dataset: Any,
        *,
        generator: str | Mapping[str, Any] | NegativeGenerator,
        query_ids: Sequence[str] | None = None,
        database_ids: Sequence[str] | None = None,
        negatives_per_query: int = 1,
        seed: int = 0,
        config: Mapping[str, Any] | None = None,
    ) -> TaskArtifact:
        queries = tuple(
            sorted(map(str, query_ids if query_ids is not None else dataset_ids(dataset)))
        )
        database = tuple(
            sorted(map(str, database_ids if database_ids is not None else dataset_ids(dataset)))
        )
        if not queries or not database:
            raise TaskConstructionError("negative task requires non-empty query and database IDs")
        if negatives_per_query < 1:
            raise TaskConstructionError("negatives_per_query must be positive")
        negative_generator = self.registry.create(generator)
        records: list[dict[str, Any]] = []
        reasons: list[str] = []
        reports: list[Mapping[str, Any]] = []
        for _query_index, query_id in enumerate(queries):
            result = negative_generator.generate(
                query_id,
                database,
                dataset=dataset,
                count=negatives_per_query,
                seed=int(hash_payload([seed, query_id, negative_generator.name])[:16], 16)
                % (2**63 - 1),
                config=config,
            )
            reports.append(result.report.to_dict())
            if not result.candidates:
                reasons.append("no_qualifying_negative")
                continue
            candidate_ids = tuple(candidate.candidate_id for candidate in result.candidates)
            records.append(
                {
                    "task_id": f"negative:{negative_generator.name}:{query_id}",
                    "query_id": query_id,
                    "candidate_ids": candidate_ids,
                    "negative_ids": candidate_ids,
                    "negative_type": negative_generator.name,
                    "achieved_constraints": tuple(
                        candidate.to_dict() for candidate in result.candidates
                    ),
                    "construction_report": result.report.to_dict(),
                }
            )
        resolved_config = dict(config or {})
        resolved_config.update(
            {
                "query_ids": queries,
                "database_ids": database,
                "negatives_per_query": negatives_per_query,
                "generator": negative_generator.name,
            }
        )
        return TaskArtifact(
            task_type="negative",
            schema_version="1.0",
            records=tuple(records),
            generator=self.name,
            generator_version=self.version,
            seed=int(seed),
            config=resolved_config,
            quality=make_quality(
                len(queries),
                len(records),
                reasons,
                required_count=len(queries),
                minimum_yield=float(resolved_config.get("minimum_yield", 0.0)),
            ),
            metadata={
                "generator": negative_generator.name,
                "generator_version": negative_generator.version,
                "query_id_hash": hash_payload(queries),
                "database_id_hash": hash_payload(database),
                "construction_reports": reports,
            },
        )


def generate_negative_task(dataset: Any, **kwargs: Any) -> TaskArtifact:
    return NegativeTaskGenerator().build(dataset, **kwargs)
