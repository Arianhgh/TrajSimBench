"""Fixed query/database retrieval task artifacts and relevance providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trajsimbench.perturbations.result import hash_payload

from .base import TaskArtifact, TaskConstructionError, dataset_ids, make_quality


class RetrievalTaskGenerator:
    name = "retrieval"
    version = "1.0"

    def build(
        self,
        dataset: Any,
        *,
        query_ids: Sequence[str] | None = None,
        database_ids: Sequence[str] | None = None,
        relevant_ids: Mapping[str, Sequence[str]] | None = None,
        exclude_self: bool = True,
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
            raise TaskConstructionError("retrieval task requires non-empty query and database IDs")
        records = []
        for query_id in queries:
            candidates = tuple(
                candidate for candidate in database if not (exclude_self and candidate == query_id)
            )
            rel = tuple(
                sorted(set(map(str, (relevant_ids or {}).get(query_id, ()))) & set(candidates))
            )
            records.append(
                {
                    "task_id": f"retrieval:{query_id}",
                    "query_id": query_id,
                    "candidate_ids": candidates,
                    "relevant_ids": rel,
                    "self_excluded": query_id not in candidates,
                }
            )
        artifact_config = dict(config or {})
        artifact_config.update(
            {"query_ids": queries, "database_ids": database, "exclude_self": exclude_self}
        )
        return TaskArtifact(
            task_type="retrieval",
            schema_version="1.0",
            records=tuple(records),
            generator=self.name,
            generator_version=self.version,
            seed=int(seed),
            config=artifact_config,
            quality=make_quality(len(queries), len(records)),
            metadata={
                "query_id_hash": hash_payload(queries),
                "database_id_hash": hash_payload(database),
            },
        )


def generate_retrieval_task(dataset: Any, **kwargs: Any) -> TaskArtifact:
    return RetrievalTaskGenerator().build(dataset, **kwargs)
