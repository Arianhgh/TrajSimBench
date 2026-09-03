"""Fixed-universe oracle approximation task construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from trajsimbench.perturbations.result import hash_payload

from .base import TaskArtifact, TaskConstructionError, dataset_ids, get_trajectory, make_quality


def _distance_value(value: Any) -> float:
    if hasattr(value, "distance"):
        value = value.distance
    value = float(value)
    if not np.isfinite(value):
        raise TaskConstructionError("oracle distance must be finite")
    return value


def _lookup_distance(
    source: Any,
    query_id: str,
    candidate_id: str,
    query_index: int,
    candidate_index: int,
    query_view: Any | None = None,
    candidate_view: Any | None = None,
) -> float:
    if callable(source):
        try:
            return _distance_value(
                source(query_view, candidate_view)
                if query_view is not None and candidate_view is not None
                else source(query_id, candidate_id)
            )
        except (TypeError, AttributeError, KeyError, IndexError):
            return _distance_value(source(query_id, candidate_id))
    if hasattr(source, "distance") and callable(source.distance):
        try:
            return _distance_value(
                source.distance(query_view, candidate_view)
                if query_view is not None and candidate_view is not None
                else source.distance(query_id, candidate_id)
            )
        except (TypeError, AttributeError, KeyError, IndexError):
            return _distance_value(source.distance(query_id, candidate_id))
    if isinstance(source, Mapping):
        for key in ((query_id, candidate_id), (str(query_id), str(candidate_id)), query_id):
            if key in source:
                value = source[key]
                if isinstance(value, Mapping):
                    return _distance_value(value[candidate_id])
                return _distance_value(value)
        raise KeyError((query_id, candidate_id))
    matrix = np.asarray(source)
    return _distance_value(matrix[query_index, candidate_index])


def stable_rank(
    candidate_distances: Mapping[str, float], *, tie_tolerance: float = 0.0
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    ordered = sorted(
        ((str(candidate), float(distance)) for candidate, distance in candidate_distances.items()),
        key=lambda item: (item[1], item[0]),
    )
    ranking = tuple(candidate for candidate, _ in ordered)
    groups: list[tuple[str, ...]] = []
    for candidate, distance in ordered:
        if not groups:
            groups.append((candidate,))
            previous = distance
        elif abs(distance - previous) <= tie_tolerance:
            groups[-1] = groups[-1] + (candidate,)
        else:
            groups.append((candidate,))
            previous = distance
    return ranking, tuple(groups)


class OracleApproximationTaskGenerator:
    name = "oracle_approximation"
    version = "1.0"

    def __init__(self, *, tie_tolerance: float = 0.0, exclude_self: bool = True) -> None:
        self.tie_tolerance = float(tie_tolerance)
        self.exclude_self = bool(exclude_self)
        if self.tie_tolerance < 0:
            raise TaskConstructionError("tie_tolerance must be non-negative")

    def build(
        self,
        dataset: Any,
        *,
        query_ids: Sequence[str] | None = None,
        database_ids: Sequence[str] | None = None,
        oracle_distances: Any | None = None,
        oracle: Any | None = None,
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
            raise TaskConstructionError("oracle task requires non-empty query and database ID sets")
        distance_source = oracle_distances if oracle_distances is not None else oracle
        records: list[dict[str, Any]] = []
        reasons: list[str] = []
        database_positions = {candidate: index for index, candidate in enumerate(database)}
        query_positions = {query: index for index, query in enumerate(queries)}
        for query_id in queries:
            candidates = tuple(
                candidate
                for candidate in database
                if not (self.exclude_self and candidate == query_id)
            )
            distances: dict[str, float] = {}
            if distance_source is None:
                reasons.append("oracle_distances_not_supplied")
            else:
                for candidate_id in candidates:
                    try:
                        distances[candidate_id] = _lookup_distance(
                            distance_source,
                            query_id,
                            candidate_id,
                            query_positions.get(query_id, 0),
                            database_positions[candidate_id],
                            get_trajectory(dataset, query_id),
                            get_trajectory(dataset, candidate_id),
                        )
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        raise TaskConstructionError(
                            f"missing oracle distance for {query_id!r}, {candidate_id!r}"
                        ) from exc
            if distances:
                ranking, tie_groups = stable_rank(distances, tie_tolerance=self.tie_tolerance)
            else:
                ranking, tie_groups = tuple(sorted(candidates)), ()
            records.append(
                {
                    "task_id": f"oracle:{query_id}",
                    "query_id": query_id,
                    "candidate_ids": candidates,
                    "oracle_distances": distances,
                    "oracle_ranked_candidate_ids": ranking,
                    "oracle_tie_groups": tie_groups,
                    "self_excluded": query_id not in candidates,
                }
            )
        artifact_config = dict(config or {})
        artifact_config.update(
            {
                "query_ids": queries,
                "database_ids": database,
                "tie_tolerance": self.tie_tolerance,
                "exclude_self": self.exclude_self,
            }
        )
        return TaskArtifact(
            task_type="oracle",
            schema_version="1.0",
            records=tuple(records),
            generator=self.name,
            generator_version=self.version,
            seed=int(seed),
            config=artifact_config,
            quality=make_quality(len(queries), len(records), reasons),
            metadata={
                "query_id_hash": hash_payload(queries),
                "database_id_hash": hash_payload(database),
                "distance_policy": "lower_is_more_similar; stable candidate ID tie break",
            },
        )


def generate_oracle_task(dataset: Any, **kwargs: Any) -> TaskArtifact:
    return OracleApproximationTaskGenerator(
        tie_tolerance=float(kwargs.pop("tie_tolerance", 0.0)),
        exclude_self=bool(kwargs.pop("exclude_self", True)),
    ).build(dataset, **kwargs)
