"""Versioned relevance providers.

Relevance is deliberately kept outside metric implementations so the same
retrieval rankings can be evaluated under multiple similarity notions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RelevanceProvider(ABC):
    name: str = "provider"
    version: str = "1"
    config: Mapping[str, Any] = field(default_factory=dict)

    @abstractmethod
    def relevance(
        self, query_id: Any, candidate_id: Any, *, context: Mapping[str, Any] | None = None
    ) -> float:
        """Return a non-negative relevance grade; zero means irrelevant."""

    def for_query(
        self,
        query_id: Any,
        candidate_ids: Sequence[Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        values = [
            self.relevance(query_id, candidate_id, context=context)
            for candidate_id in candidate_ids
        ]
        arr = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("relevance values must be finite and non-negative")
        return arr

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "config": dict(self.config)}


@dataclass(frozen=True, slots=True)
class StaticRelevance(RelevanceProvider):
    values: Mapping[Any, Mapping[Any, float]] = field(default_factory=dict)
    name: str = "static"

    def relevance(
        self, query_id: Any, candidate_id: Any, *, context: Mapping[str, Any] | None = None
    ) -> float:
        return float(self.values.get(query_id, {}).get(candidate_id, 0.0))


@dataclass(frozen=True, slots=True)
class SameSourceRelevance(RelevanceProvider):
    source_by_id: Mapping[Any, Any] = field(default_factory=dict)
    query_source_by_id: Mapping[Any, Any] | None = None
    same_query_is_irrelevant: bool = True
    name: str = "same_source"
    version: str = "1.0"

    def relevance(
        self, query_id: Any, candidate_id: Any, *, context: Mapping[str, Any] | None = None
    ) -> float:
        if self.same_query_is_irrelevant and query_id == candidate_id:
            return 0.0
        query_source = (self.query_source_by_id or self.source_by_id).get(query_id)
        candidate_source = self.source_by_id.get(candidate_id)
        return 1.0 if query_source is not None and query_source == candidate_source else 0.0


@dataclass(frozen=True, slots=True)
class OracleTopKRelevance(RelevanceProvider):
    oracle_rankings: Mapping[Any, Sequence[Any]] = field(default_factory=dict)
    graded: bool = False
    name: str = "oracle_top_k"
    version: str = "1.0"

    def relevance(
        self, query_id: Any, candidate_id: Any, *, context: Mapping[str, Any] | None = None
    ) -> float:
        ranked = list(self.oracle_rankings.get(query_id, ()))
        if candidate_id not in ranked:
            return 0.0
        rank = ranked.index(candidate_id)
        return float(1.0 / (rank + 1)) if self.graded else 1.0


@dataclass(frozen=True, slots=True)
class GradedOracleRelevance(OracleTopKRelevance):
    graded: bool = True
    name: str = "graded_oracle"


@dataclass(frozen=True, slots=True)
class TripletRelevance(RelevanceProvider):
    """Relevance grades derived from a query's expected anchor choice(s)."""

    preferred_by_query: Mapping[Any, Sequence[Any]] = field(default_factory=dict)
    grade: float = 1.0
    name: str = "triplet"
    version: str = "1.0"

    def relevance(
        self, query_id: Any, candidate_id: Any, *, context: Mapping[str, Any] | None = None
    ) -> float:
        return (
            float(self.grade) if candidate_id in self.preferred_by_query.get(query_id, ()) else 0.0
        )


@dataclass(frozen=True, slots=True)
class ExternalLabelRelevance(RelevanceProvider):
    labels: Mapping[tuple[Any, Any], float] = field(default_factory=dict)
    name: str = "external_labels"

    def relevance(
        self, query_id: Any, candidate_id: Any, *, context: Mapping[str, Any] | None = None
    ) -> float:
        return float(self.labels.get((query_id, candidate_id), 0.0))


def apply_empty_relevance_policy(values: Sequence[float], policy: str) -> tuple[bool, str | None]:
    """Return ``(valid, reason)`` for a query with no positive relevance."""

    if policy not in {"skip", "zero", "raise"}:
        raise ValueError("empty relevance policy must be 'skip', 'zero', or 'raise'")
    if np.any(np.asarray(values, dtype=float) > 0):
        return True, None
    if policy == "raise":
        raise ValueError("query has no relevant candidates")
    return policy == "zero", "no_relevant_candidates"


def provider_from_config(
    config: Mapping[str, Any], *, context: Mapping[str, Any] | None = None
) -> RelevanceProvider:
    """Construct the built-in provider family from a resolved config."""

    name = str(config.get("name", config.get("provider", "same_source"))).lower()
    context = context or {}
    if name in {"same_source", "equivalence"}:
        return SameSourceRelevance(
            source_by_id=context.get("source_by_id", config.get("source_by_id", {})),
            query_source_by_id=context.get("query_source_by_id"),
            config=dict(config),
        )
    if name in {"oracle_top_k", "oracle"}:
        return OracleTopKRelevance(
            oracle_rankings=context.get("oracle_rankings", config.get("oracle_rankings", {})),
            config=dict(config),
        )
    if name in {"graded_oracle", "oracle_graded"}:
        return GradedOracleRelevance(
            oracle_rankings=context.get("oracle_rankings", config.get("oracle_rankings", {})),
            config=dict(config),
        )
    if name == "triplet":
        return TripletRelevance(
            preferred_by_query=context.get(
                "preferred_by_query", config.get("preferred_by_query", {})
            ),
            config=dict(config),
        )
    if name in {"external", "external_labels"}:
        return ExternalLabelRelevance(
            labels=context.get("labels", config.get("labels", {})), config=dict(config)
        )
    raise ValueError(f"unknown relevance provider: {name}")
