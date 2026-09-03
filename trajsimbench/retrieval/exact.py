"""Correctness-first exact retrieval.

Distances are always oriented so that lower is better.  Candidate ties are
resolved by the supplied candidate id, which makes full and chunked retrieval
bit-for-bit equivalent even when a database is processed in different blocks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

import numpy as np


def _identifier_key(value: Any) -> tuple[str, str]:
    """Return a total-order key that is stable across mixed id types."""

    return (type(value).__name__, str(value))


def _distance_from_result(value: Any) -> tuple[float, float]:
    if hasattr(value, "distance"):
        distance = float(value.distance)
        raw = float(getattr(value, "raw_score", distance))
    elif isinstance(value, Mapping) and "distance" in value:
        distance = float(value["distance"])
        raw = float(value.get("raw_score", distance))
    elif isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
        distance = float(value[0])
        raw = float(value[1])
    else:
        distance = float(value)
        raw = distance
    if not np.isfinite(distance):
        raise ValueError("distance function returned a non-finite distance")
    return distance, raw


def _resolve_distance_fn(
    measure: Any = None, distance_fn: Callable[..., Any] | None = None
) -> Callable[..., Any]:
    if distance_fn is not None:
        return distance_fn
    if measure is not None and callable(getattr(measure, "distance", None)):
        return measure.distance
    raise TypeError("provide either measure.distance or distance_fn")


@dataclass(frozen=True, slots=True)
class TopKResult:
    """Stable exact Top-K result for one query."""

    candidate_ids: np.ndarray
    distances: np.ndarray
    raw_scores: np.ndarray
    ranks: np.ndarray
    runtime_ns: int
    candidate_count: int
    query_id: Any = None

    def __post_init__(self) -> None:
        if len(self.candidate_ids) != len(self.distances) or len(self.distances) != len(self.ranks):
            raise ValueError("TopKResult arrays must have equal lengths")

    @property
    def ids(self) -> np.ndarray:
        return self.candidate_ids

    def as_rows(
        self, *, method: str | None = None, task: str | None = None
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, (candidate_id, distance, raw, rank) in enumerate(
            zip(self.candidate_ids, self.distances, self.raw_scores, self.ranks, strict=True)
        ):
            row = {
                "query_id": self.query_id,
                "candidate_id": candidate_id.item()
                if isinstance(candidate_id, np.generic)
                else candidate_id,
                "rank": int(rank),
                "distance": float(distance),
                "raw_score": float(raw),
                "retrieval_runtime_ns": int(self.runtime_ns) if idx == 0 else None,
            }
            if method is not None:
                row["method"] = method
            if task is not None:
                row["task"] = task
            rows.append(row)
        return rows


def _stable_order(ids: Sequence[Any], distances: Sequence[float]) -> list[int]:
    return sorted(range(len(ids)), key=lambda i: (float(distances[i]), _identifier_key(ids[i])))


def exact_top_k(
    query: Any,
    candidates: Sequence[Any] | Iterable[Any],
    *,
    k: int,
    candidate_ids: Sequence[Any] | None = None,
    distance_fn: Callable[..., Any] | None = None,
    measure: Any = None,
    chunk_size: int | None = None,
    exclude_ids: set[Any] | Sequence[Any] | None = None,
    query_id: Any = None,
) -> TopKResult:
    """Compute exact stable Top-K, retaining at most ``k`` entries per chunk.

    ``candidates`` may be a list, NumPy array, or any re-iterable sequence.
    For generators, materialize once because candidate count and final ordering
    are part of the result contract.
    """

    if not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)
    items = list(candidates)
    n = len(items)
    ids = list(range(n)) if candidate_ids is None else list(candidate_ids)
    if len(ids) != n:
        raise ValueError("candidate_ids length must equal candidates length")
    excluded = set(exclude_ids or ())
    fn = _resolve_distance_fn(measure, distance_fn)
    start = perf_counter_ns()
    if chunk_size is None or chunk_size < 1:
        chunk_size = n or 1

    kept_ids: list[Any] = []
    kept_distances: list[float] = []
    kept_raw: list[float] = []
    for offset in range(0, n, int(chunk_size)):
        block = [
            (i, item)
            for i, item in enumerate(items[offset : offset + int(chunk_size)], start=offset)
            if ids[i] not in excluded
        ]
        if not block:
            continue
        block_ids = [ids[i] for i, _ in block]
        block_distances: list[float] = []
        block_raw: list[float] = []
        for _, item in block:
            d, raw = _distance_from_result(fn(query, item))
            block_distances.append(d)
            block_raw.append(raw)

        take = min(k, len(block))
        if take < len(block):
            # argpartition is used for the requested partial Top-K; the final
            # stable sort below is still the authority for tie ordering.  Keep
            # every item at the kth distance so a tie can never be lost merely
            # because NumPy chose an arbitrary partition member.
            partition = np.argpartition(np.asarray(block_distances, dtype=np.float64), take - 1)
            threshold = block_distances[int(partition[take - 1])]
            selected = [i for i, distance in enumerate(block_distances) if distance <= threshold]
        else:
            selected = list(range(len(block)))
        selected.sort(key=lambda i: (block_distances[i], _identifier_key(block_ids[i])))
        kept_ids.extend(block_ids[i] for i in selected)
        kept_distances.extend(block_distances[i] for i in selected)
        kept_raw.extend(block_raw[i] for i in selected)
        if len(kept_ids) > 2 * k:
            order = _stable_order(kept_ids, kept_distances)[:k]
            kept_ids = [kept_ids[i] for i in order]
            kept_distances = [kept_distances[i] for i in order]
            kept_raw = [kept_raw[i] for i in order]

    order = _stable_order(kept_ids, kept_distances)[:k]
    ordered_ids = [kept_ids[i] for i in order]
    ordered_distances = [kept_distances[i] for i in order]
    ordered_raw = [kept_raw[i] for i in order]
    return TopKResult(
        candidate_ids=np.asarray(ordered_ids, dtype=object),
        distances=np.asarray(ordered_distances, dtype=np.float64),
        raw_scores=np.asarray(ordered_raw, dtype=np.float64),
        ranks=np.arange(1, len(order) + 1, dtype=np.int64),
        runtime_ns=perf_counter_ns() - start,
        candidate_count=n - sum(identifier in excluded for identifier in ids),
        query_id=query_id,
    )


chunked_top_k = exact_top_k


def rank_candidates(
    query: Any,
    candidates: Sequence[Any],
    *,
    candidate_ids: Sequence[Any] | None = None,
    distance_fn: Callable[..., Any] | None = None,
    measure: Any = None,
    exclude_ids: set[Any] | Sequence[Any] | None = None,
    query_id: Any = None,
) -> TopKResult:
    """Return a complete deterministic ranking using the same Top-K path."""

    items = list(candidates)
    return exact_top_k(
        query,
        items,
        k=max(1, len(items)),
        candidate_ids=candidate_ids,
        distance_fn=distance_fn,
        measure=measure,
        chunk_size=None,
        exclude_ids=exclude_ids,
        query_id=query_id,
    )


def pairwise_distances(
    query: Any,
    candidates: Sequence[Any],
    *,
    distance_fn: Callable[..., Any] | None = None,
    measure: Any = None,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full ``(distance, raw_score)`` arrays in candidate order."""

    items = list(candidates)
    fn = _resolve_distance_fn(measure, distance_fn)
    distances = np.empty(len(items), dtype=np.float64)
    raw_scores = np.empty(len(items), dtype=np.float64)
    block_size = len(items) if chunk_size is None or chunk_size < 1 else int(chunk_size)
    for offset in range(0, len(items), block_size):
        for local, item in enumerate(items[offset : offset + block_size], start=offset):
            distances[local], raw_scores[local] = _distance_from_result(fn(query, item))
    return distances, raw_scores


def retrieve_many(
    queries: Sequence[Any],
    candidates: Sequence[Any],
    *,
    k: int,
    query_ids: Sequence[Any] | None = None,
    candidate_ids: Sequence[Any] | None = None,
    distance_fn: Callable[..., Any] | None = None,
    measure: Any = None,
    chunk_size: int | None = None,
) -> list[TopKResult]:
    """Deterministic single-process batch retrieval."""

    qids = list(range(len(queries))) if query_ids is None else list(query_ids)
    if len(qids) != len(queries):
        raise ValueError("query_ids length must equal queries length")
    return [
        exact_top_k(
            query,
            candidates,
            k=k,
            candidate_ids=candidate_ids,
            distance_fn=distance_fn,
            measure=measure,
            chunk_size=chunk_size,
            query_id=qid,
        )
        for query, qid in zip(queries, qids, strict=True)
    ]


# Compatibility spellings used by experiment notebooks and older draft APIs.
exact_topk = exact_top_k
chunked_exact_topk = exact_top_k
retrieve_top_k = exact_top_k
