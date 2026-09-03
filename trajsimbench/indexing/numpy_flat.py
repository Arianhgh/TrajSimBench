"""Exact NumPy flat embedding index for the CPU baseline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from .base import EmbeddingIndex, IndexMetadata, make_metadata, read_metadata, write_metadata


class NumpyFlatIndex(EmbeddingIndex):
    def __init__(self, *, metric: str = "l2", normalize: bool = False) -> None:
        if metric not in {"l2", "cosine"}:
            raise ValueError("metric must be 'l2' or 'cosine'")
        self.metric = metric
        self.normalize = bool(normalize)
        self.ids: np.ndarray | None = None
        self.embeddings: np.ndarray | None = None
        self.metadata: IndexMetadata | None = None

    def build(self, ids: Sequence[Any], embeddings: np.ndarray) -> NumpyFlatIndex:
        values = np.asarray(embeddings)
        if values.ndim != 2 or values.shape[0] != len(ids):
            raise ValueError("embeddings must be a 2D array with one row per id")
        if not np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float32)
        values = np.ascontiguousarray(values)
        if not np.all(np.isfinite(values)):
            raise ValueError("embeddings must be finite")
        if self.normalize or self.metric == "cosine":
            values = _normalize(values)
        self.ids = np.asarray(list(ids), dtype=object)
        self.embeddings = values
        started = perf_counter_ns()
        self.metadata = make_metadata(
            self.__class__.__name__,
            self.metric,
            ids,
            values,
            config={"normalize": self.normalize},
            start_ns=started,
        )
        self.metadata = IndexMetadata(
            **{**self.metadata.to_dict(), "build_time_ns": perf_counter_ns() - started}
        )
        return self

    def _check(self) -> tuple[np.ndarray, np.ndarray]:
        if self.ids is None or self.embeddings is None or self.metadata is None:
            raise RuntimeError("index has not been built")
        return self.ids, self.embeddings

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        ids, values = self._check()
        if not isinstance(k, (int, np.integer)) or int(k) < 1:
            raise ValueError("k must be a positive integer")
        queries_array = np.asarray(queries, dtype=values.dtype)
        if queries_array.ndim == 1:
            queries_array = queries_array[None, :]
        if queries_array.ndim != 2 or queries_array.shape[1] != values.shape[1]:
            raise ValueError("queries must have the same embedding dimension as the index")
        if not np.all(np.isfinite(queries_array)):
            raise ValueError("queries must be finite")
        if self.normalize or self.metric == "cosine":
            queries_array = _normalize(queries_array)
        if self.metric == "l2":
            distances = np.sum((queries_array[:, None, :] - values[None, :, :]) ** 2, axis=2)
        else:
            distances = 1.0 - queries_array @ values.T
        take = min(int(k), len(ids))
        output_ids: list[list[Any]] = []
        output_distances: list[list[float]] = []
        for row in distances:
            order = sorted(range(len(ids)), key=lambda i: (float(row[i]), str(ids[i])))[:take]
            output_ids.append([ids[i] for i in order])
            output_distances.append([float(row[i]) for i in order])
        return np.asarray(output_ids, dtype=object), np.asarray(output_distances, dtype=np.float64)

    def save(self, path: Path) -> Path:
        ids, values = self._check()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=ids, embeddings=values)
        assert self.metadata is not None
        metadata_path = path.with_suffix(path.suffix + ".json")
        current = self.metadata.to_dict()
        current["file_size_bytes"] = path.stat().st_size
        self.metadata = IndexMetadata(**current)
        write_metadata(metadata_path, self.metadata)
        return path

    @classmethod
    def load(cls, path: Path) -> NumpyFlatIndex:
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            ids = data["ids"]
            embeddings = data["embeddings"]
        metadata = read_metadata(path.with_suffix(path.suffix + ".json"))
        index = cls(metric=metadata.metric, normalize=bool(metadata.config.get("normalize", False)))
        index.ids = ids.astype(object)
        index.embeddings = np.ascontiguousarray(embeddings)
        index.metadata = metadata
        return index


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm embeddings cannot be normalized")
    return values / norms
