"""Shared index contracts and provenance metadata."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def array_hash(array: np.ndarray) -> str:
    data = np.ascontiguousarray(array)
    return hashlib.sha256(data.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    index_type: str
    metric: str
    dimension: int
    dtype: str
    ids_hash: str
    embeddings_hash: str
    config: dict[str, Any] = field(default_factory=dict)
    library_version: str = "numpy"
    build_hardware: dict[str, Any] = field(default_factory=dict)
    build_time_ns: int | None = None
    memory_bytes: int | None = None
    file_size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmbeddingIndex(ABC):
    """Minimal common interface for exact and approximate vector indexes."""

    metadata: IndexMetadata | None

    @abstractmethod
    def build(self, ids: Sequence[Any], embeddings: np.ndarray) -> EmbeddingIndex:
        raise NotImplementedError

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(ids, distances)`` in stable rank order."""

    @abstractmethod
    def save(self, path: Path) -> Path:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> EmbeddingIndex:
        raise NotImplementedError


def _hardware() -> dict[str, Any]:
    import os
    import platform

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
    }


def make_metadata(
    index_type: str,
    metric: str,
    ids: Sequence[Any],
    embeddings: np.ndarray,
    *,
    config: dict[str, Any] | None = None,
    start_ns: int | None = None,
) -> IndexMetadata:
    ids_array = np.asarray([str(value) for value in ids], dtype="U")
    return IndexMetadata(
        index_type=index_type,
        metric=metric,
        dimension=int(embeddings.shape[1]),
        dtype=str(embeddings.dtype),
        ids_hash=array_hash(ids_array),
        embeddings_hash=array_hash(embeddings),
        config=dict(config or {}),
        build_hardware=_hardware(),
        build_time_ns=start_ns,
        memory_bytes=int(embeddings.nbytes + ids_array.nbytes),
    )


def write_metadata(path: Path, metadata: IndexMetadata) -> None:
    path.write_text(
        json.dumps(metadata.to_dict(), sort_keys=True, indent=2, default=str), encoding="utf-8"
    )


def read_metadata(path: Path) -> IndexMetadata:
    return IndexMetadata(**json.loads(path.read_text(encoding="utf-8")))
