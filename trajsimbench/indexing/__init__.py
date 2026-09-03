"""Embedding index implementations."""

from .base import IndexMetadata
from .numpy_flat import NumpyFlatIndex

try:  # optional FAISS is intentionally not part of the CPU core
    from .faiss_flat import FaissFlatIndex
except ImportError:  # pragma: no cover - depends on environment
    FaissFlatIndex = None  # type: ignore[assignment,misc]

__all__ = ["IndexMetadata", "NumpyFlatIndex", "FaissFlatIndex"]
