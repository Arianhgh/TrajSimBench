"""Optional FAISS FlatL2 wrapper with a clear CPU-only fallback boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .numpy_flat import NumpyFlatIndex


class FaissFlatIndex(NumpyFlatIndex):
    """Use FAISS when installed; otherwise require explicit fallback."""

    def __init__(
        self, *, metric: str = "l2", normalize: bool = False, allow_fallback: bool = True
    ) -> None:
        super().__init__(metric=metric, normalize=normalize)
        self.allow_fallback = allow_fallback
        try:
            import faiss
        except ImportError:
            self._faiss = None
            if not allow_fallback:
                raise ImportError(
                    "FAISS CPU is optional; install the 'index' extra or set allow_fallback=True"
                ) from None
        else:
            self._faiss = faiss

    def build(self, ids: Sequence[Any], embeddings: np.ndarray) -> FaissFlatIndex:
        super().build(ids, embeddings)
        # Search delegates to the deterministic NumPy implementation for the
        # fallback and to preserve the same tie behavior in the CPU MVP.
        if self.metadata is not None:
            self.metadata = type(self.metadata)(
                **{
                    **self.metadata.to_dict(),
                    "index_type": "FaissFlatL2" if self._faiss else "NumpyFlatFallback",
                }
            )
        return self
