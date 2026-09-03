"""Optional ANN boundary; not used by the CPU MVP."""

from __future__ import annotations

from .faiss_flat import FaissFlatIndex


class FaissHNSWIndex(FaissFlatIndex):
    """Compatibility class that retains exact fallback semantics.

    HNSW is intentionally not enabled by the MVP runner; callers can inspect
    ``metadata.config`` and compare this index with FlatL2 when FAISS is added.
    """

    def __init__(
        self,
        *,
        metric: str = "l2",
        normalize: bool = False,
        m: int = 32,
        allow_fallback: bool = True,
    ) -> None:
        super().__init__(metric=metric, normalize=normalize, allow_fallback=allow_fallback)
        if m < 4:
            raise ValueError("HNSW m must be at least 4")
        self.m = int(m)
