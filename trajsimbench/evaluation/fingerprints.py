"""Public fingerprint helpers kept separate for downstream imports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .diagnostics import FINGERPRINT_DIMENSIONS, build_similarity_fingerprint


def fingerprint_rows(
    scores_by_method: Mapping[str, Mapping[str, float]],
    *,
    notion: str | None = None,
    version: str = "1.0",
) -> list[dict[str, Any]]:
    return [
        build_similarity_fingerprint(scores, method=method, notion=notion, version=version)
        for method, scores in sorted(scores_by_method.items())
    ]


__all__ = ["FINGERPRINT_DIMENSIONS", "build_similarity_fingerprint", "fingerprint_rows"]
