from __future__ import annotations

from pathlib import Path
from typing import Any

from trajsimbench.dashboard.services.interactive import retrieval_disagreement


def render(
    run_dir: str | Path, query_id: Any, method_a: str | None = None, method_b: str | None = None
) -> dict[str, Any]:
    return retrieval_disagreement(run_dir, query_id, method_a, method_b)
