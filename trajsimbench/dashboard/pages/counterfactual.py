from __future__ import annotations

from pathlib import Path
from typing import Any

from trajsimbench.dashboard.services.results import read_result_table


def render(
    run_dir: str | Path, *, source_id: Any | None = None, perturbation: str | None = None
) -> dict[str, Any]:
    rows = read_result_table(run_dir, "variants")
    if source_id is not None:
        rows = [row for row in rows if row.get("source_id") == source_id]
    if perturbation is not None:
        rows = [row for row in rows if row.get("perturbation") == perturbation]
    return {"variants": rows, "count": len(rows), "bounded": True}
