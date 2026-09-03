from __future__ import annotations

from pathlib import Path
from typing import Any

from trajsimbench.dashboard.services.results import read_result_table


def render(run_dir: str | Path, *, perturbation: str | None = None) -> dict[str, Any]:
    rows = read_result_table(run_dir, "robustness")
    if perturbation is not None:
        rows = [row for row in rows if row.get("perturbation") == perturbation]
    return {"rows": rows, "count": len(rows), "severity_axis": "severity_normalized"}
