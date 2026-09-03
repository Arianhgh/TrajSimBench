"""Dataset badges and distribution summaries from saved query artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .results import read_result_table


def dataset_summary(run_dir: str | Path) -> dict[str, Any]:
    rows = read_result_table(run_dir, "queries")
    if not rows:
        return {
            "trajectory_count": 0,
            "datasets": [],
            "versions": [],
            "license": None,
            "status": "empty",
        }
    return {
        "trajectory_count": len(rows),
        "datasets": sorted({row.get("dataset") for row in rows if row.get("dataset") is not None}),
        "versions": sorted(
            {row.get("dataset_version") for row in rows if row.get("dataset_version") is not None}
        ),
        "splits": dict(Counter(str(row.get("split")) for row in rows)),
        "license": None,
        "status": "available",
    }
