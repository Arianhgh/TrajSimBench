from __future__ import annotations

from pathlib import Path
from typing import Any

from trajsimbench.dashboard.services.results import read_result_table


def render(run_dir: str | Path, *, method: str | None = None) -> dict[str, Any]:
    rows = read_result_table(run_dir, "fingerprints")
    if method is not None:
        rows = [row for row in rows if row.get("method") == method]
    return {
        "rows": rows,
        "count": len(rows),
        "dimensions": [
            "sampling_invariance",
            "gps_noise_robustness",
            "location_sensitivity",
            "shape_sensitivity",
            "direction_sensitivity",
            "temporal_sensitivity",
            "detour_sensitivity",
            "same_od_hard_negative_accuracy",
        ],
    }
