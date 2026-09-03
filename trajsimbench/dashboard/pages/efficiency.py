from __future__ import annotations

from pathlib import Path
from typing import Any

from trajsimbench.dashboard.services.results import read_result_table


def render(run_dir: str | Path) -> dict[str, Any]:
    rows = read_result_table(run_dir, "systems")
    return {
        "rows": rows,
        "count": len(rows),
        "metrics": ["p95_latency_ns", "throughput", "memory_bytes", "index_size_bytes"],
    }
