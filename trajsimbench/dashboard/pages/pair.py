from __future__ import annotations

from pathlib import Path
from typing import Any

from trajsimbench.dashboard.services.interactive import pair_detail


def render(run_dir: str | Path, first_id: Any, second_id: Any) -> dict[str, Any]:
    return pair_detail(run_dir, first_id, second_id)
