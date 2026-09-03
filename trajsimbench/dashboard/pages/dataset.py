from __future__ import annotations

from pathlib import Path
from typing import Any

from trajsimbench.dashboard.services.datasets import dataset_summary


def render(run_dir: str | Path) -> dict[str, Any]:
    return dataset_summary(run_dir)
