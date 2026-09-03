"""Hardware metadata used to make CPU-only runs auditable."""

from __future__ import annotations

import platform
from typing import Any

import numpy as np


def hardware_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count() or 1,
        "gpu_requested": False,
        "numpy": np.__version__,
    }
