from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not config.get("experiment_id"):
        errors.append("experiment_id is required")
    methods = config.get("methods", [])
    if not isinstance(methods, list) or not methods:
        errors.append("methods must be a non-empty list")
    return {"valid": not errors, "errors": errors}


def render(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    current = dict(config or {})
    return {
        "config": current,
        "validation": validate_config(current),
        "download_text": json.dumps(current, indent=2, sort_keys=True),
    }
