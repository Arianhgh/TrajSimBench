"""Safe path helpers for local benchmark artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir


def project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "IMPLEMENTATION_PLAN.md").exists() or (
            candidate / "pyproject.toml"
        ).exists():
            return candidate
    return current


def cache_dir(app_name: str = "trajsimbench") -> Path:
    override = os.environ.get("TRAJSIMBENCH_CACHE")
    path = Path(override).expanduser() if override else Path(user_cache_dir(app_name))
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_under(root: str | Path, relative: str | Path) -> Path:
    """Resolve a path and reject traversal outside the configured root."""

    root_path = Path(root).resolve()
    target = (root_path / relative).resolve()
    try:
        target.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"path escapes configured root: {relative}") from exc
    return target
