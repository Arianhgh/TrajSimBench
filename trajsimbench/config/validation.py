"""Canonicalization, hashing, and explicit cross-field validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from trajsimbench.config.models import ExperimentConfig


class ConfigValidationError(ValueError):
    """Actionable configuration error raised at the public boundary."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_config(config: ExperimentConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Return a sorted, JSON-compatible representation of a resolved config."""

    data = config.model_dump(mode="json") if isinstance(config, ExperimentConfig) else dict(config)
    return _jsonable(data)


def canonical_json(config: ExperimentConfig | Mapping[str, Any]) -> str:
    return json.dumps(
        canonical_config(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def hash_resolved_config(config: ExperimentConfig | Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def validate_config_data(data: Mapping[str, Any]) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(dict(data))
    except ValidationError as exc:
        raise ConfigValidationError(str(exc)) from exc


def dump_resolved_yaml(config: ExperimentConfig, path: Path) -> None:
    """Write a stable resolved YAML file for a run manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_config(config)
    path.write_text(yaml.safe_dump(payload, sort_keys=True, allow_unicode=False), encoding="utf-8")
