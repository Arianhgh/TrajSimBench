"""YAML loading and fragment resolution for benchmark configurations."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trajsimbench.config.models import DatasetReference
from trajsimbench.config.validation import (
    ConfigValidationError,
    canonical_config,
    hash_resolved_config,
    validate_config_data,
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{label} must be a YAML mapping")
    return dict(value)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigValidationError(f"configuration file does not exist: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    return _mapping(data, str(path))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _fragment_path(
    root: Path, category: str, name: str, explicit: str | None = None
) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        explicit_path = Path(explicit)
        candidates.append(explicit_path if explicit_path.is_absolute() else root / explicit_path)
    candidates.extend(
        [
            root / category / f"{name}.yaml",
            root.parent / category / f"{name}.yaml",
            Path("configs") / category / f"{name}.yaml",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _unwrap_fragment(data: dict[str, Any], category: str) -> dict[str, Any]:
    if category in data and isinstance(data[category], dict):
        return dict(data[category])
    return data


def _resolve_dataset(raw: Any, root: Path) -> dict[str, Any]:
    if isinstance(raw, str):
        value: dict[str, Any] = {"name": raw}
    else:
        value = _mapping(raw, "dataset")
    name = value.get("name")
    if not isinstance(name, str):
        raise ConfigValidationError("dataset.name is required")
    fragment = _fragment_path(root, "datasets", name, value.get("config_path"))
    if fragment is None:
        return value
    fragment_data = _unwrap_fragment(read_yaml(fragment), "dataset")
    return _deep_merge(fragment_data, value)


def _resolve_methods(raw: Any, root: Path) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigValidationError("methods must be a list of names or mappings")
    resolved: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            value: dict[str, Any] = {"name": item}
        else:
            value = _mapping(item, "method entry")
        name = value.get("name")
        if not isinstance(name, str):
            raise ConfigValidationError("every method entry needs a name")
        fragment = _fragment_path(root, "methods", name, value.get("config_path"))
        if fragment is not None:
            fragment_data = _unwrap_fragment(read_yaml(fragment), "method")
            value = _deep_merge(fragment_data, value)
        # A fragment may use arbitrary method parameters, but they are kept
        # under the explicitly extensible config field in the resolved model.
        if "parameters" in value and "config" not in value:
            value["config"] = value.pop("parameters")
        known = {"name", "version", "config"}
        extras = {key: value.pop(key) for key in list(value) if key not in known}
        if extras:
            value.setdefault("config", {}).update(extras)
        resolved.append(value)
    return resolved


def _resolve_notion(raw: Any, root: Path) -> dict[str, Any]:
    value: dict[str, Any] = {"name": raw} if isinstance(raw, str) else _mapping(raw or {}, "notion")
    name = value.get("name", "v1")
    fragment = _fragment_path(root, "notions", str(name), value.get("config_path"))
    if fragment is not None:
        fragment_data = _unwrap_fragment(read_yaml(fragment), "notion")
        if "notions" in fragment_data:
            # The repository notion fragment is a versioned collection. Keep
            # the closed NotionSpec surface while retaining the full table in
            # its explicitly extensible config mapping.
            fragment_data = {
                "name": name,
                "version": str(fragment_data.get("schema_version", "1")),
                "config": {"notions": fragment_data["notions"]},
            }
        else:
            fragment_data.pop("schema_version", None)
        value = _deep_merge(fragment_data, value)
    extras = {
        key: value.pop(key) for key in list(value) if key not in {"name", "version", "config"}
    }
    if extras:
        config_value = value.get("config")
        if not isinstance(config_value, dict):
            config_value = {}
        config_value.update(extras)
        value["config"] = config_value
    return value


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Resolved model plus its stable serialized identity."""

    config: Any
    source_path: Path
    canonical: dict[str, Any]
    config_hash: str

    def __getattr__(self, name: str) -> Any:
        return getattr(self.config, name)

    def __getitem__(self, key: str) -> Any:
        return self.canonical[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.canonical.get(key, default)

    @property
    def resolved_config_hash(self) -> str:
        """Alias matching run-manifest terminology."""

        return self.config_hash

    def to_dict(self) -> dict[str, Any]:
        return dict(self.canonical)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.config.model_dump(*args, **kwargs)


def resolve_config(path: str | Path) -> ResolvedConfig:
    source_path = Path(path).resolve()
    raw = read_yaml(source_path)
    root = source_path.parent
    data = dict(raw)
    data["dataset"] = _resolve_dataset(data.get("dataset"), root)
    data["methods"] = _resolve_methods(data.get("methods", []), root)
    if "notion" in data:
        data["notion"] = _resolve_notion(data["notion"], root)
    if "notion_spec" in data:
        data["notion"] = _resolve_notion(data.pop("notion_spec"), root)
    try:
        config = validate_config_data(data)
    except ConfigValidationError:
        raise
    canonical = canonical_config(config)
    return ResolvedConfig(config, source_path, canonical, hash_resolved_config(config))


def load_config(path: str | Path) -> ResolvedConfig:
    """Load and resolve a YAML config; alias kept as the public convenience API."""

    return resolve_config(path)


def load_model(path: str | Path) -> Any:
    """Return only the validated Pydantic model for callers that do not need metadata."""

    return resolve_config(path).config


def load_dataset_config(path: str | Path) -> DatasetReference:
    """Load a standalone dataset fragment with the same strict model."""

    data = read_yaml(Path(path))
    data = _unwrap_fragment(data, "dataset")
    try:
        return DatasetReference.model_validate(data)
    except Exception as exc:
        raise ConfigValidationError(f"invalid dataset config {path}: {exc}") from exc
