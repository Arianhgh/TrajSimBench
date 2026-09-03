"""Registry and file loader for notion v1 specifications."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .models import NotionValidationError, SimilarityNotion


def _load_yaml_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise NotionValidationError(
                "PyYAML is required for non-JSON YAML; the bundled v1 file is JSON-compatible YAML"
            ) from exc
    try:
        return yaml.safe_load(text)
    except Exception as exc:  # pragma: no cover - parser-specific exception types vary
        raise NotionValidationError(f"could not parse notion file {path}: {exc}") from exc


def _notions_from_document(document: Any) -> list[SimilarityNotion]:
    if isinstance(document, Mapping):
        values = document.get("notions", document)
        if isinstance(values, Mapping):
            values = [
                dict(value, notion_id=key)
                if isinstance(value, Mapping) and "notion_id" not in value
                else value
                for key, value in values.items()
            ]
    else:
        values = document
    if not isinstance(values, list):
        raise NotionValidationError("notion document must contain a list under 'notions'")
    result = []
    for item in values:
        if not isinstance(item, Mapping):
            raise NotionValidationError("each notion must be a mapping")
        required = {"notion_id", "version", "definition", "expected_outcomes"}
        missing = sorted(required - set(item))
        if missing:
            raise NotionValidationError(f"notion is missing required fields: {', '.join(missing)}")
        allowed = required | {
            "exclusions",
            "properties",
            "tie_tolerance",
            "minimum_margin",
            "citations",
            "decision_notes",
            "status",
        }
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise NotionValidationError(f"unknown notion fields: {', '.join(unknown)}")
        result.append(
            SimilarityNotion(
                notion_id=str(item["notion_id"]),
                version=str(item["version"]),
                definition=str(item["definition"]),
                exclusions=tuple(item.get("exclusions", ())),
                properties=item.get("properties", {}),
                expected_outcomes=item["expected_outcomes"],
                tie_tolerance=float(item.get("tie_tolerance", 0.0)),
                minimum_margin=float(item.get("minimum_margin", 0.0)),
                citations=tuple(item.get("citations", ())),
                decision_notes=tuple(item.get("decision_notes", ())),
                status=str(item.get("status", "active")),
            )
        )
    return result


class NotionRegistry:
    def __init__(self, notions: Iterable[SimilarityNotion] = ()) -> None:
        self._notions: dict[tuple[str, str], SimilarityNotion] = {}
        for notion in notions:
            self.register(notion)

    def register(self, notion: SimilarityNotion) -> None:
        key = (notion.notion_id, notion.version)
        if key in self._notions:
            raise NotionValidationError(f"duplicate notion: {notion.key}")
        self._notions[key] = notion

    def get(self, notion_id: str, version: str | None = None) -> SimilarityNotion:
        if version is not None:
            try:
                return self._notions[(str(notion_id), str(version))]
            except KeyError as exc:
                raise KeyError(f"unknown notion {notion_id}@{version}") from exc
        matches = [value for (key, _), value in self._notions.items() if key == str(notion_id)]
        if not matches:
            raise KeyError(f"unknown notion {notion_id}")
        return sorted(matches, key=lambda value: value.version)[-1]

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(notion.key for notion in self._notions.values()))

    def values(self) -> tuple[SimilarityNotion, ...]:
        return tuple(self._notions[key] for key in sorted(self._notions))

    def __len__(self) -> int:
        return len(self._notions)

    @classmethod
    def from_file(cls, path: str | Path) -> NotionRegistry:
        return cls(_notions_from_document(_load_yaml_or_json(Path(path))))


def load_notion_file(path: str | Path) -> tuple[SimilarityNotion, ...]:
    return NotionRegistry.from_file(path).values()


def default_notion_registry() -> NotionRegistry:
    notions = [
        SimilarityNotion(
            "geometric_shape",
            "1.0",
            "Similarity of the projected polyline shape independent of absolute placement.",
            exclusions=("does not assert the same geographic location",),
            properties={
                "spatial": True,
                "location": False,
                "route": True,
                "direction": False,
                "temporal": False,
                "sampling": True,
                "observation": True,
            },
            expected_outcomes={
                "gps_noise": "preserve",
                "gps_drift": "preserve",
                "random_point_loss": "preserve",
                "contiguous_outage": "preserve",
                "sampling_reduction": "preserve",
                "spatial_quantization": "small_change",
                "temporal_jitter": "preserve",
                "speed_distortion": "preserve",
                "truncation": "small_change",
                "reversal": "preserve",
                "spatial_translation": "preserve",
                "free_space_detour": "small_change",
            },
            tie_tolerance=1e-9,
        ),
        SimilarityNotion(
            "absolute_geographic_route",
            "1.0",
            "Similarity of a route in its absolute geographic location.",
            properties={
                "spatial": True,
                "location": True,
                "route": True,
                "direction": False,
                "temporal": False,
                "sampling": True,
                "observation": False,
            },
            expected_outcomes={
                "gps_noise": "preserve",
                "gps_drift": "preserve",
                "random_point_loss": "preserve",
                "contiguous_outage": "preserve",
                "sampling_reduction": "preserve",
                "spatial_quantization": "small_change",
                "temporal_jitter": "preserve",
                "speed_distortion": "preserve",
                "truncation": "small_change",
                "reversal": "depends",
                "spatial_translation": "change",
                "free_space_detour": "change",
            },
        ),
        SimilarityNotion(
            "temporal_dynamics",
            "1.0",
            "Similarity of the movement's timing and speed dynamics.",
            properties={
                "spatial": False,
                "location": False,
                "route": False,
                "direction": True,
                "temporal": True,
                "sampling": False,
                "observation": True,
            },
            expected_outcomes={
                "gps_noise": "preserve",
                "gps_drift": "preserve",
                "random_point_loss": "preserve",
                "contiguous_outage": "preserve",
                "sampling_reduction": "preserve",
                "spatial_quantization": "preserve",
                "temporal_jitter": "change",
                "speed_distortion": "change",
                "truncation": "small_change",
                "reversal": "change",
                "spatial_translation": "preserve",
                "free_space_detour": "depends",
            },
        ),
        SimilarityNotion(
            "same_underlying_movement",
            "1.0",
            "Whether two observations can plausibly be views of one movement.",
            properties={
                "spatial": True,
                "location": True,
                "route": True,
                "direction": True,
                "temporal": True,
                "sampling": True,
                "observation": True,
            },
            expected_outcomes={
                "gps_noise": "preserve",
                "gps_drift": "preserve",
                "random_point_loss": "preserve",
                "contiguous_outage": "preserve",
                "sampling_reduction": "preserve",
                "spatial_quantization": "small_change",
                "temporal_jitter": "small_change",
                "speed_distortion": "change",
                "truncation": "change",
                "reversal": "change",
                "spatial_translation": "change",
                "free_space_detour": "change",
            },
        ),
        SimilarityNotion(
            "direction_aware_movement",
            "1.0",
            "Similarity that distinguishes the ordered direction of movement.",
            properties={
                "spatial": True,
                "location": True,
                "route": True,
                "direction": True,
                "temporal": True,
                "sampling": True,
                "observation": True,
            },
            expected_outcomes={
                "gps_noise": "preserve",
                "gps_drift": "preserve",
                "random_point_loss": "preserve",
                "contiguous_outage": "preserve",
                "sampling_reduction": "preserve",
                "spatial_quantization": "small_change",
                "temporal_jitter": "small_change",
                "speed_distortion": "change",
                "truncation": "small_change",
                "reversal": "change",
                "spatial_translation": "change",
                "free_space_detour": "change",
            },
        ),
        SimilarityNotion(
            "route_path_structure",
            "1.0",
            "Similarity of the ordered path geometry and detour structure.",
            properties={
                "spatial": True,
                "location": False,
                "route": True,
                "direction": True,
                "temporal": False,
                "sampling": True,
                "observation": False,
            },
            expected_outcomes={
                "gps_noise": "preserve",
                "gps_drift": "preserve",
                "random_point_loss": "preserve",
                "contiguous_outage": "preserve",
                "sampling_reduction": "preserve",
                "spatial_quantization": "small_change",
                "temporal_jitter": "preserve",
                "speed_distortion": "preserve",
                "truncation": "small_change",
                "reversal": "depends",
                "spatial_translation": "preserve",
                "free_space_detour": "change",
            },
        ),
    ]
    return NotionRegistry(notions)
