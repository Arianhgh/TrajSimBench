"""Small dependency-light schema for scientific similarity notions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from trajsimbench.perturbations.result import hash_payload


class NotionValidationError(ValueError):
    """Malformed or scientifically incomplete notion specification."""


class Expectation(StrEnum):
    PRESERVE = "preserve"
    SMALL_CHANGE = "small_change"
    CHANGE = "change"
    MAJOR_CHANGE = "major_change"
    DEPENDS = "depends"
    NOT_APPLICABLE = "not_applicable"


VALID_STATUSES = {"active", "experimental", "disabled"}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class SimilarityNotion:
    notion_id: str
    version: str
    definition: str
    exclusions: tuple[str, ...] = ()
    properties: Mapping[str, Any] = field(default_factory=dict)
    expected_outcomes: Mapping[str, Expectation | str] = field(default_factory=dict)
    tie_tolerance: float = 0.0
    minimum_margin: float = 0.0
    citations: tuple[str, ...] = ()
    decision_notes: tuple[str, ...] = ()
    status: str = "active"

    def __post_init__(self) -> None:
        notion_id = str(self.notion_id).strip()
        version = str(self.version).strip()
        if not notion_id or not version:
            raise NotionValidationError("notion_id and version are required")
        if not str(self.definition).strip():
            raise NotionValidationError(f"notion {notion_id!r} requires a definition")
        if self.status not in VALID_STATUSES:
            raise NotionValidationError(f"invalid status {self.status!r}")
        if self.tie_tolerance < 0 or self.minimum_margin < 0:
            raise NotionValidationError("tie_tolerance and minimum_margin must be non-negative")
        outcomes: dict[str, Expectation] = {}
        for transformation, expectation in self.expected_outcomes.items():
            try:
                outcomes[str(transformation)] = (
                    expectation
                    if isinstance(expectation, Expectation)
                    else Expectation(str(expectation))
                )
            except ValueError as exc:
                raise NotionValidationError(
                    f"invalid expectation {expectation!r} for transformation {transformation!r}"
                ) from exc
        object.__setattr__(self, "notion_id", notion_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "exclusions", tuple(map(str, self.exclusions)))
        object.__setattr__(self, "properties", _freeze(self.properties))
        object.__setattr__(self, "expected_outcomes", MappingProxyType(outcomes))
        object.__setattr__(self, "citations", tuple(map(str, self.citations)))
        object.__setattr__(self, "decision_notes", tuple(map(str, self.decision_notes)))

    @property
    def key(self) -> str:
        return f"{self.notion_id}@{self.version}"

    def expectation_for(self, transformation: str) -> Expectation:
        value = self.expected_outcomes.get(str(transformation))
        if value is None:
            return Expectation.NOT_APPLICABLE
        return value if isinstance(value, Expectation) else Expectation(str(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "notion_id": self.notion_id,
            "version": self.version,
            "definition": self.definition,
            "exclusions": list(self.exclusions),
            "properties": dict(self.properties),
            "expected_outcomes": {
                key: (
                    value.value if isinstance(value, Expectation) else Expectation(str(value)).value
                )
                for key, value in self.expected_outcomes.items()
            },
            "tie_tolerance": self.tie_tolerance,
            "minimum_margin": self.minimum_margin,
            "citations": list(self.citations),
            "decision_notes": list(self.decision_notes),
            "status": self.status,
        }

    @property
    def content_hash(self) -> str:
        return hash_payload(self.to_dict())

    def triplet_label(self, first: str, second: str, *, margin: float | None = None) -> str:
        """Translate an expectation comparison into an explicit triplet label.

        ``first`` and ``second`` are candidate transformation names.  A
        notion with no ordering information yields ``unspecified`` rather than
        manufacturing a binary label.
        """

        a = self.expectation_for(first)
        b = self.expectation_for(second)
        if a in {Expectation.NOT_APPLICABLE, Expectation.DEPENDS} or b in {
            Expectation.NOT_APPLICABLE,
            Expectation.DEPENDS,
        }:
            return "unspecified"
        rank = {
            Expectation.PRESERVE: 0,
            Expectation.SMALL_CHANGE: 1,
            Expectation.CHANGE: 2,
            Expectation.MAJOR_CHANGE: 3,
        }
        difference = rank[a] - rank[b]
        if margin is not None and abs(float(margin)) < self.minimum_margin:
            return "tie"
        if difference < 0:
            return "a_closer"
        if difference > 0:
            return "b_closer"
        return "tie"
