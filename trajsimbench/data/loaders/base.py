"""Shared loader contracts and preparation reporting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LoaderInspection:
    raw_path: Path
    total_records: int = 0
    accepted_records: int = 0
    rejected_records: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected_records += 1
        self.rejected_by_reason[reason] = self.rejected_by_reason.get(reason, 0) + 1


@dataclass(frozen=True, slots=True)
class PreparationResult:
    output_path: Path
    inspection: LoaderInspection


class BaseLoader(ABC):
    name: str

    @abstractmethod
    def inspect_raw(self, raw_path: str | Path, **kwargs: Any) -> LoaderInspection:
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self, raw_path: str | Path, output_path: str | Path, **kwargs: Any
    ) -> PreparationResult:
        raise NotImplementedError

    @abstractmethod
    def describe_license(self) -> Mapping[str, Any]:
        raise NotImplementedError
