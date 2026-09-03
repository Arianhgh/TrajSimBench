"""Explicitly disabled Germany dataset gate."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trajsimbench.data.loaders.base import BaseLoader, LoaderInspection, PreparationResult


class DatasetGateError(RuntimeError):
    """Raised when a dataset lacks an approved source/license decision."""


class GermanyLoader(BaseLoader):
    name = "germany"

    def inspect_raw(self, raw_path: str | Path, **kwargs: Any) -> LoaderInspection:
        raise DatasetGateError(
            "Germany is disabled: approve the exact dataset title, official source URL, "
            "coordinate schema, license, and comparison protocol before enabling it."
        )

    def prepare(
        self, raw_path: str | Path, output_path: str | Path, **kwargs: Any
    ) -> PreparationResult:
        raise DatasetGateError(
            "Germany is disabled (status: requires_source_decision); update the dataset config "
            "only after the supervisor approves a source and license."
        )

    def describe_license(self) -> Mapping[str, Any]:
        return {
            "dataset": "germany",
            "enabled": False,
            "status": "requires_source_decision",
            "reason": "Germany is not a uniquely identifiable dataset.",
            "required_fields": [
                "exact title",
                "official source URL",
                "schema",
                "license",
                "protocol",
            ],
        }
