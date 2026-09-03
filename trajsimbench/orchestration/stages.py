"""Explicit stage statuses used by the runner and resume logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class StageName(StrEnum):
    VALIDATE = "validate"
    LOAD_DATA = "load_data"
    MATERIALIZE_TASKS = "materialize_tasks"
    FIT_METHODS = "fit_methods"
    BUILD_INDEX = "build_index"
    EVALUATE = "evaluate"
    METRICS = "metrics"
    COMMIT = "commit"
    ANALYZE = "analyze"
    FINALIZE = "finalize"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageRecord:
    stage: str
    status: str = StageStatus.PENDING
    input_fingerprint: str | None = None
    outputs: list[str] = field(default_factory=list)
    output_checksums: dict[str, str] = field(default_factory=dict)
    started_at: str | None = None
    ended_at: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None

    def begin(self, fingerprint: str | None = None) -> None:
        self.status = StageStatus.RUNNING
        self.input_fingerprint = fingerprint
        self.started_at = datetime.now(UTC).isoformat()

    def complete(self, outputs: list[str] | None = None) -> None:
        self.status = StageStatus.COMPLETE
        self.outputs = outputs or self.outputs
        self.ended_at = datetime.now(UTC).isoformat()

    def fail(self, exc: BaseException, traceback_text: str | None = None) -> None:
        self.status = StageStatus.FAILED
        self.ended_at = datetime.now(UTC).isoformat()
        self.error_type = type(exc).__name__
        self.error_message = str(exc)
        self.traceback = traceback_text

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
