"""Run-local paths and immutable experiment identity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunContext:
    experiment_id: str
    run_id: str
    run_dir: Path

    @classmethod
    def create(cls, experiment_id: str, run_id: str, output_root: Path) -> RunContext:
        root = Path(output_root)
        directory = root / str(experiment_id) / str(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tasks").mkdir(exist_ok=True)
        (directory / "analysis" / "figures").mkdir(parents=True, exist_ok=True)
        (directory / "analysis" / "tables").mkdir(parents=True, exist_ok=True)
        return cls(str(experiment_id), str(run_id), directory)

    def path(self, *parts: str) -> Path:
        path = self.run_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
