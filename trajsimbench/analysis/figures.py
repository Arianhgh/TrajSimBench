"""Portable JSON figure specifications derived from benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tables import TABLE_NAMES, generate_tables

FIGURE_NAMES = (
    "benchmark_architecture",
    "dataset_composition",
    "retrieval_quality",
    "agreement",
    "robustness",
    "latency_quality_pareto",
    "runtime_breakdown",
    "memory_usage",
    "failure_rates",
    "ranking_examples",
    "reproducibility_overview",
)


def generate_figures(results_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Write stable JSON specifications for the standard report figures.

    The specifications deliberately point at generated CSV tables rather than
    baking chart pixels into a result run. A dashboard or paper script can use
    the same inputs to render the final visual style.
    """
    root = Path(results_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = generate_tables(root, output.parent / "tables")
    outputs: dict[str, Path] = {}
    for name in FIGURE_NAMES:
        payload: dict[str, Any] = {
            "figure_schema_version": "1.0",
            "figure": name,
            "results_root": str(root),
            "data_tables": {key: str(path) for key, path in tables.items() if key in TABLE_NAMES},
        }
        path = output / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs[name] = path
    return outputs
