"""Print approved acquisition instructions without guessing a raw-data source.

Restricted mobility datasets are intentionally not downloaded by default.  A
user supplies the official file and license acceptance, then places it below
``data/raw`` before preparation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def acquisition_report(dataset: str, config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else Path("configs/datasets") / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"dataset config does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"dataset config must be a mapping: {path}")
    if value.get("enabled") is False:
        return {
            "dataset": dataset,
            "status": value.get("status", "disabled"),
            "downloaded": False,
            "message": "This dataset is gated; obtain an explicit source/license decision first.",
        }
    return {
        "dataset": dataset,
        "status": "instructions_only",
        "downloaded": False,
        "raw_path": value.get("raw_path"),
        "source_url": value.get("source_url"),
        "source_license": value.get("source_license"),
        "message": (
            "Download from the official source after accepting its terms; raw data is not "
            "redistributed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", choices=("synthetic", "porto", "geolife", "tdrive", "ais", "germany")
    )
    parser.add_argument("--config")
    args = parser.parse_args()
    import json

    print(json.dumps(acquisition_report(args.dataset, args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
