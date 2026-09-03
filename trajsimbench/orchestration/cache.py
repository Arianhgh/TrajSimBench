"""Content fingerprints and cache validity checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_valid(stage_record: dict[str, Any], *, input_fingerprint: str, root: Path) -> bool:
    if (
        stage_record.get("status") != "complete"
        or stage_record.get("input_fingerprint") != input_fingerprint
    ):
        return False
    return all((Path(root) / output).exists() for output in stage_record.get("outputs", []))


def load_stage_cache(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_stage_cache(path: Path, records: dict[str, Any]) -> None:
    target = Path(path)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(records, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(target)
