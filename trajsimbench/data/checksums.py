"""SHA-256 manifests for canonical datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_dataset_files(dataset_path: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in dataset_path.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        ),
        key=lambda path: path.relative_to(dataset_path).as_posix(),
    )


def write_checksums(dataset_path: Path) -> dict[str, str]:
    checksums = {
        path.relative_to(dataset_path).as_posix(): sha256_file(path)
        for path in iter_dataset_files(dataset_path)
    }
    lines = [f"{digest}  {name}" for name, digest in sorted(checksums.items())]
    (dataset_path / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksums


def read_checksums(dataset_path: Path) -> dict[str, str]:
    path = dataset_path / "checksums.sha256"
    if not path.exists():
        raise FileNotFoundError(f"missing checksum manifest: {path}")
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, name = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line {line_number} in {path}") from exc
        result[name] = digest
    return result


def verify_checksums(dataset_path: Path) -> list[str]:
    """Return actionable checksum errors; an empty list means success."""

    try:
        expected = read_checksums(dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        return [str(exc)]
    errors: list[str] = []
    actual_files = {
        path.relative_to(dataset_path).as_posix() for path in iter_dataset_files(dataset_path)
    }
    for name, expected_digest in expected.items():
        path = dataset_path / name
        if not path.exists():
            errors.append(f"checksum manifest references missing file: {name}")
            continue
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            errors.append(
                f"checksum mismatch for {name}: expected {expected_digest}, got {actual_digest}"
            )
    for name in sorted(actual_files - set(expected)):
        errors.append(f"file is absent from checksum manifest: {name}")
    return errors
