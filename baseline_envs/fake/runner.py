"""Deterministic fake external adapter used only for protocol tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

PROTOCOL_VERSION = "1.0"


def run(request_path: Path, output_dir: Path) -> dict[str, object]:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter_ns()
    ids_path = Path(request["query_ids"])
    ids = np.load(ids_path, allow_pickle=False)
    embeddings = np.asarray(
        [[float(index), float(len(str(value)))] for index, value in enumerate(ids)],
        dtype=np.float32,
    )
    embedding_path = output_dir / "embeddings.npy"
    np.save(embedding_path, embeddings)
    metadata = {
        "method": "fake",
        "version": "1.0",
        "embedding_dim": embeddings.shape[1],
        "dtype": str(embeddings.dtype),
        "embedding_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
        "cpu_only": True,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "timings.json").write_text(
        json.dumps({"encode_ns": perf_counter_ns() - started}, indent=2), encoding="utf-8"
    )
    status = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "outputs": ["embeddings.npy", "metadata.json", "timings.json"],
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    run(Path(args.request), Path(args.output_dir))


if __name__ == "__main__":
    main()
