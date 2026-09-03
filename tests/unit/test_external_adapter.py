from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from trajsimbench.measures.learned.external import (
    AdapterRequest,
    run_external_adapter,
    validate_adapter_output,
)


def test_fake_adapter_round_trip_and_output_validation(tmp_path: Path) -> None:
    query_ids = tmp_path / "query_ids.npy"
    np.save(query_ids, np.asarray(["q0", "q1"], dtype=str), allow_pickle=False)
    output_dir = tmp_path / "adapter-output"
    request = AdapterRequest(
        operation="encode",
        output_dir=output_dir,
        query_ids=query_ids,
        expected_outputs=("embeddings.npy", "metadata.json", "timings.json"),
    )
    result = run_external_adapter(
        [sys.executable, "baseline_envs/fake/runner.py", str(request.path), str(output_dir)],
        request,
        timeout_seconds=30,
    )
    assert result.valid, result.validation
    assert validate_adapter_output(output_dir, request)["valid"]
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["embedding_dim"] == 2


def test_adapter_validation_rejects_nan_embeddings(tmp_path: Path) -> None:
    output_dir = tmp_path / "bad"
    output_dir.mkdir()
    (output_dir / "status.json").write_text(
        json.dumps({"protocol_version": "1.0", "status": "complete"}), encoding="utf-8"
    )
    np.save(output_dir / "embeddings.npy", np.asarray([[np.nan]], dtype=np.float32))
    report = validate_adapter_output(output_dir, expected_query_ids=["q0"])
    assert not report["valid"]
    assert any("non-finite" in error for error in report["errors"])
