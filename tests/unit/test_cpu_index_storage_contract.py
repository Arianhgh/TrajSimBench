from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from trajsimbench.indexing import FaissFlatIndex, NumpyFlatIndex
from trajsimbench.indexing.base import (
    IndexMetadata,
    array_hash,
    make_metadata,
    read_metadata,
    write_metadata,
)
from trajsimbench.storage.artifacts import (
    artifact_fingerprint,
    validate_run_directory,
    write_artifacts,
)
from trajsimbench.storage.duckdb import connect_result_root, create_duckdb_views
from trajsimbench.storage.manifest import (
    ManifestBuilder,
    build_manifest,
    load_manifest,
    sha256_file,
)
from trajsimbench.storage.parquet import read_parquet, write_parquet, write_partitioned
from trajsimbench.storage.schemas import normalize_rows, validate_rows


def test_numpy_index_build_search_save_and_load(tmp_path: Path) -> None:
    ids = ["b", "a", "c"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    index = NumpyFlatIndex(metric="l2").build(ids, embeddings)
    found, distances = index.search(np.array([1.0, 0.0], dtype=np.float32), 5)
    assert found.tolist() == [["b", "c", "a"]]
    np.testing.assert_allclose(distances[0], [0.0, 1.0, 2.0])
    assert index.metadata is not None
    assert index.metadata.dimension == 2
    path = index.save(tmp_path / "index.npz")
    loaded = NumpyFlatIndex.load(path)
    assert loaded.metadata == index.metadata
    np.testing.assert_array_equal(loaded.search(embeddings[:1], 1)[0], [["b"]])
    with pytest.raises(RuntimeError, match="not been built"):
        NumpyFlatIndex().search(np.ones((1, 2)), 1)


def test_index_metrics_normalization_and_input_contracts(tmp_path: Path) -> None:
    cosine = NumpyFlatIndex(metric="cosine").build([1, 2], [[3.0, 0.0], [0.0, 4.0]])
    found, distances = cosine.search([[1.0, 0.0]], 1)
    assert found.tolist() == [[1]]
    np.testing.assert_allclose(distances, [[0.0]])
    with pytest.raises(ValueError, match="metric"):
        NumpyFlatIndex(metric="dot")
    with pytest.raises(ValueError, match="one row"):
        NumpyFlatIndex().build(["a"], np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        NumpyFlatIndex().build(["a"], [[np.nan, 1.0]])
    with pytest.raises(ValueError, match="zero-norm"):
        NumpyFlatIndex(metric="cosine").build(["a"], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="positive"):
        cosine.search([[1.0, 0.0]], 0)
    with pytest.raises(ValueError, match="dimension"):
        cosine.search([[1.0, 0.0, 0.0]], 1)
    with pytest.raises(ValueError, match="finite"):
        cosine.search([[np.inf, 0.0]], 1)
    saved = cosine.save(tmp_path / "cosine.npz")
    assert saved.with_suffix(".npz.json").exists()


def test_faiss_flat_has_explicit_cpu_fallback_and_metadata(tmp_path: Path) -> None:
    index = FaissFlatIndex(allow_fallback=True).build(["x"], np.array([[1.0, 2.0]]))
    assert index.metadata is not None
    assert index.metadata.index_type in {"FaissFlatL2", "NumpyFlatFallback"}
    if index.metadata.index_type == "NumpyFlatFallback":
        with pytest.raises(ImportError, match="FAISS CPU"):
            FaissFlatIndex(allow_fallback=False)


def test_index_metadata_helpers_round_trip(tmp_path: Path) -> None:
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    metadata = make_metadata("test", "l2", ["id"], values, config={"x": 1}, start_ns=4)
    assert array_hash(values) == array_hash(values.copy())
    path = tmp_path / "metadata.json"
    write_metadata(path, metadata)
    assert read_metadata(path) == metadata
    assert IndexMetadata(**metadata.to_dict()).to_dict() == metadata.to_dict()


def test_parquet_round_trip_partitions_dataframe_and_json_types(tmp_path: Path) -> None:
    rows = [{"query_id": "q", "candidate_id": "c", "rank": 1, "distance": 0.5, "raw_score": 0.5}]
    path = write_parquet(
        tmp_path / "rankings.parquet",
        rows,
        table="rankings",
        common={"run_id": "run"},
    )
    loaded = read_parquet(path)
    assert loaded[0]["schema_version"] == "1.0"
    assert loaded[0]["run_id"] == "run"
    assert read_parquet(path, as_dataframe=True).shape[0] == 1
    partitions = write_partitioned(
        tmp_path / "parts",
        [
            {"stage": "search", "value": np.int64(1), "when": date(2024, 1, 1)},
            {"stage": "build", "value": np.int64(2), "when": date(2024, 1, 2)},
        ],
        table="systems",
        partition_by="stage",
    )
    assert [path.parent.name for path in partitions] == ["stage=build", "stage=search"]
    assert write_partitioned(tmp_path / "single", rows, table="rankings", partition_by=())[
        0
    ].exists()


def test_schema_normalization_validation_and_fallback_reader(tmp_path: Path) -> None:
    normalized = normalize_rows([{"query_id": "q"}], table="queries", common={"dataset": "d"})
    assert normalized[0]["schema_version"] == "1.0"
    assert normalized[0]["dataset"] == "d"
    assert validate_rows([], table="queries")["valid"]
    assert not validate_rows(
        [{"schema_version": "1.0", "query_id": "q", "rank": 0}], table="queries"
    )["valid"]
    with pytest.raises(ValueError, match="unknown result table"):
        normalize_rows([], table="nope")
    fallback = tmp_path / "legacy.parquet"
    fallback.write_text('{"value": 3}\n', encoding="utf-8")
    assert read_parquet(fallback) == [{"value": 3}]
    with pytest.raises(ValueError, match="neither readable"):
        (tmp_path / "bad.parquet").write_text("not json\n", encoding="utf-8")
        read_parquet(tmp_path / "bad.parquet")


def test_duckdb_query_layer_and_portable_table_reader(tmp_path: Path) -> None:
    write_parquet(
        tmp_path / "rankings.parquet",
        [{"query_id": "q", "candidate_id": "c", "rank": 1, "distance": 0.1, "raw_score": 0.1}],
        table="rankings",
    )
    query = connect_result_root(tmp_path)
    assert query.table("rankings")[0]["candidate_id"] == "c"
    assert query.backend == "duckdb"
    assert query.query("SELECT candidate_id FROM rankings WHERE rank = ?", [1]) == [
        {"candidate_id": "c"}
    ]
    assert create_duckdb_views(tmp_path).table("rankings")
    with pytest.raises(RuntimeError, match="DuckDB"):
        query.backend = "python"
        query.query("SELECT 1")


def test_manifest_and_run_artifact_validation(tmp_path: Path) -> None:
    ranking = write_parquet(
        tmp_path / "rankings.parquet",
        [{"query_id": "q", "candidate_id": "c", "rank": 1, "distance": 0.1, "raw_score": 0.1}],
        table="rankings",
    )
    (tmp_path / "resolved_config.yaml").write_text("schema_version: '1.0'\n", encoding="utf-8")
    entries = [{"path": "rankings.parquet", "sha256": sha256_file(ranking)}]
    artifacts = write_artifacts(tmp_path, entries)
    assert json.loads(artifacts.read_text(encoding="utf-8"))["artifacts"] == entries
    builder = ManifestBuilder("run", "experiment", tmp_path, status="complete")
    manifest = builder.build(resolved_config_hash="hash", root=tmp_path)
    assert manifest["artifacts"][0]["row_count"] == 1
    manifest_path = builder.write(root=tmp_path)
    assert load_manifest(tmp_path)["run_id"] == "run"
    assert manifest_path.exists()
    assert build_manifest("run2", "exp", tmp_path)["status"] == "complete"
    result = validate_run_directory(tmp_path, require_all=False)
    assert result["valid"]
    assert validate_run_directory(tmp_path / "missing")["status"] == "missing"
    assert artifact_fingerprint([ranking]) == artifact_fingerprint([ranking])
