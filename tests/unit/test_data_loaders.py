import json
from pathlib import Path

import numpy as np
import pytest

from trajsimbench.data.loaders.geolife import GeoLifeLoader
from trajsimbench.data.loaders.germany import DatasetGateError, GermanyLoader
from trajsimbench.data.loaders.porto import PortoLoader
from trajsimbench.data.validation import validate_dataset

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_porto_loader_counts_malformed_rows_and_prepares(tmp_path: Path) -> None:
    loader = PortoLoader(
        {
            "min_points": 2,
            "timestamp_field": "TIMESTAMP",
            "timestamp_semantics": "start_time_plus_interval",
            "sampling_interval_s": 15,
            "projected_crs": "EPSG:32629",
        }
    )
    inspection = loader.inspect_raw(FIXTURES / "porto_sample.csv")
    assert inspection.total_records == 4
    assert inspection.accepted_records == 2
    assert inspection.rejected_by_reason["malformed_or_missing_polyline"] == 1
    result = loader.prepare(FIXTURES / "porto_sample.csv", tmp_path / "porto")
    assert result.output_path.exists()
    assert validate_dataset(result.output_path).ok


def test_porto_loader_writes_disjoint_standard_temporal_and_retrieval_splits(tmp_path: Path) -> None:
    raw = tmp_path / "porto.csv"
    rows = ["TRIP_ID,TIMESTAMP,POLYLINE"]
    for index in range(12):
        rows.append(
            f'p{index},{1700000000 + index * 60},"[[-8.61,41.15],[-8.60,41.16]]"'
        )
    raw.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loader = PortoLoader(
        {
            "timestamp_field": "TIMESTAMP",
            "timestamp_semantics": "start_time_plus_interval",
            "sampling_interval_s": 15,
            "projected_crs": "EPSG:32629",
            "retrieval_scales": [{"name": "smoke", "database_count": 2, "query_count": 1}],
        }
    )
    output = loader.prepare(raw, tmp_path / "prepared").output_path
    standard = {
        partition: set(np.load(output / "splits/standard" / f"{partition}.npy", allow_pickle=False))
        for partition in ("train", "val", "test")
    }
    assert len(standard["train"] | standard["val"] | standard["test"]) == 12
    assert not (standard["train"] & standard["val"] or standard["train"] & standard["test"])
    temporal_train = np.load(output / "splits/temporal/train.npy", allow_pickle=False).tolist()
    assert temporal_train == [f"porto:p{index}" for index in range(8)]
    database = set(np.load(output / "splits/retrieval_smoke/database.npy", allow_pickle=False))
    queries = set(np.load(output / "splits/retrieval_smoke/query.npy", allow_pickle=False))
    assert database.isdisjoint(queries)
    assert database | queries <= standard["test"]
    manifest = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert len(manifest["preprocessing_config_hash"]) == 64


def test_geolife_loader_deduplicates_and_creates_user_split(tmp_path: Path) -> None:
    loader = GeoLifeLoader({"min_points": 2, "projected_crs": "EPSG:32650", "seed": 0})
    inspection = loader.inspect_raw(FIXTURES / "geolife_sample")
    assert inspection.accepted_records == 1
    assert inspection.details["deduplicated_points"] == 1
    result = loader.prepare(FIXTURES / "geolife_sample", tmp_path / "geolife")
    assert "user_held_out" in {path.name for path in (result.output_path / "splits").iterdir()}
    assert validate_dataset(result.output_path).ok


def test_germany_is_an_explicit_gate() -> None:
    loader = GermanyLoader()
    assert loader.describe_license()["enabled"] is False
    with pytest.raises(DatasetGateError, match="requires_source_decision"):
        loader.prepare("unused", "unused")
