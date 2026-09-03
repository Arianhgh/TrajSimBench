from pathlib import Path

from trajsimbench.data.dataset import TrajectoryDataset
from trajsimbench.data.loaders.synthetic import prepare_synthetic


def test_synthetic_preparation_is_end_to_end_and_idempotent(tmp_path: Path) -> None:
    output = prepare_synthetic(tmp_path / "synthetic")
    first_checksums = (output / "checksums.sha256").read_text(encoding="utf-8")
    second = prepare_synthetic(tmp_path / "synthetic")
    assert output == second
    assert first_checksums == (second / "checksums.sha256").read_text(encoding="utf-8")
    dataset = TrajectoryDataset.open(output)
    assert len(dataset) >= 10
    assert {"standard", "user_held_out"}.issubset(set(dataset.split_names()))
    assert dataset.by_id("synthetic:route_a").points.shape[1] == 5
