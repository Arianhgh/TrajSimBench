from pathlib import Path

import pytest

from scripts.download.acquire import acquisition_report


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("dataset", ["tdrive", "ais"])
def test_new_dataset_gates_do_not_claim_downloads(dataset: str) -> None:
    report = acquisition_report(dataset, REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml")

    assert report["downloaded"] is False
    assert report["status"] in {"requires_license_confirmation", "requires_source_decision"}


def test_shared_protocol_records_the_required_boundaries() -> None:
    protocol = (REPO_ROOT / "docs" / "benchmark-v1-protocol.md").read_text(encoding="utf-8")

    assert "70% train, 10% validation, and 20% test" in protocol
    assert "Create perturbations only after splitting" in protocol
    assert "The main benchmark remains free-space GPS" in protocol
    assert "paper-reproduced only after" in protocol
