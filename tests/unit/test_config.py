from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from trajsimbench.config import load_config, load_dataset_config
from trajsimbench.config.models import ExperimentConfig
from trajsimbench.config.validation import ConfigValidationError
from trajsimbench.orchestration.runner import resolve_config


def _minimal() -> dict:
    return {
        "schema_version": "1.0",
        "experiment_id": "unit_config",
        "dataset": {"name": "synthetic", "has_timestamps": True},
        "methods": [{"name": "euclidean", "config": {"coordinate_columns": ["x_m", "y_m"]}}],
        "tasks": [{"name": "oracle"}],
    }


def test_unknown_root_key_is_rejected() -> None:
    value = _minimal()
    value["misspelled_experiment_field"] = True
    with pytest.raises(
        (ConfigValidationError, ValidationError), match="misspelled_experiment_field"
    ):
        ExperimentConfig.model_validate(value)


def test_duplicate_severity_and_unknown_method_are_rejected() -> None:
    value = _minimal()
    value["perturbations"] = [{"name": "noise", "severities": [0.0, 0.0], "unit": "m"}]
    with pytest.raises(ValidationError, match="unique"):
        ExperimentConfig.model_validate(value)
    value = _minimal()
    value["methods"] = [{"name": "not_a_measure"}]
    with pytest.raises(ValidationError, match="unknown method"):
        ExperimentConfig.model_validate(value)


def test_fragment_resolution_and_hash_are_stable() -> None:
    first = load_config(Path("configs/ci/tiny_synthetic.yaml"))
    second = load_config(Path("configs/ci/tiny_synthetic.yaml"))
    assert first.dataset.name == "synthetic"
    assert first.config_hash == second.config_hash
    assert first.canonical == second.canonical
    assert first.methods[0].config


def test_schema_migration_is_rejected(tmp_path: Path) -> None:
    value = _minimal()
    value["schema_version"] = "2.0"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="unsupported schema_version"):
        load_config(path)


def test_standalone_dataset_fragment_is_strictly_loadable() -> None:
    dataset = load_dataset_config(Path("configs/datasets/germany.yaml"))
    assert dataset.enabled is False
    assert dataset.status == "requires_source_decision"


def test_compact_methods_and_metrics_are_normalized_by_model() -> None:
    value = _minimal()
    value["methods"] = ["euclidean", "dtw"]
    value["metrics"] = ["recall@1", "mrr"]
    resolved = ExperimentConfig.model_validate(value)
    assert [method.name for method in resolved.methods] == ["euclidean", "dtw"]
    assert resolved.metrics.enabled == ["recall@1", "mrr"]


def test_runner_preserves_authoritative_file_config_hash() -> None:
    path = Path("configs/ci/tiny_synthetic.yaml")
    assert resolve_config(path)["resolved_config_hash"] == load_config(path).config_hash
