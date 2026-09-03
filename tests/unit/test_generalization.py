import pytest

from trajsimbench.tasks.generalization import (
    build_generalization_task,
    validate_generalization_partitions,
)


def test_generalization_partitions_are_disjoint_and_versioned() -> None:
    report = validate_generalization_partitions(["a"], ["b"], ["c"], mode="user_held_out")
    assert report["id_overlap"] == {}
    task = build_generalization_task(
        train_ids=["a"], val_ids=["b"], test_ids=["c"], mode="user_held_out"
    )
    assert task.records[0]["query_id"] == "c"
    assert task.config["mode"] == "user_held_out"


def test_generalization_rejects_leakage() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_generalization_partitions(["a"], ["a"], [], mode="in_domain")
