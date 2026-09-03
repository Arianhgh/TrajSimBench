from dataclasses import dataclass

import pytest

from trajsimbench.data.splitting import (
    SCALE_LIMITS,
    select_scale,
    standard_split,
    user_held_out_split,
)


@dataclass
class Record:
    trajectory_id: str
    user_id: str


def test_standard_split_is_deterministic_and_disjoint() -> None:
    ids = [f"id:{index}" for index in range(20)]
    first = standard_split(ids, seed=12)
    second = standard_split(list(reversed(ids)), seed=12)
    assert first == second
    assert len(set(first["train"]) | set(first["val"]) | set(first["test"])) == len(ids)


def test_user_held_out_split_has_no_user_overlap() -> None:
    records = [Record(f"id:{index}", f"u{index // 2}") for index in range(12)]
    split = user_held_out_split(records, seed=0)
    owners = {
        partition: {record.user_id for record in records if record.trajectory_id in ids}
        for partition, ids in split.items()
    }
    assert not (owners["train"] & owners["val"])
    assert not (owners["train"] & owners["test"])
    assert not (owners["val"] & owners["test"])


def test_scale_never_samples_with_replacement() -> None:
    with pytest.raises(ValueError, match="without"):
        select_scale(["a", "b"], "tiny")
    selected = select_scale(["a", "b"], "tiny", allow_reduced=True)
    assert len(selected.database_ids) + len(selected.query_ids) <= 2
    assert SCALE_LIMITS["tiny"] == (1000, 100)
