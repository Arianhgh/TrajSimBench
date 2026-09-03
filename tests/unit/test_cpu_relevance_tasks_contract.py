from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trajsimbench.retrieval.relevance import (
    ExternalLabelRelevance,
    GradedOracleRelevance,
    OracleTopKRelevance,
    SameSourceRelevance,
    StaticRelevance,
    TripletRelevance,
    apply_empty_relevance_policy,
    provider_from_config,
)
from trajsimbench.tasks import (
    TaskArtifact,
    TaskConstructionError,
    TaskQualityReport,
    build_systems_task,
    generate_diagnostic_triplets,
    generate_retrieval_task,
)
from trajsimbench.tasks.base import (
    dataset_ids,
    freeze_value,
    get_trajectory,
    make_quality,
    thaw_value,
)
from trajsimbench.tasks.diagnostics import DiagnosticTaskGenerator
from trajsimbench.tasks.oracle import stable_rank
from trajsimbench.tasks.systems import SystemsWorkload


def small_dataset() -> dict[str, SimpleNamespace]:
    result: dict[str, SimpleNamespace] = {}
    for index, offset in enumerate((0.0, 200.0)):
        values = np.arange(8, dtype=float)
        points = np.zeros((8, 5), dtype=float)
        points[:, 0] = -73.0 + (offset + values * 20.0) / 111_000
        points[:, 1] = 45.0 + np.sin(values) * 20.0 / 111_000
        points[:, 2] = offset + values * 20.0
        points[:, 3] = np.sin(values) * 20.0
        points[:, 4] = values * 10.0
        result[str(index)] = SimpleNamespace(
            trajectory_id=str(index), points=points, metadata={"user_id": f"u{index}"}
        )
    return result


def test_relevance_providers_cover_builtins_and_empty_policies() -> None:
    static = StaticRelevance(values={"q": {"a": 2.0}})
    assert static.for_query("q", ["a", "b"]).tolist() == [2.0, 0.0]
    assert static.metadata()["name"] == "static"
    same = SameSourceRelevance(source_by_id={"q": "s", "a": "s", "b": "t"})
    assert same.for_query("q", ["q", "a", "b"]).tolist() == [0.0, 1.0, 0.0]
    explicit_query_source = SameSourceRelevance(
        source_by_id={"a": "s", "b": "t"}, query_source_by_id={"q": "s"}
    )
    assert explicit_query_source.relevance("q", "a") == 1.0
    oracle = OracleTopKRelevance(oracle_rankings={"q": ["a", "b"]})
    graded = GradedOracleRelevance(oracle_rankings={"q": ["a", "b"]})
    assert oracle.relevance("q", "a") == 1.0 and graded.relevance("q", "b") == 0.5
    assert TripletRelevance(preferred_by_query={"q": ["a"]}, grade=2).relevance("q", "a") == 2.0
    assert ExternalLabelRelevance(labels={("q", "a"): 3}).relevance("q", "a") == 3.0
    assert apply_empty_relevance_policy([0, 1], "skip") == (True, None)
    assert apply_empty_relevance_policy([0], "skip") == (False, "no_relevant_candidates")
    assert apply_empty_relevance_policy([0], "zero") == (True, "no_relevant_candidates")
    with pytest.raises(ValueError, match="no relevant"):
        apply_empty_relevance_policy([0], "raise")
    with pytest.raises(ValueError, match="empty relevance"):
        apply_empty_relevance_policy([0], "bad")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("same_source", SameSourceRelevance),
        ("oracle", OracleTopKRelevance),
        ("graded_oracle", GradedOracleRelevance),
        ("triplet", TripletRelevance),
        ("external", ExternalLabelRelevance),
    ],
)
def test_provider_factory_resolves_configured_provider(name, expected) -> None:
    context = {
        "source_by_id": {"q": "s"},
        "oracle_rankings": {"q": ["a"]},
        "preferred_by_query": {"q": ["a"]},
        "labels": {("q", "a"): 1.0},
    }
    assert isinstance(provider_from_config({"name": name}, context=context), expected)
    with pytest.raises(ValueError, match="unknown relevance"):
        provider_from_config({"name": "unknown"})
    with pytest.raises(ValueError, match="finite"):
        StaticRelevance(values={"q": {"a": np.inf}}).for_query("q", ["a"])


def test_task_base_freezes_values_quality_and_serialization() -> None:
    value = {"items": [np.int64(2), {"x": True}]}
    frozen = freeze_value(value)
    assert thaw_value(frozen) == {"items": [2, {"x": True}]}
    assert dataset_ids(small_dataset()) == ("0", "1")
    assert get_trajectory(small_dataset(), "1").trajectory_id == "1"
    quality = make_quality(4, 3, ["bad", "bad"], required_count=3, minimum_yield=0.5)
    assert quality.yield_rate == 0.75 and quality.rejection_rate == 0.25
    assert quality.to_dict()["rejection_reasons"] == {"bad": 2}
    artifact = TaskArtifact(
        task_type="retrieval",
        schema_version="1.0",
        records=({"task_id": "t", "query_id": "q", "candidate_ids": ("a",)},),
        generator="test",
        generator_version="1",
        seed=1,
    )
    assert len(artifact) == 1 and list(artifact)[0]["task_id"] == "t"
    assert artifact.to_dict()["content_hash"] == artifact.content_hash
    assert '"task_type":"retrieval"' in artifact.to_json()
    with pytest.raises(TaskConstructionError, match="duplicate candidate"):
        TaskArtifact(
            "retrieval",
            "1.0",
            ({"task_id": "t", "query_id": "q", "candidate_ids": ["a", "a"]},),
            "g",
            "1",
            0,
        )
    with pytest.raises(TaskConstructionError, match="invalid diagnostic"):
        TaskArtifact("diagnostic", "1.0", ({"task_id": "t", "expected_order": "bad"},), "g", "1", 0)
    with pytest.raises(TaskConstructionError, match="quality report"):
        TaskArtifact(
            "retrieval",
            "1.0",
            (),
            "g",
            "1",
            0,
            quality=TaskQualityReport(0, 0, 0, required_count=1, quality_gate_passed=True),
        )


def test_retrieval_and_systems_tasks_resolve_ids_and_exclusions() -> None:
    artifact = generate_retrieval_task(
        small_dataset(),
        query_ids=["1", "0"],
        database_ids=["0", "1"],
        relevant_ids={"0": ["0", "1", "missing"]},
    )
    assert artifact.records[0]["candidate_ids"] == ("1",)
    assert artifact.records[0]["relevant_ids"] == ("1",)
    with pytest.raises(TaskConstructionError, match="self match"):
        generate_retrieval_task(
            small_dataset(),
            query_ids=["0"],
            database_ids=["0"],
            exclude_self=False,
            relevant_ids={"0": ["0"]},
        )
    workload = SystemsWorkload(("a",), ("q",), warmup_count=0, repetitions=1, worker_count=1)
    assert workload.to_dict()["database_size"] == 1
    systems = build_systems_task(["a", "b"], ["q"], config={"warmup": 2, "repetitions": 2})
    assert systems.config["warmup_count"] == 2 and systems.records[0]["database_size"] == 2
    with pytest.raises(ValueError, match="positive"):
        SystemsWorkload(("a",), ("q",), repetitions=0)
    with pytest.raises(ValueError, match="non-negative"):
        SystemsWorkload(("a",), ("q",), warmup_count=-1)


@pytest.mark.parametrize(
    ("family", "notion"),
    [
        ("downsampled_vs_distinct", "geometric_shape"),
        ("noise_scale", "geometric_shape"),
        ("detour_scale", "geometric_shape"),
        ("translated_vs_nearby", "absolute_geographic_route"),
        ("original_vs_reversed", "geometric_shape"),
        ("time_warp", "temporal_dynamics"),
        ("same_od_route", "absolute_geographic_route"),
        ("partial_overlap", "geometric_shape"),
    ],
)
def test_diagnostic_families_generate_versioned_triplets(family: str, notion: str) -> None:
    artifact = generate_diagnostic_triplets(
        small_dataset(), family=family, notion=notion, count=1, seed=5
    )
    assert len(artifact.records) == 1
    assert artifact.records[0]["notion_id"] == notion
    assert artifact.records[0]["expected_order"] in {"a_closer", "b_closer", "tie", "unspecified"}
    assert artifact.quality.quality_gate_passed


def test_diagnostic_aliases_and_invalid_inputs_are_typed() -> None:
    generator = DiagnosticTaskGenerator()
    notion = generator._resolve_notion("geometric_shape@1.0")
    assert notion.notion_id == "geometric_shape"
    assert generator._expected_order("translated_vs_nearby", notion, "a", "b") == "a_closer"
    assert generator._expected_order("original_vs_reversed", notion, "a", "b") == "tie"
    assert stable_rank({"b": 1.0, "a": 1.0, "c": 2.0}, tie_tolerance=0.0) == (
        ("a", "b", "c"),
        (("a", "b"), ("c",)),
    )
    with pytest.raises(TaskConstructionError, match="positive"):
        generate_diagnostic_triplets(
            small_dataset(), family="noise_scale", notion="geometric_shape", count=0
        )
    invalid = generate_diagnostic_triplets(small_dataset(), family="bad", notion="geometric_shape")
    assert len(invalid) == 0
    assert invalid.quality.rejection_reasons["unknown diagnostic family 'bad'"] == 1
