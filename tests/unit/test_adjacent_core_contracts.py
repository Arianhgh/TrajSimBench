from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trajsimbench.measures.classical._common import (
    pair_points,
    point_distance_matrix,
    resample_by_arclength,
)
from trajsimbench.measures.classical._common import (
    path_length as common_path_length,
)
from trajsimbench.notions import (
    Expectation,
    NotionRegistry,
    NotionValidationError,
    SimilarityNotion,
    default_notion_registry,
    load_notion_file,
)
from trajsimbench.perturbations.result import (
    PerturbationProvenance,
    PerturbationResult,
    canonical_json,
    hash_array,
)
from trajsimbench.retrieval.ranking import (
    agreement_matrix,
    average_ranks,
    compare_rankings,
    kendall_tau_b,
    pairwise_ordering_agreement,
    rank_biased_overlap,
    spearman_rho,
    top_k_jaccard,
)


def test_classical_common_helpers_resample_real_paths() -> None:
    first = np.array([[0.0, 0.0], [3.0, 4.0]])
    second = np.array([[0.0, 0.0], [0.0, 4.0]])
    left, right = pair_points(first, second)
    np.testing.assert_array_equal(left, first)
    np.testing.assert_array_equal(right, second)
    np.testing.assert_allclose(point_distance_matrix(first, second), [[0, 4], [5, 3]])
    assert common_path_length(first) == 5.0
    assert common_path_length(first[:1]) == 0.0
    np.testing.assert_allclose(resample_by_arclength(first, 3), [[0, 0], [1.5, 2], [3, 4]])
    np.testing.assert_array_equal(resample_by_arclength(first[:1], 3), [[0, 0]] * 3)
    np.testing.assert_array_equal(resample_by_arclength(first, 1), [[0, 0]])
    repeated = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(resample_by_arclength(repeated, 3), [[0, 0], [1, 0], [2, 0]])
    np.testing.assert_array_equal(resample_by_arclength(np.zeros((3, 2)), 3), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="at least one"):
        resample_by_arclength(first, 0)


def test_notion_schema_expectations_labels_and_registry_file_loading(tmp_path: Path) -> None:
    notion = SimilarityNotion(
        "demo",
        "1.0",
        "A test notion",
        exclusions=["x"],
        properties={"spatial": True},
        expected_outcomes={"a": "preserve", "b": Expectation.CHANGE, "c": "depends"},
        minimum_margin=0.2,
        citations=["paper"],
        decision_notes=["note"],
        status="experimental",
    )
    assert notion.key == "demo@1.0"
    assert notion.expectation_for("missing") is Expectation.NOT_APPLICABLE
    assert notion.expectation_for("a") is Expectation.PRESERVE
    assert notion.triplet_label("a", "b") == "a_closer"
    assert notion.triplet_label("b", "a") == "b_closer"
    assert notion.triplet_label("a", "a") == "tie"
    assert notion.triplet_label("a", "b", margin=0.1) == "tie"
    assert notion.triplet_label("c", "a") == "unspecified"
    assert notion.content_hash == SimilarityNotion(**notion.to_dict()).content_hash
    with pytest.raises(NotionValidationError, match="required"):
        SimilarityNotion("", "1", "definition")
    with pytest.raises(NotionValidationError, match="definition"):
        SimilarityNotion("x", "1", " ")
    with pytest.raises(NotionValidationError, match="invalid status"):
        SimilarityNotion("x", "1", "definition", status="bad")
    with pytest.raises(NotionValidationError, match="non-negative"):
        SimilarityNotion("x", "1", "definition", tie_tolerance=-1)
    with pytest.raises(NotionValidationError, match="invalid expectation"):
        SimilarityNotion("x", "1", "definition", expected_outcomes={"a": "bad"})

    document = {
        "notions": [
            {
                "notion_id": "file",
                "version": "1",
                "definition": "d",
                "expected_outcomes": {"x": "preserve"},
            }
        ]
    }
    path = tmp_path / "notions.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = load_notion_file(path)
    assert loaded[0].key == "file@1"
    registry = NotionRegistry(loaded)
    assert registry.get("file").key == "file@1"
    assert registry.keys() == ("file@1",) and len(registry) == 1
    assert registry.values()[0].notion_id == "file"
    with pytest.raises(NotionValidationError, match="duplicate"):
        registry.register(loaded[0])
    with pytest.raises(KeyError, match="unknown notion"):
        registry.get("nope")
    with pytest.raises(KeyError, match="unknown notion"):
        registry.get("file", "2")
    assert len(default_notion_registry()) >= 6
    mapping_document = {"mapped": {"version": "1", "definition": "d", "expected_outcomes": {}}}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping_document), encoding="utf-8")
    assert load_notion_file(mapping_path)[0].key == "mapped@1"
    for bad_document in (
        {"notions": [{"notion_id": "x"}]},
        {
            "notions": [
                {
                    "notion_id": "x",
                    "version": "1",
                    "definition": "d",
                    "expected_outcomes": {},
                    "extra": 1,
                }
            ]
        },
        {"notions": [1]},
    ):
        bad = tmp_path / f"bad-{len(list(tmp_path.iterdir()))}.json"
        bad.write_text(json.dumps(bad_document), encoding="utf-8")
        with pytest.raises(NotionValidationError):
            load_notion_file(bad)


def test_perturbation_provenance_mapping_and_result_contract() -> None:
    assert canonical_json({"b": 1, "a": np.int64(2)}) == '{"a":2,"b":1}'
    assert hash_array(None) is None
    points = np.array([[1.0, 2.0]])
    provenance = PerturbationProvenance(
        "v",
        "source",
        "demo",
        {"x": [1]},
        "units",
        {"p": [2]},
        1,
        {"n": "preserve"},
        "1",
        "in",
        "out",
        ("q",),
    )
    restored = PerturbationProvenance.from_dict(provenance.to_dict())
    assert restored == provenance
    assert provenance["source_id"] == "source" and len(provenance) > 5
    from_json = PerturbationProvenance.from_dict(
        {
            "variant_id": "v",
            "transformation": "demo",
            "parameters_json": '{"p": 2}',
            "input_hash": "in",
            "output_hash": None,
        }
    )
    assert from_json.parameters["p"] == 2
    result = PerturbationResult("generated", "source", points, provenance)
    assert result.generated and result.variant_id == "v"
    result.validate(metadata={})
    assert result.to_dict(include_points=True)["points"] == [[1.0, 2.0]]
    rejected = PerturbationResult("not_generated", "source", None, provenance, "reason")
    assert not rejected.generated and rejected.to_dict()["reason"] == "reason"
    with pytest.raises(ValueError, match="status"):
        PerturbationResult("bad", "source", None, provenance)
    with pytest.raises(ValueError, match="requires points"):
        PerturbationResult("generated", "source", None, provenance)
    with pytest.raises(ValueError, match="requires a reason"):
        PerturbationResult("not_generated", "source", None, provenance)


def test_ranking_agreement_mapping_ties_universes_and_edges() -> None:
    assert average_ranks({"a": 2, "b": 1, "c": 1}) == {"b": 1.5, "c": 1.5, "a": 3.0}
    assert average_ranks({"a": 1, "b": 3}, descending=True) == {"b": 1.0, "a": 2.0}
    assert average_ranks(["a", "a", "b"]) == {"a": 1.0, "b": 3.0}
    assert kendall_tau_b(["a"], ["a"]) == 1.0
    with pytest.raises(ValueError, match="identical"):
        spearman_rho(["a"], ["b"])
    assert top_k_jaccard([], [], k=1) == 1.0
    assert rank_biased_overlap([], [], k=0) == 0.0
    assert (
        pairwise_ordering_agreement({"a": 1, "b": 1}, {"a": 1, "b": 2}, tie_policy="exclude") != 0
    )
    assert pairwise_ordering_agreement({"a": 1, "b": 1}, {"a": 1, "b": 2}, tie_policy="half") == 0.5
    assert (
        pairwise_ordering_agreement({"a": 1, "b": 1}, {"a": 1, "b": 2}, tie_policy="disagree")
        == 0.0
    )
    comparison = compare_rankings({"a": 1, "b": 2}, {"a": 1, "b": 2}, top_k=1)
    assert comparison["candidate_count"] == 2 and comparison["comparison_depth"] == 1
    assert (
        agreement_matrix({"left": {"q": ["a", "b"]}, "right": {"q": ["b", "a"]}})["left"]["right"]
        < 1
    )
    with pytest.raises(ValueError, match="identical"):
        kendall_tau_b(["a"], ["b"])
    with pytest.raises(ValueError, match="unique"):
        kendall_tau_b(["a", "b"], ["a", "b"], candidate_universe=["a", "a"])
    with pytest.raises(ValueError, match="duplicate"):
        spearman_rho(["a", "a", "b"], ["a", "b", "b"])
    with pytest.raises(ValueError, match="positive"):
        top_k_jaccard(["a"], ["a"], k=0)
    with pytest.raises(ValueError, match="persistence"):
        rank_biased_overlap(["a"], ["a"], persistence=1)
    with pytest.raises(ValueError, match="tie_policy"):
        pairwise_ordering_agreement(["a"], ["a"], tie_policy="bad")
    with pytest.raises(ValueError, match="positive"):
        compare_rankings(["a"], ["a"], top_k=0)
