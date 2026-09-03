from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trajsimbench.analysis.break_even import break_even_curve, break_even_quantity
from trajsimbench.analysis.figures import generate_figures
from trajsimbench.analysis.pareto import pareto_frontier, pareto_summary
from trajsimbench.analysis.tables import generate_tables, load_result_rows
from trajsimbench.evaluation.agreement import (
    agreement_distance,
    build_agreement_matrix,
    evaluate_agreement,
)
from trajsimbench.evaluation.diagnostics import (
    build_similarity_fingerprint,
    evaluate_triplets,
    triplet_accuracy,
)
from trajsimbench.evaluation.fingerprints import FINGERPRINT_DIMENSIONS, fingerprint_rows
from trajsimbench.evaluation.robustness import (
    hard_negative_gap,
    monotonicity_violation_rate,
    robustness_auc,
    robustness_curve,
)
from trajsimbench.evaluation.statistics import (
    bootstrap_ci,
    holm_correction,
    paired_permutation_test,
    summarize_samples,
)
from trajsimbench.evaluation.systems import summarize_timings, timed_stage


def test_analysis_cost_and_pareto_helpers_cover_edge_policies() -> None:
    result = break_even_quantity(10, 3, 1)
    assert result["break_even_queries"] == 5.0 and result["finite"]
    no_crossing = break_even_quantity(10, 1, 3)
    assert no_crossing["break_even_queries"] is None
    assert len(break_even_curve(10, 3, 1, [0, 5])) == 2
    with pytest.raises(ValueError, match="non-negative"):
        break_even_quantity(-1, 1, 0)
    rows = [
        {"method": "fast", "quality": 0.8, "latency_ns": 5},
        {"method": "slow", "quality": 0.9, "latency_ns": 10},
        {"method": "best", "quality": 0.9, "latency_ns": 4},
        {"method": "bad", "quality": np.nan, "latency_ns": 1},
    ]
    frontier = pareto_frontier(rows)
    assert {row["method"] for row in frontier} == {"best"}
    summary = pareto_summary(rows)
    assert summary["input_count"] == 4 and summary["frontier_count"] == 1


def test_evaluation_agreement_and_diagnostics_are_explicit_about_ties() -> None:
    rankings = {
        "a": {"q": ["x", "y", "z"]},
        "b": {"q": ["y", "x", "z"], "other": ["z"]},
    }
    rows = evaluate_agreement(rankings, top_k=2, tie_policy="half")
    assert len(rows) == 5 and {row["method_a"] for row in rows} == {"a"}
    matrix = build_agreement_matrix(rankings)
    assert matrix["a"]["a"] == 1.0
    assert agreement_distance(-1.0) == 1.0
    assert agreement_distance(0.5, metric="pairwise_ordering_agreement") == 0.5
    assert np.isnan(agreement_distance(float("nan")))
    with pytest.raises(ValueError, match="unknown agreement"):
        evaluate_agreement(rankings, metrics=("nope",))

    triplets = [
        {"anchor_id": "q", "a_id": "a", "b_id": "b", "expectation": "a_closer"},
        {"query_id": "q", "positive_id": "b", "negative_id": "a", "expected": "tie"},
        {"query_id": "q", "positive_id": "a", "negative_id": "b", "expected": "unspecified"},
    ]
    distances = {("q", "a"): 1.0, ("q", "b"): 1.0}
    strict = triplet_accuracy(triplets, distances)
    assert strict["valid_triplet_count"] == 2 and strict["accuracy"] == 0.5
    tolerant = triplet_accuracy(triplets, distances, tie_aware=True)
    assert tolerant["accuracy"] == 1.0
    evaluated = evaluate_triplets(triplets, distances, bootstrap_resamples=10, seed=4)
    assert evaluated["bootstrap_resamples"] == 10
    callable_result = triplet_accuracy(
        [triplets[0]], lambda anchor, candidate: distances[(anchor, candidate)]
    )
    assert callable_result["accuracy"] == 0.0


def test_fingerprint_rows_and_similarity_fingerprint_preserve_dimensions() -> None:
    rows = fingerprint_rows({"m": {"sampling_invariance": 0.5}}, notion="n")
    assert rows[0]["method"] == "m"
    assert set(rows[0]) >= set(FINGERPRINT_DIMENSIONS)
    fingerprint = build_similarity_fingerprint(
        {"sampling_invariance": 1.0}, method="m", notion="shape"
    )
    assert fingerprint["sampling_invariance"] == 1.0
    assert np.isnan(fingerprint["temporal_sensitivity"])
    with pytest.raises(ValueError, match="unknown fingerprint"):
        build_similarity_fingerprint({"not_a_dimension": 1.0})


def test_robustness_curves_auc_monotonicity_and_hard_negative_gap() -> None:
    curve = robustness_curve([0, 1, 2], 10.0, [10.0, 8.0, 5.0], severity_unit="meters")
    assert curve[1]["severity_normalized"] == 0.5
    assert curve[1]["normalized_value"] == 0.8
    assert robustness_auc(curve) == pytest.approx(0.775)
    multi = robustness_curve([0, 1], [10.0, 20.0], [[10.0, 9.0], [20.0, 18.0]], mode="sensitivity")
    assert len(multi) == 4 and multi[-1]["normalized_value"] == -2.0
    zero = robustness_curve([1, 1], 0.0, [0.0, 1.0])
    assert not zero[1]["valid"]
    assert np.isnan(robustness_auc(zero))
    assert monotonicity_violation_rate({"x": [1, 3, 2]})["violations"] == 1
    assert monotonicity_violation_rate([[3, 2, 1]], expected="nonincreasing")["rate"] == 0.0
    gap = hard_negative_gap([1, 2], [0.5, 1.0], direction="error")
    assert gap["delta_hard"] == 0.75 and gap["sample_size"] == 2
    with pytest.raises(ValueError, match="ordered"):
        robustness_curve([1, 0], 1, [1, 0])
    with pytest.raises(ValueError, match="equal shape"):
        hard_negative_gap([1], [1, 2])


def test_statistics_are_seeded_and_validate_arguments() -> None:
    assert bootstrap_ci([1, 2, 3], resamples=20, seed=3) == bootstrap_ci(
        [1, 2, 3], resamples=20, seed=3
    )
    empty_ci = bootstrap_ci([], resamples=20)
    assert np.isnan(empty_ci[0]) and np.isnan(empty_ci[1])
    assert bootstrap_ci([1, 2], resamples=20, statistic="median")[0] <= 2
    result = paired_permutation_test([2, 3], [1, 1], permutations=20, seed=2, alternative="greater")
    assert result["effect"] == 1.5 and result["sample_size"] == 2
    assert paired_permutation_test([], [])["sample_size"] == 0
    corrected = holm_correction([0.01, 0.2, 0.03])
    np.testing.assert_allclose(corrected, [0.03, 0.2, 0.06])
    summary = summarize_samples([1.0, 2.0], resamples=10, seed=1)
    assert summary["sample_size"] == 2
    with pytest.raises(ValueError, match="statistic"):
        bootstrap_ci([1], statistic="mode")
    with pytest.raises(ValueError, match="invalid"):
        paired_permutation_test([1], [1], alternative="bad")
    with pytest.raises(ValueError, match="p-values"):
        holm_correction([1.1])


def test_timing_summary_and_context_record_duration() -> None:
    summary = summarize_timings("search", [100, 200, 300], workload=10, metadata={"cpu": "test"})
    assert summary.median_ns == 200 and summary.throughput == 50_000_000.0
    assert summary.as_dict()["stage"] == "search"
    with timed_stage("stage", metadata={"x": 1}) as record:
        record["payload"] = "done"
    assert record["end_ns"] >= record["start_ns"]
    assert record["duration_ns"] >= 0
    with pytest.raises(ValueError, match="cannot be empty"):
        summarize_timings("empty", [])


def test_table_and_figure_generation_read_authoritative_parquet(tmp_path: Path) -> None:
    from trajsimbench.storage.parquet import write_parquet

    run = tmp_path / "run"
    run.mkdir()
    write_parquet(
        run / "rankings.parquet",
        [
            {
                "query_id": "q",
                "candidate_id": "c",
                "rank": 1,
                "distance": 0.1,
                "raw_score": 0.1,
                "dataset": "d",
                "dataset_version": "1",
                "method": "m",
            }
        ],
        table="rankings",
    )
    write_parquet(
        run / "systems.parquet",
        [{"stage": "search", "quality": 0.9, "latency_ns": 2}],
        table="systems",
    )
    (run / "manifest.json").write_text(
        __import__("json").dumps({"run_id": "r", "experiment_id": "e", "status": "complete"}),
        encoding="utf-8",
    )
    assert load_result_rows(tmp_path, "rankings")[0]["query_id"] == "q"
    outputs = generate_tables(tmp_path, tmp_path / "tables", parameters={"seed": 1})
    assert (tmp_path / "tables" / "dataset_statistics.csv").exists()
    assert len(outputs) == 11
    figures = generate_figures(tmp_path, tmp_path / "figures")
    assert len(figures) == 11
    payload = __import__("json").loads(
        figures["benchmark_architecture"].read_text(encoding="utf-8")
    )
    assert payload["figure_schema_version"] == "1.0"
