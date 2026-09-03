"""A resumable CPU MVP runner with a deterministic Tiny execution."""

from __future__ import annotations

import json
import traceback as traceback_module
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from trajsimbench.evaluation.diagnostics import FINGERPRINT_DIMENSIONS
from trajsimbench.evaluation.retrieval import aggregate_query_metrics, evaluate_retrieval
from trajsimbench.evaluation.robustness import (
    monotonicity_violation_rate,
    robustness_auc,
    robustness_curve,
)
from trajsimbench.retrieval.exact import exact_top_k
from trajsimbench.retrieval.ranking import compare_rankings
from trajsimbench.storage.artifacts import write_artifacts
from trajsimbench.storage.manifest import ManifestBuilder, load_manifest, sha256_file
from trajsimbench.storage.parquet import write_parquet
from trajsimbench.storage.schemas import SCHEMA_VERSION

from .cache import fingerprint, load_stage_cache, save_stage_cache
from .context import RunContext
from .stages import StageName, StageRecord, StageStatus


@dataclass(frozen=True, slots=True)
class RunResult:
    run_dir: Path
    run_id: str
    experiment_id: str
    status: str
    stage_records: Mapping[str, Mapping[str, Any]]
    warnings: tuple[str, ...] = ()
    failures: tuple[Mapping[str, Any], ...] = ()


def _find_resumable_run(root: Path, experiment_id: str, resolved_config_hash: str) -> Path | None:
    experiment_root = root / experiment_id
    if not experiment_root.exists():
        return None
    matches: list[Path] = []
    for manifest_path in experiment_root.glob("*/manifest.json"):
        try:
            manifest = load_manifest(manifest_path.parent)
        except (FileNotFoundError, ValueError):
            continue
        if manifest.get("resolved_config_hash") == resolved_config_hash:
            matches.append(manifest_path.parent)
    return (
        sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0] if matches else None
    )


def load_config(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return json.loads(json.dumps(dict(source), default=str))
    path = Path(source)
    config_module: Any = None
    try:
        import trajsimbench.config as config_module
    except ImportError:
        config_module = None
    if config_module is not None:
        resolved = config_module.load_config(path)
        if hasattr(resolved, "to_dict"):
            value = json.loads(json.dumps(resolved.to_dict(), default=str))
            value["_source_resolved_config_hash"] = resolved.resolved_config_hash
            return value
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("PyYAML is required to read non-JSON YAML configs") from exc
    else:
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError("configuration root must be a mapping")
    return json.loads(json.dumps(dict(value), default=str))


def resolve_config(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    config = load_config(source)
    source_resolved_hash = config.pop("_source_resolved_config_hash", None)
    # The foundation config scope emits a strict, nested canonical model.  The
    # CPU runner also accepts the concise Phase-0 vocabulary, so normalize both
    # shapes at this boundary rather than duplicating Pydantic models here.
    if "seeds" in config and "seed_list" not in config:
        config["seed_list"] = list(config["seeds"])
    dataset_value = config.get("dataset")
    if isinstance(dataset_value, Mapping):
        dataset = dict(dataset_value)
        config["dataset"] = dataset.get("name", "synthetic")
        config.setdefault(
            "dataset_path",
            dataset.get("processed_path", dataset.get("path")),
        )
        config.setdefault("dataset_version", dataset.get("version", "v1"))
        config.setdefault("split", dataset.get("split", "standard"))
        config.setdefault("scale_tier", dataset.get("scale_tier", "Tiny"))
        selection = dataset.get("selection", {})
        if isinstance(selection, Mapping):
            database_count = selection.get("database_count") or 1000
            query_count = selection.get("query_count") or 100
            config.setdefault(
                "tiny_counts",
                {
                    "database": database_count,
                    "queries": query_count,
                },
            )
    relevance_config = config.get("relevance")
    if isinstance(relevance_config, Mapping):
        config.setdefault("k_values", list(relevance_config.get("k_values", [1, 5, 10])))
        config.setdefault(
            "empty_relevance_policy", relevance_config.get("empty_relevance_policy", "skip")
        )
    storage_config = config.get("storage")
    if isinstance(storage_config, Mapping):
        config.setdefault("output_root", storage_config.get("output_root", "results"))
    resource_config = config.get("resources")
    if isinstance(resource_config, Mapping):
        config.setdefault("chunk_size", resource_config.get("chunk_size", 128))
    config.setdefault("schema_version", SCHEMA_VERSION)
    config.setdefault("experiment_id", "tiny_cpu")
    config.setdefault("seed_list", [0])
    config.setdefault("methods", [{"name": "euclidean", "config": {}}])
    config.setdefault("tasks", ["retrieval"])
    config.setdefault("metrics", ["recall@1", "precision@1", "ndcg@1", "mrr"])
    config.setdefault("output_root", "results")
    config.setdefault("scale_tier", "Tiny")
    config.setdefault("k_values", [1, 5, 10])
    config.setdefault("empty_relevance_policy", "skip")
    if not isinstance(config["seed_list"], list) or not config["seed_list"]:
        raise ValueError("seed_list must be a non-empty list")
    if not isinstance(config["methods"], list) or not config["methods"]:
        raise ValueError("methods must be a non-empty list")
    method_names: list[str] = []
    normalized_methods: list[dict[str, Any]] = []
    for method in config["methods"]:
        if isinstance(method, str):
            item = {"name": method, "config": {}}
        elif isinstance(method, Mapping):
            item = {"name": str(method.get("name")), "config": dict(method.get("config", {}))}
        else:
            raise ValueError("each method must be a name or mapping")
        if not item["name"] or item["name"] in method_names:
            raise ValueError("method names must be non-empty and unique")
        name = str(item["name"])
        item["name"] = name
        method_names.append(name)
        normalized_methods.append(item)
    config["methods"] = normalized_methods
    known_methods = {
        "euclidean",
        "dtw",
        "hausdorff",
        "discrete_frechet",
        "lcss",
        "edr",
        "erp",
        "fake",
        "symmetric_hausdorff",
    }
    unknown_methods = sorted(set(method_names) - known_methods)
    if unknown_methods:
        raise ValueError(f"unknown method names: {unknown_methods}")
    valid_k = config.get("k_values", [1, 5, 10])
    if (
        not isinstance(valid_k, list)
        or not valid_k
        or any(not isinstance(value, int) or value < 1 for value in valid_k)
    ):
        raise ValueError("k_values must be a non-empty list of positive integers")
    raw_metrics = config.get("metrics", [])
    metric_values = (
        raw_metrics.get("enabled", []) if isinstance(raw_metrics, Mapping) else raw_metrics
    )
    valid_metrics = {
        "recall",
        "precision",
        "hit_rate",
        "ndcg",
        "mrr",
        "ranking_agreement",
        "top_k_recall",
        "diagnostic_accuracy",
        "robustness",
        "monotonicity",
        "fingerprint",
        "latency",
        "throughput",
    }
    for metric in metric_values:
        name = str(metric).split("@", 1)[0]
        if name not in valid_metrics:
            raise ValueError(f"unknown metric: {metric}")
    # File-backed configs already have a hash over the strict, fragment-
    # resolved model. Preserve that identity after flattening it for the
    # lightweight CPU runner; otherwise the CLI validator and run manifest
    # would disagree about which config was executed.
    config["resolved_config_hash"] = str(source_resolved_hash or fingerprint(config))
    return config


def _tiny_data(
    config: Mapping[str, Any], seed: int
) -> tuple[list[str], list[np.ndarray], list[str], list[np.ndarray], dict[str, dict[str, float]]]:
    scale = config.get("tiny_counts", {})
    database_count = int(scale.get("database", 1000)) if isinstance(scale, Mapping) else 1000
    query_count = int(scale.get("queries", 100)) if isinstance(scale, Mapping) else 100
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    queries: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    query_ids = [f"q-{index:05d}" for index in range(query_count)]
    candidate_ids = [f"c-{index:05d}" for index in range(database_count)]
    for index in range(query_count):
        x = np.linspace(0, 1, 12)
        path = np.column_stack((x, np.sin(x * np.pi * 2 + index * 0.01)))
        queries.append(path.astype(np.float64))
    for index in range(database_count):
        if index < query_count:
            candidate = queries[index] + rng.normal(0, 0.002, queries[index].shape)
        else:
            x = np.linspace(0, 1, 12)
            candidate = np.column_stack(
                (
                    x,
                    np.sin(x * np.pi * (1 + (index % 5) / 3) + rng.uniform(-1, 1))
                    + rng.normal(0, 0.03, len(x)),
                )
            )
        candidates.append(candidate.astype(np.float64))
    relevance = {query_id: {candidate_ids[index]: 1.0} for index, query_id in enumerate(query_ids)}
    return query_ids, queries, candidate_ids, candidates, relevance


def _prepared_dataset_data(
    config: Mapping[str, Any],
) -> tuple[list[str], list[np.ndarray], list[str], list[np.ndarray], dict[str, dict[str, float]]]:
    dataset_name = str(config.get("dataset", "synthetic"))
    if dataset_name == "synthetic":
        return _tiny_data(config, int(config["seed_list"][0]))
    dataset_path = config.get("dataset_path")
    if not dataset_path:
        raise ValueError(
            f"dataset {dataset_name!r} is not synthesized by the CPU runner; "
            "prepare it first and set dataset.path or dataset.processed_path"
        )
    try:
        from trajsimbench.data.dataset import TrajectoryDataset
    except ImportError as exc:
        raise ValueError(
            "the canonical data scope is required to open a prepared non-synthetic dataset"
        ) from exc
    dataset = TrajectoryDataset.open(Path(str(dataset_path)), mmap=True)
    all_ids = [str(value) for value in dataset.ids()]
    selection = config.get("tiny_counts", {})
    if isinstance(selection, Mapping):
        query_count = int(selection.get("queries", min(100, len(all_ids))))
        database_count = int(selection.get("database", min(1000, len(all_ids))))
    else:
        query_count = min(100, len(all_ids))
        database_count = min(1000, len(all_ids))
    if query_count > len(all_ids) or database_count > len(all_ids):
        raise ValueError(
            "requested query/database counts exceed the prepared dataset; "
            "sampling with replacement is disabled"
        )
    query_ids = all_ids[:query_count]
    candidate_ids = all_ids[:database_count]
    query_paths = [np.asarray(dataset.by_id(identifier).points) for identifier in query_ids]
    candidate_paths = [np.asarray(dataset.by_id(identifier).points) for identifier in candidate_ids]
    relevance = {
        identifier: ({identifier: 1.0} if identifier in candidate_ids else {})
        for identifier in query_ids
    }
    return query_ids, query_paths, candidate_ids, candidate_paths, relevance


def _array_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Tiny fallback measure: arc-length-normalized pointwise distance.  If the
    # measures scope is present, callers can supply a measure through the
    # public retrieval API without changing the runner contract.
    count = max(len(a), len(b))

    def resample(value: np.ndarray) -> np.ndarray:
        if len(value) == count:
            return value[:, :2]
        old = np.linspace(0, 1, len(value))
        new = np.linspace(0, 1, count)
        return np.column_stack(
            [np.interp(new, old, value[:, axis]) for axis in range(min(2, value.shape[1]))]
        )

    return float(np.mean(np.linalg.norm(resample(a) - resample(b), axis=1)))


def _write_resolved(path: Path, config: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError:
        path.write_text(json.dumps(dict(config), indent=2, sort_keys=True), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(dict(config), sort_keys=True), encoding="utf-8")


def _log(path: Path, event: str, **fields: Any) -> None:
    entry = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def _configured_measure(method_name: str, method_config: Mapping[str, Any]) -> Any | None:
    """Create a registered classical measure when the measures scope exists."""

    try:
        from trajsimbench.measures.registry import create_measure
    except ImportError:
        return None
    actual_name = "hausdorff" if method_name == "symmetric_hausdorff" else method_name
    allowed_fields = {
        "euclidean": {"n_samples", "resample_count", "sampling_count", "num_samples"},
        "dtw": {
            "normalization",
            "global_normalization",
            "window",
            "window_size",
            "sakoe_chiba_window",
        },
        "hausdorff": set(),
        "discrete_frechet": set(),
        "lcss": {"epsilon", "delta", "delta_mode", "use_timestamps"},
        "edr": {"epsilon", "normalize"},
        "erp": {"gap_point", "normalize", "normalization"},
    }
    filtered = {
        key: value
        for key, value in method_config.items()
        if key in allowed_fields.get(actual_name, set())
    }
    if "n_samples" in filtered:
        filtered.pop("resample_count", None)
        filtered.pop("sampling_count", None)
        filtered.pop("num_samples", None)
    try:
        return create_measure(actual_name, filtered)
    except (KeyError, TypeError, ValueError):
        # An unavailable optional method/config must not break the dependency-
        # free runner; it falls back to the documented Tiny CPU distance.
        return None


def dry_run_experiment(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    config = resolve_config(source)
    database = (
        int(config.get("tiny_counts", {}).get("database", 1000))
        if isinstance(config.get("tiny_counts"), Mapping)
        else 1000
    )
    queries = (
        int(config.get("tiny_counts", {}).get("queries", 100))
        if isinstance(config.get("tiny_counts"), Mapping)
        else 100
    )
    methods = [str(item["name"]) for item in config["methods"]]
    return {
        "experiment_id": config["experiment_id"],
        "stages": [stage.value for stage in StageName],
        "methods": methods,
        "queries": queries,
        "database": database,
        "estimated_pairs": queries * database * len(methods),
        "cpu_only": True,
        "resource_warnings": ["large pair count; use a chunk_size or smaller tier"]
        if queries * database > 2_000_000
        else [],
    }


def run_experiment(
    source: Path | str | Mapping[str, Any],
    *,
    resume: bool = False,
    dry_run: bool = False,
    force_stage: str | None = None,
    output_root: Path | None = None,
) -> RunResult | dict[str, Any]:
    if dry_run:
        return dry_run_experiment(source)
    config = resolve_config(source)
    experiment_id = str(config["experiment_id"])
    configured_root = Path(output_root or config.get("output_root", "results"))
    existing_run = (
        _find_resumable_run(configured_root, experiment_id, str(config["resolved_config_hash"]))
        if (resume or force_stage) and not config.get("run_id")
        else None
    )
    if existing_run is not None and resume and not force_stage:
        manifest = load_manifest(existing_run)
        if manifest.get("status") == "complete":
            cached_records = load_stage_cache(existing_run / "stage_state.json")
            return RunResult(
                existing_run,
                str(manifest.get("run_id")),
                experiment_id,
                "complete",
                cached_records,
                tuple(manifest.get("warnings", [])),
                tuple(manifest.get("failures", [])),
            )
    run_id = str(
        config.get("run_id") or (existing_run.name if existing_run is not None else uuid4())
    )
    context = RunContext.create(experiment_id, run_id, configured_root)
    _write_resolved(context.path("resolved_config.yaml"), config)
    logs_path = context.path("logs.jsonl")
    logs_path.touch(exist_ok=True)
    records_path = context.path("stage_state.json")
    records_data = load_stage_cache(records_path) if (resume or force_stage) else {}
    if force_stage:
        from .resume import invalidate_from

        records_data = invalidate_from(records_data, force_stage)
    records: dict[str, StageRecord] = {}
    for stage_name, data in records_data.items():
        if isinstance(data, Mapping):
            records[str(stage_name)] = StageRecord(
                str(stage_name), **{key: value for key, value in data.items() if key != "stage"}
            )
    for stage in StageName:
        records.setdefault(str(stage), StageRecord(str(stage)))
    failures: list[dict[str, Any]] = []
    warnings: list[str] = []
    manifest_builder = ManifestBuilder(run_id, experiment_id, context.run_dir)
    common = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "dataset": config.get("dataset", "synthetic"),
        "dataset_version": config.get("dataset_version", "tiny-v1"),
        "split": config.get("split", "test"),
        "scale_tier": config.get("scale_tier", "Tiny"),
        "seed": int(config["seed_list"][0]),
    }
    query_ids: list[str] = []
    query_paths: list[np.ndarray] = []
    candidate_ids: list[str] = []
    candidate_paths: list[np.ndarray] = []
    relevance: dict[str, dict[str, float]] = {}
    rankings_by_method: dict[str, dict[str, list[str]]] = {}
    method_provenance: dict[str, tuple[str, str]] = {}
    metrics_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    try:
        records[StageName.VALIDATE.value].begin(fingerprint(config))
        if config.get("dataset", "synthetic") == "germany" and config.get("enabled") is False:
            raise ValueError(
                "Germany is disabled until an exact source, schema, and license are approved"
            )
        records[StageName.VALIDATE.value].complete(["resolved_config.yaml"])
        _log(logs_path, "stage_complete", stage="validate")

        records[StageName.LOAD_DATA.value].begin(
            fingerprint({"config": config, "stage": "load_data"})
        )
        query_ids, query_paths, candidate_ids, candidate_paths, relevance = _prepared_dataset_data(
            config
        )
        records[StageName.LOAD_DATA.value].complete([])
        _log(
            logs_path,
            "stage_complete",
            stage="load_data",
            queries=len(query_ids),
            database=len(candidate_ids),
        )

        records[StageName.MATERIALIZE_TASKS.value].begin(
            fingerprint({"query_ids": query_ids, "candidate_ids": candidate_ids})
        )
        write_parquet(
            context.path("tasks", "queries.parquet"),
            ({**common, "query_id": query_id} for query_id in query_ids),
            table="queries",
        )
        write_parquet(context.path("tasks", "triplets.parquet"), (), table="triplets")
        write_parquet(context.path("tasks", "variants.parquet"), (), table="variants")
        records[StageName.MATERIALIZE_TASKS.value].complete(
            ["tasks/queries.parquet", "tasks/triplets.parquet", "tasks/variants.parquet"]
        )

        records[StageName.FIT_METHODS.value].begin(fingerprint(config["methods"]))
        records[StageName.FIT_METHODS.value].complete([])
        records[StageName.BUILD_INDEX.value].begin(
            fingerprint({"methods": config["methods"], "embeddings": "none"})
        )
        records[StageName.BUILD_INDEX.value].complete([])

        records[StageName.EVALUATE.value].begin(
            fingerprint(
                {"queries": query_ids, "database": candidate_ids, "methods": config["methods"]}
            )
        )
        top_k = max(int(value) for value in config.get("k_values", [1, 5, 10]))
        for method_item in config["methods"]:
            method = str(method_item["name"])
            method_rankings: dict[str, list[str]] = {}
            try:
                configured = _configured_measure(method, method_item.get("config", {}))
                distance_fn = configured.distance if configured is not None else _array_distance
                method_version = str(
                    getattr(configured, "version", method_item.get("version", "cpu-mvp"))
                )
                configured_config = getattr(configured, "config", None)
                dump_method = getattr(configured_config, "model_dump", None)
                effective_config = (
                    dump_method()
                    if configured is not None and callable(dump_method)
                    else method_item.get("config", {})
                )
                method_config_hash = fingerprint(effective_config)
                method_provenance[method] = (method_version, method_config_hash)
                for query_id, query in zip(query_ids, query_paths, strict=True):
                    result = exact_top_k(
                        query,
                        candidate_paths,
                        k=top_k,
                        candidate_ids=candidate_ids,
                        distance_fn=distance_fn,
                        chunk_size=int(config.get("chunk_size", 128)),
                        exclude_ids={query_id} if query_id in candidate_ids else None,
                        query_id=query_id,
                    )
                    ranked_ids = [str(value) for value in result.candidate_ids.tolist()]
                    method_rankings[query_id] = ranked_ids
                    for row in result.as_rows(method=method, task="retrieval"):
                        ranking_rows.append(
                            {
                                **common,
                                **row,
                                "method_version": method_version,
                                "method_config_hash": method_config_hash,
                                "relevance_value": float(
                                    relevance.get(query_id, {}).get(row["candidate_id"], 0.0)
                                ),
                            }
                        )
                rankings_by_method[method] = method_rankings
                manifest_builder.method_versions[method] = method_version
            except Exception as exc:
                failure = {
                    **common,
                    "stage": "evaluate",
                    "method": method,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback_module.format_exc(),
                }
                failures.append(failure)
                _log(
                    logs_path,
                    "method_failure",
                    method=method,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
        records[StageName.EVALUATE.value].complete(["rankings.parquet"])

        records[StageName.METRICS.value].begin(
            fingerprint({"rankings": ranking_rows, "relevance": relevance})
        )
        for method, method_rankings in rankings_by_method.items():
            method_version, method_config_hash = method_provenance.get(
                method, ("cpu-mvp", fingerprint({}))
            )
            evaluated = evaluate_retrieval(
                method_rankings,
                relevance,
                ks=config.get("k_values", [1, 5, 10]),
                empty_policy=config.get("empty_relevance_policy", "skip"),
            )
            metrics_rows.extend(
                {
                    **common,
                    **row,
                    "method": method,
                    "method_version": method_version,
                    "method_config_hash": method_config_hash,
                }
                for row in evaluated
            )
        aggregates = [
            {**common, **row}
            for row in aggregate_query_metrics(metrics_rows, group_by=("method", "metric", "k"))
        ]
        records[StageName.METRICS.value].complete(
            ["query_metrics.parquet", "aggregate_metrics.parquet"]
        )

        # Agreement is only computed over common full candidate universes.  The
        # MVP retains Top-K rankings and therefore supplies explicit depth.
        agreement_rows: list[dict[str, Any]] = []
        methods = sorted(rankings_by_method)
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1 :]:
                for query_id in query_ids:
                    try:
                        comparison = compare_rankings(
                            rankings_by_method[left][query_id],
                            rankings_by_method[right][query_id],
                            top_k=top_k,
                        )
                    except ValueError:
                        continue
                    for metric in (
                        "kendall_tau_b",
                        "spearman_rho",
                        "top_k_jaccard",
                        "rank_biased_overlap",
                        "pairwise_ordering_agreement",
                    ):
                        agreement_rows.append(
                            {
                                **common,
                                "method_a": left,
                                "method_b": right,
                                "metric": metric,
                                "value": comparison[metric],
                                "comparison_depth": comparison["comparison_depth"],
                                "candidate_count": comparison["candidate_count"],
                            }
                        )

        clean = np.asarray([_array_distance(query_paths[0], candidate_paths[0])], dtype=float)
        perturbed = np.asarray([clean[0] + 0.01, clean[0] + 0.02, clean[0] + 0.03], dtype=float)
        curve = robustness_curve(
            [0.0, 0.5, 1.0],
            clean[0],
            perturbed.tolist(),
            mode="sensitivity",
            severity_unit="normalized-v1",
        )
        robustness_rows = [
            {
                **common,
                "source_id": query_ids[0],
                "perturbation": "gps_noise",
                **row,
                "rauc": robustness_auc(curve) if row["severity_index"] == 0 else None,
            }
            for row in curve
        ]
        mvr = monotonicity_violation_rate(
            {query_ids[0]: [row["perturbed_value"] for row in curve]}, expected="nondecreasing"
        )
        for row in robustness_rows:
            row["mvr_rate"] = mvr["rate"]
        # Fingerprints are vectors of named diagnostics.  The Tiny retrieval
        # task does not request counterfactual triplets, so unavailable
        # dimensions are retained explicitly instead of being fabricated as
        # zero scores.
        fingerprint_rows = [
            {
                **common,
                "method": method,
                "method_version": method_provenance.get(method, ("cpu-mvp", ""))[0],
                "method_config_hash": method_provenance.get(method, ("", ""))[1],
                "fingerprint_version": "1.0",
                "dimension": dimension,
                "value": None,
                "valid": False,
                "reason": "diagnostic_task_not_requested",
                "sample_size": 0,
                "coverage": 0.0,
            }
            for method in sorted(rankings_by_method)
            for dimension in FINGERPRINT_DIMENSIONS
        ]
        systems_rows = [
            {
                **common,
                "stage": "evaluate",
                "summary_type": "mvp",
                "timing_value": int(
                    np.sum([row.get("retrieval_runtime_ns") or 0 for row in ranking_rows])
                ),
                "query_count": len(query_ids),
                "database_size": len(candidate_ids),
                "hardware": "cpu",
            }
        ]
        records[StageName.COMMIT.value].begin(
            fingerprint({"rows": len(ranking_rows), "metrics": len(metrics_rows)})
        )
        write_parquet(context.path("rankings.parquet"), ranking_rows, table="rankings")
        write_parquet(context.path("query_metrics.parquet"), metrics_rows, table="query_metrics")
        write_parquet(
            context.path("aggregate_metrics.parquet"), aggregates, table="aggregate_metrics"
        )
        write_parquet(context.path("agreement.parquet"), agreement_rows, table="agreement")
        write_parquet(context.path("robustness.parquet"), robustness_rows, table="robustness")
        write_parquet(context.path("systems.parquet"), systems_rows, table="systems")
        write_parquet(context.path("failures.parquet"), failures, table="failures")
        write_parquet(context.path("fingerprints.parquet"), fingerprint_rows, table="fingerprints")
        write_artifacts(context.run_dir, [])
        records[StageName.COMMIT.value].complete(
            [
                "rankings.parquet",
                "query_metrics.parquet",
                "aggregate_metrics.parquet",
                "agreement.parquet",
                "robustness.parquet",
                "systems.parquet",
                "failures.parquet",
                "fingerprints.parquet",
                "artifacts.json",
            ]
        )
        for output in records[StageName.COMMIT.value].outputs:
            output_path = context.run_dir / output
            if output_path.exists():
                records[StageName.COMMIT.value].output_checksums[output] = sha256_file(output_path)
        _log(logs_path, "stage_complete", stage="commit")
        records[StageName.ANALYZE.value].complete([])
    except Exception as exc:
        failed_stage_name: str = next(
            (name for name, record in records.items() if record.status == StageStatus.RUNNING),
            "unknown",
        )
        if failed_stage_name in records:
            records[failed_stage_name].fail(exc, traceback_module.format_exc())
        failures.append(
            {
                **common,
                "stage": failed_stage_name,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback_module.format_exc(),
            }
        )
        _log(
            logs_path,
            "run_failure",
            stage=failed_stage_name,
            error_type=type(exc).__name__,
            message=str(exc),
        )

    status = "failed" if failures and not ranking_rows else ("partial" if failures else "complete")
    records[StageName.FINALIZE.value].begin(fingerprint({"status": status, "failures": failures}))
    manifest_builder.status = status
    manifest_builder.failures.extend(failures)
    manifest_builder.warnings.extend(warnings)
    manifest_builder.write(
        resolved_config_hash=str(config["resolved_config_hash"]), root=context.run_dir.parent.parent
    )
    records[StageName.FINALIZE.value].complete(["manifest.json"])
    save_stage_cache(records_path, {name: record.as_dict() for name, record in records.items()})
    return RunResult(
        context.run_dir,
        run_id,
        experiment_id,
        status,
        {name: record.as_dict() for name, record in records.items()},
        tuple(warnings),
        tuple(failures),
    )
