"""Pydantic models for resolved benchmark configuration.

The models intentionally use ``extra='forbid'``. Extensible method-specific
settings belong below ``MethodSpec.config`` so a typo in the experiment-level
contract cannot silently change a run.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


SUPPORTED_SCHEMA_MAJOR = 1
SUPPORTED_SCALE_TIERS = {"tiny", "standard", "medium", "large"}
SUPPORTED_METHODS = {
    "euclidean",
    "dtw",
    "hausdorff",
    "symmetric_hausdorff",
    "discrete_frechet",
    "lcss",
    "edr",
    "erp",
}
SUPPORTED_TASKS = {
    "oracle",
    "oracle_approximation",
    "equivalence",
    "diagnostics",
    "diagnostic",
    "counterfactual_diagnostics",
    "retrieval",
    "agreement",
    "robustness",
    "hard_negatives",
    "hard_negative_retrieval",
    "systems",
    "generalization",
}
SUPPORTED_METRICS = {
    "recall",
    "precision",
    "hit_rate",
    "ndcg",
    "ranking_agreement",
    "top_k_recall",
    "mrr",
    "diagnostic_accuracy",
    "robustness",
    "monotonicity",
    "fingerprint",
    "latency",
    "throughput",
}


def _schema_major(value: str | int | float) -> int:
    text = str(value)
    try:
        major = int(text.split(".", 1)[0])
    except ValueError as exc:
        raise ValueError(
            "schema_version must be a supported numeric version such as '1.0'"
        ) from exc
    if major != SUPPORTED_SCHEMA_MAJOR:
        raise ValueError(
            f"unsupported schema_version {value!r}; supported major version is "
            f"{SUPPORTED_SCHEMA_MAJOR}.0"
        )
    return major


class SelectionConfig(StrictModel):
    """Explicit query/database selection limits.

    A limit is never interpreted as permission to sample with replacement.
    """

    query_count: int | None = Field(default=None, ge=0)
    database_count: int | None = Field(default=None, ge=0)
    query_ids: list[str] | None = None
    database_ids: list[str] | None = None

    @model_validator(mode="after")
    def unique_ids(self) -> SelectionConfig:
        for label, values in (("query_ids", self.query_ids), ("database_ids", self.database_ids)):
            if values is not None and len(values) != len(set(values)):
                raise ValueError(f"{label} must not contain duplicate IDs")
        return self


class DatasetReference(StrictModel):
    name: str
    version: str = "v1"
    config_path: str | None = None
    enabled: bool = True
    status: str | None = None
    split: str = "standard"
    scale_tier: str = "tiny"
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    min_points: int = Field(default=2, ge=1)
    max_points: int | None = Field(default=None, ge=1)
    projected_crs: str = "EPSG:32610"
    raw_path: str | None = None
    polyline_field: str = "POLYLINE"
    trip_id_field: str = "TRIP_ID"
    timestamp_field: str | None = None
    timestamp_semantics: str | None = None
    sampling_interval_s: float = Field(default=15.0, gt=0)
    missing_data_sentinel: str | None = None
    encoding: str = "utf-8"
    user_id_field: str = "user_id"
    mobility_mode_field: str = "mobility_mode"
    source_url: str | None = None
    source_name: str | None = None
    source_license: str | None = None
    redistribution_policy: str | None = None
    preprocessing_config_hash: str | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    has_timestamps: bool = True
    available_trajectories: int | None = Field(default=None, ge=0)
    user_held_out: bool = False
    deduplicate_policy: str = "drop_consecutive"

    @field_validator("name", "version", "split", "scale_tier")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text fields must not be empty")
        return value.lower() if value in {"Tiny", "Standard", "Medium", "Large"} else value

    @field_validator("scale_tier")
    @classmethod
    def known_scale_tier(cls, value: str) -> str:
        value = value.lower()
        if value not in SUPPORTED_SCALE_TIERS:
            raise ValueError(
                f"unknown scale_tier {value!r}; choose one of "
                f"{', '.join(sorted(SUPPORTED_SCALE_TIERS))}"
            )
        return value

    @field_validator("bounding_box")
    @classmethod
    def valid_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return value
        west, south, east, north = value
        if not (
            -180 <= west <= 180
            and -180 <= east <= 180
            and -90 <= south <= 90
            and -90 <= north <= 90
        ):
            raise ValueError("bounding_box must be (west, south, east, north) in WGS84 degrees")
        if west > east or south > north:
            raise ValueError("bounding_box lower bounds must not exceed upper bounds")
        return value

    @model_validator(mode="after")
    def check_point_range(self) -> DatasetReference:
        if self.max_points is not None and self.max_points < self.min_points:
            raise ValueError("max_points must be greater than or equal to min_points")
        if self.status == "requires_source_decision" and self.enabled:
            raise ValueError("a dataset requiring a source decision must have enabled=false")
        if self.selection.query_count is not None and self.available_trajectories is not None:
            if self.selection.query_count > self.available_trajectories:
                raise ValueError("query_count exceeds available_trajectories")
        if self.selection.database_count is not None and self.available_trajectories is not None:
            if self.selection.database_count > self.available_trajectories:
                raise ValueError("database_count exceeds available_trajectories")
        return self


class MethodSpec(StrictModel):
    name: str
    version: str = "1"
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def known_method(cls, value: str) -> str:
        if value not in SUPPORTED_METHODS:
            allowed = ", ".join(sorted(SUPPORTED_METHODS))
            raise ValueError(f"unknown method {value!r}; choose one of: {allowed}")
        return value


class TaskSpec(StrictModel):
    name: str
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def known_task(cls, value: str) -> str:
        if value not in SUPPORTED_TASKS:
            raise ValueError(
                f"unknown task {value!r}; choose one of: {', '.join(sorted(SUPPORTED_TASKS))}"
            )
        return value


class NotionSpec(StrictModel):
    name: str = "v1"
    version: str = "1"
    config: dict[str, Any] = Field(default_factory=dict)


class PerturbationSpec(StrictModel):
    name: str
    severities: list[float] = Field(default_factory=list)
    unit: str = "m"
    seed_offset: int = 0
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("severities")
    @classmethod
    def valid_severities(cls, values: list[float]) -> list[float]:
        if len(values) != len(set(values)):
            raise ValueError("perturbation severities must be unique and ordered")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("perturbation severities must be finite and non-negative")
        if values != sorted(values):
            raise ValueError("perturbation severities must be in ascending order")
        return values

    @field_validator("unit")
    @classmethod
    def valid_unit(cls, value: str) -> str:
        if value not in {
            "m",
            "meter",
            "meters",
            "s",
            "second",
            "seconds",
            "fraction",
            "count",
            "unitless",
            "normalized-v1",
        }:
            raise ValueError("unit must be a distance, time, fraction, or count unit")
        return value


class RelevanceConfig(StrictModel):
    providers: list[str] = Field(default_factory=list)
    k_values: list[int] = Field(default_factory=lambda: [1, 5, 10])
    empty_relevance_policy: str = "skip"

    @model_validator(mode="after")
    def unique_positive_k(self) -> RelevanceConfig:
        if len(self.k_values) != len(set(self.k_values)) or any(k <= 0 for k in self.k_values):
            raise ValueError("relevance k_values must be unique positive integers")
        return self

    @field_validator("empty_relevance_policy")
    @classmethod
    def valid_empty_policy(cls, value: str) -> str:
        if value not in {"skip", "zero", "raise"}:
            raise ValueError("empty_relevance_policy must be 'skip', 'zero', or 'raise'")
        return value


class MetricsConfig(StrictModel):
    enabled: list[str] = Field(default_factory=lambda: ["ranking_agreement"])
    bootstrap_replicates: int = Field(default=1000, ge=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    correction: str = "holm"

    @field_validator("enabled")
    @classmethod
    def known_metrics(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("metrics.enabled must not contain duplicates")
        unknown = sorted({value.split("@", 1)[0] for value in values} - SUPPORTED_METRICS)
        if unknown:
            raise ValueError(f"unknown metric(s): {', '.join(unknown)}")
        return values


class SystemsConfig(StrictModel):
    warmup: int = Field(default=1, ge=0)
    repetitions: int = Field(default=3, ge=1)
    worker_count: int = Field(default=1, ge=1)
    record_memory: bool = True


class StorageConfig(StrictModel):
    output_root: str = "results"
    format: str = "parquet"
    keep_intermediate: bool = False

    @field_validator("format")
    @classmethod
    def only_parquet(cls, value: str) -> str:
        if value != "parquet":
            raise ValueError("the authoritative result format is parquet")
        return value


class CacheConfig(StrictModel):
    enabled: bool = True
    resume: bool = True
    cache_root: str | None = None
    invalidate_on_dirty_code: bool = True


class ResourceConfig(StrictModel):
    max_pair_count: int = Field(default=1_000_000, ge=1)
    max_dp_cells: int = Field(default=50_000_000, ge=1)
    timeout_seconds: float = Field(default=3600, gt=0)
    max_workers: int = Field(default=1, ge=1)
    chunk_size: int = Field(default=128, ge=1)
    allow_large_run: bool = False


class ExperimentConfig(StrictModel):
    """Fully resolved experiment settings."""

    CURRENT_SCHEMA: ClassVar[str] = "1.0"

    schema_version: str = "1.0"
    experiment_id: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=lambda: [0])
    dataset: DatasetReference
    methods: list[MethodSpec] = Field(default_factory=list)
    tasks: list[TaskSpec] = Field(default_factory=list)
    notion: NotionSpec = Field(default_factory=NotionSpec)
    perturbations: list[PerturbationSpec] = Field(default_factory=list)
    negative_policies: list[dict[str, Any]] = Field(default_factory=list)
    relevance: RelevanceConfig = Field(default_factory=RelevanceConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    systems: SystemsConfig = Field(default_factory=SystemsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_and_compact_forms(cls, value: Any) -> Any:
        """Normalize the concise Phase-0 config vocabulary before strict parsing."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "seed_list" in data and "seeds" not in data:
            data["seeds"] = data["seed_list"]
        data.pop("seed_list", None)

        dataset = data.get("dataset")
        if isinstance(dataset, str):
            dataset = {"name": dataset}
        elif isinstance(dataset, dict):
            dataset = dict(dataset)
        if dataset is not None:
            for old_key, new_key in (
                ("dataset_version", "version"),
                ("split", "split"),
                ("scale_tier", "scale_tier"),
            ):
                if old_key in data and new_key not in dataset:
                    dataset[new_key] = data[old_key]
            if "scale_tier" in dataset and isinstance(dataset["scale_tier"], str):
                dataset["scale_tier"] = dataset["scale_tier"].lower()
            tiny_counts = data.get("tiny_counts")
            if isinstance(tiny_counts, dict):
                selection = dict(dataset.get("selection", {}))
                if "database" in tiny_counts:
                    selection.setdefault("database_count", tiny_counts["database"])
                if "queries" in tiny_counts:
                    selection.setdefault("query_count", tiny_counts["queries"])
                dataset["selection"] = selection
            data["dataset"] = dataset
        for old_key in ("dataset_version", "split", "scale_tier", "tiny_counts"):
            data.pop(old_key, None)

        tasks = data.get("tasks")
        if isinstance(tasks, list):
            data["tasks"] = [task if isinstance(task, dict) else {"name": task} for task in tasks]
        perturbations = data.get("perturbations")
        if isinstance(perturbations, list):
            normalized_perturbations: list[Any] = []
            for perturbation in perturbations:
                if isinstance(perturbation, dict):
                    item = dict(perturbation)
                    if "severity_values" in item and "severities" not in item:
                        item["severities"] = item.pop("severity_values")
                    normalized_perturbations.append(item)
                else:
                    normalized_perturbations.append(perturbation)
            data["perturbations"] = normalized_perturbations
        if "k_values" in data:
            relevance = dict(data.get("relevance", {}))
            relevance.setdefault("k_values", data["k_values"])
            data["relevance"] = relevance
            data.pop("k_values", None)
        if "empty_relevance_policy" in data:
            relevance = dict(data.get("relevance", {}))
            relevance.setdefault("empty_relevance_policy", data["empty_relevance_policy"])
            data["relevance"] = relevance
            data.pop("empty_relevance_policy", None)
        if "chunk_size" in data:
            resources = dict(data.get("resources", {}))
            resources.setdefault("chunk_size", data["chunk_size"])
            data["resources"] = resources
            data.pop("chunk_size", None)
        if "output_root" in data:
            storage = dict(data.get("storage", {}))
            storage.setdefault("output_root", data["output_root"])
            data["storage"] = storage
            data.pop("output_root", None)

        # Accept the concise YAML vocabulary used by the plan while keeping
        # the resolved model strict and fully typed.  Method names and metric
        # names are intentionally normalized here rather than in the CLI so
        # direct ``ExperimentConfig.model_validate`` callers get the same
        # contract as file-based config resolution.
        methods = data.get("methods")
        if isinstance(methods, list):
            data["methods"] = [
                method if isinstance(method, dict) else {"name": method} for method in methods
            ]
        metrics = data.get("metrics")
        if isinstance(metrics, list):
            data["metrics"] = {"enabled": metrics}
        elif isinstance(metrics, dict):
            metrics = dict(metrics)
            if "names" in metrics and "enabled" not in metrics:
                metrics["enabled"] = metrics.pop("names")
            data["metrics"] = metrics
        return data

    @field_validator("schema_version", mode="before")
    @classmethod
    def supported_schema(cls, value: str | int | float) -> str:
        _schema_major(value)
        return str(value)

    @field_validator("experiment_id")
    @classmethod
    def valid_experiment_id(cls, value: str) -> str:
        if not value.strip() or any(char.isspace() for char in value):
            raise ValueError("experiment_id must be non-empty and contain no whitespace")
        return value

    @field_validator("seeds")
    @classmethod
    def unique_seeds(cls, values: list[int]) -> list[int]:
        if not values or len(values) != len(set(values)):
            raise ValueError("seeds must contain at least one unique integer")
        return values

    @model_validator(mode="after")
    def cross_field_constraints(self) -> ExperimentConfig:
        method_names = [method.name for method in self.methods]
        if not method_names:
            raise ValueError("methods must contain at least one enabled method")
        if len(method_names) != len(set(method_names)):
            raise ValueError("methods must not contain duplicate names")
        task_names = [task.name for task in self.tasks]
        if len(task_names) != len(set(task_names)):
            raise ValueError("tasks must not contain duplicate names")
        temporal_methods = {"dtw", "lcss", "edr", "erp"}
        if temporal_methods.intersection(method_names) and not self.dataset.has_timestamps:
            # The geometric implementations may still use these names, but a
            # config explicitly declaring no timestamps must not enable a
            # temporal task without a clear override.
            for task in self.tasks:
                if task.config.get("use_timestamps", False):
                    raise ValueError(
                        "timestamp-dependent task requires dataset.has_timestamps=true"
                    )
        selection = self.dataset.selection
        if selection.query_ids and selection.database_ids:
            overlap = set(selection.query_ids).intersection(selection.database_ids)
            if overlap and not self.dataset.user_held_out:
                raise ValueError(
                    "query_ids and database_ids overlap; exclude self-matches explicitly"
                )
        for perturbation in self.perturbations:
            if not perturbation.severities:
                raise ValueError(f"perturbation {perturbation.name!r} must specify severities")
        return self

    @property
    def resolved_hash(self) -> str:
        from trajsimbench.config.validation import hash_resolved_config

        return hash_resolved_config(self)

    @property
    def config_hash(self) -> str:
        return self.resolved_hash
