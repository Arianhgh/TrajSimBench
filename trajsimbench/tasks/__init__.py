"""Deterministic benchmark task artifacts."""

from .base import (
    TaskArtifact,
    TaskConstructionError,
    TaskQualityReport,
    dataset_ids,
    get_trajectory,
)
from .diagnostics import DiagnosticTaskGenerator, generate_diagnostic_triplets
from .equivalence import EquivalenceTaskGenerator, generate_equivalence_tasks
from .generalization import (
    GeneralizationMode,
    build_generalization_task,
    validate_generalization_partitions,
)
from .negatives import NegativeTaskGenerator, generate_negative_task
from .oracle import OracleApproximationTaskGenerator, generate_oracle_task
from .retrieval import RetrievalTaskGenerator, generate_retrieval_task
from .systems import SystemsWorkload, build_systems_task

__all__ = [
    "TaskArtifact",
    "TaskConstructionError",
    "TaskQualityReport",
    "dataset_ids",
    "get_trajectory",
    "OracleApproximationTaskGenerator",
    "generate_oracle_task",
    "EquivalenceTaskGenerator",
    "generate_equivalence_tasks",
    "DiagnosticTaskGenerator",
    "generate_diagnostic_triplets",
    "RetrievalTaskGenerator",
    "generate_retrieval_task",
    "NegativeTaskGenerator",
    "generate_negative_task",
    "GeneralizationMode",
    "build_generalization_task",
    "validate_generalization_partitions",
    "SystemsWorkload",
    "build_systems_task",
]
