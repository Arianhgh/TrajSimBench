"""Evaluation metrics for retrieval, agreement, diagnostics, and systems."""

from .agreement import agreement_distance, compute_agreement, evaluate_agreement
from .diagnostics import evaluate_triplets, similarity_fingerprint, triplet_accuracy
from .retrieval import (
    aggregate_query_metrics,
    evaluate_retrieval,
    hit_rate,
    hit_rate_at_k,
    mrr,
    mrr_score,
    ndcg,
    ndcg_at_k,
    precision,
    precision_at_k,
    recall,
    recall_at_k,
)
from .robustness import (
    hard_negative_gap,
    monotonicity_violation_rate,
    robustness_auc,
    robustness_curve,
)
from .statistics import bootstrap_ci, holm_correction, paired_permutation_test

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "ndcg_at_k",
    "mrr",
    "evaluate_retrieval",
    "aggregate_query_metrics",
    "recall",
    "precision",
    "hit_rate",
    "ndcg",
    "mrr_score",
    "evaluate_agreement",
    "compute_agreement",
    "agreement_distance",
    "triplet_accuracy",
    "evaluate_triplets",
    "similarity_fingerprint",
    "robustness_curve",
    "robustness_auc",
    "monotonicity_violation_rate",
    "hard_negative_gap",
    "bootstrap_ci",
    "paired_permutation_test",
    "holm_correction",
]
