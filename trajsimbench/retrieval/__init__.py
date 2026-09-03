"""Retrieval primitives and relevance providers.

The module deliberately imports only the small CPU core.  Optional index
implementations are available from :mod:`trajsimbench.indexing`.
"""

from .exact import (
    TopKResult,
    chunked_exact_topk,
    chunked_top_k,
    exact_top_k,
    exact_topk,
    rank_candidates,
    retrieve_top_k,
)
from .ranking import (
    kendall_tau_b,
    pairwise_ordering_agreement,
    rank_biased_overlap,
    spearman_rho,
    top_k_jaccard,
)
from .relevance import (
    GradedOracleRelevance,
    RelevanceProvider,
    SameSourceRelevance,
    StaticRelevance,
)

__all__ = [
    "TopKResult",
    "chunked_top_k",
    "exact_top_k",
    "exact_topk",
    "chunked_exact_topk",
    "rank_candidates",
    "retrieve_top_k",
    "kendall_tau_b",
    "pairwise_ordering_agreement",
    "rank_biased_overlap",
    "spearman_rho",
    "top_k_jaccard",
    "RelevanceProvider",
    "SameSourceRelevance",
    "GradedOracleRelevance",
    "StaticRelevance",
]
