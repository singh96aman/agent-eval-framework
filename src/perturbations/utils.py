"""
Perturbation utilities - consolidated exports for storage, balancing, and schema.

This module re-exports the key utilities from internal modules for a cleaner API.
Internal implementation files (placebo, fine_grained, coarse_grained) are used
by generator_v2 but not directly exported here.
"""

# Schema - all perturbation data structures
from src.perturbations.schema import (
    PerturbationClass,
    PerturbationFamily,
    PerturbationType,
    PerturbationRecord,
    PerturbationIndex,
)

# Storage - saving and exporting perturbations
from src.perturbations.storage import (
    PerturbationStorage,
    PerturbationExporter,
    build_index_from_perturbations,
    group_by_benchmark,
    filter_valid_perturbations,
)

# Balancer - batch-level distribution balancing
from src.perturbations.balancer import (
    BatchDistribution,
    PerturbationBalancer,
    balance_perturbation_batch,
)

# Tool similarity - for near-neighbor tool selection
from src.perturbations.tool_similarity import (
    ToolSimilarityMatcher,
)

__all__ = [
    # Schema
    "PerturbationClass",
    "PerturbationFamily",
    "PerturbationType",
    "PerturbationRecord",
    "PerturbationIndex",
    # Storage
    "PerturbationStorage",
    "PerturbationExporter",
    "build_index_from_perturbations",
    "group_by_benchmark",
    "filter_valid_perturbations",
    # Balancer
    "BatchDistribution",
    "PerturbationBalancer",
    "balance_perturbation_batch",
    # Tool similarity
    "ToolSimilarityMatcher",
]
