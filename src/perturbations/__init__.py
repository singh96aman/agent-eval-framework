"""
Section 3: Controlled Perturbations Module.

Main entry points:
- generator_v2: PerturbationGeneratorV2, generate_perturbations_for_batch
- qc: PerturbationQC for quality control
- utils: Schema, storage, balancing utilities

Usage:
    from src.perturbations import (
        PerturbationGeneratorV2,
        generate_perturbations_for_batch,
        balance_perturbation_batch,
        PerturbationStorage,
        PerturbationQC,
    )
"""

# Main generator
from src.perturbations.generator_v2 import (
    PerturbationGeneratorV2,
    generate_perturbations_for_batch,
)

# Quality control
from src.perturbations.qc import (
    PerturbationQC,
    SchemaValidator,
    DiffValidator,
    ClassFamilyValidator,
)

# Utilities (schema, storage, balancer)
from src.perturbations.utils import (
    # Schema
    PerturbationClass,
    PerturbationFamily,
    PerturbationType,
    PerturbationRecord,
    PerturbationIndex,
    # Storage
    PerturbationStorage,
    PerturbationExporter,
    build_index_from_perturbations,
    group_by_benchmark,
    filter_valid_perturbations,
    # Balancer
    BatchDistribution,
    PerturbationBalancer,
    balance_perturbation_batch,
    # Tool similarity
    ToolSimilarityMatcher,
)

__all__ = [
    # Generator
    "PerturbationGeneratorV2",
    "generate_perturbations_for_batch",
    # QC
    "PerturbationQC",
    "SchemaValidator",
    "DiffValidator",
    "ClassFamilyValidator",
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
