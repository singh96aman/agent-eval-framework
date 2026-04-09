"""
Evaluation module for Section 4: Evaluation Unit.

This module provides schema definitions and utilities for creating
and managing evaluation units - the atomic unit of analysis in this study.
"""

from src.evaluation.blinding import (
    generate_blinding_assignment,
    generate_blinding_key,
    get_blinded_trajectory_order,
    verify_balance,
)
from src.evaluation.derived_cache import (
    build_derived_cache,
    rebuild_derived_cache,
    verify_cache_consistency,
)
from src.evaluation.index import (
    build_evaluation_unit_index,
    save_index,
)
from src.evaluation.storage import (
    create_data_directories,
    export_evaluation_units_to_json,
    load_evaluation_units_from_mongodb,
    save_evaluation_units_to_mongodb,
)
from src.evaluation.unit_assembler import (
    assemble_all_units,
    assemble_evaluation_unit,
    migrate_step_identity,
    reconstruct_baseline,
)
from src.evaluation.schema import (
    BaselineData,
    BlindingAssignment,
    DerivedCache,
    EvaluationCapabilities,
    EvaluationUnit,
    PerturbedData,
)
from src.evaluation.sampling import (
    check_coverage_minimums,
    get_default_coverage_config,
    get_sampling_report,
    sample_human_evaluation_set,
)

__all__ = [
    # Schema classes
    "BaselineData",
    "BlindingAssignment",
    "DerivedCache",
    "EvaluationCapabilities",
    "EvaluationUnit",
    "PerturbedData",
    # Blinding functions
    "generate_blinding_assignment",
    "generate_blinding_key",
    "get_blinded_trajectory_order",
    "verify_balance",
    # Derived cache functions
    "build_derived_cache",
    "rebuild_derived_cache",
    "verify_cache_consistency",
    # Index functions
    "build_evaluation_unit_index",
    "save_index",
    # Storage functions
    "create_data_directories",
    "export_evaluation_units_to_json",
    "load_evaluation_units_from_mongodb",
    "save_evaluation_units_to_mongodb",
    # Unit assembler functions
    "assemble_all_units",
    "assemble_evaluation_unit",
    "migrate_step_identity",
    "reconstruct_baseline",
    # Sampling functions
    "check_coverage_minimums",
    "get_default_coverage_config",
    "get_sampling_report",
    "sample_human_evaluation_set",
]
