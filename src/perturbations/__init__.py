"""
Perturbation generation framework for agent trajectories.

This module generates realistic errors in agent trajectories to test
judge calibration to error criticality.
"""

from .generator import PerturbationGenerator
from .strategies import (
    PlanningErrorStrategy,
    ToolSelectionErrorStrategy,
    ParameterErrorStrategy,
)
from .tool_similarity import ToolSimilarityMatcher
from .qc import (
    PerturbationQC,
    run_qc,
    run_qc_batch,
    get_qc_statistics,
    SchemaValidator,
    DiffValidator,
    CoherenceValidator,
    ClassFamilyValidator,
    ImpactDeriver,
    ValidationResult,
)
from .coarse_grained import (
    BaseCoarseGrainedGenerator,
    SkippedPrerequisiteGenerator,
    WrongPlanGenerator,
    FalseTerminalGenerator,
    PrematureTerminationGenerator,
    WrongToolFamilyGenerator,
    get_coarse_grained_generator,
    get_all_coarse_grained_generators,
)
from .fine_grained import (
    BaseFineGrainedGenerator,
    DataReferenceGenerator,
    ParameterGenerator,
    ToolSelectionNearNeighborGenerator,
    get_fine_grained_generator,
    generate_data_reference_perturbation,
    generate_parameter_perturbation,
    generate_tool_selection_perturbation,
)

__all__ = [
    # Original exports
    "PerturbationGenerator",
    "PlanningErrorStrategy",
    "ToolSelectionErrorStrategy",
    "ParameterErrorStrategy",
    "ToolSimilarityMatcher",
    # QC Pipeline
    "PerturbationQC",
    "run_qc",
    "run_qc_batch",
    "get_qc_statistics",
    "SchemaValidator",
    "DiffValidator",
    "CoherenceValidator",
    "ClassFamilyValidator",
    "ImpactDeriver",
    "ValidationResult",
    # Coarse-grained generators
    "BaseCoarseGrainedGenerator",
    "SkippedPrerequisiteGenerator",
    "WrongPlanGenerator",
    "FalseTerminalGenerator",
    "PrematureTerminationGenerator",
    "WrongToolFamilyGenerator",
    "get_coarse_grained_generator",
    "get_all_coarse_grained_generators",
    # Fine-grained generators (Phase 2B)
    "BaseFineGrainedGenerator",
    "DataReferenceGenerator",
    "ParameterGenerator",
    "ToolSelectionNearNeighborGenerator",
    "get_fine_grained_generator",
    "generate_data_reference_perturbation",
    "generate_parameter_perturbation",
    "generate_tool_selection_perturbation",
]
