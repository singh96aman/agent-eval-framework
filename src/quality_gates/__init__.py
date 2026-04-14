"""
Quality Gates Module.

Provides validation checks that must pass for data to proceed through
the pipeline. All gates use ONLY regex/parsing - NO LLM calls.

Usage:
    from src.quality_gates import GateRunner, get_gate

    # Create runner for a phase
    runner = GateRunner(phase="perturb")

    # Add gates
    runner.add_gate(get_gate("no_synthetic_markers"))
    runner.add_gate(get_gate("json_validity"))

    # Run checks
    report = runner.run(perturbations)
    if report.all_passed:
        print("All gates passed!")
    else:
        print(f"Failed: {report.fail_count} gates")
"""

from src.quality_gates.base import (
    BaseGate,
    GateResult,
    GateReport,
    GateRunner,
    GateStatus,
)

from src.quality_gates.pipeline_gates import (
    PIPELINE_GATES,
    get_gate,
    # Individual gates
    TrajectoryCountGate,
    GraderPassRateGate,
    NoSyntheticMarkersGate,
    NoStructuralCorruptionGate,
    JSONValidityGate,
    PositionDistributionGate,
    ClassDistributionGate,
    BlindingBalanceGate,
    LengthPreservationGate,
    OutcomeVarianceGate,
    MinStepsGate,
    MaxStepsGate,
    TaskSuccessGate,
    UniqueIDsGate,
)

from src.quality_gates.prompt_gates import (
    PROMPT_GATES,
    get_prompt_gate,
    PrimingDetectionGate,
    BlindingIntegrityGate,
    VocabularyAlignmentGate,
    SchemaAlignmentGate,
    NeutralityGate,
    OutputParsabilityGate,
)

from src.quality_gates.perturbation_gates import (
    PERTURBATION_GATES,
    get_perturbation_gate,
    GateNoSyntheticMarkers,
    GateNoStructuralCorruption,
    GateJSONValidity,
    GatePlaceboPreservesSemantics,
    GateLengthPreservation,
    GatePositionDistribution,
    GateNonPlaceboMeaningful,
)

__all__ = [
    # Base
    "BaseGate",
    "GateResult",
    "GateReport",
    "GateRunner",
    "GateStatus",
    # Registries
    "PIPELINE_GATES",
    "PROMPT_GATES",
    "PERTURBATION_GATES",
    # Factory functions
    "get_gate",
    "get_prompt_gate",
    "get_perturbation_gate",
    # Pipeline gates
    "TrajectoryCountGate",
    "GraderPassRateGate",
    "NoSyntheticMarkersGate",
    "NoStructuralCorruptionGate",
    "JSONValidityGate",
    "PositionDistributionGate",
    "ClassDistributionGate",
    "BlindingBalanceGate",
    "LengthPreservationGate",
    "OutcomeVarianceGate",
    "MinStepsGate",
    "MaxStepsGate",
    "TaskSuccessGate",
    "UniqueIDsGate",
    # Prompt gates
    "PrimingDetectionGate",
    "BlindingIntegrityGate",
    "VocabularyAlignmentGate",
    "SchemaAlignmentGate",
    "NeutralityGate",
    "OutputParsabilityGate",
    # Perturbation gates
    "GateNoSyntheticMarkers",
    "GateNoStructuralCorruption",
    "GateJSONValidity",
    "GatePlaceboPreservesSemantics",
    "GateLengthPreservation",
    "GatePositionDistribution",
    "GateNonPlaceboMeaningful",
]
