"""
Section 5C: Outcome Evidence Module.

Provides objective ground truth for the actual impact of perturbations.
By grading trajectories, we measure the real-world consequence of errors -
independent of human or judge perception.

Usage:
    from src.outcome_evidence import (
        OutcomeRecord,
        compute_outcome_degradation,
        get_grader,
    )

    # Grade a trajectory
    grader = get_grader("gaia")
    result = grader.grade(trajectory)

    # Compute outcome degradation
    od = compute_outcome_degradation(baseline_score=1.0, perturbed_score=0.5)

    # Create outcome record
    record = OutcomeRecord.create(
        evaluation_unit_id="eval::gaia_122::001",
        replay_tier=3,
        evidence_method=EvidenceMethod.FINAL_ANSWER_GRADING,
        baseline=baseline_outcome,
        perturbed=perturbed_outcome,
        metrics=outcome_metrics,
    )
"""

# Schema classes
from .schema import (
    EvidenceMethod,
    SWEBenchVerifierOutput,
    GAIAVerifierOutput,
    ToolBenchVerifierOutput,
    BaselineOutcome,
    PerturbedOutcome,
    OutcomeMetrics,
    PropagationTrace,
    RecoveryDetails,
    ExecutionMetadata,
    OutcomeRecord,
)

# Metrics functions
from .metrics import (
    compute_outcome_degradation,
    compute_outcome_degradation_binary,
    compute_propagation_depth,
    compute_recovery_cost,
    derive_true_impact,
    categorize_od,
    compare_to_expected_impact,
)

# Tier 3 grading
from .tier_3 import (
    BaseGrader,
    ExactMatchGrader,
    FuzzyMatchGrader,
    HeuristicGrader,
    LLMGrader,
    get_grader,
    GradingResult,
)

# Storage functions
from .storage import (
    create_outcome_directories,
    save_outcome_evidence,
    load_outcome_evidence,
    save_tier_results,
    load_tier_results,
    save_tier_assignments,
    load_tier_assignments,
    save_execution_log,
    save_logs_for_unit,
    save_step_outputs_for_unit,
)

__all__ = [
    # Schema
    "EvidenceMethod",
    "SWEBenchVerifierOutput",
    "GAIAVerifierOutput",
    "ToolBenchVerifierOutput",
    "BaselineOutcome",
    "PerturbedOutcome",
    "OutcomeMetrics",
    "PropagationTrace",
    "RecoveryDetails",
    "ExecutionMetadata",
    "OutcomeRecord",
    # Metrics
    "compute_outcome_degradation",
    "compute_outcome_degradation_binary",
    "compute_propagation_depth",
    "compute_recovery_cost",
    "derive_true_impact",
    "categorize_od",
    "compare_to_expected_impact",
    # Tier 3 Grading
    "BaseGrader",
    "ExactMatchGrader",
    "FuzzyMatchGrader",
    "HeuristicGrader",
    "LLMGrader",
    "get_grader",
    "GradingResult",
    # Storage
    "create_outcome_directories",
    "save_outcome_evidence",
    "load_outcome_evidence",
    "save_tier_results",
    "load_tier_results",
    "save_tier_assignments",
    "load_tier_assignments",
    "save_execution_log",
    "save_logs_for_unit",
    "save_step_outputs_for_unit",
]
