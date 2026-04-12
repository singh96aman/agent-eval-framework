"""
Schema definitions for Section 5C: Outcome Evidence.

Defines OutcomeRecord, verifier outputs, metrics, and supporting dataclasses
for objective ground truth of perturbation impact.

The Outcome Evidence module provides:
- Ground truth impact for calibration metrics (6B)
- Objective success/failure for claim validation (6C)
- Baseline for comparing judge predictions against reality
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class EvidenceMethod(Enum):
    """Evidence collection method corresponding to replay tiers."""

    FULL_REPLAY = "full_replay"  # Tier 1
    DOWNSTREAM_REGENERATION = "downstream_regeneration"  # Tier 2
    FINAL_ANSWER_GRADING = "final_answer_grading"  # Tier 3


# === Verifier Outputs by Benchmark ===


@dataclass
class SWEBenchVerifierOutput:
    """
    Verifier output for SWE-bench trajectories.

    Contains test suite execution results.
    """

    test_results: Dict[str, int]  # total_tests, passed_tests, failed_tests, error_tests
    test_pass_rate: float
    failing_test_names: List[str]
    patch_applied: bool
    patch_diff: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_results": self.test_results,
            "test_pass_rate": self.test_pass_rate,
            "failing_test_names": self.failing_test_names,
            "patch_applied": self.patch_applied,
            "patch_diff": self.patch_diff,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SWEBenchVerifierOutput":
        """Create SWEBenchVerifierOutput from dictionary."""
        return cls(
            test_results=data["test_results"],
            test_pass_rate=data["test_pass_rate"],
            failing_test_names=data["failing_test_names"],
            patch_applied=data["patch_applied"],
            patch_diff=data.get("patch_diff"),
        )


@dataclass
class GAIAVerifierOutput:
    """
    Verifier output for GAIA trajectories.

    Contains exact match grading results.
    """

    expected_answer: str
    predicted_answer: str
    exact_match: bool
    normalized_expected: Optional[str] = None
    normalized_predicted: Optional[str] = None
    partial_match_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "expected_answer": self.expected_answer,
            "predicted_answer": self.predicted_answer,
            "exact_match": self.exact_match,
            "normalized_expected": self.normalized_expected,
            "normalized_predicted": self.normalized_predicted,
            "partial_match_score": self.partial_match_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAIAVerifierOutput":
        """Create GAIAVerifierOutput from dictionary."""
        return cls(
            expected_answer=data["expected_answer"],
            predicted_answer=data["predicted_answer"],
            exact_match=data["exact_match"],
            normalized_expected=data.get("normalized_expected"),
            normalized_predicted=data.get("normalized_predicted"),
            partial_match_score=data.get("partial_match_score", 0.0),
        )


@dataclass
class ToolBenchVerifierOutput:
    """
    Verifier output for ToolBench trajectories.

    Contains task completion heuristics.
    """

    task_completed: bool
    success_heuristic: float  # 0-1
    final_action: str
    api_errors_count: int
    reached_goal_state: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_completed": self.task_completed,
            "success_heuristic": self.success_heuristic,
            "final_action": self.final_action,
            "api_errors_count": self.api_errors_count,
            "reached_goal_state": self.reached_goal_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolBenchVerifierOutput":
        """Create ToolBenchVerifierOutput from dictionary."""
        return cls(
            task_completed=data["task_completed"],
            success_heuristic=data["success_heuristic"],
            final_action=data["final_action"],
            api_errors_count=data["api_errors_count"],
            reached_goal_state=data.get("reached_goal_state"),
        )


# === Outcome Data Structures ===


@dataclass
class BaselineOutcome:
    """
    Outcome data for the baseline (unperturbed) trajectory.
    """

    trajectory_variant_id: str
    outcome_score: float  # 0-1 normalized
    outcome_binary: bool  # pass/fail
    verifier_output: Dict[str, Any]  # Verifier-specific details

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trajectory_variant_id": self.trajectory_variant_id,
            "outcome_score": self.outcome_score,
            "outcome_binary": self.outcome_binary,
            "verifier_output": self.verifier_output,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineOutcome":
        """Create BaselineOutcome from dictionary."""
        return cls(
            trajectory_variant_id=data["trajectory_variant_id"],
            outcome_score=data["outcome_score"],
            outcome_binary=data["outcome_binary"],
            verifier_output=data["verifier_output"],
        )


@dataclass
class PerturbedOutcome:
    """
    Outcome data for the perturbed trajectory.
    """

    trajectory_variant_id: str
    outcome_score: float  # 0-1 normalized
    outcome_binary: bool  # pass/fail
    verifier_output: Dict[str, Any]  # Verifier-specific details

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trajectory_variant_id": self.trajectory_variant_id,
            "outcome_score": self.outcome_score,
            "outcome_binary": self.outcome_binary,
            "verifier_output": self.verifier_output,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerturbedOutcome":
        """Create PerturbedOutcome from dictionary."""
        return cls(
            trajectory_variant_id=data["trajectory_variant_id"],
            outcome_score=data["outcome_score"],
            outcome_binary=data["outcome_binary"],
            verifier_output=data["verifier_output"],
        )


@dataclass
class OutcomeMetrics:
    """
    Core outcome metrics computed from baseline and perturbed outcomes.
    """

    outcome_degradation: float  # OD = baseline_score - perturbed_score
    outcome_degradation_binary: int  # 1, 0, or -1
    propagation_depth: Optional[int] = None  # Tier 1-2 only
    recovery_cost: Optional[Dict[str, Any]] = None  # Tier 1 only

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "outcome_degradation": self.outcome_degradation,
            "outcome_degradation_binary": self.outcome_degradation_binary,
            "propagation_depth": self.propagation_depth,
            "recovery_cost": self.recovery_cost,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutcomeMetrics":
        """Create OutcomeMetrics from dictionary."""
        return cls(
            outcome_degradation=data["outcome_degradation"],
            outcome_degradation_binary=data["outcome_degradation_binary"],
            propagation_depth=data.get("propagation_depth"),
            recovery_cost=data.get("recovery_cost"),
        )


@dataclass
class PropagationTrace:
    """
    Detailed propagation analysis for Tier 1-2 evidence.

    Tracks which steps diverged between baseline and perturbed trajectories.
    """

    first_divergent_step: Optional[str]  # canonical_step_id
    divergent_steps: List[str]  # canonical_step_ids
    convergent_steps: List[str]  # canonical_step_ids

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "first_divergent_step": self.first_divergent_step,
            "divergent_steps": self.divergent_steps,
            "convergent_steps": self.convergent_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PropagationTrace":
        """Create PropagationTrace from dictionary."""
        return cls(
            first_divergent_step=data.get("first_divergent_step"),
            divergent_steps=data.get("divergent_steps", []),
            convergent_steps=data.get("convergent_steps", []),
        )


@dataclass
class RecoveryDetails:
    """
    Recovery cost details for Tier 1 (full replay) evidence.

    Measures additional resources required to recover from perturbation.
    """

    extra_steps: Optional[int] = None
    extra_tool_calls: Optional[int] = None
    retries: Optional[int] = None
    recovery_successful: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "extra_steps": self.extra_steps,
            "extra_tool_calls": self.extra_tool_calls,
            "retries": self.retries,
            "recovery_successful": self.recovery_successful,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecoveryDetails":
        """Create RecoveryDetails from dictionary."""
        return cls(
            extra_steps=data.get("extra_steps"),
            extra_tool_calls=data.get("extra_tool_calls"),
            retries=data.get("retries"),
            recovery_successful=data.get("recovery_successful"),
        )


@dataclass
class ExecutionMetadata:
    """
    Metadata about the evidence collection execution.
    """

    method: str  # e.g., 'swebench_harness', 'gaia_exact_match'
    duration_seconds: float
    errors: Optional[List[str]] = None
    logs: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
            "logs": self.logs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionMetadata":
        """Create ExecutionMetadata from dictionary."""
        return cls(
            method=data["method"],
            duration_seconds=data["duration_seconds"],
            errors=data.get("errors"),
            logs=data.get("logs"),
        )


# === Main Outcome Record ===


@dataclass
class OutcomeRecord:
    """
    Complete outcome evidence record for an evaluation unit.

    Per Section 5C.4, this is the main output schema that combines:
    - Baseline and perturbed outcomes
    - Core metrics (OD, PD, RC)
    - Propagation analysis (Tier 1-2)
    - Recovery details (Tier 1)
    - Execution metadata
    """

    # === Identity ===
    outcome_id: str
    evaluation_unit_id: str
    created_at: str  # ISO 8601

    # === Tier Information ===
    replay_tier: int  # 1, 2, or 3
    evidence_method: str  # EvidenceMethod value

    # === Outcomes ===
    baseline: BaselineOutcome
    perturbed: PerturbedOutcome

    # === Core Metrics ===
    metrics: OutcomeMetrics

    # === Tier-specific Details ===
    propagation_trace: Optional[PropagationTrace] = None  # Tier 1-2
    recovery_details: Optional[RecoveryDetails] = None  # Tier 1

    # === Execution Metadata ===
    execution: Optional[ExecutionMetadata] = None

    @classmethod
    def create(
        cls,
        evaluation_unit_id: str,
        replay_tier: int,
        evidence_method: EvidenceMethod,
        baseline: BaselineOutcome,
        perturbed: PerturbedOutcome,
        metrics: OutcomeMetrics,
        propagation_trace: Optional[PropagationTrace] = None,
        recovery_details: Optional[RecoveryDetails] = None,
        execution: Optional[ExecutionMetadata] = None,
    ) -> "OutcomeRecord":
        """Factory method to create a new OutcomeRecord with auto-generated ID and timestamp."""
        return cls(
            outcome_id=f"outcome_{uuid.uuid4().hex[:12]}",
            evaluation_unit_id=evaluation_unit_id,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            replay_tier=replay_tier,
            evidence_method=evidence_method.value,
            baseline=baseline,
            perturbed=perturbed,
            metrics=metrics,
            propagation_trace=propagation_trace,
            recovery_details=recovery_details,
            execution=execution,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "outcome_id": self.outcome_id,
            "evaluation_unit_id": self.evaluation_unit_id,
            "created_at": self.created_at,
            "replay_tier": self.replay_tier,
            "evidence_method": self.evidence_method,
            "baseline": self.baseline.to_dict(),
            "perturbed": self.perturbed.to_dict(),
            "metrics": self.metrics.to_dict(),
            "propagation_trace": (
                self.propagation_trace.to_dict() if self.propagation_trace else None
            ),
            "recovery_details": (
                self.recovery_details.to_dict() if self.recovery_details else None
            ),
            "execution": self.execution.to_dict() if self.execution else None,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutcomeRecord":
        """Create OutcomeRecord from dictionary."""
        return cls(
            outcome_id=data["outcome_id"],
            evaluation_unit_id=data["evaluation_unit_id"],
            created_at=data["created_at"],
            replay_tier=data["replay_tier"],
            evidence_method=data["evidence_method"],
            baseline=BaselineOutcome.from_dict(data["baseline"]),
            perturbed=PerturbedOutcome.from_dict(data["perturbed"]),
            metrics=OutcomeMetrics.from_dict(data["metrics"]),
            propagation_trace=(
                PropagationTrace.from_dict(data["propagation_trace"])
                if data.get("propagation_trace")
                else None
            ),
            recovery_details=(
                RecoveryDetails.from_dict(data["recovery_details"])
                if data.get("recovery_details")
                else None
            ),
            execution=(
                ExecutionMetadata.from_dict(data["execution"])
                if data.get("execution")
                else None
            ),
        )

    def get_true_impact(self) -> int:
        """
        Derive true impact level from outcome degradation.

        Returns:
            0: No impact (OD = 0)
            1: Minor (0 < OD <= 0.25)
            2: Moderate (0.25 < OD <= 0.5)
            3: Critical (OD > 0.5)
        """
        od = self.metrics.outcome_degradation
        if od == 0:
            return 0
        elif od <= 0.25:
            return 1
        elif od <= 0.5:
            return 2
        else:
            return 3

    def is_negative_od(self) -> bool:
        """Check if this outcome has negative OD (perturbed better than baseline)."""
        return self.metrics.outcome_degradation < 0

    def get_evidence_method_enum(self) -> EvidenceMethod:
        """Get evidence method as enum."""
        return EvidenceMethod(self.evidence_method)
