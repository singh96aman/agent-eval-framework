"""
Schema definitions for Section 6 Analysis Results.

Defines AnalysisResult and supporting dataclasses for per-unit analysis.
One document per (evaluation_unit_id, judge_model) pair.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid


@dataclass
class GroundTruth:
    """
    Ground truth data combined from evaluation unit, outcome evidence, and human labels.
    """

    # From evaluation unit derived_cache
    perturbation_class: str  # placebo | fine_grained | coarse_grained
    perturbation_family: (
        str  # data_reference | parameter | tool_selection | structural | terminal_flag
    )
    perturbation_type: str
    target_step_canonical_id: str
    expected_impact: int  # 0-3
    expected_detectability: int  # 0-2
    benchmark: str

    # From outcome evidence (5C)
    outcome_degradation: Optional[float]  # OD = baseline - perturbed
    true_impact_level: Optional[int]  # 0-3 derived from OD
    baseline_outcome_binary: Optional[bool]
    perturbed_outcome_binary: Optional[bool]

    # From human labels (5A) - if available
    human_error_detected: Optional[bool] = None
    human_error_step_id: Optional[str] = None
    human_impact_tier: Optional[float] = None
    human_error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "perturbation_class": self.perturbation_class,
            "perturbation_family": self.perturbation_family,
            "perturbation_type": self.perturbation_type,
            "target_step_canonical_id": self.target_step_canonical_id,
            "expected_impact": self.expected_impact,
            "expected_detectability": self.expected_detectability,
            "benchmark": self.benchmark,
            "outcome_degradation": self.outcome_degradation,
            "true_impact_level": self.true_impact_level,
            "baseline_outcome_binary": self.baseline_outcome_binary,
            "perturbed_outcome_binary": self.perturbed_outcome_binary,
            "human_error_detected": self.human_error_detected,
            "human_error_step_id": self.human_error_step_id,
            "human_impact_tier": self.human_impact_tier,
            "human_error_type": self.human_error_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruth":
        """Create from dictionary."""
        return cls(
            perturbation_class=data["perturbation_class"],
            perturbation_family=data["perturbation_family"],
            perturbation_type=data["perturbation_type"],
            target_step_canonical_id=data["target_step_canonical_id"],
            expected_impact=data["expected_impact"],
            expected_detectability=data["expected_detectability"],
            benchmark=data["benchmark"],
            outcome_degradation=data.get("outcome_degradation"),
            true_impact_level=data.get("true_impact_level"),
            baseline_outcome_binary=data.get("baseline_outcome_binary"),
            perturbed_outcome_binary=data.get("perturbed_outcome_binary"),
            human_error_detected=data.get("human_error_detected"),
            human_error_step_id=data.get("human_error_step_id"),
            human_impact_tier=data.get("human_impact_tier"),
            human_error_type=data.get("human_error_type"),
        )


@dataclass
class JudgeOutputSummary:
    """
    Summary of judge output relevant for analysis.
    """

    error_detected: bool
    error_confidence: float
    predicted_step_canonical_id: Optional[str]
    predicted_error_type: Optional[str]
    localization_confidence: Optional[float]
    predicted_impact_score: float
    predicted_failure_prob: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "error_detected": self.error_detected,
            "error_confidence": self.error_confidence,
            "predicted_step_canonical_id": self.predicted_step_canonical_id,
            "predicted_error_type": self.predicted_error_type,
            "localization_confidence": self.localization_confidence,
            "predicted_impact_score": self.predicted_impact_score,
            "predicted_failure_prob": self.predicted_failure_prob,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgeOutputSummary":
        """Create from dictionary."""
        return cls(
            error_detected=data["error_detected"],
            error_confidence=data["error_confidence"],
            predicted_step_canonical_id=data.get("predicted_step_canonical_id"),
            predicted_error_type=data.get("predicted_error_type"),
            localization_confidence=data.get("localization_confidence"),
            predicted_impact_score=data["predicted_impact_score"],
            predicted_failure_prob=data["predicted_failure_prob"],
        )


@dataclass
class DetectionEval:
    """
    Detection evaluation results (6A metrics at per-unit level).
    """

    # Binary classification results
    detection_correct: (
        bool  # True if (detected AND non-placebo) OR (not detected AND placebo)
    )
    is_true_positive: bool  # detected AND non-placebo
    is_false_positive: bool  # detected AND placebo
    is_true_negative: bool  # not detected AND placebo
    is_false_negative: bool  # not detected AND non-placebo

    # Localization (only if detected)
    localization_correct: Optional[bool] = None  # predicted_step == target_step
    localization_distance: Optional[int] = None  # |predicted_idx - target_idx|
    localization_near: Optional[bool] = None  # distance <= 1

    # Type identification (only if detected)
    type_correct: Optional[bool] = None  # predicted_type matches perturbation_family

    # Critical error detection
    is_critical_detected: Optional[bool] = None  # detected AND expected_impact == 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "detection_correct": self.detection_correct,
            "is_true_positive": self.is_true_positive,
            "is_false_positive": self.is_false_positive,
            "is_true_negative": self.is_true_negative,
            "is_false_negative": self.is_false_negative,
            "localization_correct": self.localization_correct,
            "localization_distance": self.localization_distance,
            "localization_near": self.localization_near,
            "type_correct": self.type_correct,
            "is_critical_detected": self.is_critical_detected,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionEval":
        """Create from dictionary."""
        return cls(
            detection_correct=data["detection_correct"],
            is_true_positive=data["is_true_positive"],
            is_false_positive=data["is_false_positive"],
            is_true_negative=data["is_true_negative"],
            is_false_negative=data["is_false_negative"],
            localization_correct=data.get("localization_correct"),
            localization_distance=data.get("localization_distance"),
            localization_near=data.get("localization_near"),
            type_correct=data.get("type_correct"),
            is_critical_detected=data.get("is_critical_detected"),
        )


@dataclass
class CalibrationEval:
    """
    Calibration evaluation results (6B metrics at per-unit level).
    """

    # Consequence Calibration Error
    cce: Optional[float]  # predicted_impact_score - normalized_OD (signed error)
    abs_cce: Optional[float]  # |cce|

    # Over/Under Reaction
    over_reaction: Optional[bool]  # predicted > 0.5 AND true_impact <= 1
    under_reaction: Optional[bool]  # predicted < 0.5 AND true_impact == 3

    # Failure prediction
    failure_predicted: bool  # predicted_failure_prob > 0.5
    failure_actual: Optional[bool]  # perturbed_outcome_binary == False
    failure_correct: Optional[bool]  # failure_predicted == failure_actual

    # Impact tier prediction
    impact_tier_predicted: int  # 0-3 from predicted_impact_score
    impact_tier_error: Optional[int]  # |predicted_tier - true_tier|

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "cce": self.cce,
            "abs_cce": self.abs_cce,
            "over_reaction": self.over_reaction,
            "under_reaction": self.under_reaction,
            "failure_predicted": self.failure_predicted,
            "failure_actual": self.failure_actual,
            "failure_correct": self.failure_correct,
            "impact_tier_predicted": self.impact_tier_predicted,
            "impact_tier_error": self.impact_tier_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationEval":
        """Create from dictionary."""
        return cls(
            cce=data.get("cce"),
            abs_cce=data.get("abs_cce"),
            over_reaction=data.get("over_reaction"),
            under_reaction=data.get("under_reaction"),
            failure_predicted=data["failure_predicted"],
            failure_actual=data.get("failure_actual"),
            failure_correct=data.get("failure_correct"),
            impact_tier_predicted=data["impact_tier_predicted"],
            impact_tier_error=data.get("impact_tier_error"),
        )


@dataclass
class HumanComparison:
    """
    Comparison between judge output and human labels.
    Only populated when human labels are available.
    """

    detection_agrees: Optional[bool]  # judge error_detected == human error_detected
    localization_agrees: Optional[bool]  # judge step == human step
    type_agrees: Optional[bool]  # judge type == human type
    impact_tier_diff: Optional[float]  # judge tier - human tier

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "detection_agrees": self.detection_agrees,
            "localization_agrees": self.localization_agrees,
            "type_agrees": self.type_agrees,
            "impact_tier_diff": self.impact_tier_diff,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanComparison":
        """Create from dictionary."""
        return cls(
            detection_agrees=data.get("detection_agrees"),
            localization_agrees=data.get("localization_agrees"),
            type_agrees=data.get("type_agrees"),
            impact_tier_diff=data.get("impact_tier_diff"),
        )


@dataclass
class AnalysisResult:
    """
    Complete analysis result for one (evaluation_unit_id, judge_model) pair.

    This is the main output schema stored in MongoDB analysis_results collection.
    """

    # Identity
    analysis_id: str
    experiment_id: str
    evaluation_unit_id: str
    judge_model: str
    created_at: str  # ISO 8601

    # Components
    ground_truth: GroundTruth
    judge_output: JudgeOutputSummary
    detection_eval: DetectionEval
    calibration_eval: CalibrationEval
    human_comparison: Optional[HumanComparison]

    @classmethod
    def create(
        cls,
        experiment_id: str,
        evaluation_unit_id: str,
        judge_model: str,
        ground_truth: GroundTruth,
        judge_output: JudgeOutputSummary,
        detection_eval: DetectionEval,
        calibration_eval: CalibrationEval,
        human_comparison: Optional[HumanComparison] = None,
    ) -> "AnalysisResult":
        """Factory method with auto-generated ID and timestamp."""
        return cls(
            analysis_id=f"analysis_{uuid.uuid4().hex[:12]}",
            experiment_id=experiment_id,
            evaluation_unit_id=evaluation_unit_id,
            judge_model=judge_model,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            ground_truth=ground_truth,
            judge_output=judge_output,
            detection_eval=detection_eval,
            calibration_eval=calibration_eval,
            human_comparison=human_comparison,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "analysis_id": self.analysis_id,
            "experiment_id": self.experiment_id,
            "evaluation_unit_id": self.evaluation_unit_id,
            "judge_model": self.judge_model,
            "created_at": self.created_at,
            "ground_truth": self.ground_truth.to_dict(),
            "judge_output": self.judge_output.to_dict(),
            "detection_eval": self.detection_eval.to_dict(),
            "calibration_eval": self.calibration_eval.to_dict(),
            "human_comparison": (
                self.human_comparison.to_dict() if self.human_comparison else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Create from dictionary."""
        human_comparison = None
        if data.get("human_comparison"):
            human_comparison = HumanComparison.from_dict(data["human_comparison"])

        return cls(
            analysis_id=data["analysis_id"],
            experiment_id=data["experiment_id"],
            evaluation_unit_id=data["evaluation_unit_id"],
            judge_model=data["judge_model"],
            created_at=data["created_at"],
            ground_truth=GroundTruth.from_dict(data["ground_truth"]),
            judge_output=JudgeOutputSummary.from_dict(data["judge_output"]),
            detection_eval=DetectionEval.from_dict(data["detection_eval"]),
            calibration_eval=CalibrationEval.from_dict(data["calibration_eval"]),
            human_comparison=human_comparison,
        )


# === Utility Functions ===


def derive_impact_level(outcome_degradation: Optional[float]) -> Optional[int]:
    """
    Derive impact level (0-3) from outcome degradation.

    Args:
        outcome_degradation: OD value (typically -1 to 1)

    Returns:
        0: No impact (OD <= 0)
        1: Minor (0 < OD <= 0.25)
        2: Moderate (0.25 < OD <= 0.5)
        3: Critical (OD > 0.5)
        None: If input is None
    """
    if outcome_degradation is None:
        return None
    if outcome_degradation <= 0:
        return 0
    elif outcome_degradation <= 0.25:
        return 1
    elif outcome_degradation <= 0.5:
        return 2
    else:
        return 3


def impact_score_to_tier(impact_score: float) -> int:
    """
    Convert predicted impact score (0-1) to tier (0-3).

    Args:
        impact_score: Predicted impact score from judge (0-1)

    Returns:
        Impact tier 0-3
    """
    if impact_score <= 0.25:
        return 0
    elif impact_score <= 0.5:
        return 1
    elif impact_score <= 0.75:
        return 2
    else:
        return 3


def map_error_type_to_family(predicted_type: Optional[str]) -> Optional[str]:
    """
    Map judge's predicted error type to perturbation family.

    Args:
        predicted_type: Judge's prediction (planning, tool_selection, parameter, data_reference, other)

    Returns:
        Mapped perturbation family or None
    """
    if predicted_type is None:
        return None

    # Direct mappings
    type_to_family = {
        "planning": "structural",
        "tool_selection": "tool_selection",
        "parameter": "parameter",
        "data_reference": "data_reference",
        "other": None,  # Cannot match
    }

    return type_to_family.get(predicted_type.lower(), None)
