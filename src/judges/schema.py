"""
Judge output schema and data structures.

Defines the structured format for LLM judge evaluations of agent trajectories.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class ErrorSeverity(Enum):
    """Severity levels for step errors."""

    CRITICAL = "critical"  # Blocks task completion
    MAJOR = "major"  # Significant impact on quality
    MINOR = "minor"  # Small issue, task still succeeds
    NONE = "none"  # No error


class JudgeMode(Enum):
    """Judge evaluation modes."""

    SINGLE_TRAJECTORY = "single_trajectory"
    BLINDED_PAIR = "blinded_pair"
    LABELED_PAIR = "labeled_pair"


class ErrorType(Enum):
    """Error type categories for localization."""

    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    PARAMETER = "parameter"
    DATA_REFERENCE = "data_reference"
    OTHER = "other"


class ParseStatus(Enum):
    """Parse status for judge responses."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class StepError:
    """
    Represents an error identified in a specific step.
    """

    step_index: int
    error_type: str  # e.g., "planning", "tool_selection", "parameter", "reasoning"
    severity: ErrorSeverity
    description: str
    impacts_task_success: bool  # Does this error prevent task completion?

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "step_index": self.step_index,
            "error_type": self.error_type,
            "severity": self.severity.value,
            "description": self.description,
            "impacts_task_success": self.impacts_task_success,
        }


@dataclass
class JudgeOutput:
    """
    Complete output from a judge evaluation.

    This aligns with the 5 metrics defined in the POC experiment config:
    - task_success: binary (0 or 1)
    - completeness: percentage (0-100)
    - efficiency: count of unnecessary tool calls
    - hallucination: binary (0 or 1)
    - sycophancy: binary (0 or 1) - if agent agreed with false user premises
    """

    trajectory_id: str
    judge_name: str
    model_id: str

    # Core metrics (from config)
    task_success: int  # 0 or 1
    completeness: float  # 0-100
    efficiency_errors: int  # Count of unnecessary/redundant calls
    hallucination: int  # 0 or 1
    sycophancy: int  # 0 or 1

    # Overall score (0-100, higher is better)
    overall_score: float

    # Detailed analysis
    step_errors: List[StepError] = field(default_factory=list)
    reasoning: str = ""

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    evaluation_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "trajectory_id": self.trajectory_id,
            "judge_name": self.judge_name,
            "model_id": self.model_id,
            "task_success": self.task_success,
            "completeness": self.completeness,
            "efficiency_errors": self.efficiency_errors,
            "hallucination": self.hallucination,
            "sycophancy": self.sycophancy,
            "overall_score": self.overall_score,
            "step_errors": [error.to_dict() for error in self.step_errors],
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "evaluation_time_ms": self.evaluation_time_ms,
            "tokens_used": self.tokens_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgeOutput":
        """Create from dictionary (e.g., from MongoDB)."""
        step_errors = [
            StepError(
                step_index=err["step_index"],
                error_type=err["error_type"],
                severity=ErrorSeverity(err["severity"]),
                description=err["description"],
                impacts_task_success=err["impacts_task_success"],
            )
            for err in data.get("step_errors", [])
        ]

        return cls(
            trajectory_id=data["trajectory_id"],
            judge_name=data["judge_name"],
            model_id=data["model_id"],
            task_success=data["task_success"],
            completeness=data["completeness"],
            efficiency_errors=data["efficiency_errors"],
            hallucination=data["hallucination"],
            sycophancy=data["sycophancy"],
            overall_score=data["overall_score"],
            step_errors=step_errors,
            reasoning=data.get("reasoning", ""),
            timestamp=data.get("timestamp", datetime.utcnow()),
            evaluation_time_ms=data.get("evaluation_time_ms"),
            tokens_used=data.get("tokens_used"),
        )


@dataclass
class EvaluationBatch:
    """
    Represents a batch of evaluations to process.
    """

    experiment_id: str
    perturbations: List[Dict[str, Any]]  # List of perturbation dicts
    judge_name: str
    batch_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvaluationResults:
    """
    Summary results from a judge evaluation run.
    """

    experiment_id: str
    judge_name: str
    total_evaluated: int
    successful: int
    failed: int
    total_time_seconds: float
    total_tokens: int
    average_score: float
    evaluation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# LLM Judge Output Schema Extensions
# =============================================================================


@dataclass
class JudgeConfig:
    """
    Configuration for a judge evaluation run.
    """

    prompt_template_id: str
    temperature: float
    seed: Optional[int] = None
    max_tokens: int = 2000
    sample_index: int = 0  # 0 for deterministic, 1-n for variance samples

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_template_id": self.prompt_template_id,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "sample_index": self.sample_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgeConfig":
        """Create from dictionary."""
        return cls(
            prompt_template_id=data["prompt_template_id"],
            temperature=data["temperature"],
            seed=data.get("seed"),
            max_tokens=data.get("max_tokens", 2000),
            sample_index=data.get("sample_index", 0),
        )


@dataclass
class InputView:
    """
    Information about the input view provided to the judge.
    """

    view_file: str
    trajectory_variant_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "view_file": self.view_file,
            "trajectory_variant_ids": self.trajectory_variant_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputView":
        """Create from dictionary."""
        return cls(
            view_file=data["view_file"],
            trajectory_variant_ids=data["trajectory_variant_ids"],
        )


@dataclass
class DetectionOutput:
    """
    Detection signals from judge evaluation.
    """

    overall_score: float  # 0-100
    error_detected: bool
    error_confidence: float  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "error_detected": self.error_detected,
            "error_confidence": self.error_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionOutput":
        """Create from dictionary."""
        return cls(
            overall_score=float(data["overall_score"]),
            error_detected=bool(data["error_detected"]),
            error_confidence=float(data["error_confidence"]),
        )


@dataclass
class LocalizationOutput:
    """
    Error localization output from judge evaluation.
    Only populated when error_detected is True.
    """

    predicted_error_step: Optional[int] = None  # display_step_index
    predicted_error_step_canonical: Optional[str] = None  # canonical_step_id
    localization_confidence: Optional[float] = None  # 0-1
    predicted_error_type: Optional[str] = None  # ErrorType value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_error_step": self.predicted_error_step,
            "predicted_error_step_canonical": self.predicted_error_step_canonical,
            "localization_confidence": self.localization_confidence,
            "predicted_error_type": self.predicted_error_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalizationOutput":
        """Create from dictionary."""
        return cls(
            predicted_error_step=data.get("predicted_error_step"),
            predicted_error_step_canonical=data.get("predicted_error_step_canonical"),
            localization_confidence=data.get("localization_confidence"),
            predicted_error_type=data.get("predicted_error_type"),
        )


@dataclass
class ImpactOutput:
    """
    Impact prediction output from judge evaluation.
    """

    predicted_impact_score: float  # 0-1
    predicted_failure_prob: float  # 0-1
    impact_explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_impact_score": self.predicted_impact_score,
            "predicted_failure_prob": self.predicted_failure_prob,
            "impact_explanation": self.impact_explanation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImpactOutput":
        """Create from dictionary."""
        return cls(
            predicted_impact_score=float(data["predicted_impact_score"]),
            predicted_failure_prob=float(data["predicted_failure_prob"]),
            impact_explanation=data.get("impact_explanation"),
        )


@dataclass
class PairComparisonOutput:
    """
    Pair comparison output for blinded_pair and labeled_pair modes.
    Only populated in pair evaluation modes.
    """

    error_trajectory: Optional[str] = None  # "A", "B", "neither", "both"
    preference: Optional[str] = None  # "A", "B", "tie"
    comparison_explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_trajectory": self.error_trajectory,
            "preference": self.preference,
            "comparison_explanation": self.comparison_explanation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PairComparisonOutput":
        """Create from dictionary."""
        if data is None:
            return None
        return cls(
            error_trajectory=data.get("error_trajectory"),
            preference=data.get("preference"),
            comparison_explanation=data.get("comparison_explanation"),
        )


@dataclass
class Section5JudgeOutput:
    """
    Complete judge output following the specified schema.

    This is the primary output format for exp_trajectory_sampling_v7.
    """

    # Identity
    judge_output_id: str
    evaluation_unit_id: str
    judge_model: str
    judge_mode: str  # JudgeMode value
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Configuration
    config: JudgeConfig = None

    # View Information
    input_view: InputView = None

    # Core Outputs
    detection: DetectionOutput = None
    localization: LocalizationOutput = None
    impact: ImpactOutput = None
    pair_comparison: Optional[PairComparisonOutput] = None

    # Raw Response
    raw_response: str = ""
    parse_status: str = "success"  # ParseStatus value
    parse_errors: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "judge_output_id": self.judge_output_id,
            "evaluation_unit_id": self.evaluation_unit_id,
            "judge_model": self.judge_model,
            "judge_mode": self.judge_mode,
            "created_at": (
                self.created_at.isoformat()
                if isinstance(self.created_at, datetime)
                else self.created_at
            ),
            "config": self.config.to_dict() if self.config else None,
            "input_view": self.input_view.to_dict() if self.input_view else None,
            "detection": self.detection.to_dict() if self.detection else None,
            "localization": self.localization.to_dict() if self.localization else None,
            "impact": self.impact.to_dict() if self.impact else None,
            "pair_comparison": (
                self.pair_comparison.to_dict() if self.pair_comparison else None
            ),
            "raw_response": self.raw_response,
            "parse_status": self.parse_status,
            "parse_errors": self.parse_errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Section5JudgeOutput":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()

        return cls(
            judge_output_id=data["judge_output_id"],
            evaluation_unit_id=data["evaluation_unit_id"],
            judge_model=data["judge_model"],
            judge_mode=data["judge_mode"],
            created_at=created_at,
            config=(
                JudgeConfig.from_dict(data["config"]) if data.get("config") else None
            ),
            input_view=(
                InputView.from_dict(data["input_view"])
                if data.get("input_view")
                else None
            ),
            detection=(
                DetectionOutput.from_dict(data["detection"])
                if data.get("detection")
                else None
            ),
            localization=(
                LocalizationOutput.from_dict(data["localization"])
                if data.get("localization")
                else None
            ),
            impact=(
                ImpactOutput.from_dict(data["impact"]) if data.get("impact") else None
            ),
            pair_comparison=PairComparisonOutput.from_dict(data.get("pair_comparison")),
            raw_response=data.get("raw_response", ""),
            parse_status=data.get("parse_status", "success"),
            parse_errors=data.get("parse_errors"),
        )


# =============================================================================
# Aggregation Schema
# =============================================================================


@dataclass
class AggregatedDetection:
    """
    Aggregated detection signals across multiple samples.
    """

    mean_overall_score: float
    std_overall_score: float
    error_detected_rate: float  # Fraction of samples detecting error
    mean_error_confidence: float
    std_error_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_overall_score": self.mean_overall_score,
            "std_overall_score": self.std_overall_score,
            "error_detected_rate": self.error_detected_rate,
            "mean_error_confidence": self.mean_error_confidence,
            "std_error_confidence": self.std_error_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedDetection":
        """Create from dictionary."""
        return cls(
            mean_overall_score=float(data["mean_overall_score"]),
            std_overall_score=float(data["std_overall_score"]),
            error_detected_rate=float(data["error_detected_rate"]),
            mean_error_confidence=float(data["mean_error_confidence"]),
            std_error_confidence=float(data["std_error_confidence"]),
        )


@dataclass
class AggregatedLocalization:
    """
    Aggregated localization signals across multiple samples.
    """

    modal_predicted_step: Optional[str] = None  # canonical_step_id
    step_agreement_rate: float = 0.0
    modal_predicted_type: Optional[str] = None
    type_agreement_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "modal_predicted_step": self.modal_predicted_step,
            "step_agreement_rate": self.step_agreement_rate,
            "modal_predicted_type": self.modal_predicted_type,
            "type_agreement_rate": self.type_agreement_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedLocalization":
        """Create from dictionary."""
        return cls(
            modal_predicted_step=data.get("modal_predicted_step"),
            step_agreement_rate=float(data.get("step_agreement_rate", 0.0)),
            modal_predicted_type=data.get("modal_predicted_type"),
            type_agreement_rate=float(data.get("type_agreement_rate", 0.0)),
        )


@dataclass
class AggregatedImpact:
    """
    Aggregated impact predictions across multiple samples.
    """

    mean_impact_score: float
    std_impact_score: float
    mean_failure_prob: float
    std_failure_prob: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_impact_score": self.mean_impact_score,
            "std_impact_score": self.std_impact_score,
            "mean_failure_prob": self.mean_failure_prob,
            "std_failure_prob": self.std_failure_prob,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedImpact":
        """Create from dictionary."""
        return cls(
            mean_impact_score=float(data["mean_impact_score"]),
            std_impact_score=float(data["std_impact_score"]),
            mean_failure_prob=float(data["mean_failure_prob"]),
            std_failure_prob=float(data["std_failure_prob"]),
        )


@dataclass
class AggregatedJudgeOutput:
    """
    Aggregated judge output across samples and/or judges.
    """

    evaluation_unit_id: str
    judge_model: Optional[str] = None  # None for cross-judge aggregation

    # Per-sample aggregation
    aggregated_detection: AggregatedDetection = None
    aggregated_localization: AggregatedLocalization = None
    aggregated_impact: AggregatedImpact = None

    # Cross-judge aggregation (when judge_model is None)
    cross_judge: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evaluation_unit_id": self.evaluation_unit_id,
            "judge_model": self.judge_model,
            "aggregated_detection": (
                self.aggregated_detection.to_dict()
                if self.aggregated_detection
                else None
            ),
            "aggregated_localization": (
                self.aggregated_localization.to_dict()
                if self.aggregated_localization
                else None
            ),
            "aggregated_impact": (
                self.aggregated_impact.to_dict() if self.aggregated_impact else None
            ),
            "cross_judge": self.cross_judge,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedJudgeOutput":
        """Create from dictionary."""
        return cls(
            evaluation_unit_id=data["evaluation_unit_id"],
            judge_model=data.get("judge_model"),
            aggregated_detection=(
                AggregatedDetection.from_dict(data["aggregated_detection"])
                if data.get("aggregated_detection")
                else None
            ),
            aggregated_localization=(
                AggregatedLocalization.from_dict(data["aggregated_localization"])
                if data.get("aggregated_localization")
                else None
            ),
            aggregated_impact=(
                AggregatedImpact.from_dict(data["aggregated_impact"])
                if data.get("aggregated_impact")
                else None
            ),
            cross_judge=data.get("cross_judge"),
        )
