"""
Schema definitions for Section 5A: Human Labels.

Defines AnnotationRecord, AggregatedLabel, and supporting dataclasses
for ground-truth human annotations per 5A_Requirements_HumanLabels.MD.

Human labels provide the ground-truth reference for evaluating LLM judge
performance. Trained annotators assess detectability, localization, error type,
impact, and propagation across a stratified sample of evaluation units.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class AnnotationMode(Enum):
    """Annotation task modes corresponding to Section 4's view types."""

    # Label collection modes
    DETECTABILITY = "detectability"
    CONSEQUENCE = "consequence"
    PREFERENCE = "preference"
    # Display modes (how trajectories are shown)
    SINGLE_TRAJECTORY = "single_trajectory"
    BLINDED_PAIR = "blinded_pair"
    LABELED_PAIR = "labeled_pair"


class ErrorTrajectory(Enum):
    """Which trajectory contains the error."""

    A = "A"
    B = "B"
    NEITHER = "neither"
    BOTH = "both"


class ErrorType(Enum):
    """Type classification for detected errors."""

    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    PARAMETER = "parameter"
    DATA_REFERENCE = "data_reference"
    UNCLEAR = "unclear"


class Correctness(Enum):
    """Correctness assessment for trajectory answers."""

    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


class Preference(Enum):
    """Overall trajectory preference."""

    A = "A"
    B = "B"
    TIE = "tie"


class LocalizationAccuracy(Enum):
    """How precise is the error localization."""

    EXACT = "exact"
    NEAR = "near"
    WRONG = "wrong"


@dataclass
class DetectabilityLabels:
    """
    Mode A: Detectability annotation labels.

    Collected when annotator evaluates error detection without knowing
    the expected answer.

    Attributes:
        error_detected: Does the annotator believe an error exists?
        error_trajectory: Which trajectory contains the error?
        error_step_id: Which step contains the error? (if detected)
        confidence: How confident is the annotator? (1-5)
    """

    error_detected: bool
    error_trajectory: str  # ErrorTrajectory value
    error_step_id: Optional[str]
    confidence: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_detected": self.error_detected,
            "error_trajectory": self.error_trajectory,
            "error_step_id": self.error_step_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectabilityLabels":
        """Create DetectabilityLabels from dictionary."""
        return cls(
            error_detected=data["error_detected"],
            error_trajectory=data["error_trajectory"],
            error_step_id=data.get("error_step_id"),
            confidence=data["confidence"],
        )


@dataclass
class ConsequenceLabels:
    """
    Mode B: Consequence annotation labels.

    Collected when annotator evaluates error impact with access to
    the expected answer.

    Attributes:
        error_type: What kind of error? (planning, tool_selection, etc.)
        impact_tier: 0=none, 1=minor, 2=moderate, 3=critical
        propagation_depth: How many downstream steps affected? (0-3)
        correctness_a: Is trajectory A's answer correct?
        correctness_b: Is trajectory B's answer correct?
    """

    error_type: Optional[str]  # ErrorType value
    impact_tier: Optional[int]
    propagation_depth: Optional[int]
    correctness_a: Optional[str]  # Correctness value
    correctness_b: Optional[str]  # Correctness value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_type": self.error_type,
            "impact_tier": self.impact_tier,
            "propagation_depth": self.propagation_depth,
            "correctness_a": self.correctness_a,
            "correctness_b": self.correctness_b,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsequenceLabels":
        """Create ConsequenceLabels from dictionary."""
        return cls(
            error_type=data.get("error_type"),
            impact_tier=data.get("impact_tier"),
            propagation_depth=data.get("propagation_depth"),
            correctness_a=data.get("correctness_a"),
            correctness_b=data.get("correctness_b"),
        )


@dataclass
class PreferenceLabels:
    """
    Mode C: Preference annotation labels.

    Collected when annotator evaluates overall trajectory quality.

    Attributes:
        preference: Which trajectory is better overall? (A, B, tie)
        preference_reason: Brief justification for the preference
    """

    preference: Optional[str]  # Preference value
    preference_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "preference": self.preference,
            "preference_reason": self.preference_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceLabels":
        """Create PreferenceLabels from dictionary."""
        return cls(
            preference=data.get("preference"),
            preference_reason=data.get("preference_reason"),
        )


@dataclass
class AnnotationRecord:
    """
    Complete annotation record from a single annotator.

    Packages identity info, view information, and all label types
    (detectability, consequence, preference) based on annotation mode.

    Attributes:
        annotation_id: Unique identifier (uuid)
        evaluation_unit_id: Links to the evaluation unit
        annotator_id: Who made this annotation
        annotation_mode: detectability | consequence | preference
        created_at: ISO 8601 timestamp
        view_file: Path to view JSON used
        trajectory_a_variant_id: Variant ID assigned to position A
        trajectory_b_variant_id: Variant ID assigned to position B
        detectability: Mode A labels (if applicable)
        consequence: Mode B labels (if applicable)
        preference: Mode C labels (if applicable)
        time_spent_seconds: How long the annotation took
        notes: Optional annotator notes
        flagged_for_review: Whether flagged for expert review
    """

    # Identity
    annotation_id: str
    evaluation_unit_id: str
    annotator_id: str
    annotation_mode: str  # AnnotationMode value
    created_at: str  # ISO 8601

    # View information
    view_file: str
    trajectory_a_variant_id: str
    trajectory_b_variant_id: str

    # Labels by mode
    detectability: Optional[DetectabilityLabels]
    consequence: Optional[ConsequenceLabels]
    preference: Optional[PreferenceLabels]

    # Metadata
    time_spent_seconds: int
    notes: Optional[str]
    flagged_for_review: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "annotation_id": self.annotation_id,
            "evaluation_unit_id": self.evaluation_unit_id,
            "annotator_id": self.annotator_id,
            "annotation_mode": self.annotation_mode,
            "created_at": self.created_at,
            "view_file": self.view_file,
            "trajectory_a_variant_id": self.trajectory_a_variant_id,
            "trajectory_b_variant_id": self.trajectory_b_variant_id,
            "detectability": (
                self.detectability.to_dict() if self.detectability else None
            ),
            "consequence": self.consequence.to_dict() if self.consequence else None,
            "preference": self.preference.to_dict() if self.preference else None,
            "time_spent_seconds": self.time_spent_seconds,
            "notes": self.notes,
            "flagged_for_review": self.flagged_for_review,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationRecord":
        """Create AnnotationRecord from dictionary."""
        detectability = None
        if data.get("detectability"):
            detectability = DetectabilityLabels.from_dict(data["detectability"])

        consequence = None
        if data.get("consequence"):
            consequence = ConsequenceLabels.from_dict(data["consequence"])

        preference = None
        if data.get("preference"):
            preference = PreferenceLabels.from_dict(data["preference"])

        return cls(
            annotation_id=data["annotation_id"],
            evaluation_unit_id=data["evaluation_unit_id"],
            annotator_id=data["annotator_id"],
            annotation_mode=data["annotation_mode"],
            created_at=data["created_at"],
            view_file=data["view_file"],
            trajectory_a_variant_id=data["trajectory_a_variant_id"],
            trajectory_b_variant_id=data["trajectory_b_variant_id"],
            detectability=detectability,
            consequence=consequence,
            preference=preference,
            time_spent_seconds=data["time_spent_seconds"],
            notes=data.get("notes"),
            flagged_for_review=data.get("flagged_for_review", False),
        )


@dataclass
class AggregatedDetectability:
    """
    Aggregated detectability labels across multiple annotators.

    Attributes:
        error_detected_majority: Majority vote on error detection
        error_detected_agreement: Agreement rate (0-1)
        error_trajectory_majority: Majority vote on which trajectory
        error_trajectory_agreement: Agreement rate (0-1)
        error_step_id_majority: Majority vote on error step
        localization_agreement: How well annotators agree on location
        mean_confidence: Average confidence across annotators
    """

    error_detected_majority: bool
    error_detected_agreement: float
    error_trajectory_majority: str
    error_trajectory_agreement: float
    error_step_id_majority: Optional[str]
    localization_agreement: str  # LocalizationAccuracy value or "mixed"
    mean_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_detected_majority": self.error_detected_majority,
            "error_detected_agreement": self.error_detected_agreement,
            "error_trajectory_majority": self.error_trajectory_majority,
            "error_trajectory_agreement": self.error_trajectory_agreement,
            "error_step_id_majority": self.error_step_id_majority,
            "localization_agreement": self.localization_agreement,
            "mean_confidence": self.mean_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedDetectability":
        """Create AggregatedDetectability from dictionary."""
        return cls(
            error_detected_majority=data["error_detected_majority"],
            error_detected_agreement=data["error_detected_agreement"],
            error_trajectory_majority=data["error_trajectory_majority"],
            error_trajectory_agreement=data["error_trajectory_agreement"],
            error_step_id_majority=data.get("error_step_id_majority"),
            localization_agreement=data["localization_agreement"],
            mean_confidence=data["mean_confidence"],
        )


@dataclass
class AggregatedConsequence:
    """
    Aggregated consequence labels across multiple annotators.

    Attributes:
        error_type_majority: Majority vote on error type
        error_type_agreement: Agreement rate (0-1)
        mean_impact_tier: Average impact tier
        impact_tier_std: Standard deviation of impact tier
        mean_propagation_depth: Average propagation depth
        correctness_a_majority: Majority vote on A's correctness
        correctness_b_majority: Majority vote on B's correctness
    """

    error_type_majority: Optional[str]
    error_type_agreement: float
    mean_impact_tier: float
    impact_tier_std: float
    mean_propagation_depth: float
    correctness_a_majority: str
    correctness_b_majority: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_type_majority": self.error_type_majority,
            "error_type_agreement": self.error_type_agreement,
            "mean_impact_tier": self.mean_impact_tier,
            "impact_tier_std": self.impact_tier_std,
            "mean_propagation_depth": self.mean_propagation_depth,
            "correctness_a_majority": self.correctness_a_majority,
            "correctness_b_majority": self.correctness_b_majority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedConsequence":
        """Create AggregatedConsequence from dictionary."""
        return cls(
            error_type_majority=data.get("error_type_majority"),
            error_type_agreement=data["error_type_agreement"],
            mean_impact_tier=data["mean_impact_tier"],
            impact_tier_std=data["impact_tier_std"],
            mean_propagation_depth=data["mean_propagation_depth"],
            correctness_a_majority=data["correctness_a_majority"],
            correctness_b_majority=data["correctness_b_majority"],
        )


@dataclass
class AggregatedPreference:
    """
    Aggregated preference labels across multiple annotators.

    Attributes:
        preference_majority: Majority vote on preference (A, B, tie)
        preference_agreement: Agreement rate (0-1)
    """

    preference_majority: str
    preference_agreement: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "preference_majority": self.preference_majority,
            "preference_agreement": self.preference_agreement,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedPreference":
        """Create AggregatedPreference from dictionary."""
        return cls(
            preference_majority=data["preference_majority"],
            preference_agreement=data["preference_agreement"],
        )


@dataclass
class AggregatedLabel:
    """
    Post-aggregation label combining all annotators' responses.

    This is the final ground-truth label used for downstream analysis
    (6A, 6B, 6C).

    Attributes:
        evaluation_unit_id: Links to the evaluation unit
        annotation_ids: List of contributing annotation IDs
        num_annotators: Number of annotators who labeled this unit
        aggregated_detectability: Combined detectability labels
        aggregated_consequence: Combined consequence labels
        aggregated_preference: Combined preference labels
        low_agreement_flag: True if agreement is below threshold
        needs_adjudication: True if expert review needed
    """

    evaluation_unit_id: str
    annotation_ids: List[str]
    num_annotators: int
    aggregated_detectability: Optional[AggregatedDetectability]
    aggregated_consequence: Optional[AggregatedConsequence]
    aggregated_preference: Optional[AggregatedPreference]
    low_agreement_flag: bool
    needs_adjudication: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "evaluation_unit_id": self.evaluation_unit_id,
            "annotation_ids": self.annotation_ids,
            "num_annotators": self.num_annotators,
            "aggregated_detectability": (
                self.aggregated_detectability.to_dict()
                if self.aggregated_detectability
                else None
            ),
            "aggregated_consequence": (
                self.aggregated_consequence.to_dict()
                if self.aggregated_consequence
                else None
            ),
            "aggregated_preference": (
                self.aggregated_preference.to_dict()
                if self.aggregated_preference
                else None
            ),
            "low_agreement_flag": self.low_agreement_flag,
            "needs_adjudication": self.needs_adjudication,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregatedLabel":
        """Create AggregatedLabel from dictionary."""
        agg_detect = None
        if data.get("aggregated_detectability"):
            agg_detect = AggregatedDetectability.from_dict(
                data["aggregated_detectability"]
            )

        agg_conseq = None
        if data.get("aggregated_consequence"):
            agg_conseq = AggregatedConsequence.from_dict(data["aggregated_consequence"])

        agg_pref = None
        if data.get("aggregated_preference"):
            agg_pref = AggregatedPreference.from_dict(data["aggregated_preference"])

        return cls(
            evaluation_unit_id=data["evaluation_unit_id"],
            annotation_ids=data["annotation_ids"],
            num_annotators=data["num_annotators"],
            aggregated_detectability=agg_detect,
            aggregated_consequence=agg_conseq,
            aggregated_preference=agg_pref,
            low_agreement_flag=data.get("low_agreement_flag", False),
            needs_adjudication=data.get("needs_adjudication", False),
        )
