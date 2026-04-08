"""
Schema definitions for Section 3: Controlled Perturbations.

Defines PerturbationRecord, enums for class/family/type,
and supporting dataclasses per 3_Requirements_ControlledPerturbations.MD.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class PerturbationClass(Enum):
    """Perturbation class - the experimental condition."""

    PLACEBO = "placebo"
    FINE_GRAINED = "fine_grained"
    COARSE_GRAINED = "coarse_grained"


class PerturbationFamily(Enum):
    """Perturbation family - what kind of slot mutation."""

    DATA_REFERENCE = "data_reference"
    PARAMETER = "parameter"
    TOOL_SELECTION = "tool_selection"
    STRUCTURAL = "structural"
    TERMINAL_FLAG = "terminal_flag"


class PerturbationType(Enum):
    """Specific perturbation type within each family."""

    # Placebo types
    PARAPHRASE = "paraphrase"
    FORMATTING = "formatting"
    SYNONYM = "synonym"
    REORDER_ARGS = "reorder_args"
    TRIVIAL_RENAME = "trivial_rename"

    # Data reference types (fine-grained)
    WRONG_VALUE = "wrong_value"
    OFF_BY_ONE = "off_by_one"
    TYPO_IN_ID = "typo_in_id"
    WRONG_FIELD = "wrong_field"
    COPIED_VALUE_ERROR = "copied_value_error"

    # Parameter types (fine-grained)
    THRESHOLD_SHIFT = "threshold_shift"
    QUERY_DRIFT = "query_drift"
    WRONG_PARAMETER = "wrong_parameter"

    # Tool selection types
    NEAR_NEIGHBOR_TOOL = "near_neighbor_tool"  # fine-grained
    WRONG_TOOL_FAMILY = "wrong_tool_family"  # coarse-grained

    # Structural types (coarse-grained)
    SKIPPED_PREREQUISITE = "skipped_prerequisite"
    WRONG_PLAN = "wrong_plan"
    WRONG_BRANCH = "wrong_branch"
    HALLUCINATED_DEPENDENCY = "hallucinated_dependency"

    # Terminal flag types (coarse-grained)
    PREMATURE_TERMINATION = "premature_termination"
    FALSE_TERMINAL = "false_terminal"


class GenerationStatus(Enum):
    """QC status for generated perturbations."""

    VALID = "valid"
    INVALID = "invalid"
    BORDERLINE = "borderline"


class ReviewSource(Enum):
    """Source of QC review."""

    HEURISTIC = "heuristic"
    LLM = "llm"
    HUMAN = "human"


# Valid class-family combinations per Section 3.1
VALID_CLASS_FAMILY_COMBINATIONS = {
    PerturbationClass.PLACEBO: [
        PerturbationFamily.DATA_REFERENCE,
        PerturbationFamily.PARAMETER,
    ],
    PerturbationClass.FINE_GRAINED: [
        PerturbationFamily.DATA_REFERENCE,
        PerturbationFamily.PARAMETER,
        PerturbationFamily.TOOL_SELECTION,
    ],
    PerturbationClass.COARSE_GRAINED: [
        PerturbationFamily.TOOL_SELECTION,
        PerturbationFamily.STRUCTURAL,
        PerturbationFamily.TERMINAL_FLAG,
    ],
}

# Valid perturbation types per family
VALID_TYPES_BY_FAMILY = {
    PerturbationFamily.DATA_REFERENCE: [
        PerturbationType.WRONG_VALUE,
        PerturbationType.OFF_BY_ONE,
        PerturbationType.TYPO_IN_ID,
        PerturbationType.WRONG_FIELD,
        PerturbationType.COPIED_VALUE_ERROR,
        # Placebo types also allowed for data_reference slots
        PerturbationType.PARAPHRASE,
        PerturbationType.FORMATTING,
        PerturbationType.SYNONYM,
    ],
    PerturbationFamily.PARAMETER: [
        PerturbationType.THRESHOLD_SHIFT,
        PerturbationType.QUERY_DRIFT,
        PerturbationType.WRONG_PARAMETER,
        # Placebo types also allowed for parameter slots
        PerturbationType.PARAPHRASE,
        PerturbationType.FORMATTING,
        PerturbationType.REORDER_ARGS,
    ],
    PerturbationFamily.TOOL_SELECTION: [
        PerturbationType.NEAR_NEIGHBOR_TOOL,  # fine-grained
        PerturbationType.WRONG_TOOL_FAMILY,  # coarse-grained
    ],
    PerturbationFamily.STRUCTURAL: [
        PerturbationType.SKIPPED_PREREQUISITE,
        PerturbationType.WRONG_PLAN,
        PerturbationType.WRONG_BRANCH,
        PerturbationType.HALLUCINATED_DEPENDENCY,
    ],
    PerturbationFamily.TERMINAL_FLAG: [
        PerturbationType.PREMATURE_TERMINATION,
        PerturbationType.FALSE_TERMINAL,
    ],
}


@dataclass
class PerturbationRecord:
    """
    Complete record of a single perturbation.

    Per Section 3.6 Final Schema.
    """

    # === Core identification ===
    perturbation_id: str
    original_trajectory_id: str
    generation_timestamp: str  # ISO 8601

    # === Classification (two-field system) ===
    perturbation_class: str  # PerturbationClass value
    perturbation_family: str  # PerturbationFamily value
    perturbation_type: str  # PerturbationType value

    # === Target specification ===
    target_step_index: int
    target_slot: str  # JSON path to modified field
    original_value: Any
    perturbed_value: Any
    mutation_method: str  # How mutation was generated

    # === Generation-time predictions (derived from Section 2) ===
    expected_impact: int = 0  # 0=none, 1=minor, 2=moderate, 3=critical
    expected_detectability: int = 1  # 0=invisible, 1=subtle, 2=obvious
    propagation_risk: List[str] = field(default_factory=list)  # Downstream step IDs
    verified_semantic_change: bool = True  # True for non-placebo, False for placebo
    notes: Optional[str] = None

    # === Post-generation QC ===
    generation_status: str = "valid"  # GenerationStatus value
    qc_notes: Optional[str] = None
    review_source: str = "heuristic"  # ReviewSource value
    reviewed_at: Optional[str] = None

    # === QC check results ===
    qc_checks_passed: List[str] = field(default_factory=list)
    qc_checks_failed: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        original_trajectory_id: str,
        perturbation_class: PerturbationClass,
        perturbation_family: PerturbationFamily,
        perturbation_type: PerturbationType,
        target_step_index: int,
        target_slot: str,
        original_value: Any,
        perturbed_value: Any,
        mutation_method: str,
        expected_impact: int = 0,
        notes: Optional[str] = None,
    ) -> "PerturbationRecord":
        """Factory method to create a new PerturbationRecord with auto-generated ID and timestamp."""
        return cls(
            perturbation_id=f"pert_{original_trajectory_id}_{uuid.uuid4().hex[:8]}",
            original_trajectory_id=original_trajectory_id,
            generation_timestamp=datetime.utcnow().isoformat() + "Z",
            perturbation_class=perturbation_class.value,
            perturbation_family=perturbation_family.value,
            perturbation_type=perturbation_type.value,
            target_step_index=target_step_index,
            target_slot=target_slot,
            original_value=original_value,
            perturbed_value=perturbed_value,
            mutation_method=mutation_method,
            expected_impact=expected_impact,
            verified_semantic_change=(perturbation_class != PerturbationClass.PLACEBO),
            notes=notes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "perturbation_id": self.perturbation_id,
            "original_trajectory_id": self.original_trajectory_id,
            "generation_timestamp": self.generation_timestamp,
            "perturbation_class": self.perturbation_class,
            "perturbation_family": self.perturbation_family,
            "perturbation_type": self.perturbation_type,
            "target_step_index": self.target_step_index,
            "target_slot": self.target_slot,
            "original_value": self.original_value,
            "perturbed_value": self.perturbed_value,
            "mutation_method": self.mutation_method,
            "expected_impact": self.expected_impact,
            "expected_detectability": self.expected_detectability,
            "propagation_risk": self.propagation_risk,
            "verified_semantic_change": self.verified_semantic_change,
            "notes": self.notes,
            "generation_status": self.generation_status,
            "qc_notes": self.qc_notes,
            "review_source": self.review_source,
            "reviewed_at": self.reviewed_at,
            "qc_checks_passed": self.qc_checks_passed,
            "qc_checks_failed": self.qc_checks_failed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerturbationRecord":
        """Create PerturbationRecord from dictionary."""
        return cls(**data)

    def is_valid(self) -> bool:
        """Check if perturbation passed QC."""
        return self.generation_status == GenerationStatus.VALID.value

    def get_class(self) -> PerturbationClass:
        """Get perturbation class as enum."""
        return PerturbationClass(self.perturbation_class)

    def get_family(self) -> PerturbationFamily:
        """Get perturbation family as enum."""
        return PerturbationFamily(self.perturbation_family)

    def get_type(self) -> PerturbationType:
        """Get perturbation type as enum."""
        return PerturbationType(self.perturbation_type)


@dataclass
class PerturbationIndex:
    """
    Summary index of all perturbations per Section 3.8.
    """

    total_perturbations: int = 0
    by_class: Dict[str, int] = field(default_factory=dict)
    by_family: Dict[str, int] = field(default_factory=dict)
    by_benchmark: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    by_generation_status: Dict[str, int] = field(default_factory=dict)
    perturbations: List[Dict[str, Any]] = field(default_factory=list)

    def add_perturbation(
        self, record: PerturbationRecord, benchmark: str, file_path: str
    ):
        """Add a perturbation to the index."""
        self.total_perturbations += 1

        # Update counts
        self.by_class[record.perturbation_class] = (
            self.by_class.get(record.perturbation_class, 0) + 1
        )
        self.by_family[record.perturbation_family] = (
            self.by_family.get(record.perturbation_family, 0) + 1
        )
        self.by_benchmark[benchmark] = self.by_benchmark.get(benchmark, 0) + 1
        self.by_type[record.perturbation_type] = (
            self.by_type.get(record.perturbation_type, 0) + 1
        )
        self.by_generation_status[record.generation_status] = (
            self.by_generation_status.get(record.generation_status, 0) + 1
        )

        # Add summary entry
        self.perturbations.append(
            {
                "perturbation_id": record.perturbation_id,
                "original_trajectory_id": record.original_trajectory_id,
                "perturbation_class": record.perturbation_class,
                "perturbation_family": record.perturbation_family,
                "perturbation_type": record.perturbation_type,
                "expected_impact": record.expected_impact,
                "generation_status": record.generation_status,
                "file": file_path,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_perturbations": self.total_perturbations,
            "by_class": self.by_class,
            "by_family": self.by_family,
            "by_benchmark": self.by_benchmark,
            "by_type": self.by_type,
            "by_generation_status": self.by_generation_status,
            "perturbations": self.perturbations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerturbationIndex":
        """Create PerturbationIndex from dictionary."""
        return cls(**data)

    def get_distribution_report(self) -> str:
        """Get a human-readable distribution report."""
        lines = [
            f"Total Perturbations: {self.total_perturbations}",
            "",
            "By Class:",
        ]
        for cls, count in sorted(self.by_class.items()):
            pct = (
                count / self.total_perturbations * 100
                if self.total_perturbations > 0
                else 0
            )
            lines.append(f"  {cls}: {count} ({pct:.1f}%)")

        lines.append("")
        lines.append("By Family:")
        for fam, count in sorted(self.by_family.items()):
            pct = (
                count / self.total_perturbations * 100
                if self.total_perturbations > 0
                else 0
            )
            lines.append(f"  {fam}: {count} ({pct:.1f}%)")

        lines.append("")
        lines.append("By Status:")
        for status, count in sorted(self.by_generation_status.items()):
            pct = (
                count / self.total_perturbations * 100
                if self.total_perturbations > 0
                else 0
            )
            lines.append(f"  {status}: {count} ({pct:.1f}%)")

        return "\n".join(lines)


def validate_class_family_combination(
    perturbation_class: PerturbationClass, perturbation_family: PerturbationFamily
) -> bool:
    """Check if a class-family combination is valid per Section 3.1."""
    allowed_families = VALID_CLASS_FAMILY_COMBINATIONS.get(perturbation_class, [])
    return perturbation_family in allowed_families


def validate_family_type_combination(
    perturbation_family: PerturbationFamily, perturbation_type: PerturbationType
) -> bool:
    """Check if a family-type combination is valid."""
    allowed_types = VALID_TYPES_BY_FAMILY.get(perturbation_family, [])
    return perturbation_type in allowed_types


def get_class_for_tool_selection_type(
    perturbation_type: PerturbationType,
) -> PerturbationClass:
    """
    Determine class for tool_selection perturbations per Section 3.1 rules.

    - near_neighbor_tool → fine_grained
    - wrong_tool_family → coarse_grained
    """
    if perturbation_type == PerturbationType.NEAR_NEIGHBOR_TOOL:
        return PerturbationClass.FINE_GRAINED
    elif perturbation_type == PerturbationType.WRONG_TOOL_FAMILY:
        return PerturbationClass.COARSE_GRAINED
    else:
        raise ValueError(f"Unknown tool_selection type: {perturbation_type}")
