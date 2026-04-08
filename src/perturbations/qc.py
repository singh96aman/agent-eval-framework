"""
Heuristic QC Pipeline for Perturbation Validation.

Phase 4 of Section 3 (Controlled Perturbations).
All checks are deterministic or rule-based - NO LLM in QC.

Validates:
- Schema-valid (required fields, enum values)
- Actually changed content (non-placebo) or preserved semantics (placebo)
- Coherent (trajectory still parses)
- Correctly labeled (class/family/type consistent)
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

from src.perturbations.schema import (
    GenerationStatus,
    PerturbationClass,
    PerturbationFamily,
    PerturbationRecord,
    PerturbationType,
    ReviewSource,
    validate_class_family_combination,
    validate_family_type_combination,
)
from src.typing.schema import TypedStep, TypedTrajectory

# Placebo perturbation types that should preserve semantics
PLACEBO_TYPES = {
    PerturbationType.PARAPHRASE.value,
    PerturbationType.FORMATTING.value,
    PerturbationType.SYNONYM.value,
    PerturbationType.REORDER_ARGS.value,
    PerturbationType.TRIVIAL_RENAME.value,
}


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    message: str
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        status = "PASS" if self.passed else f"FAIL ({self.severity})"
        return f"{self.check_name}: {status} - {self.message}"


class BaseValidator(ABC):
    """Base class for all validators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this validator."""
        pass

    @abstractmethod
    def validate(
        self,
        record: PerturbationRecord,
        perturbed_trajectory: TypedTrajectory,
        original_step: TypedStep,
    ) -> List[ValidationResult]:
        """
        Run validation checks.

        Args:
            record: The perturbation record to validate
            perturbed_trajectory: The trajectory after perturbation
            original_step: The original step before perturbation

        Returns:
            List of ValidationResult objects
        """
        pass


class SchemaValidator(BaseValidator):
    """
    Validates schema correctness of PerturbationRecord.

    Checks:
    - All required fields present
    - perturbation_class in valid enum
    - perturbation_family in valid enum
    - perturbation_type in valid enum
    - target_step_index is int >= 0
    """

    @property
    def name(self) -> str:
        return "schema"

    # Required fields that must be present and non-None
    REQUIRED_FIELDS = [
        "perturbation_id",
        "original_trajectory_id",
        "generation_timestamp",
        "perturbation_class",
        "perturbation_family",
        "perturbation_type",
        "target_step_index",
        "target_slot",
        "mutation_method",
    ]

    def validate(
        self,
        record: PerturbationRecord,
        perturbed_trajectory: TypedTrajectory,
        original_step: TypedStep,
    ) -> List[ValidationResult]:
        results = []

        # Check required fields
        missing_fields = []
        for field_name in self.REQUIRED_FIELDS:
            value = getattr(record, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_fields.append(field_name)

        if missing_fields:
            results.append(
                ValidationResult(
                    check_name="schema_required_fields",
                    passed=False,
                    message=f"Missing required fields: {missing_fields}",
                    severity="error",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="schema_required_fields",
                    passed=True,
                    message="All required fields present",
                )
            )

        # Check perturbation_class is valid enum
        try:
            PerturbationClass(record.perturbation_class)
            results.append(
                ValidationResult(
                    check_name="schema_perturbation_class",
                    passed=True,
                    message=f"Valid perturbation_class: {record.perturbation_class}",
                )
            )
        except ValueError:
            results.append(
                ValidationResult(
                    check_name="schema_perturbation_class",
                    passed=False,
                    message=f"Invalid perturbation_class: {record.perturbation_class}",
                    severity="error",
                )
            )

        # Check perturbation_family is valid enum
        try:
            PerturbationFamily(record.perturbation_family)
            results.append(
                ValidationResult(
                    check_name="schema_perturbation_family",
                    passed=True,
                    message=f"Valid perturbation_family: {record.perturbation_family}",
                )
            )
        except ValueError:
            results.append(
                ValidationResult(
                    check_name="schema_perturbation_family",
                    passed=False,
                    message=f"Invalid perturbation_family: {record.perturbation_family}",
                    severity="error",
                )
            )

        # Check perturbation_type is valid enum
        try:
            PerturbationType(record.perturbation_type)
            results.append(
                ValidationResult(
                    check_name="schema_perturbation_type",
                    passed=True,
                    message=f"Valid perturbation_type: {record.perturbation_type}",
                )
            )
        except ValueError:
            results.append(
                ValidationResult(
                    check_name="schema_perturbation_type",
                    passed=False,
                    message=f"Invalid perturbation_type: {record.perturbation_type}",
                    severity="error",
                )
            )

        # Check target_step_index is int >= 0
        if isinstance(record.target_step_index, int) and record.target_step_index >= 0:
            results.append(
                ValidationResult(
                    check_name="schema_target_step_index",
                    passed=True,
                    message=f"Valid target_step_index: {record.target_step_index}",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="schema_target_step_index",
                    passed=False,
                    message=f"Invalid target_step_index: {record.target_step_index} (must be int >= 0)",
                    severity="error",
                )
            )

        return results


class DiffValidator(BaseValidator):
    """
    Validates that content actually changed.

    Checks:
    - Non-placebo: original_value != perturbed_value
    - Placebo: semantic markers preserved (numbers, tool names, file paths)
    """

    @property
    def name(self) -> str:
        return "diff"

    # Patterns to extract semantic markers
    NUMERIC_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
    FILEPATH_PATTERN = re.compile(r"(?:/[\w.-]+)+(?:\.\w+)?")
    TOOL_NAME_PATTERN = re.compile(
        r"\b(?:read|write|search|grep|find|cat|ls|git|bash|execute|submit)\b",
        re.IGNORECASE,
    )

    def _extract_semantic_markers(self, value: Any) -> Dict[str, set]:
        """Extract semantic markers from a value."""
        if value is None:
            return {"numbers": set(), "filepaths": set(), "tools": set()}

        text = str(value)
        return {
            "numbers": set(self.NUMERIC_PATTERN.findall(text)),
            "filepaths": set(self.FILEPATH_PATTERN.findall(text)),
            "tools": set(self.TOOL_NAME_PATTERN.findall(text.lower())),
        }

    def _values_are_equal(self, v1: Any, v2: Any) -> bool:
        """Check if two values are semantically equal."""
        # Handle None cases
        if v1 is None and v2 is None:
            return True
        if v1 is None or v2 is None:
            return False

        # Direct equality check
        if v1 == v2:
            return True

        # String normalization for comparison
        s1 = str(v1).strip().lower()
        s2 = str(v2).strip().lower()
        return s1 == s2

    def validate(
        self,
        record: PerturbationRecord,
        perturbed_trajectory: TypedTrajectory,
        original_step: TypedStep,
    ) -> List[ValidationResult]:
        results = []

        is_placebo = record.perturbation_class == PerturbationClass.PLACEBO.value
        is_placebo_type = record.perturbation_type in PLACEBO_TYPES

        original = record.original_value
        perturbed = record.perturbed_value

        if is_placebo or is_placebo_type:
            # Placebo: should preserve semantic markers
            original_markers = self._extract_semantic_markers(original)
            perturbed_markers = self._extract_semantic_markers(perturbed)

            # Check numbers preserved
            if original_markers["numbers"] == perturbed_markers["numbers"]:
                results.append(
                    ValidationResult(
                        check_name="diff_placebo_numbers",
                        passed=True,
                        message="Numeric values preserved in placebo perturbation",
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        check_name="diff_placebo_numbers",
                        passed=False,
                        message=f"Numeric values changed: {original_markers['numbers']} -> {perturbed_markers['numbers']}",
                        severity="warning",
                    )
                )

            # Check tool names preserved
            if original_markers["tools"] == perturbed_markers["tools"]:
                results.append(
                    ValidationResult(
                        check_name="diff_placebo_tools",
                        passed=True,
                        message="Tool names preserved in placebo perturbation",
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        check_name="diff_placebo_tools",
                        passed=False,
                        message=f"Tool names changed: {original_markers['tools']} -> {perturbed_markers['tools']}",
                        severity="warning",
                    )
                )

            # Check file paths preserved
            if original_markers["filepaths"] == perturbed_markers["filepaths"]:
                results.append(
                    ValidationResult(
                        check_name="diff_placebo_filepaths",
                        passed=True,
                        message="File paths preserved in placebo perturbation",
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        check_name="diff_placebo_filepaths",
                        passed=False,
                        message=f"File paths changed: {original_markers['filepaths']} -> {perturbed_markers['filepaths']}",
                        severity="warning",
                    )
                )

        else:
            # Non-placebo: must have actual change
            if not self._values_are_equal(original, perturbed):
                results.append(
                    ValidationResult(
                        check_name="diff_content_changed",
                        passed=True,
                        message="Content actually changed in non-placebo perturbation",
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        check_name="diff_content_changed",
                        passed=False,
                        message=f"No actual change detected: original={original}, perturbed={perturbed}",
                        severity="error",
                    )
                )

        return results


class CoherenceValidator(BaseValidator):
    """
    Validates trajectory structure coherence.

    Checks:
    - All steps have required fields
    - Step indices are sequential
    - Tool call steps have tool_name
    - No empty content
    """

    @property
    def name(self) -> str:
        return "coherence"

    # Required fields for all steps
    REQUIRED_STEP_FIELDS = ["step_index", "step_role", "raw_text"]

    def validate(
        self,
        record: PerturbationRecord,
        perturbed_trajectory: TypedTrajectory,
        original_step: TypedStep,
    ) -> List[ValidationResult]:
        results = []

        if not perturbed_trajectory.steps:
            results.append(
                ValidationResult(
                    check_name="coherence_has_steps",
                    passed=False,
                    message="Trajectory has no steps",
                    severity="error",
                )
            )
            return results

        results.append(
            ValidationResult(
                check_name="coherence_has_steps",
                passed=True,
                message=f"Trajectory has {len(perturbed_trajectory.steps)} steps",
            )
        )

        # Check required fields on all steps
        steps_missing_fields = []
        for step in perturbed_trajectory.steps:
            missing = []
            for field_name in self.REQUIRED_STEP_FIELDS:
                value = getattr(step, field_name, None)
                if value is None:
                    missing.append(field_name)
            if missing:
                steps_missing_fields.append((step.step_index, missing))

        if steps_missing_fields:
            results.append(
                ValidationResult(
                    check_name="coherence_step_fields",
                    passed=False,
                    message=f"Steps missing required fields: {steps_missing_fields}",
                    severity="error",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="coherence_step_fields",
                    passed=True,
                    message="All steps have required fields",
                )
            )

        # Check step indices are sequential
        indices = [step.step_index for step in perturbed_trajectory.steps]
        expected = list(range(len(indices)))
        if sorted(indices) == expected:
            results.append(
                ValidationResult(
                    check_name="coherence_sequential_indices",
                    passed=True,
                    message="Step indices are sequential",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="coherence_sequential_indices",
                    passed=False,
                    message=f"Step indices not sequential: {indices}",
                    severity="warning",
                )
            )

        # Check tool_call steps have tool_name
        tool_steps_without_name = []
        for step in perturbed_trajectory.steps:
            if step.step_role == "tool_call" and not step.tool_name:
                tool_steps_without_name.append(step.step_index)

        if tool_steps_without_name:
            results.append(
                ValidationResult(
                    check_name="coherence_tool_names",
                    passed=False,
                    message=f"Tool call steps without tool_name: {tool_steps_without_name}",
                    severity="warning",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="coherence_tool_names",
                    passed=True,
                    message="All tool call steps have tool_name",
                )
            )

        # Check no empty content
        empty_content_steps = []
        for step in perturbed_trajectory.steps:
            if not step.raw_text or not step.raw_text.strip():
                empty_content_steps.append(step.step_index)

        if empty_content_steps:
            results.append(
                ValidationResult(
                    check_name="coherence_no_empty",
                    passed=False,
                    message=f"Steps with empty content: {empty_content_steps}",
                    severity="warning",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="coherence_no_empty",
                    passed=True,
                    message="No steps have empty content",
                )
            )

        return results


class ClassFamilyValidator(BaseValidator):
    """
    Validates taxonomy consistency.

    Checks:
    - Class-family combination valid per Section 3.1
    - Family-type combination valid
    - tool_selection: near_neighbor -> fine_grained, wrong_family -> coarse_grained
    """

    @property
    def name(self) -> str:
        return "class_family"

    def validate(
        self,
        record: PerturbationRecord,
        perturbed_trajectory: TypedTrajectory,
        original_step: TypedStep,
    ) -> List[ValidationResult]:
        results = []

        try:
            p_class = PerturbationClass(record.perturbation_class)
            p_family = PerturbationFamily(record.perturbation_family)
            p_type = PerturbationType(record.perturbation_type)
        except ValueError as e:
            results.append(
                ValidationResult(
                    check_name="class_family_enum_parse",
                    passed=False,
                    message=f"Cannot parse enum values: {e}",
                    severity="error",
                )
            )
            return results

        # Check class-family combination
        if validate_class_family_combination(p_class, p_family):
            results.append(
                ValidationResult(
                    check_name="class_family_combination",
                    passed=True,
                    message=f"Valid class-family: {p_class.value} + {p_family.value}",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="class_family_combination",
                    passed=False,
                    message=f"Invalid class-family combination: {p_class.value} + {p_family.value}",
                    severity="error",
                )
            )

        # Check family-type combination
        if validate_family_type_combination(p_family, p_type):
            results.append(
                ValidationResult(
                    check_name="family_type_combination",
                    passed=True,
                    message=f"Valid family-type: {p_family.value} + {p_type.value}",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="family_type_combination",
                    passed=False,
                    message=f"Invalid family-type combination: {p_family.value} + {p_type.value}",
                    severity="error",
                )
            )

        # Special rule for tool_selection family
        if p_family == PerturbationFamily.TOOL_SELECTION:
            if p_type == PerturbationType.NEAR_NEIGHBOR_TOOL:
                # near_neighbor -> fine_grained
                if p_class == PerturbationClass.FINE_GRAINED:
                    results.append(
                        ValidationResult(
                            check_name="tool_selection_class_rule",
                            passed=True,
                            message="near_neighbor_tool correctly maps to fine_grained",
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            check_name="tool_selection_class_rule",
                            passed=False,
                            message=f"near_neighbor_tool should be fine_grained, got {p_class.value}",
                            severity="error",
                        )
                    )
            elif p_type == PerturbationType.WRONG_TOOL_FAMILY:
                # wrong_family -> coarse_grained
                if p_class == PerturbationClass.COARSE_GRAINED:
                    results.append(
                        ValidationResult(
                            check_name="tool_selection_class_rule",
                            passed=True,
                            message="wrong_tool_family correctly maps to coarse_grained",
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            check_name="tool_selection_class_rule",
                            passed=False,
                            message=f"wrong_tool_family should be coarse_grained, got {p_class.value}",
                            severity="error",
                        )
                    )

        return results


class ImpactDeriver:
    """
    Derives expected_impact from Section 2 data.

    Uses:
    - critical_path_score.value from TypedStep
    - affects_final_answer.value from TypedStep
    - recoverable_if_wrong.value from TypedStep
    - Placebo always -> 0
    - Heuristic formula: score based on criticality + outcome linkage
    """

    def derive_impact(
        self,
        record: PerturbationRecord,
        original_step: TypedStep,
    ) -> Tuple[int, str]:
        """
        Derive expected impact score.

        Args:
            record: The perturbation record
            original_step: The original step being perturbed

        Returns:
            Tuple of (impact_score 0-3, explanation string)
        """
        # Placebo always has 0 impact
        if record.perturbation_class == PerturbationClass.PLACEBO.value:
            return 0, "Placebo perturbation - no semantic change"

        # Check if perturbation type is placebo-like
        if record.perturbation_type in PLACEBO_TYPES:
            return (
                0,
                f"Placebo-type perturbation ({record.perturbation_type}) - no semantic change",
            )

        # Extract values from provenance fields
        critical_score = self._get_provenance_value(
            original_step.critical_path_score, 0.5
        )
        affects_final = self._get_provenance_value(
            original_step.affects_final_answer, False
        )
        recoverable = self._get_provenance_value(
            original_step.recoverable_if_wrong, True
        )

        # Heuristic formula:
        # Base score from criticality (0-3 scale)
        base_score = self._criticality_to_impact(critical_score)

        # Adjustments
        explanation_parts = [f"Base criticality score: {base_score}"]

        # If affects final answer, increase impact
        if affects_final:
            base_score = min(3, base_score + 1)
            explanation_parts.append("Affects final answer (+1)")

        # If recoverable, decrease impact
        if recoverable and base_score > 0:
            base_score = max(0, base_score - 1)
            explanation_parts.append("Recoverable if wrong (-1)")

        # Coarse-grained perturbations have higher impact floor
        if record.perturbation_class == PerturbationClass.COARSE_GRAINED.value:
            if base_score < 2:
                base_score = 2
                explanation_parts.append("Coarse-grained perturbation (min 2)")

        explanation = "; ".join(explanation_parts)
        return int(base_score), explanation

    def _get_provenance_value(self, field, default):
        """Safely extract value from ProvenanceField."""
        if field is None:
            return default
        if hasattr(field, "value"):
            return field.value
        return default

    def _criticality_to_impact(self, score: float) -> int:
        """Convert criticality score (0-1) to impact (0-3)."""
        if score < 0.25:
            return 0  # none
        elif score < 0.5:
            return 1  # minor
        elif score < 0.75:
            return 2  # moderate
        else:
            return 3  # critical


class PerturbationQC:
    """
    QC Orchestrator for perturbation validation.

    Runs all validators and updates the PerturbationRecord with:
    - generation_status
    - qc_notes
    - qc_checks_passed
    - qc_checks_failed
    - expected_impact (derived)
    """

    def __init__(self):
        self.validators: List[BaseValidator] = [
            SchemaValidator(),
            DiffValidator(),
            CoherenceValidator(),
            ClassFamilyValidator(),
        ]
        self.impact_deriver = ImpactDeriver()

    def validate(
        self,
        record: PerturbationRecord,
        perturbed_trajectory: TypedTrajectory,
        original_step: TypedStep,
    ) -> PerturbationRecord:
        """
        Run all validators, update record with QC results.

        Args:
            record: The perturbation record to validate
            perturbed_trajectory: The trajectory after perturbation
            original_step: The original step before perturbation

        Returns:
            Updated PerturbationRecord with QC fields populated
        """
        all_results: List[ValidationResult] = []

        # Run all validators
        for validator in self.validators:
            results = validator.validate(record, perturbed_trajectory, original_step)
            all_results.extend(results)

        # Categorize results
        passed_checks = [r.check_name for r in all_results if r.passed]
        failed_checks = [r.check_name for r in all_results if not r.passed]

        # Collect error and warning messages
        error_messages = [
            r.message for r in all_results if not r.passed and r.severity == "error"
        ]
        warning_messages = [
            r.message for r in all_results if not r.passed and r.severity == "warning"
        ]

        # Determine generation status based on which validators failed
        schema_failed = any(
            r.check_name.startswith("schema_") and not r.passed
            for r in all_results
            if r.severity == "error"
        )
        diff_failed = any(
            r.check_name.startswith("diff_content_changed") and not r.passed
            for r in all_results
            if r.severity == "error"
        )
        coherence_failed = any(
            r.check_name.startswith("coherence_") and not r.passed
            for r in all_results
            if r.severity == "error"
        )
        class_family_failed = any(
            r.check_name.startswith("class_family") and not r.passed
            for r in all_results
            if r.severity == "error"
        )
        tool_selection_failed = any(
            r.check_name.startswith("tool_selection") and not r.passed
            for r in all_results
            if r.severity == "error"
        )

        # Status assignment logic:
        # - invalid: schema or diff failed (core problems)
        # - borderline: coherence or class_family failed (structural problems)
        # - valid: all passed
        if schema_failed or diff_failed:
            status = GenerationStatus.INVALID.value
        elif coherence_failed or class_family_failed or tool_selection_failed:
            status = GenerationStatus.BORDERLINE.value
        else:
            status = GenerationStatus.VALID.value

        # Build QC notes
        qc_note_parts = []
        if error_messages:
            qc_note_parts.append(f"Errors: {'; '.join(error_messages)}")
        if warning_messages:
            qc_note_parts.append(f"Warnings: {'; '.join(warning_messages)}")
        if not qc_note_parts:
            qc_note_parts.append("All QC checks passed")

        qc_notes = " | ".join(qc_note_parts)

        # Derive expected impact
        expected_impact, impact_explanation = self.impact_deriver.derive_impact(
            record, original_step
        )

        # Append impact explanation to notes
        qc_notes += f" | Impact: {expected_impact} ({impact_explanation})"

        # Update record fields
        record.generation_status = status
        record.qc_notes = qc_notes
        record.qc_checks_passed = passed_checks
        record.qc_checks_failed = failed_checks
        record.expected_impact = expected_impact
        record.review_source = ReviewSource.HEURISTIC.value
        record.reviewed_at = datetime.utcnow().isoformat() + "Z"

        return record

    def get_validation_summary(
        self,
        record: PerturbationRecord,
        perturbed_trajectory: TypedTrajectory,
        original_step: TypedStep,
    ) -> Dict[str, Any]:
        """
        Get detailed validation summary without modifying record.

        Args:
            record: The perturbation record to validate
            perturbed_trajectory: The trajectory after perturbation
            original_step: The original step before perturbation

        Returns:
            Dict with detailed validation results
        """
        all_results: List[ValidationResult] = []

        for validator in self.validators:
            results = validator.validate(record, perturbed_trajectory, original_step)
            all_results.extend(results)

        expected_impact, impact_explanation = self.impact_deriver.derive_impact(
            record, original_step
        )

        return {
            "perturbation_id": record.perturbation_id,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                }
                for r in all_results
            ],
            "passed_count": sum(1 for r in all_results if r.passed),
            "failed_count": sum(1 for r in all_results if not r.passed),
            "expected_impact": expected_impact,
            "impact_explanation": impact_explanation,
        }


def run_qc(
    record: PerturbationRecord,
    trajectory: TypedTrajectory,
    step: TypedStep,
) -> PerturbationRecord:
    """
    Run QC pipeline on a perturbation record.

    Convenience function that creates a PerturbationQC instance and validates.

    Args:
        record: The perturbation record to validate
        trajectory: The perturbed trajectory
        step: The original step before perturbation

    Returns:
        Updated PerturbationRecord with QC fields populated
    """
    qc = PerturbationQC()
    return qc.validate(record, trajectory, step)


def run_qc_batch(
    records: List[Tuple[PerturbationRecord, TypedTrajectory, TypedStep]],
) -> List[PerturbationRecord]:
    """
    Run QC pipeline on multiple perturbation records.

    Args:
        records: List of tuples (record, trajectory, step)

    Returns:
        List of updated PerturbationRecords
    """
    qc = PerturbationQC()
    results = []
    for record, trajectory, step in records:
        validated = qc.validate(record, trajectory, step)
        results.append(validated)
    return results


def get_qc_statistics(records: List[PerturbationRecord]) -> Dict[str, Any]:
    """
    Get aggregate QC statistics for a batch of records.

    Args:
        records: List of PerturbationRecords that have been through QC

    Returns:
        Dict with aggregate statistics
    """
    total = len(records)
    if total == 0:
        return {"total": 0, "valid": 0, "invalid": 0, "borderline": 0}

    valid_count = sum(
        1 for r in records if r.generation_status == GenerationStatus.VALID.value
    )
    invalid_count = sum(
        1 for r in records if r.generation_status == GenerationStatus.INVALID.value
    )
    borderline_count = sum(
        1 for r in records if r.generation_status == GenerationStatus.BORDERLINE.value
    )

    # Aggregate check failures
    check_failure_counts: Dict[str, int] = {}
    for record in records:
        for check in record.qc_checks_failed:
            check_failure_counts[check] = check_failure_counts.get(check, 0) + 1

    # Impact distribution
    impact_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for record in records:
        impact = record.expected_impact
        if impact in impact_counts:
            impact_counts[impact] += 1

    return {
        "total": total,
        "valid": valid_count,
        "valid_pct": valid_count / total * 100,
        "invalid": invalid_count,
        "invalid_pct": invalid_count / total * 100,
        "borderline": borderline_count,
        "borderline_pct": borderline_count / total * 100,
        "check_failure_counts": check_failure_counts,
        "impact_distribution": impact_counts,
    }
