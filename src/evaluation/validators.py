"""
Comprehensive validation checks for evaluation units.

This module implements the validation rules defined in Section 4.9 of the
requirements documentation. Validators ensure data integrity and consistency
across evaluation units before they are used in experiments.

Validators return structured results with pass/fail status and detailed
error/warning information to support debugging and quality assurance.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.evaluation.ids import (
    CANONICAL_STEP_PATTERN,
    EVALUATION_UNIT_PATTERN,
    VARIANT_BASE_PATTERN,
    VARIANT_PERT_PATTERN,
)
from src.evaluation.derived_cache import verify_cache_consistency
from src.evaluation.tier_assignment import assign_replay_tier


@dataclass
class ValidationResult:
    """
    Result of a single validation check.

    Attributes:
        passed: Whether the validation passed (no errors)
        validator_name: Name of the validator that produced this result
        errors: List of error messages (validation failures)
        warnings: List of warning messages (non-fatal issues)
    """

    passed: bool
    validator_name: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# Required top-level fields for an evaluation unit
REQUIRED_TOP_LEVEL_FIELDS = [
    "evaluation_unit_id",
    "experiment_id",
    "created_at",
    "source_trajectory_id",
    "benchmark",
    "task_id",
    "task_text",
    "baseline",
    "perturbed",
    "derived_cache",
    "evaluation_capabilities",
    "replay_tier",
    "blinding",
]


def validate_id_format(unit: Dict[str, Any]) -> ValidationResult:
    """
    Validate that all IDs in the evaluation unit match expected patterns.

    Checks:
    - evaluation_unit_id matches pattern `eval::{source}::{index}`
    - baseline variant_id matches pattern `{source}::base`
    - perturbed variant_id matches pattern `{source}::pert::{index}`
    - all canonical_step_ids in steps match pattern `{source}::step::{position}`

    Args:
        unit: The evaluation unit dictionary to validate

    Returns:
        ValidationResult with pass/fail status and any errors/warnings
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check evaluation_unit_id format
    eval_unit_id = unit.get("evaluation_unit_id", "")
    if not eval_unit_id:
        errors.append("evaluation_unit_id is missing or empty")
    elif not EVALUATION_UNIT_PATTERN.match(eval_unit_id):
        errors.append(
            f"evaluation_unit_id '{eval_unit_id}' does not match pattern "
            f"'eval::{{source}}::{{index}}'"
        )

    # Extract source_trajectory_id for cross-validation
    source_id = unit.get("source_trajectory_id", "")

    # Check baseline variant_id format
    baseline = unit.get("baseline", {})
    baseline_variant_id = baseline.get("trajectory_variant_id", "")
    if not baseline_variant_id:
        errors.append("baseline.trajectory_variant_id is missing or empty")
    elif not VARIANT_BASE_PATTERN.match(baseline_variant_id):
        errors.append(
            f"baseline variant_id '{baseline_variant_id}' does not match pattern "
            f"'{{source}}::base'"
        )
    else:
        # Check that source matches
        match = VARIANT_BASE_PATTERN.match(baseline_variant_id)
        if match and source_id and match.group(1) != source_id:
            errors.append(
                f"baseline variant_id source '{match.group(1)}' does not match "
                f"source_trajectory_id '{source_id}'"
            )

    # Check perturbed variant_id format
    perturbed = unit.get("perturbed", {})
    perturbed_variant_id = perturbed.get("trajectory_variant_id", "")
    if not perturbed_variant_id:
        errors.append("perturbed.trajectory_variant_id is missing or empty")
    elif not VARIANT_PERT_PATTERN.match(perturbed_variant_id):
        errors.append(
            f"perturbed variant_id '{perturbed_variant_id}' does not match pattern "
            f"'{{source}}::pert::{{index}}'"
        )
    else:
        # Check that source matches
        match = VARIANT_PERT_PATTERN.match(perturbed_variant_id)
        if match and source_id and match.group(1) != source_id:
            errors.append(
                f"perturbed variant_id source '{match.group(1)}' does not match "
                f"source_trajectory_id '{source_id}'"
            )

    # Check canonical_step_ids in baseline steps
    baseline_trajectory = baseline.get("trajectory", {})
    baseline_steps = baseline_trajectory.get("steps", [])
    for i, step in enumerate(baseline_steps):
        step_id = step.get("canonical_step_id", "")
        if step_id and not CANONICAL_STEP_PATTERN.match(step_id):
            errors.append(
                f"baseline step {i} canonical_step_id '{step_id}' does not match "
                f"pattern '{{source}}::step::{{position}}'"
            )
        elif step_id:
            # Check that source matches
            match = CANONICAL_STEP_PATTERN.match(step_id)
            if match and source_id and match.group(1) != source_id:
                errors.append(
                    f"baseline step {i} canonical_step_id source '{match.group(1)}' "
                    f"does not match source_trajectory_id '{source_id}'"
                )

    # Check canonical_step_ids in perturbed steps
    perturbed_trajectory = perturbed.get("trajectory", {})
    perturbed_steps = perturbed_trajectory.get("steps", [])
    for i, step in enumerate(perturbed_steps):
        step_id = step.get("canonical_step_id", "")
        if step_id and not CANONICAL_STEP_PATTERN.match(step_id):
            errors.append(
                f"perturbed step {i} canonical_step_id '{step_id}' does not match "
                f"pattern '{{source}}::step::{{position}}'"
            )
        elif step_id:
            # Check that source matches
            match = CANONICAL_STEP_PATTERN.match(step_id)
            if match and source_id and match.group(1) != source_id:
                errors.append(
                    f"perturbed step {i} canonical_step_id source '{match.group(1)}' "
                    f"does not match source_trajectory_id '{source_id}'"
                )

    return ValidationResult(
        passed=len(errors) == 0,
        validator_name="id_format",
        errors=errors,
        warnings=warnings,
    )


def validate_structure(unit: Dict[str, Any]) -> ValidationResult:
    """
    Validate the structural integrity of an evaluation unit.

    Checks:
    - baseline.trajectory exists and has steps
    - perturbed.trajectory exists and has steps
    - perturbation_record exists in perturbed
    - all required top-level fields exist

    Args:
        unit: The evaluation unit dictionary to validate

    Returns:
        ValidationResult with pass/fail status and any errors/warnings
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check required top-level fields
    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        if field_name not in unit:
            errors.append(f"Missing required top-level field: '{field_name}'")

    # Check baseline structure
    baseline = unit.get("baseline")
    if baseline is None:
        errors.append("baseline is None")
    elif not isinstance(baseline, dict):
        errors.append(f"baseline is not a dict, got {type(baseline).__name__}")
    else:
        baseline_trajectory = baseline.get("trajectory")
        if baseline_trajectory is None:
            errors.append("baseline.trajectory is missing")
        elif not isinstance(baseline_trajectory, dict):
            errors.append(
                f"baseline.trajectory is not a dict, got {type(baseline_trajectory).__name__}"
            )
        else:
            baseline_steps = baseline_trajectory.get("steps")
            if baseline_steps is None:
                errors.append("baseline.trajectory.steps is missing")
            elif not isinstance(baseline_steps, list):
                errors.append(
                    f"baseline.trajectory.steps is not a list, got {type(baseline_steps).__name__}"
                )
            elif len(baseline_steps) == 0:
                warnings.append("baseline.trajectory.steps is empty")

    # Check perturbed structure
    perturbed = unit.get("perturbed")
    if perturbed is None:
        errors.append("perturbed is None")
    elif not isinstance(perturbed, dict):
        errors.append(f"perturbed is not a dict, got {type(perturbed).__name__}")
    else:
        perturbed_trajectory = perturbed.get("trajectory")
        if perturbed_trajectory is None:
            errors.append("perturbed.trajectory is missing")
        elif not isinstance(perturbed_trajectory, dict):
            errors.append(
                f"perturbed.trajectory is not a dict, got {type(perturbed_trajectory).__name__}"
            )
        else:
            perturbed_steps = perturbed_trajectory.get("steps")
            if perturbed_steps is None:
                errors.append("perturbed.trajectory.steps is missing")
            elif not isinstance(perturbed_steps, list):
                errors.append(
                    f"perturbed.trajectory.steps is not a list, got {type(perturbed_steps).__name__}"
                )
            elif len(perturbed_steps) == 0:
                warnings.append("perturbed.trajectory.steps is empty")

        # Check perturbation_record
        perturbation_record = perturbed.get("perturbation_record")
        if perturbation_record is None:
            errors.append("perturbed.perturbation_record is missing")
        elif not isinstance(perturbation_record, dict):
            errors.append(
                f"perturbed.perturbation_record is not a dict, "
                f"got {type(perturbation_record).__name__}"
            )

    return ValidationResult(
        passed=len(errors) == 0,
        validator_name="structure",
        errors=errors,
        warnings=warnings,
    )


def validate_step_identity(unit: Dict[str, Any]) -> ValidationResult:
    """
    Validate step identity fields for consistency.

    Checks:
    - all steps have canonical_step_id
    - all steps have display_step_index
    - display_step_index is sequential (1, 2, 3, ...)
    - canonical_step_ids are consistent between baseline and perturbed
      (same step has same canonical_id)
    - target_step_canonical_id from perturbation_record exists in baseline

    Args:
        unit: The evaluation unit dictionary to validate

    Returns:
        ValidationResult with pass/fail status and any errors/warnings
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Extract trajectories
    baseline = unit.get("baseline", {})
    perturbed = unit.get("perturbed", {})
    baseline_trajectory = baseline.get("trajectory", {})
    perturbed_trajectory = perturbed.get("trajectory", {})
    baseline_steps = baseline_trajectory.get("steps", [])
    perturbed_steps = perturbed_trajectory.get("steps", [])

    # Build set of baseline canonical_step_ids for reference
    baseline_canonical_ids = set()

    # Validate baseline steps
    for i, step in enumerate(baseline_steps):
        # Check canonical_step_id exists
        if "canonical_step_id" not in step:
            errors.append(f"baseline step {i} missing canonical_step_id")
        else:
            baseline_canonical_ids.add(step["canonical_step_id"])

        # Check display_step_index exists
        if "display_step_index" not in step:
            errors.append(f"baseline step {i} missing display_step_index")
        else:
            expected_display_index = i + 1  # 1-indexed
            actual_display_index = step["display_step_index"]
            if actual_display_index != expected_display_index:
                errors.append(
                    f"baseline step {i} display_step_index is {actual_display_index}, "
                    f"expected {expected_display_index}"
                )

    # Validate perturbed steps
    for i, step in enumerate(perturbed_steps):
        # Check canonical_step_id exists
        if "canonical_step_id" not in step:
            errors.append(f"perturbed step {i} missing canonical_step_id")

        # Check display_step_index exists
        if "display_step_index" not in step:
            errors.append(f"perturbed step {i} missing display_step_index")
        else:
            expected_display_index = i + 1  # 1-indexed
            actual_display_index = step["display_step_index"]
            if actual_display_index != expected_display_index:
                errors.append(
                    f"perturbed step {i} display_step_index is {actual_display_index}, "
                    f"expected {expected_display_index}"
                )

    # Check canonical_step_id consistency for corresponding positions
    # Steps at the same position (before perturbation) should have same canonical_id
    perturbation_record = perturbed.get("perturbation_record", {})
    target_step_index = perturbation_record.get("target_step_index", -1)

    # Check steps before the target position
    min_len = min(len(baseline_steps), len(perturbed_steps))
    for i in range(min(min_len, target_step_index)):
        baseline_step = baseline_steps[i]
        perturbed_step = perturbed_steps[i]
        baseline_canonical = baseline_step.get("canonical_step_id")
        perturbed_canonical = perturbed_step.get("canonical_step_id")
        if baseline_canonical and perturbed_canonical:
            if baseline_canonical != perturbed_canonical:
                warnings.append(
                    f"step {i} canonical_step_id mismatch before target: "
                    f"baseline='{baseline_canonical}', perturbed='{perturbed_canonical}'"
                )

    # Check target_step_canonical_id exists in baseline
    derived_cache = unit.get("derived_cache", {})
    target_step_canonical_id = derived_cache.get("target_step_canonical_id", "")
    if target_step_canonical_id:
        if target_step_canonical_id not in baseline_canonical_ids:
            errors.append(
                f"target_step_canonical_id '{target_step_canonical_id}' "
                f"not found in baseline steps"
            )
    elif target_step_index >= 0:
        warnings.append("target_step_canonical_id not set in derived_cache")

    return ValidationResult(
        passed=len(errors) == 0,
        validator_name="step_identity",
        errors=errors,
        warnings=warnings,
    )


def validate_derived_cache(unit: Dict[str, Any]) -> ValidationResult:
    """
    Validate that derived_cache fields are consistent with canonical data.

    Uses verify_cache_consistency() from derived_cache module to check
    that all cached fields match their canonical sources.

    Args:
        unit: The evaluation unit dictionary to validate

    Returns:
        ValidationResult with pass/fail status and any errors/warnings
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check if derived_cache exists
    if "derived_cache" not in unit:
        errors.append("derived_cache is missing")
        return ValidationResult(
            passed=False,
            validator_name="derived_cache",
            errors=errors,
            warnings=warnings,
        )

    try:
        is_consistent, mismatches = verify_cache_consistency(unit)
        if not is_consistent:
            for mismatch in mismatches:
                errors.append(f"Cache inconsistency: {mismatch}")
    except Exception as e:
        errors.append(f"Error verifying cache consistency: {str(e)}")

    return ValidationResult(
        passed=len(errors) == 0,
        validator_name="derived_cache",
        errors=errors,
        warnings=warnings,
    )


def validate_capabilities_tier(unit: Dict[str, Any]) -> ValidationResult:
    """
    Validate evaluation capabilities and replay tier.

    Checks:
    - all capability flags are boolean
    - replay_tier is derived correctly from capabilities
    - replay_tier is 1, 2, 3, or None

    Args:
        unit: The evaluation unit dictionary to validate

    Returns:
        ValidationResult with pass/fail status and any errors/warnings
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Required capability fields
    capability_fields = [
        "has_objective_verifier",
        "can_replay",
        "can_regenerate_downstream",
        "environment_accessible",
        "ground_truth_available",
    ]

    # Check evaluation_capabilities exists
    capabilities = unit.get("evaluation_capabilities")
    if capabilities is None:
        errors.append("evaluation_capabilities is missing")
        return ValidationResult(
            passed=False,
            validator_name="capabilities_tier",
            errors=errors,
            warnings=warnings,
        )

    if not isinstance(capabilities, dict):
        errors.append(
            f"evaluation_capabilities is not a dict, got {type(capabilities).__name__}"
        )
        return ValidationResult(
            passed=False,
            validator_name="capabilities_tier",
            errors=errors,
            warnings=warnings,
        )

    # Check all capability fields are present and boolean
    for cap_field in capability_fields:
        if cap_field not in capabilities:
            errors.append(f"evaluation_capabilities.{cap_field} is missing")
        elif not isinstance(capabilities[cap_field], bool):
            errors.append(
                f"evaluation_capabilities.{cap_field} is not a boolean, "
                f"got {type(capabilities[cap_field]).__name__}"
            )

    # Check replay_tier validity
    replay_tier = unit.get("replay_tier")
    valid_tiers = [1, 2, 3, None]
    if replay_tier not in valid_tiers:
        errors.append(f"replay_tier is {replay_tier!r}, must be one of {valid_tiers}")

    # Verify replay_tier matches computed value from capabilities
    if all(
        cap in capabilities and isinstance(capabilities[cap], bool)
        for cap in capability_fields
    ):
        try:
            expected_tier = assign_replay_tier(capabilities)
            if replay_tier != expected_tier:
                errors.append(
                    f"replay_tier is {replay_tier}, but computed from capabilities is {expected_tier}"
                )
        except Exception as e:
            errors.append(f"Error computing replay tier: {str(e)}")

    return ValidationResult(
        passed=len(errors) == 0,
        validator_name="capabilities_tier",
        errors=errors,
        warnings=warnings,
    )


def validate_blinding(unit: Dict[str, Any]) -> ValidationResult:
    """
    Validate blinding assignment consistency.

    Checks:
    - blinding assignment exists
    - variant IDs in blinding match unit's variants
    - is_a_baseline and is_a_perturbed are consistent (one True, one False)

    Args:
        unit: The evaluation unit dictionary to validate

    Returns:
        ValidationResult with pass/fail status and any errors/warnings
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check blinding exists
    blinding = unit.get("blinding")
    if blinding is None:
        errors.append("blinding assignment is missing")
        return ValidationResult(
            passed=False,
            validator_name="blinding",
            errors=errors,
            warnings=warnings,
        )

    if not isinstance(blinding, dict):
        errors.append(f"blinding is not a dict, got {type(blinding).__name__}")
        return ValidationResult(
            passed=False,
            validator_name="blinding",
            errors=errors,
            warnings=warnings,
        )

    # Get variant IDs from unit
    baseline = unit.get("baseline", {})
    perturbed = unit.get("perturbed", {})
    baseline_variant_id = baseline.get("trajectory_variant_id", "")
    perturbed_variant_id = perturbed.get("trajectory_variant_id", "")

    # Get variant IDs from blinding
    trajectory_a_variant_id = blinding.get("trajectory_a_variant_id", "")
    trajectory_b_variant_id = blinding.get("trajectory_b_variant_id", "")

    # Check that blinding variant IDs match unit's variants
    unit_variant_ids = {baseline_variant_id, perturbed_variant_id}
    blinding_variant_ids = {trajectory_a_variant_id, trajectory_b_variant_id}

    if unit_variant_ids != blinding_variant_ids:
        # Remove empty strings for cleaner error message
        unit_ids_clean = {v for v in unit_variant_ids if v}
        blinding_ids_clean = {v for v in blinding_variant_ids if v}
        errors.append(
            f"blinding variant IDs {blinding_ids_clean} do not match "
            f"unit variant IDs {unit_ids_clean}"
        )

    # Check is_a_baseline and is_a_perturbed flags
    is_a_baseline = blinding.get("is_a_baseline")
    is_a_perturbed = blinding.get("is_a_perturbed")

    if is_a_baseline is None:
        errors.append("blinding.is_a_baseline is missing")
    elif not isinstance(is_a_baseline, bool):
        errors.append(
            f"blinding.is_a_baseline is not a boolean, got {type(is_a_baseline).__name__}"
        )

    if is_a_perturbed is None:
        errors.append("blinding.is_a_perturbed is missing")
    elif not isinstance(is_a_perturbed, bool):
        errors.append(
            f"blinding.is_a_perturbed is not a boolean, got {type(is_a_perturbed).__name__}"
        )

    # Check consistency: is_a_baseline and is_a_perturbed should be opposites
    if isinstance(is_a_baseline, bool) and isinstance(is_a_perturbed, bool):
        if is_a_baseline == is_a_perturbed:
            errors.append(
                f"blinding flags inconsistent: is_a_baseline={is_a_baseline}, "
                f"is_a_perturbed={is_a_perturbed} (should be opposites)"
            )

        # Verify that A/B assignment is consistent with flags
        if is_a_baseline:
            if trajectory_a_variant_id != baseline_variant_id:
                errors.append(
                    f"is_a_baseline is True but trajectory_a_variant_id "
                    f"'{trajectory_a_variant_id}' != baseline '{baseline_variant_id}'"
                )
            if trajectory_b_variant_id != perturbed_variant_id:
                errors.append(
                    f"is_a_baseline is True but trajectory_b_variant_id "
                    f"'{trajectory_b_variant_id}' != perturbed '{perturbed_variant_id}'"
                )
        else:
            if trajectory_a_variant_id != perturbed_variant_id:
                errors.append(
                    f"is_a_baseline is False but trajectory_a_variant_id "
                    f"'{trajectory_a_variant_id}' != perturbed '{perturbed_variant_id}'"
                )
            if trajectory_b_variant_id != baseline_variant_id:
                errors.append(
                    f"is_a_baseline is False but trajectory_b_variant_id "
                    f"'{trajectory_b_variant_id}' != baseline '{baseline_variant_id}'"
                )

    # Check evaluation_unit_id in blinding matches unit
    blinding_unit_id = blinding.get("evaluation_unit_id", "")
    unit_id = unit.get("evaluation_unit_id", "")
    if blinding_unit_id and unit_id and blinding_unit_id != unit_id:
        errors.append(
            f"blinding.evaluation_unit_id '{blinding_unit_id}' does not match "
            f"unit's evaluation_unit_id '{unit_id}'"
        )

    return ValidationResult(
        passed=len(errors) == 0,
        validator_name="blinding",
        errors=errors,
        warnings=warnings,
    )


def validate_all(unit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all validators on an evaluation unit and return a summary.

    Runs the following validators:
    - validate_id_format
    - validate_structure
    - validate_step_identity
    - validate_derived_cache
    - validate_capabilities_tier
    - validate_blinding

    Args:
        unit: The evaluation unit dictionary to validate

    Returns:
        A summary dictionary:
        {
            "unit_id": "...",
            "passed": true/false,
            "validators": {
                "id_format": {"passed": true, "errors": [], "warnings": []},
                "structure": {"passed": true, "errors": [], "warnings": []},
                ...
            },
            "total_errors": 0,
            "total_warnings": 0
        }
    """
    # Run all validators
    validators = [
        validate_id_format,
        validate_structure,
        validate_step_identity,
        validate_derived_cache,
        validate_capabilities_tier,
        validate_blinding,
    ]

    results: Dict[str, Dict[str, Any]] = {}
    total_errors = 0
    total_warnings = 0
    all_passed = True

    for validator in validators:
        result = validator(unit)
        results[result.validator_name] = {
            "passed": result.passed,
            "errors": result.errors,
            "warnings": result.warnings,
        }
        total_errors += len(result.errors)
        total_warnings += len(result.warnings)
        if not result.passed:
            all_passed = False

    return {
        "unit_id": unit.get("evaluation_unit_id", "<unknown>"),
        "passed": all_passed,
        "validators": results,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
    }
