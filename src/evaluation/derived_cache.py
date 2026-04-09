"""
Derived cache building and verification for evaluation units.

This module provides functions to build, verify, and rebuild the derived_cache
component of evaluation units. The derived_cache contains denormalized fields
extracted from canonical objects for quick filtering. These fields are NOT
the source of truth - they must be regeneratable from canonical data.

The canonical sources are:
- baseline["trajectory"] for baseline trajectory data
- perturbed["trajectory"] for perturbed trajectory data
- perturbation_record for perturbation metadata
"""

from typing import Any, Dict, List, Tuple

from src.evaluation.ids import generate_canonical_step_id

# Fields that must match between derived_cache and canonical sources
DERIVED_CACHE_FIELDS = [
    "baseline_outcome",
    "baseline_num_steps",
    "perturbed_num_steps",
    "perturbation_class",
    "perturbation_family",
    "perturbation_type",
    "target_step_canonical_id",
    "expected_impact",
    "expected_detectability",
]


def build_derived_cache(
    baseline: Dict[str, Any],
    perturbed: Dict[str, Any],
    perturbation_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a derived cache from canonical objects.

    Extracts denormalized fields from baseline trajectory, perturbed trajectory,
    and perturbation record for quick filtering. The derived cache is NOT the
    source of truth - it is regeneratable from the canonical objects.

    Args:
        baseline: The baseline data dict containing "trajectory" with
            "baseline_outcome" and "steps" fields.
        perturbed: The perturbed data dict containing "trajectory" with
            "steps" field.
        perturbation_record: The perturbation record dict containing
            perturbation metadata fields.

    Returns:
        A dictionary containing derived cache fields:
            - _warning: Reminder that this is derived data
            - baseline_outcome: From baseline trajectory (default 0.0)
            - baseline_num_steps: Number of steps in baseline trajectory
            - perturbed_num_steps: Number of steps in perturbed trajectory
            - perturbation_class: From perturbation record
            - perturbation_family: From perturbation record
            - perturbation_type: From perturbation record
            - target_step_canonical_id: Generated from source_trajectory_id and target_step_index
            - expected_impact: From perturbation record (default 0)
            - expected_detectability: From perturbation record (default 1)

    Examples:
        >>> baseline = {
        ...     "trajectory": {
        ...         "baseline_outcome": 1.0,
        ...         "steps": [{"step_index": 0}, {"step_index": 1}]
        ...     }
        ... }
        >>> perturbed = {
        ...     "trajectory": {
        ...         "steps": [{"step_index": 0}, {"step_index": 1}, {"step_index": 2}]
        ...     }
        ... }
        >>> perturbation_record = {
        ...     "source_trajectory_id": "gaia_122",
        ...     "target_step_index": 1,
        ...     "perturbation_class": "semantic",
        ...     "perturbation_family": "tool_selection",
        ...     "perturbation_type": "wrong_tool",
        ...     "expected_impact": 2,
        ...     "expected_detectability": 1
        ... }
        >>> cache = build_derived_cache(baseline, perturbed, perturbation_record)
        >>> cache["baseline_outcome"]
        1.0
        >>> cache["baseline_num_steps"]
        2
        >>> cache["target_step_canonical_id"]
        'gaia_122::step::1'
    """
    # Extract baseline trajectory data
    baseline_trajectory = baseline.get("trajectory", {})
    baseline_outcome = baseline_trajectory.get("baseline_outcome", 0.0)
    baseline_steps = baseline_trajectory.get("steps", [])
    baseline_num_steps = len(baseline_steps)

    # Extract perturbed trajectory data
    perturbed_trajectory = perturbed.get("trajectory", {})
    perturbed_steps = perturbed_trajectory.get("steps", [])
    perturbed_num_steps = len(perturbed_steps)

    # Extract perturbation record data
    perturbation_class = perturbation_record.get("perturbation_class", "")
    perturbation_family = perturbation_record.get("perturbation_family", "")
    perturbation_type = perturbation_record.get("perturbation_type", "")
    expected_impact = perturbation_record.get("expected_impact", 0)
    expected_detectability = perturbation_record.get("expected_detectability", 1)

    # Generate target step canonical ID
    # Note: perturbation_record uses "original_trajectory_id" not "source_trajectory_id"
    source_trajectory_id = perturbation_record.get("original_trajectory_id", "")
    target_step_index = perturbation_record.get("target_step_index", 0)
    target_step_canonical_id = generate_canonical_step_id(
        source_trajectory_id, target_step_index
    )

    return {
        "_warning": "Derived from canonical objects. Regenerate, never hand-edit.",
        "baseline_outcome": baseline_outcome,
        "baseline_num_steps": baseline_num_steps,
        "perturbed_num_steps": perturbed_num_steps,
        "perturbation_class": perturbation_class,
        "perturbation_family": perturbation_family,
        "perturbation_type": perturbation_type,
        "target_step_canonical_id": target_step_canonical_id,
        "expected_impact": expected_impact,
        "expected_detectability": expected_detectability,
    }


def verify_cache_consistency(
    evaluation_unit: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Verify that derived_cache fields are consistent with canonical objects.

    Rebuilds the cache from canonical objects and compares each field to
    detect any inconsistencies caused by manual edits or data corruption.

    Args:
        evaluation_unit: The full evaluation unit dict containing:
            - baseline: Baseline data with trajectory
            - perturbed: Perturbed data with trajectory and perturbation_record
            - derived_cache: The cache to verify

    Returns:
        A tuple of (is_consistent, mismatches) where:
            - is_consistent: True if all fields match, False otherwise
            - mismatches: List of mismatch descriptions, empty if consistent

    Examples:
        >>> unit = {
        ...     "baseline": {
        ...         "trajectory": {"baseline_outcome": 1.0, "steps": [{}, {}]}
        ...     },
        ...     "perturbed": {
        ...         "trajectory": {"steps": [{}, {}, {}]},
        ...         "perturbation_record": {
        ...             "source_trajectory_id": "gaia_122",
        ...             "target_step_index": 1,
        ...             "perturbation_class": "semantic",
        ...             "perturbation_family": "tool_selection",
        ...             "perturbation_type": "wrong_tool",
        ...             "expected_impact": 2,
        ...             "expected_detectability": 1
        ...         }
        ...     },
        ...     "derived_cache": {
        ...         "baseline_outcome": 1.0,
        ...         "baseline_num_steps": 2,
        ...         "perturbed_num_steps": 3,
        ...         "perturbation_class": "semantic",
        ...         "perturbation_family": "tool_selection",
        ...         "perturbation_type": "wrong_tool",
        ...         "target_step_canonical_id": "gaia_122::step::1",
        ...         "expected_impact": 2,
        ...         "expected_detectability": 1
        ...     }
        ... }
        >>> is_consistent, mismatches = verify_cache_consistency(unit)
        >>> is_consistent
        True
        >>> len(mismatches)
        0
    """
    # Extract canonical objects
    baseline = evaluation_unit.get("baseline", {})
    perturbed = evaluation_unit.get("perturbed", {})
    perturbation_record = perturbed.get("perturbation_record", {})
    existing_cache = evaluation_unit.get("derived_cache", {})

    # Rebuild the cache from canonical sources
    rebuilt_cache = build_derived_cache(baseline, perturbed, perturbation_record)

    # Compare each field
    mismatches: List[str] = []
    for field in DERIVED_CACHE_FIELDS:
        existing_value = existing_cache.get(field)
        rebuilt_value = rebuilt_cache.get(field)

        if existing_value != rebuilt_value:
            mismatches.append(
                f"Field '{field}': cached={existing_value!r}, expected={rebuilt_value!r}"
            )

    is_consistent = len(mismatches) == 0
    return (is_consistent, mismatches)


def rebuild_derived_cache(evaluation_unit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rebuild and return a new derived_cache from canonical objects.

    This is a convenience function that extracts the canonical objects from
    an evaluation unit and builds a fresh derived_cache. Use this to fix
    inconsistent caches or regenerate after canonical data changes.

    Args:
        evaluation_unit: The full evaluation unit dict containing:
            - baseline: Baseline data with trajectory
            - perturbed: Perturbed data with trajectory and perturbation_record

    Returns:
        A new derived_cache dictionary built from canonical objects.

    Examples:
        >>> unit = {
        ...     "baseline": {
        ...         "trajectory": {"baseline_outcome": 0.5, "steps": [{}]}
        ...     },
        ...     "perturbed": {
        ...         "trajectory": {"steps": [{}, {}]},
        ...         "perturbation_record": {
        ...             "source_trajectory_id": "toolbench_999",
        ...             "target_step_index": 0,
        ...             "perturbation_class": "placebo",
        ...             "perturbation_family": "formatting",
        ...             "perturbation_type": "whitespace",
        ...             "expected_impact": 0,
        ...             "expected_detectability": 0
        ...         }
        ...     },
        ...     "derived_cache": {"baseline_outcome": 999}  # Wrong, will be fixed
        ... }
        >>> new_cache = rebuild_derived_cache(unit)
        >>> new_cache["baseline_outcome"]
        0.5
        >>> new_cache["target_step_canonical_id"]
        'toolbench_999::step::0'
    """
    baseline = evaluation_unit.get("baseline", {})
    perturbed = evaluation_unit.get("perturbed", {})
    perturbation_record = perturbed.get("perturbation_record", {})

    return build_derived_cache(baseline, perturbed, perturbation_record)
