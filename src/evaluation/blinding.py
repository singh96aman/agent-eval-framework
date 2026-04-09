"""
Blinding assignment module for evaluation units.

This module provides functions for generating and managing blinded A/B
assignments for evaluation units. Blinding prevents evaluators from knowing
which trajectory has the perturbation, ensuring unbiased evaluation.

Key features:
- Deterministic A/B assignment using SHA-256 hash
- Reproducible with same seed
- Balance verification (45-55% acceptable range)
- Full blinding key document generation

The `is_a_baseline` field is sensitive and lives only in the blinding
assignment, never in evaluator views.
"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


def generate_blinding_assignment(
    evaluation_unit_id: str,
    baseline_variant_id: str,
    perturbed_variant_id: str,
    global_seed: int,
) -> Dict[str, Any]:
    """
    Generate A/B assignment for a single evaluation unit using deterministic hash.

    The assignment is deterministic: given the same evaluation_unit_id and
    global_seed, the same A/B assignment will always be produced.

    Args:
        evaluation_unit_id: Unique identifier for the evaluation unit
            (e.g., "eval::gaia_122::001")
        baseline_variant_id: Variant ID for the baseline trajectory
            (e.g., "gaia_122::base")
        perturbed_variant_id: Variant ID for the perturbed trajectory
            (e.g., "gaia_122::pert::001")
        global_seed: Seed for reproducible randomization

    Returns:
        BlindingAssignment dict with:
            - evaluation_unit_id: The evaluation unit ID
            - trajectory_a_variant_id: Variant ID assigned to position A
            - trajectory_b_variant_id: Variant ID assigned to position B
            - is_a_baseline: True if trajectory A is the baseline (PRIVATE)
            - is_a_perturbed: True if trajectory A is the perturbed (PRIVATE)
    """
    # Deterministic hash based on unit ID and seed
    combined = f"{evaluation_unit_id}:{global_seed}"
    hash_val = hashlib.sha256(combined.encode()).hexdigest()

    # Use first 8 hex characters to determine assignment
    # This gives us 2^32 possible values for good distribution
    is_a_baseline = int(hash_val[:8], 16) % 2 == 0

    if is_a_baseline:
        trajectory_a_variant_id = baseline_variant_id
        trajectory_b_variant_id = perturbed_variant_id
    else:
        trajectory_a_variant_id = perturbed_variant_id
        trajectory_b_variant_id = baseline_variant_id

    return {
        "evaluation_unit_id": evaluation_unit_id,
        "trajectory_a_variant_id": trajectory_a_variant_id,
        "trajectory_b_variant_id": trajectory_b_variant_id,
        "is_a_baseline": is_a_baseline,
        "is_a_perturbed": not is_a_baseline,
    }


def verify_balance(blinding_assignments: List[Dict[str, Any]]) -> Tuple[bool, float]:
    """
    Verify that A-is-baseline ratio is within acceptable range (45-55%).

    A well-balanced blinding assignment ensures that evaluators cannot
    infer which trajectory is baseline based on position alone.

    Args:
        blinding_assignments: List of BlindingAssignment dicts

    Returns:
        Tuple of:
            - is_balanced: True if ratio is within 0.45-0.55
            - ratio: The actual A-is-baseline ratio (0.0 to 1.0)

    Raises:
        ValueError: If blinding_assignments is empty
    """
    if not blinding_assignments:
        raise ValueError("Cannot verify balance with empty assignments list")

    total = len(blinding_assignments)
    a_is_baseline_count = sum(
        1
        for assignment in blinding_assignments
        if assignment.get("is_a_baseline", False)
    )

    ratio = a_is_baseline_count / total

    # Acceptable range is 45-55%
    is_balanced = 0.45 <= ratio <= 0.55

    return is_balanced, ratio


def generate_blinding_key(
    evaluation_units: List[Dict[str, Any]],
    global_seed: int,
) -> Dict[str, Any]:
    """
    Generate the full blinding key document for a set of evaluation units.

    The blinding key document contains all A/B assignments and balance
    verification. This document should be kept private and only used
    for unblinding after evaluation is complete.

    Args:
        evaluation_units: List of evaluation unit dicts, each containing:
            - evaluation_unit_id: Unique identifier
            - baseline.trajectory_variant_id: Baseline variant ID
            - perturbed.trajectory_variant_id: Perturbed variant ID
        global_seed: Seed for reproducible randomization

    Returns:
        BlindingKey dict with:
            - generated_at: ISO datetime of generation
            - global_seed: The seed used
            - balance_check: Balance verification results
            - assignments: List of BlindingAssignment dicts
    """
    assignments = []

    for unit in evaluation_units:
        evaluation_unit_id = unit.get("evaluation_unit_id", "")

        # Extract variant IDs from nested structure
        baseline_data = unit.get("baseline", {})
        perturbed_data = unit.get("perturbed", {})

        baseline_variant_id = baseline_data.get("trajectory_variant_id", "")
        perturbed_variant_id = perturbed_data.get("trajectory_variant_id", "")

        assignment = generate_blinding_assignment(
            evaluation_unit_id=evaluation_unit_id,
            baseline_variant_id=baseline_variant_id,
            perturbed_variant_id=perturbed_variant_id,
            global_seed=global_seed,
        )
        assignments.append(assignment)

    # Calculate balance statistics
    total = len(assignments)
    a_is_baseline_count = sum(1 for a in assignments if a.get("is_a_baseline", False))
    a_is_perturbed_count = total - a_is_baseline_count

    # Compute ratio (handle empty case)
    balance_ratio = a_is_baseline_count / total if total > 0 else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "global_seed": global_seed,
        "balance_check": {
            "a_is_baseline_count": a_is_baseline_count,
            "a_is_perturbed_count": a_is_perturbed_count,
            "total": total,
            "balance_ratio": balance_ratio,
        },
        "assignments": assignments,
    }


def get_blinded_trajectory_order(
    evaluation_unit: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
    """
    Return trajectories in blinded A/B order based on the unit's blinding assignment.

    This function extracts trajectories from an evaluation unit and returns
    them in the order specified by the blinding assignment (A first, B second).
    The caller receives trajectories without knowing which is baseline/perturbed.

    Args:
        evaluation_unit: An evaluation unit dict containing:
            - baseline: BaselineData dict with trajectory
            - perturbed: PerturbedData dict with trajectory
            - blinding: BlindingAssignment dict with A/B assignment

    Returns:
        Tuple of:
            - trajectory_a: The trajectory dict assigned to position A
            - trajectory_b: The trajectory dict assigned to position B
            - variant_a_id: The variant ID for trajectory A
            - variant_b_id: The variant ID for trajectory B

    Raises:
        KeyError: If required fields are missing from evaluation_unit
    """
    # Extract blinding assignment
    blinding = evaluation_unit.get("blinding", {})
    is_a_baseline = blinding.get("is_a_baseline", True)

    # Extract trajectory data
    baseline_data = evaluation_unit.get("baseline", {})
    perturbed_data = evaluation_unit.get("perturbed", {})

    baseline_trajectory = baseline_data.get("trajectory", {})
    perturbed_trajectory = perturbed_data.get("trajectory", {})

    baseline_variant_id = baseline_data.get("trajectory_variant_id", "")
    perturbed_variant_id = perturbed_data.get("trajectory_variant_id", "")

    # Return in blinded order
    if is_a_baseline:
        return (
            baseline_trajectory,
            perturbed_trajectory,
            baseline_variant_id,
            perturbed_variant_id,
        )
    else:
        return (
            perturbed_trajectory,
            baseline_trajectory,
            perturbed_variant_id,
            baseline_variant_id,
        )
