"""
LLM Judge view generation module.

This module provides functions to generate views of evaluation units for LLM judges.
Each view mode provides different information to the judge for evaluation.

Supported judge modes:
- single_trajectory: Judge evaluates just the perturbed trajectory
- blinded_pair: Comparative evaluation with blinded A/B ordering
- labeled_pair: Explicit baseline vs perturbed comparison (for calibration)

IMPORTANT: Views from different modes must NOT be mixed in analysis as they
are scientifically incomparable due to different information exposure.
"""

from typing import Any, Dict, List

from src.evaluation.blinding import get_blinded_trajectory_order


def _format_trajectory_for_view(
    trajectory: Dict[str, Any],
    trajectory_id: str,
) -> Dict[str, Any]:
    """
    Format a trajectory for inclusion in a judge view.

    Returns the full trajectory structure with trajectory_id, steps, and num_steps.
    Steps are included in full (not simplified) as judges need complete information.

    Args:
        trajectory: The trajectory dict containing steps and metadata.
        trajectory_id: The trajectory variant ID for this trajectory.

    Returns:
        Formatted trajectory dict for the view.
    """
    steps = trajectory.get("steps", [])
    return {
        "trajectory_id": trajectory_id,
        "steps": steps,
        "num_steps": len(steps),
    }


def generate_single_trajectory_view(unit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a single trajectory view for LLM judge evaluation.

    This is the most common mode - the judge evaluates just the perturbed
    trajectory without knowledge of the baseline. The judge must identify
    errors and assess their impact independently.

    Args:
        unit: An evaluation unit dict containing:
            - evaluation_unit_id: Unique identifier
            - task_text: The task description
            - perturbed: PerturbedData dict with trajectory

    Returns:
        A view dict with:
            - judge_mode: "single_trajectory"
            - evaluation_unit_id: The unit ID
            - task_text: The task description
            - trajectory: Full trajectory with trajectory_id, steps, num_steps
            - expected_outputs: List of expected judge output fields

    Example:
        >>> unit = {
        ...     "evaluation_unit_id": "eval::gaia_122::001",
        ...     "task_text": "Find the capital of France",
        ...     "perturbed": {
        ...         "trajectory_variant_id": "gaia_122::pert::001",
        ...         "trajectory": {"steps": [...], ...}
        ...     }
        ... }
        >>> view = generate_single_trajectory_view(unit)
        >>> view["judge_mode"]
        'single_trajectory'
    """
    evaluation_unit_id = unit.get("evaluation_unit_id", "")
    task_text = unit.get("task_text", "")

    # Extract perturbed trajectory data
    perturbed_data = unit.get("perturbed", {})
    perturbed_trajectory = perturbed_data.get("trajectory", {})
    perturbed_variant_id = perturbed_data.get("trajectory_variant_id", "")

    # Format trajectory for the view
    trajectory_view = _format_trajectory_for_view(
        perturbed_trajectory,
        perturbed_variant_id,
    )

    return {
        "judge_mode": "single_trajectory",
        "evaluation_unit_id": evaluation_unit_id,
        "task_text": task_text,
        "trajectory": trajectory_view,
        "expected_outputs": [
            "overall_score",
            "error_detected",
            "predicted_error_step",
            "predicted_impact",
        ],
    }


def generate_blinded_pair_view(unit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a blinded pair view for comparative LLM judge evaluation.

    In this mode, the judge receives both trajectories in a blinded A/B order
    determined by the unit's blinding assignment. The judge does not know
    which trajectory is baseline vs perturbed.

    This mode is useful for:
    - Preference comparisons
    - Error detection through comparison
    - Assessing relative quality

    Args:
        unit: An evaluation unit dict containing:
            - evaluation_unit_id: Unique identifier
            - task_text: The task description
            - baseline: BaselineData dict with trajectory
            - perturbed: PerturbedData dict with trajectory
            - blinding: BlindingAssignment dict with A/B order

    Returns:
        A view dict with:
            - judge_mode: "blinded_pair"
            - evaluation_unit_id: The unit ID
            - task_text: The task description
            - trajectory_a: First trajectory (blinded order)
            - trajectory_b: Second trajectory (blinded order)
            - expected_outputs: List of expected judge output fields

    Example:
        >>> view = generate_blinded_pair_view(unit)
        >>> view["judge_mode"]
        'blinded_pair'
        >>> "trajectory_a" in view and "trajectory_b" in view
        True
    """
    evaluation_unit_id = unit.get("evaluation_unit_id", "")
    task_text = unit.get("task_text", "")

    # Use blinding module to get trajectories in blinded order
    trajectory_a, trajectory_b, variant_a_id, variant_b_id = (
        get_blinded_trajectory_order(unit)
    )

    # Format trajectories for the view
    trajectory_a_view = _format_trajectory_for_view(trajectory_a, variant_a_id)
    trajectory_b_view = _format_trajectory_for_view(trajectory_b, variant_b_id)

    return {
        "judge_mode": "blinded_pair",
        "evaluation_unit_id": evaluation_unit_id,
        "task_text": task_text,
        "trajectory_a": trajectory_a_view,
        "trajectory_b": trajectory_b_view,
        "expected_outputs": [
            "preference",
            "error_trajectory",
            "error_step",
        ],
    }


def generate_labeled_pair_view(unit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a labeled pair view for calibration studies.

    In this mode, the judge explicitly knows which trajectory is baseline
    and which is perturbed, along with a summary of the perturbation.
    This is primarily used for:
    - Calibration studies
    - Testing judge sensitivity
    - Validating perturbation detectability

    WARNING: This mode exposes ground truth labels. Results from this mode
    should NOT be mixed with blinded evaluation results in analysis.

    Args:
        unit: An evaluation unit dict containing:
            - evaluation_unit_id: Unique identifier
            - task_text: The task description
            - baseline: BaselineData dict with trajectory
            - perturbed: PerturbedData dict with trajectory and perturbation_record
            - derived_cache: DerivedCache dict with perturbation metadata

    Returns:
        A view dict with:
            - judge_mode: "labeled_pair"
            - evaluation_unit_id: The unit ID
            - task_text: The task description
            - baseline_trajectory: The baseline trajectory (labeled)
            - perturbed_trajectory: The perturbed trajectory (labeled)
            - perturbation_summary: Summary of the perturbation applied
            - expected_outputs: List of expected judge output fields

    Example:
        >>> view = generate_labeled_pair_view(unit)
        >>> view["judge_mode"]
        'labeled_pair'
        >>> "baseline_trajectory" in view
        True
        >>> "perturbation_summary" in view
        True
    """
    evaluation_unit_id = unit.get("evaluation_unit_id", "")
    task_text = unit.get("task_text", "")

    # Extract baseline trajectory data
    baseline_data = unit.get("baseline", {})
    baseline_trajectory = baseline_data.get("trajectory", {})
    baseline_variant_id = baseline_data.get("trajectory_variant_id", "")

    # Extract perturbed trajectory data
    perturbed_data = unit.get("perturbed", {})
    perturbed_trajectory = perturbed_data.get("trajectory", {})
    perturbed_variant_id = perturbed_data.get("trajectory_variant_id", "")

    # Extract perturbation metadata from derived_cache
    derived_cache = unit.get("derived_cache", {})

    # Build perturbation summary from derived cache
    perturbation_summary = {
        "target_step_canonical_id": derived_cache.get("target_step_canonical_id", ""),
        "perturbation_type": derived_cache.get("perturbation_type", ""),
        "perturbation_class": derived_cache.get("perturbation_class", ""),
        "perturbation_family": derived_cache.get("perturbation_family", ""),
    }

    # Format trajectories for the view
    baseline_view = _format_trajectory_for_view(
        baseline_trajectory, baseline_variant_id
    )
    perturbed_view = _format_trajectory_for_view(
        perturbed_trajectory, perturbed_variant_id
    )

    return {
        "judge_mode": "labeled_pair",
        "evaluation_unit_id": evaluation_unit_id,
        "task_text": task_text,
        "baseline_trajectory": baseline_view,
        "perturbed_trajectory": perturbed_view,
        "perturbation_summary": perturbation_summary,
        "expected_outputs": [
            "impact_estimate",
            "detectability_estimate",
        ],
    }


def generate_view(
    unit: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    """
    Generate a judge view for the specified mode.

    This is a convenience function that dispatches to the appropriate
    view generator based on the mode parameter.

    Args:
        unit: An evaluation unit dict.
        mode: The judge mode ("single_trajectory", "blinded_pair", or "labeled_pair").

    Returns:
        The generated view dict for the specified mode.

    Raises:
        ValueError: If an invalid mode is specified.

    Example:
        >>> view = generate_view(unit, "single_trajectory")
        >>> view["judge_mode"]
        'single_trajectory'
    """
    generators = {
        "single_trajectory": generate_single_trajectory_view,
        "blinded_pair": generate_blinded_pair_view,
        "labeled_pair": generate_labeled_pair_view,
    }

    if mode not in generators:
        valid_modes = ", ".join(generators.keys())
        raise ValueError(
            f"Invalid judge mode: '{mode}'. Valid modes are: {valid_modes}"
        )

    return generators[mode](unit)


def validate_mode_consistency(views: List[Dict[str, Any]]) -> bool:
    """
    Validate that all views in a list use the same judge mode.

    Views from different modes must NOT be mixed in analysis as they
    provide different information to the judge and are scientifically
    incomparable.

    Args:
        views: List of view dicts to validate.

    Returns:
        True if all views have the same judge_mode, False otherwise.

    Raises:
        ValueError: If the views list is empty.

    Example:
        >>> views = [
        ...     {"judge_mode": "single_trajectory", ...},
        ...     {"judge_mode": "single_trajectory", ...},
        ... ]
        >>> validate_mode_consistency(views)
        True
    """
    if not views:
        raise ValueError("Cannot validate mode consistency with empty views list")

    first_mode = views[0].get("judge_mode")
    return all(view.get("judge_mode") == first_mode for view in views)


def get_supported_modes() -> List[str]:
    """
    Get the list of supported judge modes.

    Returns:
        List of valid mode strings.

    Example:
        >>> modes = get_supported_modes()
        >>> "single_trajectory" in modes
        True
    """
    return ["single_trajectory", "blinded_pair", "labeled_pair"]
