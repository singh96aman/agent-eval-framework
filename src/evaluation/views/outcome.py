"""
Outcome verification view generation for evaluation units.

This module generates views for outcome verification (Section 5C).
Outcome verification always sees both trajectories (no blinding needed)
and includes verification configuration based on benchmark type.

Key features:
- Benchmark-specific verification configurations
- Full trajectory data for replay/verification
- Complete perturbation records for impact analysis
- Replay tier information for tiered evidence collection
"""

from typing import Any, Dict, Optional


def get_verification_config(benchmark: str, unit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return verification configuration based on benchmark type.

    Different benchmarks support different verifier types:
    - swebench: Test suite verification with pytest
    - gaia: Exact match verification against expected answer
    - toolbench: Heuristic verification checking API success and final answer

    Args:
        benchmark: The benchmark name (swebench, gaia, toolbench)
        unit: The evaluation unit dictionary containing trajectory data

    Returns:
        Verification config dict with:
            - verifier_type: Type of verifier (test_suite, exact_match, heuristic)
            - verifier_params: Parameters for the verifier
            - expected_answer: Expected answer if available (from trajectory)

    Raises:
        ValueError: If benchmark is not recognized
    """
    benchmark_lower = benchmark.lower()

    if benchmark_lower == "swebench":
        return {
            "verifier_type": "test_suite",
            "verifier_params": {
                "test_command": "pytest",
                "timeout_seconds": 300,
            },
            "expected_answer": None,
        }

    if benchmark_lower == "gaia":
        # Extract expected answer from baseline trajectory if available
        expected_answer = _extract_expected_answer(unit)
        return {
            "verifier_type": "exact_match",
            "verifier_params": {
                "case_sensitive": False,
                "normalize_whitespace": True,
            },
            "expected_answer": expected_answer,
        }

    if benchmark_lower == "toolbench":
        # Extract expected answer from trajectory if available
        expected_answer = _extract_expected_answer(unit)
        return {
            "verifier_type": "heuristic",
            "verifier_params": {
                "check_api_success": True,
                "check_final_answer": True,
            },
            "expected_answer": expected_answer,
        }

    raise ValueError(
        f"Unknown benchmark '{benchmark}'. "
        f"Expected one of: swebench, gaia, toolbench"
    )


def _extract_expected_answer(unit: Dict[str, Any]) -> Optional[str]:
    """
    Extract expected answer from evaluation unit trajectory data.

    Looks in multiple locations for the expected answer:
    1. baseline.trajectory.expected_answer
    2. perturbed.trajectory.expected_answer

    Args:
        unit: The evaluation unit dictionary

    Returns:
        The expected answer string if found, None otherwise
    """
    # Try baseline trajectory first
    baseline = unit.get("baseline", {})
    baseline_trajectory = baseline.get("trajectory", {})
    expected_answer = baseline_trajectory.get("expected_answer")

    if expected_answer is not None:
        return str(expected_answer) if expected_answer else None

    # Fall back to perturbed trajectory
    perturbed = unit.get("perturbed", {})
    perturbed_trajectory = perturbed.get("trajectory", {})
    expected_answer = perturbed_trajectory.get("expected_answer")

    if expected_answer is not None:
        return str(expected_answer) if expected_answer else None

    return None


def generate_outcome_view(unit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate the full outcome verification view for replay/verification.

    This view provides everything needed for outcome evidence collection
    (Section 5C). Outcome verification always sees both trajectories
    with no blinding, as it needs the full context to execute
    verification methods.

    The view includes:
    - Full baseline trajectory (TypedTrajectory)
    - Full perturbed trajectory (TypedTrajectory)
    - Complete perturbation record for impact analysis
    - Verification configuration based on benchmark
    - Replay tier information for tiered evidence collection

    Args:
        unit: The evaluation unit dictionary containing:
            - evaluation_unit_id: Unique identifier
            - benchmark: Benchmark name
            - task_id: Task identifier
            - source_trajectory_id: Reference to original trajectory
            - replay_tier: Tier for outcome evidence (1, 2, or 3)
            - baseline: BaselineData with trajectory
            - perturbed: PerturbedData with trajectory and perturbation_record

    Returns:
        Outcome verification view dict with:
            - evaluation_unit_id: The evaluation unit ID
            - benchmark: The benchmark name
            - task_id: The task identifier
            - source_trajectory_id: Reference to source trajectory
            - replay_tier: The replay tier (1, 2, or 3)
            - baseline: Baseline trajectory data
            - perturbed: Perturbed trajectory data with perturbation record
            - verification_config: Benchmark-specific verification config

    Raises:
        KeyError: If required fields are missing from the unit
        ValueError: If benchmark is not recognized
    """
    # Extract required fields
    evaluation_unit_id = unit["evaluation_unit_id"]
    benchmark = unit["benchmark"]
    task_id = unit["task_id"]
    source_trajectory_id = unit["source_trajectory_id"]
    replay_tier = unit.get("replay_tier")

    # Extract baseline data
    baseline_data = unit["baseline"]
    baseline_variant_id = baseline_data["trajectory_variant_id"]
    baseline_trajectory = baseline_data["trajectory"]

    # Extract perturbed data
    perturbed_data = unit["perturbed"]
    perturbed_variant_id = perturbed_data["trajectory_variant_id"]
    perturbed_trajectory = perturbed_data["trajectory"]
    perturbation_record = perturbed_data["perturbation_record"]

    # Get verification config based on benchmark
    verification_config = get_verification_config(benchmark, unit)

    return {
        "evaluation_unit_id": evaluation_unit_id,
        "benchmark": benchmark,
        "task_id": task_id,
        "source_trajectory_id": source_trajectory_id,
        "replay_tier": replay_tier,
        "baseline": {
            "trajectory_variant_id": baseline_variant_id,
            "trajectory": baseline_trajectory,
        },
        "perturbed": {
            "trajectory_variant_id": perturbed_variant_id,
            "trajectory": perturbed_trajectory,
            "perturbation_record": perturbation_record,
        },
        "verification_config": verification_config,
    }


def generate_outcome_views_batch(
    units: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """
    Generate outcome verification views for a batch of evaluation units.

    Convenience function to process multiple evaluation units at once.

    Args:
        units: List of evaluation unit dictionaries

    Returns:
        List of outcome verification view dictionaries

    Raises:
        KeyError: If required fields are missing from any unit
        ValueError: If benchmark is not recognized for any unit
    """
    return [generate_outcome_view(unit) for unit in units]
