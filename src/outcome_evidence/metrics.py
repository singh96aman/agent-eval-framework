"""
Metrics computation for Section 5C: Outcome Evidence.

Provides functions for computing:
- Outcome Degradation (OD)
- Outcome Degradation Binary
- Propagation Depth (PD)
- Recovery Cost (RC)
- True Impact derivation
"""

from typing import Any, Dict, List, Optional, Tuple


def compute_outcome_degradation(baseline_score: float, perturbed_score: float) -> float:
    """
    Compute Outcome Degradation (OD).

    OD = baseline_score - perturbed_score

    Args:
        baseline_score: Baseline outcome score (0-1)
        perturbed_score: Perturbed outcome score (0-1)

    Returns:
        OD value (-1 to 1)
        - Positive: perturbation caused degradation
        - Zero: no change
        - Negative: perturbation improved outcome (flag for review)
    """
    return baseline_score - perturbed_score


def compute_outcome_degradation_binary(
    baseline_pass: bool, perturbed_pass: bool
) -> int:
    """
    Compute binary Outcome Degradation.

    Args:
        baseline_pass: Whether baseline passed
        perturbed_pass: Whether perturbed passed

    Returns:
        1: baseline passes AND perturbed fails (degradation)
        0: both pass OR both fail (no change)
        -1: baseline fails AND perturbed passes (unexpected improvement)
    """
    if baseline_pass and not perturbed_pass:
        return 1  # Degradation
    elif not baseline_pass and perturbed_pass:
        return -1  # Unexpected improvement
    else:
        return 0  # No change


def compute_propagation_depth(
    baseline_steps: List[Dict[str, Any]],
    perturbed_steps: List[Dict[str, Any]],
    perturbation_step_idx: int,
    semantic_diff_fn: Optional[callable] = None,
) -> Tuple[int, List[str], List[str]]:
    """
    Compute Propagation Depth (PD) - Tier 1-2 only.

    Counts the number of downstream steps whose outputs differ
    between baseline and perturbed trajectories.

    Args:
        baseline_steps: List of baseline step dicts
        perturbed_steps: List of perturbed step dicts
        perturbation_step_idx: Index of the perturbed step
        semantic_diff_fn: Optional function to compare step outputs
                         Signature: (baseline_output, perturbed_output) -> bool
                         Returns True if semantically different

    Returns:
        Tuple of (propagation_depth, divergent_step_ids, convergent_step_ids)
    """
    if semantic_diff_fn is None:
        semantic_diff_fn = _default_semantic_diff

    divergent_steps = []
    convergent_steps = []

    # Compare steps after perturbation point
    max_idx = min(len(baseline_steps), len(perturbed_steps))
    for i in range(perturbation_step_idx, max_idx):
        baseline_step = baseline_steps[i]
        perturbed_step = perturbed_steps[i]

        # Get step outputs
        baseline_output = _extract_step_output(baseline_step)
        perturbed_output = _extract_step_output(perturbed_step)

        # Get step ID
        step_id = baseline_step.get("canonical_step_id", f"step_{i}")

        if semantic_diff_fn(baseline_output, perturbed_output):
            divergent_steps.append(step_id)
        else:
            convergent_steps.append(step_id)

    # Handle case where perturbed has extra steps
    for i in range(max_idx, len(perturbed_steps)):
        step_id = perturbed_steps[i].get("canonical_step_id", f"step_{i}")
        divergent_steps.append(step_id)

    propagation_depth = len(divergent_steps)
    return propagation_depth, divergent_steps, convergent_steps


def _extract_step_output(step: Dict[str, Any]) -> Any:
    """Extract the output/observation from a step."""
    # Try different possible output fields
    if "observation" in step:
        return step["observation"]
    if "extracted_value" in step:
        return step["extracted_value"]
    if "tool_output" in step:
        return step["tool_output"]
    if "content" in step:
        return step["content"]
    return None


def _default_semantic_diff(baseline_output: Any, perturbed_output: Any) -> bool:
    """
    Default semantic difference check.

    Returns True if outputs are semantically different.
    """
    if baseline_output is None and perturbed_output is None:
        return False
    if baseline_output is None or perturbed_output is None:
        return True

    # String comparison (normalized)
    if isinstance(baseline_output, str) and isinstance(perturbed_output, str):
        return baseline_output.strip().lower() != perturbed_output.strip().lower()

    # Numeric comparison with tolerance
    if isinstance(baseline_output, (int, float)) and isinstance(
        perturbed_output, (int, float)
    ):
        threshold = 0.001
        return abs(baseline_output - perturbed_output) > threshold

    # Dict comparison
    if isinstance(baseline_output, dict) and isinstance(perturbed_output, dict):
        return baseline_output != perturbed_output

    # List comparison
    if isinstance(baseline_output, list) and isinstance(perturbed_output, list):
        return baseline_output != perturbed_output

    # Fallback: direct comparison
    return baseline_output != perturbed_output


def compute_recovery_cost(
    baseline_trace: Dict[str, Any],
    perturbed_trace: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute Recovery Cost (RC) - Tier 1 only.

    Measures additional resources required to recover from perturbation.

    Args:
        baseline_trace: Baseline execution trace with steps, tool_calls, etc.
        perturbed_trace: Perturbed execution trace

    Returns:
        Dict with recovery cost components:
        - extra_steps: Number of additional steps
        - extra_tool_calls: Number of additional tool calls
        - retries: Number of retry attempts
        - recovery_successful: Whether perturbed achieved same outcome as baseline
    """
    baseline_steps = baseline_trace.get("steps", [])
    perturbed_steps = perturbed_trace.get("steps", [])

    extra_steps = len(perturbed_steps) - len(baseline_steps)

    # Count tool calls
    baseline_tool_calls = _count_tool_calls(baseline_steps)
    perturbed_tool_calls = _count_tool_calls(perturbed_steps)
    extra_tool_calls = perturbed_tool_calls - baseline_tool_calls

    # Count retries (steps that are error recovery)
    baseline_retries = _count_retries(baseline_steps)
    perturbed_retries = _count_retries(perturbed_steps)
    retries = perturbed_retries - baseline_retries

    # Check if recovery was successful
    baseline_outcome = baseline_trace.get("final_outcome")
    perturbed_outcome = perturbed_trace.get("final_outcome")
    recovery_successful = baseline_outcome == perturbed_outcome

    return {
        "extra_steps": extra_steps,
        "extra_tool_calls": extra_tool_calls,
        "retries": retries,
        "recovery_successful": recovery_successful,
    }


def _count_tool_calls(steps: List[Dict[str, Any]]) -> int:
    """Count the number of tool calls in steps."""
    count = 0
    for step in steps:
        step_type = step.get("step_type", "")
        if step_type in ("tool_call", "tool_use", "action"):
            count += 1
        # Also check for tool_name field
        if step.get("tool_name"):
            count += 1
    return count


def _count_retries(steps: List[Dict[str, Any]]) -> int:
    """Count the number of retry attempts in steps."""
    count = 0
    for step in steps:
        # Check for retry indicators
        content = str(step.get("content", "")).lower()
        if any(
            indicator in content
            for indicator in ["retry", "try again", "error", "failed", "let me try"]
        ):
            count += 1
        # Check for explicit retry flag
        if step.get("is_retry"):
            count += 1
    return count


def derive_true_impact(outcome_degradation: float) -> int:
    """
    Derive true impact level from outcome degradation.

    Per Section 5C.6:
    - 0 (None): OD = 0
    - 1 (Minor): 0 < OD <= 0.25
    - 2 (Moderate): 0.25 < OD <= 0.5
    - 3 (Critical): OD > 0.5

    Args:
        outcome_degradation: OD value

    Returns:
        Impact level 0-3
    """
    if outcome_degradation == 0:
        return 0
    elif outcome_degradation <= 0.25:
        return 1
    elif outcome_degradation <= 0.5:
        return 2
    else:
        return 3


def categorize_od(outcome_degradation: float) -> str:
    """
    Categorize OD value into human-readable category.

    Args:
        outcome_degradation: OD value

    Returns:
        Category string: "none", "minor", "moderate", "critical", or "negative"
    """
    if outcome_degradation < 0:
        return "negative"
    elif outcome_degradation == 0:
        return "none"
    elif outcome_degradation <= 0.25:
        return "minor"
    elif outcome_degradation <= 0.5:
        return "moderate"
    else:
        return "critical"


def compare_to_expected_impact(true_impact: int, expected_impact: int) -> str:
    """
    Compare true impact to expected impact from perturbation record.

    Args:
        true_impact: Actual impact level (0-3)
        expected_impact: Expected impact from perturbation design (0-3)

    Returns:
        Comparison result: "accurate", "underestimated", or "overestimated"
    """
    if true_impact == expected_impact:
        return "accurate"
    elif true_impact > expected_impact:
        return "underestimated"
    else:
        return "overestimated"
