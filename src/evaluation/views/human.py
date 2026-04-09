"""
Human evaluator view generation module.

This module generates human-readable views of evaluation units for human
evaluators. There are three evaluation modes, each with different information
displayed to prevent bias:

1. Detectability: Does one trajectory contain an error?
   - NO expected_answer shown (would bias the evaluator)
   - Evaluator identifies which trajectory has an error and which step

2. Consequence: What is the impact of the error on task outcome?
   - expected_answer IS shown (needed to assess correctness)
   - Evaluator rates impact severity and final correctness

3. Preference: Which trajectory better accomplishes the task?
   - expected_answer is configurable (parameter)
   - Evaluator indicates preference between trajectories

All views use blinded A/B ordering from the evaluation unit's blinding
assignment to prevent evaluators from inferring which trajectory is
baseline vs perturbed based on position.
"""

from typing import Any, Dict, Optional

from src.evaluation.blinding import get_blinded_trajectory_order


def _truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to max_length, adding ellipsis if truncated.

    Args:
        text: The text to truncate.
        max_length: Maximum length before truncation.

    Returns:
        Truncated text with ellipsis if it exceeded max_length.
    """
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _determine_step_role(step: Dict[str, Any]) -> str:
    """
    Determine the role/type of a step for human-readable display.

    Maps step types to human-friendly role names.

    Args:
        step: The step dict from the trajectory.

    Returns:
        A human-readable step role string.
    """
    # Check for step_type field (from typed trajectory)
    step_type = step.get("step_type", "")

    # Map step types to roles
    role_mapping = {
        "planning": "planning",
        "tool_selection": "tool_call",
        "tool_execution": "tool_call",
        "reasoning": "reasoning",
        "validation": "validation",
        "final_answer": "final_answer",
        "other": "other",
    }

    if step_type and step_type in role_mapping:
        return role_mapping[step_type]

    # Infer from content if step_type not present
    tool_name = step.get("tool_name")
    if tool_name:
        return "tool_call"

    # Check raw_text content for clues
    raw_text = step.get("raw_text", step.get("content", ""))
    if raw_text:
        raw_lower = raw_text.lower()
        if any(word in raw_lower for word in ["plan", "approach", "strategy"]):
            return "planning"
        if any(word in raw_lower for word in ["answer", "final", "result"]):
            return "final_answer"
        if any(word in raw_lower for word in ["check", "verify", "validate"]):
            return "validation"

    return "reasoning"


def _create_description(step: Dict[str, Any]) -> str:
    """
    Create a human-readable description from step content.

    Args:
        step: The step dict from the trajectory.

    Returns:
        A human-readable description (truncated to 200 chars).
    """
    # Try raw_text first (typed trajectory format)
    raw_text = step.get("raw_text", "")
    if not raw_text:
        # Fall back to content (data schema format)
        raw_text = step.get("content", "")

    if not raw_text:
        # Build description from tool info if no text content
        tool_name = step.get("tool_name", "")
        if tool_name:
            tool_args = step.get("tool_arguments", step.get("tool_input", {}))
            if isinstance(tool_args, dict):
                # Try to get a query or command for description
                query = tool_args.get("query", tool_args.get("command", ""))
                if query:
                    raw_text = f"{tool_name}: {query}"
                else:
                    raw_text = f"Execute {tool_name}"
            else:
                raw_text = f"Execute {tool_name}"
        else:
            raw_text = "No description available"

    return _truncate_text(raw_text, 200)


def _summarize_tool_arguments(step: Dict[str, Any]) -> str:
    """
    Summarize tool arguments for human display.

    Args:
        step: The step dict from the trajectory.

    Returns:
        A summary string of tool arguments (first 100 chars).
    """
    # Try tool_arguments (typed trajectory) then tool_input (data schema)
    tool_args = step.get("tool_arguments", step.get("tool_input", {}))

    if not tool_args:
        return ""

    if isinstance(tool_args, dict):
        # Format as key=value pairs
        parts = []
        for key, value in tool_args.items():
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            parts.append(f"{key}={repr(value_str)}")
        summary = ", ".join(parts)
    else:
        summary = str(tool_args)

    return _truncate_text(summary, 100)


def _summarize_observation(step: Dict[str, Any]) -> str:
    """
    Summarize observation/tool output for human display.

    Args:
        step: The step dict from the trajectory.

    Returns:
        A summary string of the observation (first 150 chars).
    """
    # Try observation (typed trajectory) then tool_output (data schema)
    observation = step.get("observation", step.get("tool_output", ""))

    if not observation:
        return ""

    return _truncate_text(str(observation), 150)


def create_simplified_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a SimplifiedStep for human-readable display.

    Extracts key information from a raw step and formats it for human
    evaluators, with appropriate truncation and summarization.

    Args:
        step: A step dict from the trajectory with fields like:
            - step_index or step_number: The position in trajectory
            - canonical_step_id: The canonical ID (if present)
            - step_type: Type classification
            - raw_text or content: The step content
            - tool_name: Tool used (optional)
            - tool_arguments or tool_input: Arguments (optional)
            - observation or tool_output: Result (optional)

    Returns:
        SimplifiedStep dict with:
            - display_step_index: Position for display (1-indexed)
            - canonical_step_id: Unique step identifier
            - step_role: Human-friendly role name
            - description: Human-readable description (max 200 chars)
            - tool_name: Name of tool if applicable
            - tool_arguments_summary: Summarized arguments (max 100 chars)
            - observation_summary: Summarized observation (max 150 chars)
    """
    # Extract display index - prefer display_step_index, fall back to step_index/step_number
    display_step_index = step.get(
        "display_step_index", step.get("step_index", step.get("step_number", 1))
    )

    # Extract canonical step ID
    canonical_step_id = step.get("canonical_step_id", "")
    if not canonical_step_id:
        # Generate a placeholder if not present
        step_idx = step.get("step_index", step.get("step_number", display_step_index))
        canonical_step_id = f"step::{step_idx}"

    # Determine step role
    step_role = _determine_step_role(step)

    # Create description
    description = _create_description(step)

    # Extract tool name
    tool_name = step.get("tool_name", None)

    # Summarize tool arguments
    tool_arguments_summary = _summarize_tool_arguments(step)

    # Summarize observation
    observation_summary = _summarize_observation(step)

    return {
        "display_step_index": display_step_index,
        "canonical_step_id": canonical_step_id,
        "step_role": step_role,
        "description": description,
        "tool_name": tool_name,
        "tool_arguments_summary": tool_arguments_summary,
        "observation_summary": observation_summary,
    }


def _create_trajectory_view(
    trajectory: Dict[str, Any],
    view_label: str,
) -> Dict[str, Any]:
    """
    Create a trajectory view with simplified steps.

    Args:
        trajectory: The full trajectory dict.
        view_label: The label for this trajectory ("view_A" or "view_B").

    Returns:
        Dict with trajectory_id, steps, and num_steps.
    """
    steps = trajectory.get("steps", [])
    simplified_steps = [create_simplified_step(step) for step in steps]

    return {
        "trajectory_id": view_label,
        "steps": simplified_steps,
        "num_steps": len(simplified_steps),
    }


def generate_detectability_view(unit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a view for detectability evaluation.

    Detectability evaluation asks: Does one trajectory contain an error?
    If so, which trajectory and which step?

    CRITICAL: This view does NOT include expected_answer, as knowing the
    answer would bias the evaluator toward noticing certain error types.

    Args:
        unit: An evaluation unit dict containing:
            - evaluation_unit_id: Unique identifier
            - task_text: The task description
            - baseline: BaselineData dict with trajectory
            - perturbed: PerturbedData dict with trajectory
            - blinding: BlindingAssignment dict with A/B assignment

    Returns:
        DetectabilityView dict with:
            - view_type: "detectability"
            - evaluation_unit_id: The evaluation unit ID
            - task_text: The task description
            - trajectory_a: Blinded trajectory in position A
            - trajectory_b: Blinded trajectory in position B
            - instructions: Instructions for the evaluator
            - rating_schema: Expected response format
    """
    # Get blinded trajectory order
    traj_a, traj_b, variant_a_id, variant_b_id = get_blinded_trajectory_order(unit)

    # Create simplified trajectory views
    trajectory_a_view = _create_trajectory_view(traj_a, "view_A")
    trajectory_b_view = _create_trajectory_view(traj_b, "view_B")

    return {
        "view_type": "detectability",
        "evaluation_unit_id": unit.get("evaluation_unit_id", ""),
        "task_text": unit.get("task_text", ""),
        "trajectory_a": trajectory_a_view,
        "trajectory_b": trajectory_b_view,
        "instructions": (
            "Compare these two trajectories. Does one contain an error? "
            "If so, identify which trajectory and which step."
        ),
        "rating_schema": {
            "error_detected": "bool",
            "error_trajectory": "A | B | neither | both",
            "error_step_id": "string (canonical_step_id) | null",
            "confidence": "1-5",
        },
    }


def generate_consequence_view(unit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a view for consequence/impact evaluation.

    Consequence evaluation asks: What is the impact of the error on
    the task outcome? Did the error affect the final result?

    This view INCLUDES expected_answer because the evaluator needs to
    know the correct answer to assess whether the trajectory outcome
    is correct.

    Args:
        unit: An evaluation unit dict containing:
            - evaluation_unit_id: Unique identifier
            - task_text: The task description
            - baseline: BaselineData dict with trajectory
            - perturbed: PerturbedData dict with trajectory
            - blinding: BlindingAssignment dict with A/B assignment

    Returns:
        ConsequenceView dict with:
            - view_type: "consequence"
            - evaluation_unit_id: The evaluation unit ID
            - task_text: The task description
            - expected_answer: The correct answer for the task
            - trajectory_a: Blinded trajectory in position A
            - trajectory_b: Blinded trajectory in position B
            - instructions: Instructions for the evaluator
            - rating_schema: Expected response format
    """
    # Get blinded trajectory order
    traj_a, traj_b, variant_a_id, variant_b_id = get_blinded_trajectory_order(unit)

    # Create simplified trajectory views
    trajectory_a_view = _create_trajectory_view(traj_a, "view_A")
    trajectory_b_view = _create_trajectory_view(traj_b, "view_B")

    # Extract expected answer from trajectory ground truth or task info
    expected_answer = None

    # Try baseline trajectory's ground_truth
    baseline_traj = unit.get("baseline", {}).get("trajectory", {})
    ground_truth = baseline_traj.get("ground_truth", {})
    expected_answer = ground_truth.get("expected_answer")

    # Fall back to task-level expected_answer if present
    if not expected_answer:
        expected_answer = unit.get("expected_answer")

    return {
        "view_type": "consequence",
        "evaluation_unit_id": unit.get("evaluation_unit_id", ""),
        "task_text": unit.get("task_text", ""),
        "expected_answer": expected_answer,
        "trajectory_a": trajectory_a_view,
        "trajectory_b": trajectory_b_view,
        "instructions": (
            "Assess the impact of any errors in these trajectories on the final outcome. "
            "Compare each trajectory's result against the expected answer and rate the "
            "severity of impact."
        ),
        "rating_schema": {
            "trajectory_a_impact": "0-3 (0=no impact, 1=minor, 2=moderate, 3=severe)",
            "trajectory_b_impact": "0-3 (0=no impact, 1=minor, 2=moderate, 3=severe)",
            "trajectory_a_correct": "bool",
            "trajectory_b_correct": "bool",
            "confidence": "1-5",
        },
    }


def generate_preference_view(
    unit: Dict[str, Any],
    include_expected_answer: bool = False,
) -> Dict[str, Any]:
    """
    Generate a view for preference evaluation.

    Preference evaluation asks: Which trajectory better accomplishes
    the task? This is a pairwise comparison without requiring explicit
    error identification.

    Args:
        unit: An evaluation unit dict containing:
            - evaluation_unit_id: Unique identifier
            - task_text: The task description
            - baseline: BaselineData dict with trajectory
            - perturbed: PerturbedData dict with trajectory
            - blinding: BlindingAssignment dict with A/B assignment
        include_expected_answer: Whether to show the expected answer
            to the evaluator. Default False for unbiased preference
            judgment.

    Returns:
        PreferenceView dict with:
            - view_type: "preference"
            - evaluation_unit_id: The evaluation unit ID
            - task_text: The task description
            - expected_answer: Included only if include_expected_answer=True
            - trajectory_a: Blinded trajectory in position A
            - trajectory_b: Blinded trajectory in position B
            - instructions: Instructions for the evaluator
            - rating_schema: Expected response format
    """
    # Get blinded trajectory order
    traj_a, traj_b, variant_a_id, variant_b_id = get_blinded_trajectory_order(unit)

    # Create simplified trajectory views
    trajectory_a_view = _create_trajectory_view(traj_a, "view_A")
    trajectory_b_view = _create_trajectory_view(traj_b, "view_B")

    view = {
        "view_type": "preference",
        "evaluation_unit_id": unit.get("evaluation_unit_id", ""),
        "task_text": unit.get("task_text", ""),
        "trajectory_a": trajectory_a_view,
        "trajectory_b": trajectory_b_view,
        "instructions": (
            "Compare these two trajectories and indicate which one better "
            "accomplishes the given task. Consider the reasoning quality, "
            "tool usage, and overall approach."
        ),
        "rating_schema": {
            "preference": "A | B | tie",
            "reason": "string (brief explanation)",
            "confidence": "1-5",
        },
    }

    # Conditionally include expected answer
    if include_expected_answer:
        expected_answer = None
        baseline_traj = unit.get("baseline", {}).get("trajectory", {})
        ground_truth = baseline_traj.get("ground_truth", {})
        expected_answer = ground_truth.get("expected_answer")
        if not expected_answer:
            expected_answer = unit.get("expected_answer")
        view["expected_answer"] = expected_answer

    return view


def generate_human_view(
    unit: Dict[str, Any],
    mode: str,
    include_expected_answer: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Generate a human evaluator view for the specified mode.

    Convenience function that dispatches to the appropriate view
    generator based on the evaluation mode.

    Args:
        unit: An evaluation unit dict.
        mode: One of "detectability", "consequence", or "preference".
        include_expected_answer: For preference mode, whether to include
            the expected answer. Ignored for other modes (detectability
            never includes it, consequence always includes it).

    Returns:
        The appropriate view dict for the specified mode.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode == "detectability":
        return generate_detectability_view(unit)
    elif mode == "consequence":
        return generate_consequence_view(unit)
    elif mode == "preference":
        include_answer = (
            include_expected_answer if include_expected_answer is not None else False
        )
        return generate_preference_view(unit, include_expected_answer=include_answer)
    else:
        raise ValueError(
            f"Unknown evaluation mode: {mode}. "
            f"Expected one of: detectability, consequence, preference"
        )
