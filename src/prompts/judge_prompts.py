"""
Judge evaluation prompts for Section 5.

Contains versioned prompts for:
- Single trajectory evaluation
- Blinded pair comparison (V1 and V2 neutral)
- Labeled pair (calibration) evaluation

All prompts follow naming convention: {TYPE}_{VERSION}
"""

# =============================================================================
# Single Trajectory Evaluation Prompts
# =============================================================================

SINGLE_TRAJECTORY_SYSTEM_V1 = """You are an expert evaluator of AI agent trajectories. Your task is to evaluate a trajectory where an AI agent attempts to complete a task using tools.

Analyze the trajectory for:
1. Overall quality (score 0-100)
2. Whether any errors or mistakes exist
3. If errors exist, identify the specific step and type
4. Estimate the impact of any errors on task success

Respond in JSON format only."""


SINGLE_TRAJECTORY_USER_V1 = """Task: {{task_text}}

Trajectory:
{{formatted_trajectory}}

Evaluate this trajectory. Respond with JSON:
{
  "overall_score": <0-100>,
  "error_detected": <true/false>,
  "error_confidence": <0-1>,
  "predicted_error_step": <step number or null>,
  "predicted_error_type": <"planning"|"tool_selection"|"parameter"|"data_reference"|"other"|null>,
  "localization_confidence": <0-1 or null>,
  "predicted_impact_score": <0-1>,
  "predicted_failure_prob": <0-1>,
  "impact_explanation": "<brief explanation>"
}"""


# =============================================================================
# Blinded Pair Comparison Prompts
# =============================================================================

# V1: Original prompt (may have priming bias)
BLINDED_PAIR_SYSTEM_V1 = """You are an expert evaluator comparing two AI agent trajectories. Both trajectories attempt the same task. One may contain an error.

Your job is to:
1. Determine if either trajectory contains an error
2. If so, identify which trajectory and which step
3. Compare overall quality

You do NOT know which trajectory is "baseline" or "perturbed" - evaluate purely on quality.
Respond in JSON format only."""


# V2: Neutral prompt (removes priming bias)
BLINDED_PAIR_SYSTEM_V2 = """You are a quality evaluator for AI agent trajectories.

Your task: Compare two trajectories (A and B) that attempt the same task.
Analyze the quality of reasoning, tool usage, and execution in each.

You may find:
- Both trajectories are correct
- Both have issues
- One has issues while the other is correct
- Differences that don't affect correctness

Provide your assessment without assuming either must have errors.
Respond in JSON format only."""


BLINDED_PAIR_USER_V1 = """Task: {{task_text}}

Trajectory A:
{{formatted_trajectory_a}}

Trajectory B:
{{formatted_trajectory_b}}

Compare these trajectories. Respond with JSON:
{
  "overall_score_a": <0-100>,
  "overall_score_b": <0-100>,
  "error_detected": <true/false>,
  "error_trajectory": <"A"|"B"|"neither"|"both">,
  "error_confidence": <0-1>,
  "predicted_error_step": <step number in error trajectory, or null>,
  "predicted_error_type": <type or null>,
  "preference": <"A"|"B"|"tie">,
  "predicted_impact_score": <0-1>,
  "predicted_failure_prob": <0-1>,
  "comparison_explanation": "<brief explanation>"
}"""


# V2: Neutral user prompt (equal framing for A and B)
BLINDED_PAIR_USER_V2 = """Task: {{task_text}}

=== Trajectory A ===
{{formatted_trajectory_a}}

=== Trajectory B ===
{{formatted_trajectory_b}}

Analyze both trajectories and compare their quality. Respond with JSON:
{
  "overall_score_a": <0-100>,
  "overall_score_b": <0-100>,
  "error_detected": <true/false>,
  "error_trajectory": <"A"|"B"|"neither"|"both">,
  "error_confidence": <0-1>,
  "predicted_error_step": <step number in error trajectory, or null>,
  "predicted_error_type": <"planning"|"tool_selection"|"parameter"|"data_reference"|"other"|null>,
  "localization_confidence": <0-1 or null>,
  "preference": <"A"|"B"|"tie">,
  "predicted_impact_score": <0-1>,
  "predicted_failure_prob": <0-1>,
  "comparison_explanation": "<brief explanation of your reasoning>"
}"""


# =============================================================================
# V3: Task-Blocking Error Detection (addresses false positive issue)
# =============================================================================
# Rationale: V1/V2 prompts ask broadly about "errors or mistakes" which causes
# the judge to flag cosmetic differences, paraphrases, and pre-existing issues.
# V3 focuses exclusively on errors that would cause TASK FAILURE.

BLINDED_PAIR_SYSTEM_V3 = """You are evaluating which of two AI agent trajectories would more likely SUCCEED at completing a task.

CRITICAL: You are NOT looking for any possible issues. You are ONLY looking for errors that would cause the task to FAIL.

Types of task-blocking errors:
- Wrong tool choice that cannot accomplish the task requirement
- Parameter errors that cause API failures or incorrect results
- Planning errors that skip required steps
- Reasoning errors that lead to wrong conclusions used in subsequent steps

NOT task-blocking (ignore these):
- Stylistic differences in how steps are described
- Minor wording variations that don't change meaning
- Formatting differences
- Verbose vs. concise explanations
- Order of independent operations

Both trajectories may be perfectly valid. Different approaches to the same task are acceptable if they both would succeed.

Respond in JSON format only."""


BLINDED_PAIR_USER_V3 = """Task: {{task_text}}

=== Trajectory A ===
{{formatted_trajectory_a}}

=== Trajectory B ===
{{formatted_trajectory_b}}

Answer these questions:
1. Would Trajectory A successfully complete the task? (yes/no and why)
2. Would Trajectory B successfully complete the task? (yes/no and why)
3. If one would fail, what specific step causes the failure?

Respond with JSON:
{
  "trajectory_a_succeeds": <true/false>,
  "trajectory_b_succeeds": <true/false>,
  "error_detected": <true/false - only true if one would FAIL the task>,
  "error_trajectory": <"A"|"B"|"neither"|"both">,
  "error_confidence": <0-1>,
  "predicted_error_step": <step number causing failure, or null>,
  "predicted_error_type": <"planning"|"tool_selection"|"parameter"|"data_reference"|null>,
  "why_would_fail": "<specific reason this error causes task failure, or null>",
  "preference": <"A"|"B"|"tie" - which would succeed, or tie if both succeed/both fail>,
  "comparison_explanation": "<brief explanation focusing on task success>"
}"""


SINGLE_TRAJECTORY_SYSTEM_V2 = """You are evaluating whether an AI agent trajectory would SUCCESSFULLY complete a task.

CRITICAL: You are NOT auditing for any possible issues. You are ONLY determining whether the trajectory would SUCCEED or FAIL at the given task.

A trajectory SUCCEEDS if:
- The final result matches what the task requested
- Each step's output is used correctly in subsequent steps
- No step produces results that contradict the task requirements

A trajectory FAILS if:
- A tool call returns an error that blocks progress
- A parameter error causes wrong data to be retrieved/processed
- A planning error skips a required step
- The final result is wrong or incomplete

Do NOT flag as errors:
- Verbose explanations
- Stylistic choices
- Alternative valid approaches
- Minor inefficiencies that don't affect the result

Respond in JSON format only."""


SINGLE_TRAJECTORY_USER_V2 = """Task: {{task_text}}

Trajectory:
{{formatted_trajectory}}

Would this trajectory successfully complete the task?

Respond with JSON:
{
  "task_would_succeed": <true/false>,
  "overall_score": <0-100, where 100=perfect success, 0=complete failure>,
  "error_detected": <true/false - only true if task would FAIL>,
  "error_confidence": <0-1>,
  "predicted_error_step": <step number causing failure, or null>,
  "predicted_error_type": <"planning"|"tool_selection"|"parameter"|"data_reference"|null>,
  "why_would_fail": "<specific reason this causes task failure, or null>",
  "impact_explanation": "<brief explanation focusing on task success>"
}"""


# =============================================================================
# Labeled Pair (Calibration) Prompts
# =============================================================================

LABELED_PAIR_SYSTEM_V1 = """You are evaluating a controlled perturbation study. You will see:
- A baseline trajectory (original, no errors)
- A perturbed trajectory (with a known injected error)

Your task is to estimate the impact of the perturbation.
Respond in JSON format only."""


LABELED_PAIR_USER_V1 = """Task: {{task_text}}

Baseline Trajectory (no error):
{{formatted_baseline}}

Perturbed Trajectory (contains injected error at step {{target_step}}):
{{formatted_perturbed}}

Perturbation: {{perturbation_type}} at step {{target_step}}

Estimate the impact. Respond with JSON:
{
  "impact_estimate": <0-3, where 0=none, 1=minor, 2=moderate, 3=critical>,
  "predicted_failure_prob": <0-1>,
  "downstream_steps_affected": <list of step numbers>,
  "impact_explanation": "<brief explanation>"
}"""


# =============================================================================
# All Judge Prompts Registry
# =============================================================================

JUDGE_PROMPTS = {
    # Single trajectory
    "SINGLE_TRAJECTORY_SYSTEM_V1": SINGLE_TRAJECTORY_SYSTEM_V1,
    "SINGLE_TRAJECTORY_USER_V1": SINGLE_TRAJECTORY_USER_V1,
    "SINGLE_TRAJECTORY_SYSTEM_V2": SINGLE_TRAJECTORY_SYSTEM_V2,
    "SINGLE_TRAJECTORY_USER_V2": SINGLE_TRAJECTORY_USER_V2,
    # Blinded pair
    "BLINDED_PAIR_SYSTEM_V1": BLINDED_PAIR_SYSTEM_V1,
    "BLINDED_PAIR_SYSTEM_V2": BLINDED_PAIR_SYSTEM_V2,
    "BLINDED_PAIR_USER_V1": BLINDED_PAIR_USER_V1,
    "BLINDED_PAIR_USER_V2": BLINDED_PAIR_USER_V2,
    "BLINDED_PAIR_SYSTEM_V3": BLINDED_PAIR_SYSTEM_V3,
    "BLINDED_PAIR_USER_V3": BLINDED_PAIR_USER_V3,
    # Labeled pair
    "LABELED_PAIR_SYSTEM_V1": LABELED_PAIR_SYSTEM_V1,
    "LABELED_PAIR_USER_V1": LABELED_PAIR_USER_V1,
}
