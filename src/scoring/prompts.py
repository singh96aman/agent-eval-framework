"""
Prompts for perturbation quality scoring.

These prompts assess the VALIDITY of perturbations (not difficulty/detectability),
focusing on whether the perturbation is well-formed and realistic.

Modes:
- single: 1 LLM call per metric (N calls for N metrics)
- batch: 1 LLM call for all metrics
"""

# Individual metric prompts (for single mode - 1 call per metric)
METRIC_PROMPTS = {
    "content_changed": """You are evaluating whether a perturbation actually changed the content.

## Original Step
{original_step_content}

## Perturbed Step
{perturbed_step_content}

## Task
Score whether the content actually changed:
- 0 = Identical or trivially different (whitespace only)
- 1 = Meaningful change occurred

Return ONLY valid JSON:
{{"content_changed": 0 or 1, "reasoning": "Brief explanation"}}""",

    "syntactically_valid": """You are evaluating whether a perturbed step is syntactically valid.

## Perturbed Step
{perturbed_step_content}

## Task
Score whether the perturbed content is well-formed:
- 0 = Broken syntax, corrupted text, invalid JSON
- 1 = Valid syntax, coherent text

Return ONLY valid JSON:
{{"syntactically_valid": 0 or 1, "reasoning": "Brief explanation"}}""",

    "semantically_meaningful": """You are evaluating whether a perturbation is semantically meaningful.

## Original Step
{original_step_content}

## Perturbed Step
{perturbed_step_content}

## Task
Score whether the change is semantically meaningful:
- 0 = Change is trivial (punctuation, formatting only)
- 1 = Change affects meaning or behavior

Return ONLY valid JSON:
{{"semantically_meaningful": 0 or 1, "reasoning": "Brief explanation"}}""",

    "type_matches_intent": """You are evaluating whether a perturbation matches its intended type.

## Original Step
{original_step_content}

## Perturbed Step
{perturbed_step_content}

## Perturbation Metadata
- Intended type: {perturbation_type}
- Position: {perturbation_position}
- Description: {perturbation_description}

## Task
Score whether the error matches the intended perturbation type:
- 0 = Error is different from intended type (e.g., labeled "parameter" but changed planning)
- 1 = Error matches the intended perturbation type

Return ONLY valid JSON:
{{"type_matches_intent": 0 or 1, "reasoning": "Brief explanation"}}""",

    "realistic_error": """You are evaluating whether a perturbation represents a realistic agent error.

## Original Step
{original_step_content}

## Perturbed Step
{perturbed_step_content}

## Perturbation Metadata
- Type: {perturbation_type}
- Position: {perturbation_position}

## Task
Score whether this error could plausibly occur in a real agent:
- 0 = Impossible/nonsensical error
- 1 = Unlikely but possible
- 2 = Plausible error
- 3 = Very realistic, common agent failure mode

Return ONLY valid JSON:
{{"realistic_error": 0-3, "reasoning": "Brief explanation"}}"""
}

# Batch mode prompt (all metrics in 1 call)
QUALITY_SCORING_PROMPT_BATCH = """You are evaluating the VALIDITY of a perturbation applied to an agent trajectory.

Your task is to score how well-formed and realistic this perturbation is, NOT how difficult it would be to detect.

## Original Step
{original_step_content}

## Perturbed Step
{perturbed_step_content}

## Perturbation Metadata
- Type: {perturbation_type}
- Position: {perturbation_position}
- Intended error: {perturbation_description}

## Scoring Criteria

Rate each criterion:

1. **content_changed** (0 or 1): Did the content actually change?
   - 0 = Identical or trivially different (whitespace only)
   - 1 = Meaningful change occurred

2. **syntactically_valid** (0 or 1): Is the perturbed content well-formed?
   - 0 = Broken syntax, corrupted text, invalid JSON
   - 1 = Valid syntax, coherent text

3. **semantically_meaningful** (0 or 1): Is the change meaningful?
   - 0 = Change is trivial (punctuation, formatting only)
   - 1 = Change affects meaning or behavior

4. **type_matches_intent** (0 or 1): Does the error match the intended type?
   - 0 = Error is different from intended type (e.g., labeled "parameter" but changed planning)
   - 1 = Error matches the intended perturbation type

5. **realistic_error** (0-3): Could this error plausibly occur in a real agent?
   - 0 = Impossible/nonsensical error
   - 1 = Unlikely but possible
   - 2 = Plausible error
   - 3 = Very realistic, common agent failure mode

## Output Format (JSON)

Return ONLY valid JSON with no additional text:

{{
  "content_changed": 0 or 1,
  "syntactically_valid": 0 or 1,
  "semantically_meaningful": 0 or 1,
  "type_matches_intent": 0 or 1,
  "realistic_error": 0-3,
  "reasoning": "Brief explanation of scores"
}}"""


PERTURBATION_TYPE_DESCRIPTIONS = {
    "planning": "A planning error where the agent's reasoning or strategy is flawed",
    "tool_selection": "A tool selection error where the agent chose the wrong tool for the task",
    "parameter": "A parameter error where the agent used incorrect arguments for a tool",
    "data_reference": "A data reference error where the agent misused or misquoted data"
}


def get_metric_names():
    """Return list of metric names in order."""
    return ["content_changed", "syntactically_valid", "semantically_meaningful",
            "type_matches_intent", "realistic_error"]


def format_metric_prompt(
    metric_name: str,
    original_step_content: str,
    perturbed_step_content: str,
    perturbation_type: str = "",
    perturbation_position: str = "",
    perturbation_description: str = ""
) -> str:
    """
    Format a prompt for a single metric (single mode).

    Args:
        metric_name: Name of the metric to score
        original_step_content: Original step content
        perturbed_step_content: Perturbed step content
        perturbation_type: Type of perturbation
        perturbation_position: Position (early/middle/late)
        perturbation_description: Description of intended error

    Returns:
        Formatted prompt string for this metric
    """
    if metric_name not in METRIC_PROMPTS:
        raise ValueError(f"Unknown metric: {metric_name}")

    if not perturbation_description:
        perturbation_description = PERTURBATION_TYPE_DESCRIPTIONS.get(
            perturbation_type,
            f"A {perturbation_type} error"
        )

    return METRIC_PROMPTS[metric_name].format(
        original_step_content=original_step_content,
        perturbed_step_content=perturbed_step_content,
        perturbation_type=perturbation_type,
        perturbation_position=perturbation_position,
        perturbation_description=perturbation_description
    )


def format_batch_prompt(
    original_step_content: str,
    perturbed_step_content: str,
    perturbation_type: str,
    perturbation_position: str,
    perturbation_description: str = None
) -> str:
    """
    Format the batch mode prompt (all metrics in 1 call).

    Args:
        original_step_content: Content of the original step
        perturbed_step_content: Content of the perturbed step
        perturbation_type: Type of perturbation (planning, tool_selection, etc.)
        perturbation_position: Position (early, middle, late)
        perturbation_description: Optional custom description

    Returns:
        Formatted prompt string
    """
    if not perturbation_description:
        perturbation_description = PERTURBATION_TYPE_DESCRIPTIONS.get(
            perturbation_type,
            f"A {perturbation_type} error"
        )

    return QUALITY_SCORING_PROMPT_BATCH.format(
        original_step_content=original_step_content,
        perturbed_step_content=perturbed_step_content,
        perturbation_type=perturbation_type,
        perturbation_position=perturbation_position,
        perturbation_description=perturbation_description
    )
