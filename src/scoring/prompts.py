"""
Prompts for perturbation quality scoring.

These prompts assess the VALIDITY of perturbations (not difficulty/detectability),
focusing on whether the perturbation is well-formed and realistic.
"""

QUALITY_SCORING_PROMPT = """You are evaluating the VALIDITY of a perturbation applied to an agent trajectory.

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
  "total_score": sum of above (0-7),
  "reasoning": "Brief explanation of scores"
}}"""


PERTURBATION_TYPE_DESCRIPTIONS = {
    "planning": "A planning error where the agent's reasoning or strategy is flawed",
    "tool_selection": "A tool selection error where the agent chose the wrong tool for the task",
    "parameter": "A parameter error where the agent used incorrect arguments for a tool",
    "data_reference": "A data reference error where the agent misused or misquoted data"
}


def format_quality_scoring_prompt(
    original_step_content: str,
    perturbed_step_content: str,
    perturbation_type: str,
    perturbation_position: str,
    perturbation_description: str = None
) -> str:
    """
    Format the quality scoring prompt with perturbation details.

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

    return QUALITY_SCORING_PROMPT.format(
        original_step_content=original_step_content,
        perturbed_step_content=perturbed_step_content,
        perturbation_type=perturbation_type,
        perturbation_position=perturbation_position,
        perturbation_description=perturbation_description
    )
