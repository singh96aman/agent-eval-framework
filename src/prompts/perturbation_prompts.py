"""
Perturbation generation prompts for Section 3.

Contains versioned prompts for:
- Paraphrase generation (placebo)
- Wrong plan generation (coarse-grained)
- LLM-based value mutations (fine-grained)

All prompts follow naming convention: {TYPE}_PROMPT_{VERSION}
"""

# =============================================================================
# Placebo Generation Prompts
# =============================================================================

PARAPHRASE_PROMPT_V1 = """Paraphrase the following text while preserving its exact meaning.
You MUST preserve these items exactly as written:
{preserve_str}

Rules:
1. Keep the same meaning and intent
2. Use different words and sentence structure where possible
3. DO NOT change any technical terms, tool names, file paths, or numbers
4. DO NOT add new information or remove existing information
5. Keep the same length (within 20% of original)

Original text:
{text}

Respond with ONLY the paraphrased text, nothing else."""


# =============================================================================
# Coarse-Grained Generation Prompts
# =============================================================================

WRONG_PLAN_PROMPT_V1 = """Generate an inferior but plausible alternative plan for this task.

Original Task: {task_description}
Original Plan Steps: {original_steps}

Requirements:
1. The alternative plan should be PLAUSIBLE but SUBOPTIMAL
2. It should take a different approach that is less efficient or has gaps
3. Do NOT make it obviously wrong - it should seem reasonable at first glance
4. Include 3-5 steps that address the task but with flaws

Respond with JSON:
{{
  "alternative_steps": [
    {{"step": 1, "action": "<action description>", "flaw": "<what's wrong with this step>"}},
    ...
  ],
  "plan_flaw_type": "<missing_step|wrong_order|inefficient|incomplete>",
  "reasoning": "<why this plan is inferior>"
}}"""


# V2: Creates TASK-BLOCKING errors (task will FAIL, not just suboptimal)
WRONG_PLAN_PROMPT_V2 = """You are generating test data for AI judge evaluation. Create a WRONG plan that will cause the task to FAIL.

TASK: {task_description}

ORIGINAL CORRECT PLAN:
{original_plan}

Create a replacement plan that will FAIL to complete the task. Use ONE of these error types:

1. WRONG OBJECTIVE: Misunderstand what the task asks for
   - Example: Task asks for "Italian learning apps" but plan searches for "Italian restaurants"
   - Example: Task asks for "current price" but plan gets "historical prices"

2. MISSING CRITICAL STEP: Skip something that makes completion impossible
   - Example: Try to use API without authentication step
   - Example: Query database without connecting first

3. WRONG TOOL CHOICE: Use a tool that cannot accomplish the goal
   - Example: Use text search when numeric calculation is needed
   - Example: Use weather API when stock data is requested

4. LOGIC ERROR: Make a reasoning mistake that leads to wrong answer
   - Example: Add numbers when multiplication was needed
   - Example: Use wrong units or wrong data field

REQUIREMENTS:
- The plan should look reasonable at first glance (not gibberish)
- But it MUST fail to accomplish the actual task objective
- Keep similar length and style to the original
- Do NOT just make it "slower" or "less efficient" - it must FAIL

Generate ONLY the wrong plan text. No explanation, no JSON, just the plan."""


# =============================================================================
# Fine-Grained LLM Mutation Prompts
# =============================================================================

WRONG_PARAMETER_PROMPT_V1 = """Generate a plausible but incorrect version of this parameter value.

Tool: {tool_name}
Parameter: {param_name}
Current Value: {current_value}
Value Type: {value_type}
Task Context: {task_context}

Requirements:
- Generate a realistic error (a human might make this mistake)
- The error should be plausible in context
- Do NOT add markers like "_old", "_wrong", "_mutated", "_backup", "_test", "_v1"
- Return valid JSON only

Respond with JSON ONLY:
{{
  "mutated_value": "<the incorrect value>",
  "mutation_type": "<what kind of error: typo, similar_name, off_by_one, wrong_context, etc.>",
  "reasoning": "<brief explanation of why this is a plausible error>"
}}"""


VALUE_MUTATION_PROMPT_V1 = """Generate a plausible mutation of this value that represents a realistic error.

Original Value: {original_value}
Value Type: {value_type}
Context: {context}

Requirements:
- Generate a value that looks like a realistic mistake
- The mutation should be subtle but meaningful
- Do NOT add markers like "_old", "_wrong", "_mutated"
- Return valid JSON only

Respond with JSON ONLY:
{{
  "mutated_value": "<the mutated value>",
  "mutation_type": "<type: typo, substitution, transposition, etc.>",
  "reasoning": "<why this mutation is plausible>"
}}"""


WRONG_DATE_PROMPT_V1 = """Generate a plausible but incorrect version of this date.

Original Date: {original_date}
Format: {date_format}
Context: {context}

Requirements:
- Generate a realistic date error (off-by-one day/month, wrong year, format error)
- The error should be something a human might accidentally type
- Do NOT append markers like "_wrong"
- Return valid JSON only

Respond with JSON ONLY:
{{
  "mutated_date": "<the incorrect date>",
  "mutation_type": "<shift_day, shift_month, shift_year, format_error, typo>",
  "reasoning": "<why this is a plausible error>"
}}"""


WRONG_IDENTIFIER_PROMPT_V1 = """Generate a plausible but incorrect version of this identifier.

Original ID: {original_id}
ID Type: {id_type}
Context: {context}

Requirements:
- Generate a realistic identifier error (typo, similar name, wrong context)
- The error should look like a natural mistake
- Do NOT add suffixes like "_old", "_wrong", "_v1", "_backup", "_test"
- Return valid JSON only

Respond with JSON ONLY:
{{
  "mutated_id": "<the incorrect identifier>",
  "mutation_type": "<typo, similar_name, wrong_context, transposition>",
  "reasoning": "<why this is a plausible error>"
}}"""


# =============================================================================
# All Perturbation Prompts Registry
# =============================================================================

PERTURBATION_PROMPTS = {
    # Placebo
    "PARAPHRASE_PROMPT_V1": PARAPHRASE_PROMPT_V1,
    # Coarse-grained
    "WRONG_PLAN_PROMPT_V1": WRONG_PLAN_PROMPT_V1,
    "WRONG_PLAN_PROMPT_V2": WRONG_PLAN_PROMPT_V2,
    # Fine-grained LLM mutations
    "WRONG_PARAMETER_PROMPT_V1": WRONG_PARAMETER_PROMPT_V1,
    "VALUE_MUTATION_PROMPT_V1": VALUE_MUTATION_PROMPT_V1,
    "WRONG_DATE_PROMPT_V1": WRONG_DATE_PROMPT_V1,
    "WRONG_IDENTIFIER_PROMPT_V1": WRONG_IDENTIFIER_PROMPT_V1,
}
