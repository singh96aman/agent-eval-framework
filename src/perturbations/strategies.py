"""
Perturbation strategies for generating realistic errors.

This module implements three types of perturbations:
- Type A: Planning errors (wrong task understanding)
- Type B: Tool selection errors (wrong tool choice)
- Type C: Parameter errors (wrong arguments)
"""

import json
import random
import re
from typing import Dict, Any, Optional, List
from copy import deepcopy

from src.data.schema import Trajectory, Step
from src.perturbations.tool_similarity import ToolSimilarityMatcher


def _reduce_scope(thought: str) -> Optional[str]:
    """
    Reduce scope of multi-part task to single part.

    Transforms "X and Y" patterns to just "X".
    """
    # Pattern: "get/find/check X and Y" -> "get/find/check X"
    patterns = [
        (r"(get|find|check|retrieve|look up)\s+(.+?)\s+and\s+(.+?)(?:\.|$)",
         r"\1 \2."),
        (r"(both|also)\s+(.+?)\s+and\s+(.+?)(?:\.|$)",
         r"\2."),
        (r"(.+?)\s+as well as\s+(.+?)(?:\.|$)",
         r"\1."),
    ]

    for pattern, replacement in patterns:
        match = re.search(pattern, thought, re.IGNORECASE)
        if match:
            result = re.sub(pattern, replacement, thought, count=1,
                            flags=re.IGNORECASE)
            if result != thought:
                return result

    return None


class BasePerturbationStrategy:
    """Base class for perturbation strategies."""

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize strategy with optional random seed."""
        self.random = random.Random(random_seed)

    def perturb_step(
        self,
        step: Step,
        trajectory: Trajectory,
        system_prompt: Optional[str] = None
    ) -> Step:
        """
        Apply perturbation to a step.

        Args:
            step: Step to perturb
            trajectory: Full trajectory for context
            system_prompt: System prompt with tool definitions

        Returns:
            Perturbed step
        """
        raise NotImplementedError


class PlanningErrorStrategy(BasePerturbationStrategy):
    """
    Type A: Planning errors.

    Inject wrong reasoning that misinterprets the task SEMANTICALLY.
    Changes must be grammatically correct and represent plausible agent mistakes.

    Examples:
    - User asks for "schedule + details" → Agent only plans to get schedule
    - User asks for "latest data" → Agent plans to get historical data
    - User asks "compare X and Y" → Agent only analyzes X
    """

    # Semantic planning error strategies (not text corruption)
    PLANNING_ERROR_STRATEGIES = [
        {
            "name": "temporal_confusion",
            "trigger_words": ["schedule", "upcoming", "future", "next", "tomorrow"],
            "transformation": lambda t: re.sub(
                r"(get|find|check|look up)\s+(the\s+)?(upcoming|next|future|scheduled)",
                r"\1 \2past",
                t,
                flags=re.IGNORECASE
            ) if re.search(r"(upcoming|next|future|scheduled)", t, re.IGNORECASE) else None,
            "reasoning": "Agent confuses future with past events"
        },
        {
            "name": "recency_ignored",
            "trigger_words": ["latest", "recent", "current", "new", "today"],
            "transformation": lambda t: re.sub(
                r"(get|find|check|retrieve)\s+(the\s+)?(latest|recent|current|newest)",
                r"\1 \2available",
                t,
                flags=re.IGNORECASE
            ) if re.search(r"(latest|recent|current|newest)", t, re.IGNORECASE) else None,
            "reasoning": "Agent ignores recency requirement"
        },
        {
            "name": "scope_reduction",
            "trigger_words": ["and", "both", "also", "as well", "plus"],
            "transformation": lambda t: _reduce_scope(t),
            "reasoning": "Agent addresses only first part of multi-part task"
        },
        {
            "name": "comparison_missed",
            "trigger_words": ["compare", "difference", "versus", "vs", "between"],
            "transformation": lambda t: re.sub(
                r"(compare|find the difference between|contrast)\s+(.+?)\s+(and|with|versus|vs\.?)\s+(.+)",
                r"analyze \2",
                t,
                flags=re.IGNORECASE
            ) if re.search(r"(compare|difference|versus|vs|between)", t, re.IGNORECASE) else None,
            "reasoning": "Agent misses comparison requirement, only analyzes one item"
        },
    ]

    def perturb_step(
        self,
        step: Step,
        trajectory: Trajectory,
        system_prompt: Optional[str] = None
    ) -> Step:
        """
        Inject planning error by modifying the Thought field SEMANTICALLY.

        The perturbed thought must be:
        1. Grammatically correct
        2. A plausible agent mistake
        3. Semantically different from original (not just text insertion)

        Args:
            step: Step to perturb
            trajectory: Full trajectory for context
            system_prompt: Not used for planning errors

        Returns:
            Step with perturbed reasoning
        """
        perturbed = deepcopy(step)
        content = step.content
        task_desc = trajectory.ground_truth.task_description or ""

        # Extract Thought from content
        thought_match = re.search(r"Thought:\s*(.+?)(?:\nAction:|$)", content, re.DOTALL)

        if thought_match:
            original_thought = thought_match.group(1).strip()
            perturbed_thought = self._apply_semantic_error(original_thought, task_desc)

            if perturbed_thought and perturbed_thought != original_thought:
                # Successfully modified thought
                new_content = content.replace(
                    f"Thought: {original_thought}",
                    f"Thought: {perturbed_thought}"
                )
                perturbed.content = new_content
                perturbed.metadata["perturbation"] = {
                    "type": "planning_error",
                    "original_thought": original_thought,
                    "perturbed_thought": perturbed_thought
                }
                return perturbed

        # For steps without Thought: or where semantic modification failed,
        # create a new grammatically correct planning statement
        perturbed_content, reasoning = self._create_planning_prefix(content, task_desc)
        perturbed.content = perturbed_content
        perturbed.metadata["perturbation"] = {
            "type": "planning_error",
            "template": "semantic_prefix",
            "reasoning": reasoning
        }
        return perturbed

    def _apply_semantic_error(self, thought: str, task: str) -> Optional[str]:
        """
        Apply semantic transformation to thought.

        Returns modified thought or None if no transformation applies.
        """
        combined_text = f"{task} {thought}".lower()

        for strategy in self.PLANNING_ERROR_STRATEGIES:
            # Check if trigger words are present
            if not any(word in combined_text for word in strategy["trigger_words"]):
                continue

            # Try transformation
            result = strategy["transformation"](thought)
            if result and result != thought:
                return result

        return None

    def _create_planning_prefix(self, content: str, task_desc: str) -> tuple:
        """
        Create a semantically meaningful planning prefix.

        Returns (modified_content, reasoning).
        """
        # Analyze task to create appropriate wrong planning
        task_lower = task_desc.lower()

        # Check for multi-part tasks
        if " and " in task_lower or " also " in task_lower or "both" in task_lower:
            prefix = "I'll start by addressing the first part of this request."
            reasoning = "Agent plans to only handle first part of multi-part task"
        elif "all" in task_lower or "every" in task_lower or "each" in task_lower:
            prefix = "I'll focus on one representative example to answer this."
            reasoning = "Agent reduces scope from 'all' to 'one'"
        elif "compare" in task_lower or "difference" in task_lower:
            prefix = "I'll look up information about the first item mentioned."
            reasoning = "Agent misses comparison, only researches one item"
        elif "latest" in task_lower or "recent" in task_lower or "current" in task_lower:
            prefix = "I'll find some relevant information about this topic."
            reasoning = "Agent ignores recency constraint"
        else:
            # Generic but grammatically correct prefix
            prefix = "I'll address the main aspect of this request first."
            reasoning = "Agent simplifies task scope"

        # Construct the perturbed content
        if "Thought:" in content:
            # Insert after Thought:
            perturbed = re.sub(
                r"(Thought:\s*)",
                f"\\1{prefix} ",
                content,
                count=1
            )
        else:
            # Prepend with Thought section
            perturbed = f"Thought: {prefix}\n{content}"

        return perturbed, reasoning


class ToolSelectionErrorStrategy(BasePerturbationStrategy):
    """
    Type B: Tool selection errors.

    Replace tool call with plausible but incorrect alternative.

    Examples:
    - racecards (schedule) → results (history)
    - latest_coupons → trending_coupons
    - fixtures_by_date → results_by_date
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize with tool similarity matcher."""
        super().__init__(random_seed)
        self.tool_matcher = ToolSimilarityMatcher()

    def perturb_step(
        self,
        step: Step,
        trajectory: Trajectory,
        system_prompt: Optional[str] = None
    ) -> Step:
        """
        Replace tool call with similar but incorrect tool.

        Strategy:
        1. Extract tool name from Action field
        2. Use ToolSimilarityMatcher to find plausible substitutes
        3. Replace tool name in Action field
        4. Keep Action Input unchanged (may cause errors downstream)

        Args:
            step: Step to perturb
            trajectory: Full trajectory for context
            system_prompt: System prompt with tool definitions

        Returns:
            Step with wrong tool selection
        """
        if not step.tool_name or not system_prompt:
            # Can't perturb if no tool or no system prompt
            return deepcopy(step)

        # Index tools from system prompt
        self.tool_matcher.index_tools(system_prompt)

        # Find plausible substitutes
        substitutes = self.tool_matcher.find_plausible_substitutes(
            step.tool_name,
            max_substitutes=3
        )

        if not substitutes:
            # No plausible substitute found, return original
            return deepcopy(step)

        # Choose random substitute
        wrong_tool = self.random.choice(substitutes)

        # Create perturbed step
        perturbed = deepcopy(step)
        perturbed.tool_name = wrong_tool

        # Replace tool name in content
        content = step.content
        content = re.sub(
            rf"Action:\s*{re.escape(step.tool_name)}",
            f"Action: {wrong_tool}",
            content
        )

        perturbed.content = content
        perturbed.metadata["perturbation"] = {
            "type": "tool_selection_error",
            "original_tool": step.tool_name,
            "wrong_tool": wrong_tool,
            "alternatives_considered": substitutes
        }

        return perturbed


class ParameterErrorStrategy(BasePerturbationStrategy):
    """
    Type C: Parameter errors.

    Corrupt or omit parameters in tool calls.

    Subtypes (realism-constrained per PerturbationDiversity.MD):
    - C1: wrong_value - Replace value with incorrect but plausible value
    - C2: format_error - Correct value, wrong format (date formats, etc.)
    - C3: off_by_one - Numeric values slightly wrong (common programming error)

    Note: Missing/wrong param name excluded as they often cause API crashes.

    Examples:
    - C1: {"id_race": "53128"} → {"id_race": "99999"}
    - C2: {"date": "2024-01-15"} → {"date": "01/15/2024"}
    - C3: {"page": 1} → {"page": 0}
    """

    # Realism-constrained subtypes (execution continues with wrong data)
    ERROR_SUBTYPES = [
        "wrong_value",      # C1: Wrong but plausible value
        "format_error",     # C2: Correct value, wrong format
        "off_by_one",       # C3: Subtle numeric error
    ]

    # Legacy types for backward compatibility
    ERROR_TYPES = [
        "missing_required_param",
        "wrong_param_value",
        "wrong_param_name",
        "wrong_param_type"
    ]

    def perturb_step(
        self,
        step: Step,
        trajectory: Trajectory,
        system_prompt: Optional[str] = None,
        subtype: Optional[str] = None
    ) -> Step:
        """
        Corrupt tool parameters.

        Strategy:
        1. Parse Action Input JSON
        2. Apply one of the realism-constrained subtypes: wrong_value, format_error, off_by_one
        3. Replace in content

        Args:
            step: Step to perturb
            trajectory: Full trajectory for context
            system_prompt: Not used for parameter errors
            subtype: Specific subtype to use (wrong_value, format_error, off_by_one)
                     If None, randomly chooses one

        Returns:
            Step with corrupted parameters
        """
        if not step.tool_input:
            # No parameters to perturb
            return deepcopy(step)

        # Choose error subtype (use new realism-constrained subtypes)
        if subtype and subtype in self.ERROR_SUBTYPES:
            error_subtype = subtype
        else:
            error_subtype = self.random.choice(self.ERROR_SUBTYPES)

        # Apply error based on subtype
        perturbed = deepcopy(step)
        original_input = deepcopy(step.tool_input)

        if error_subtype == "wrong_value":
            # C1: Replace value with incorrect but plausible value
            perturbed_input = self._corrupt_param_value(original_input)
        elif error_subtype == "format_error":
            # C2: Correct value, wrong format
            perturbed_input = self._corrupt_param_format(original_input)
        elif error_subtype == "off_by_one":
            # C3: Subtle numeric error
            perturbed_input = self._off_by_one_error(original_input)
        else:
            # Fallback to legacy behavior
            error_subtype = self.random.choice(self.ERROR_TYPES)
            if error_subtype == "missing_required_param":
                perturbed_input = self._remove_random_param(original_input)
            elif error_subtype == "wrong_param_value":
                perturbed_input = self._corrupt_param_value(original_input)
            elif error_subtype == "wrong_param_name":
                perturbed_input = self._rename_random_param(original_input)
            else:
                perturbed_input = self._change_param_type(original_input)

        # Verify that perturbation actually changed the input
        if perturbed_input == original_input:
            # Perturbation failed to modify input, return original step unchanged
            return deepcopy(step)

        # Update step
        perturbed.tool_input = perturbed_input

        # Replace in content (Action Input: {...})
        content = step.content

        # Find and replace the JSON in Action Input
        action_input_match = re.search(
            r"Action Input:\s*(\{[^}]*\}|\{.*?\n\})",
            content,
            re.DOTALL
        )

        if action_input_match:
            original_json = action_input_match.group(1)
            new_json = json.dumps(perturbed_input, indent=2)
            content = content.replace(original_json, new_json)

        perturbed.content = content
        perturbed.metadata["perturbation"] = {
            "type": "parameter_error",
            "error_subtype": error_subtype,
            "original_input": original_input,
            "perturbed_input": perturbed_input
        }

        return perturbed

    def _corrupt_param_format(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        C2: Change parameter format (value correct, format wrong).

        Examples:
        - "2024-01-15" → "01/15/2024" (date format)
        - "john@email.com" → "john(at)email.com" (email format)
        - "+1-555-1234" → "5551234" (phone format)
        """
        if not params:
            return params

        new_params = params.copy()
        key = self.random.choice(list(new_params.keys()))
        value = new_params[key]

        if isinstance(value, str):
            # Date format: YYYY-MM-DD → MM/DD/YYYY
            if re.match(r"\d{4}-\d{2}-\d{2}", value):
                parts = value.split("-")
                new_params[key] = f"{parts[1]}/{parts[2]}/{parts[0]}"
            # Date format: MM/DD/YYYY → DD.MM.YYYY
            elif re.match(r"\d{2}/\d{2}/\d{4}", value):
                parts = value.split("/")
                new_params[key] = f"{parts[1]}.{parts[0]}.{parts[2]}"
            # Email format
            elif "@" in value and "." in value:
                new_params[key] = value.replace("@", "(at)")
            # Phone format: remove formatting
            elif re.match(r"[\d\-\+\(\) ]+", value) and len(value) >= 7:
                new_params[key] = re.sub(r"[^\d]", "", value)
            # URL format: add/remove www
            elif value.startswith("http"):
                if "www." in value:
                    new_params[key] = value.replace("www.", "")
                else:
                    new_params[key] = value.replace("://", "://www.")
            # Generic string: change case or add prefix
            else:
                new_params[key] = value.upper() if value.islower() else value.lower()

        return new_params

    def _off_by_one_error(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        C3: Off-by-one error in numeric values.

        Examples:
        - {"page": 1} → {"page": 0}
        - {"index": 5} → {"index": 4}
        - {"count": 10} → {"count": 9}
        """
        if not params:
            return params

        new_params = params.copy()

        # Find numeric parameters
        numeric_keys = [
            k for k, v in new_params.items()
            if isinstance(v, int) or (isinstance(v, str) and v.isdigit())
        ]

        if not numeric_keys:
            # No numeric params, try to find something to perturb
            return self._corrupt_param_value(params)

        key = self.random.choice(numeric_keys)
        value = new_params[key]

        # Apply off-by-one
        if isinstance(value, int):
            # Randomly +1 or -1
            offset = self.random.choice([-1, 1])
            new_params[key] = value + offset
        elif isinstance(value, str) and value.isdigit():
            int_val = int(value)
            offset = self.random.choice([-1, 1])
            new_params[key] = str(int_val + offset)

        return new_params

    def _remove_random_param(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a random parameter."""
        if not params:
            return params

        new_params = params.copy()
        key_to_remove = self.random.choice(list(new_params.keys()))
        del new_params[key_to_remove]
        return new_params

    def _corrupt_param_value(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Change a parameter value to something wrong."""
        if not params:
            return params

        new_params = params.copy()
        key = self.random.choice(list(new_params.keys()))
        value = new_params[key]

        # Corrupt based on type
        if isinstance(value, bool):
            # Check bool first (since bool is subclass of int in Python)
            new_params[key] = not value
        elif isinstance(value, str):
            if value.isdigit():
                # Numeric string → change number
                new_params[key] = str(int(value) + 99999)
            elif re.match(r"\d{4}-\d{2}-\d{2}", value):
                # Date format → scramble
                new_params[key] = value.replace("-", "/")[::-1]
            else:
                # Generic string → append "_wrong"
                new_params[key] = value + "_wrong"
        elif isinstance(value, int):
            new_params[key] = value + 99999
        elif isinstance(value, float):
            # Float → add significant offset
            new_params[key] = value + 100.0
        else:
            # Fallback for other types: convert to string and append "_wrong"
            new_params[key] = str(value) + "_wrong"

        return new_params

    def _rename_random_param(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rename a parameter key."""
        if not params:
            return params

        new_params = {}
        keys = list(params.keys())
        key_to_rename = self.random.choice(keys)

        for k, v in params.items():
            if k == key_to_rename:
                # Change parameter name slightly
                new_key = k + "_id" if "_id" not in k else k.replace("_id", "")
                new_params[new_key] = v
            else:
                new_params[k] = v

        return new_params

    def _change_param_type(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Change parameter type (string to int, etc.)."""
        if not params:
            return params

        new_params = params.copy()
        key = self.random.choice(list(new_params.keys()))
        value = new_params[key]

        # Convert type (check bool first since it's subclass of int)
        if isinstance(value, bool):
            new_params[key] = str(value).lower()  # True → "true"
        elif isinstance(value, str) and value.isdigit():
            new_params[key] = int(value)  # "123" → 123
        elif isinstance(value, int):
            new_params[key] = str(value)  # 123 → "123"
        elif isinstance(value, float):
            new_params[key] = str(value)  # 3.14 → "3.14"
        elif isinstance(value, str):
            new_params[key] = [value]  # "foo" → ["foo"]
        else:
            # Fallback: convert to string
            new_params[key] = str(value)

        return new_params


class DataReferenceErrorStrategy(BasePerturbationStrategy):
    """
    Type D: Data reference errors (hallucination).

    Replace a referenced value from a prior step with a plausible but wrong value.
    This tests judges' ability to track information flow across trajectory steps.

    Examples:
    - Prior step returned article_id=123, agent uses article_id=999
    - Prior step found user "john_doe", agent references "john_smith"
    - Prior step got price=$50, agent references $75

    Note: Not applicable to early position (step 1-2) since there's no prior
    step to reference.

    Realism: Very common real-world failure - agents lose track of IDs,
    names, and values across multi-step trajectories.
    """

    # Patterns that indicate referenceable values
    REFERENCE_PATTERNS = [
        # IDs and identifiers
        (r'"?(\w+_id)"?\s*[:=]\s*"?(\d+)"?', "id"),
        (r'"?id"?\s*[:=]\s*"?(\d+)"?', "id"),
        (r'"?(\w+Id)"?\s*[:=]\s*"?(\d+)"?', "id"),
        # Names and usernames
        (r'"?(\w*name)"?\s*[:=]\s*"([^"]+)"', "name"),
        (r'"?username"?\s*[:=]\s*"([^"]+)"', "username"),
        # Numeric values
        (r'"?(\w+)"?\s*[:=]\s*(\d+\.?\d*)', "number"),
        # Dates
        (r'"?(\w*date\w*)"?\s*[:=]\s*"([^"]+)"', "date"),
    ]

    def perturb_step(
        self,
        step: Step,
        trajectory: Trajectory,
        system_prompt: Optional[str] = None
    ) -> Step:
        """
        Replace a data reference with a hallucinated value.

        Strategy:
        1. Find values from prior steps (tool outputs, parameters)
        2. Find references to those values in current step
        3. Replace with plausible but wrong value

        Args:
            step: Step to perturb
            trajectory: Full trajectory for context (needed to find prior values)
            system_prompt: Not used for data reference errors

        Returns:
            Step with hallucinated data reference
        """
        # Find prior steps
        prior_steps = [
            s for s in trajectory.steps
            if s.step_number < step.step_number
        ]

        if not prior_steps:
            # No prior steps to reference - this perturbation type not applicable
            return deepcopy(step)

        # Extract referenceable values from prior steps
        prior_values = self._extract_prior_values(prior_steps)

        if not prior_values:
            # No values found to hallucinate
            return deepcopy(step)

        # Find references in current step that we can corrupt
        reference_to_corrupt = self._find_reference_in_step(step, prior_values)

        if not reference_to_corrupt:
            # Fallback: inject hallucination based on tool input
            return self._inject_hallucinated_input(step, prior_values)

        # Apply the hallucination
        perturbed = deepcopy(step)
        original_value = reference_to_corrupt["original_value"]
        hallucinated_value = self._generate_hallucinated_value(
            original_value, reference_to_corrupt["value_type"]
        )

        # Replace in content
        perturbed.content = step.content.replace(
            str(original_value), str(hallucinated_value)
        )

        # Replace in tool_input if applicable
        if perturbed.tool_input:
            perturbed.tool_input = self._replace_in_dict(
                perturbed.tool_input, original_value, hallucinated_value
            )

        perturbed.metadata["perturbation"] = {
            "type": "data_reference_error",
            "error_subtype": "hallucinated_reference",
            "original_value": original_value,
            "hallucinated_value": hallucinated_value,
            "value_type": reference_to_corrupt["value_type"],
            "source_step": reference_to_corrupt.get("source_step"),
        }

        return perturbed

    def _extract_prior_values(self, prior_steps: List[Step]) -> List[Dict[str, Any]]:
        """
        Extract referenceable values from prior steps.

        Returns list of dicts with: value, value_type, source_step
        """
        values = []

        for step in prior_steps:
            # Check tool output
            if step.tool_output:
                for pattern, value_type in self.REFERENCE_PATTERNS:
                    matches = re.findall(pattern, step.tool_output, re.IGNORECASE)
                    for match in matches:
                        # Handle tuple matches from groups
                        if isinstance(match, tuple):
                            val = match[-1]  # Last group is usually the value
                        else:
                            val = match
                        if val and len(str(val)) > 1:
                            values.append({
                                "value": val,
                                "value_type": value_type,
                                "source_step": step.step_number,
                            })

            # Check tool input (parameters might be referenced later)
            if step.tool_input:
                for key, val in step.tool_input.items():
                    if val and isinstance(val, (str, int, float)):
                        # Determine type
                        if "id" in key.lower():
                            value_type = "id"
                        elif "name" in key.lower():
                            value_type = "name"
                        elif "date" in key.lower():
                            value_type = "date"
                        elif isinstance(val, (int, float)) or str(val).isdigit():
                            value_type = "number"
                        else:
                            value_type = "string"

                        values.append({
                            "value": val,
                            "value_type": value_type,
                            "source_step": step.step_number,
                        })

        return values

    def _find_reference_in_step(
        self, step: Step, prior_values: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find a prior value that's referenced in current step.

        Returns dict with original_value, value_type, source_step or None
        """
        # Check content and tool_input for references to prior values
        content_str = str(step.content) + str(step.tool_input)

        for prior in prior_values:
            val_str = str(prior["value"])
            if val_str in content_str:
                return {
                    "original_value": prior["value"],
                    "value_type": prior["value_type"],
                    "source_step": prior["source_step"],
                }

        return None

    def _inject_hallucinated_input(
        self, step: Step, prior_values: List[Dict[str, Any]]
    ) -> Step:
        """
        Fallback: inject hallucination into tool input.

        If no direct reference found, pick a tool input value and
        replace it with a hallucinated version.
        """
        perturbed = deepcopy(step)

        if not step.tool_input:
            return perturbed

        # Find a suitable input parameter to hallucinate
        for key, val in step.tool_input.items():
            if val and isinstance(val, (str, int)):
                # Determine type
                if "id" in key.lower():
                    value_type = "id"
                elif "name" in key.lower():
                    value_type = "name"
                else:
                    value_type = "string"

                # Generate hallucinated value
                hallucinated = self._generate_hallucinated_value(val, value_type)

                # Apply to perturbed step
                perturbed.tool_input = deepcopy(step.tool_input)
                perturbed.tool_input[key] = hallucinated

                # Update content if it contains the original value
                if str(val) in step.content:
                    perturbed.content = step.content.replace(
                        str(val), str(hallucinated)
                    )

                perturbed.metadata["perturbation"] = {
                    "type": "data_reference_error",
                    "error_subtype": "hallucinated_reference",
                    "original_value": val,
                    "hallucinated_value": hallucinated,
                    "value_type": value_type,
                    "key": key,
                }

                return perturbed

        return perturbed

    def _generate_hallucinated_value(
        self, original_value: Any, value_type: str
    ) -> Any:
        """
        Generate a plausible but wrong value.

        The hallucinated value should:
        - Be the same type as original
        - Look plausible (not obviously wrong)
        - Be definitively incorrect
        """
        if value_type == "id":
            # IDs: generate different but plausible ID
            if isinstance(original_value, int):
                # Add offset that changes the ID significantly
                return original_value + self.random.randint(1000, 9999)
            elif isinstance(original_value, str) and original_value.isdigit():
                return str(int(original_value) + self.random.randint(1000, 9999))
            else:
                # String ID: modify last few characters
                base = str(original_value)[:-3] if len(str(original_value)) > 3 else ""
                return base + str(self.random.randint(100, 999))

        elif value_type == "name":
            # Names: swap with similar-looking name
            original_str = str(original_value)
            if "_" in original_str:
                # username style: change suffix
                parts = original_str.split("_")
                parts[-1] = parts[-1] + str(self.random.randint(1, 99))
                return "_".join(parts)
            else:
                # Regular name: append number or change case
                return original_str + str(self.random.randint(1, 9))

        elif value_type == "number":
            # Numbers: similar magnitude but wrong
            if isinstance(original_value, int):
                magnitude = max(1, abs(original_value) // 10)
                return original_value + self.random.randint(-magnitude, magnitude) * 2
            elif isinstance(original_value, float):
                return round(original_value * self.random.uniform(0.8, 1.3), 2)
            else:
                # String number
                try:
                    num = float(original_value)
                    return str(int(num * self.random.uniform(0.8, 1.3)))
                except ValueError:
                    return str(original_value) + "_modified"

        elif value_type == "date":
            # Dates: change day or month
            date_str = str(original_value)
            # Try common date formats
            if "-" in date_str:
                parts = date_str.split("-")
                if len(parts) == 3:
                    # Change day
                    try:
                        day = int(parts[2]) if len(parts[2]) <= 2 else int(parts[2][:2])
                        new_day = ((day + self.random.randint(1, 10)) % 28) + 1
                        parts[2] = str(new_day).zfill(2)
                        return "-".join(parts)
                    except ValueError:
                        pass
            return date_str + "_wrong"

        else:
            # Default: append suffix
            return str(original_value) + "_wrong"

    def _replace_in_dict(
        self, d: Dict[str, Any], old_val: Any, new_val: Any
    ) -> Dict[str, Any]:
        """Recursively replace value in dictionary."""
        result = {}
        for k, v in d.items():
            if v == old_val:
                result[k] = new_val
            elif isinstance(v, dict):
                result[k] = self._replace_in_dict(v, old_val, new_val)
            elif isinstance(v, list):
                result[k] = [
                    new_val if item == old_val else item
                    for item in v
                ]
            else:
                result[k] = v
        return result


class GAIAPerturbationStrategy(BasePerturbationStrategy):
    """
    GAIA-native perturbation strategies.

    GAIA trajectories are plain text steps without "Thought:" prefix.
    These perturbations create realistic errors appropriate for GAIA's format.

    Perturbation types (mapped to standard categories):
    - scope_reduction: Reduce task scope (planning analog)
    - wrong_source: Use wrong website/source (tool_selection analog)
    - wrong_query: Modify search query incorrectly (parameter analog)
    - wrong_extraction: Extract wrong data from results (data_reference analog)
    """

    # Semantic scope reduction patterns for planning errors
    SCOPE_REDUCTION_PATTERNS = [
        # "X and Y" -> just X
        (r"(.+)\s+and\s+(.+)", r"\1", "Ignoring second requirement"),
        # "both X and Y" -> just X
        (r"both\s+(.+)\s+and\s+(.+)", r"\1", "Only addressing first item"),
        # "compare X with Y" -> just X
        (r"compare\s+(.+)\s+(?:with|to|against)\s+(.+)", r"find \1", "Missing comparison"),
        # "all X" -> "one X"
        (r"all\s+(?:the\s+)?(\w+)", r"one \1", "Limiting scope to one"),
        # "each X" -> "the first X"
        (r"each\s+(\w+)", r"the first \1", "Only doing first"),
    ]

    # Wrong source patterns
    WRONG_SOURCES = [
        ("wikipedia", "a random blog"),
        ("official", "unofficial"),
        ("arxiv", "ResearchGate"),
        ("documentation", "Stack Overflow"),
        ("the source", "a cached version"),
        ("google", "Bing"),
    ]

    # Query modification patterns
    QUERY_MODIFICATIONS = [
        (r'"([^"]+)"', lambda m: f'"{m.group(1)[:len(m.group(1))//2]}"'),  # Truncate quoted query
        (r"search for (.+)", r"search for part of \1"),
        (r"look up (.+)", r"briefly check \1"),
    ]

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize with random seed."""
        super().__init__(random_seed)
        self.subtype = "scope_reduction"  # Default

    def perturb_step(
        self,
        step: Step,
        trajectory: Trajectory,
        system_prompt: Optional[str] = None,
        subtype: Optional[str] = None
    ) -> Step:
        """
        Apply GAIA-specific perturbation.

        Args:
            step: Step to perturb
            trajectory: Full trajectory for context
            system_prompt: Not used
            subtype: Perturbation subtype (scope_reduction, wrong_source, wrong_query, wrong_extraction)

        Returns:
            Perturbed step
        """
        if subtype:
            self.subtype = subtype

        if self.subtype == "scope_reduction":
            return self._scope_reduction_perturbation(step, trajectory)
        elif self.subtype == "wrong_source":
            return self._wrong_source_perturbation(step, trajectory)
        elif self.subtype == "wrong_query":
            return self._wrong_query_perturbation(step, trajectory)
        elif self.subtype == "wrong_extraction":
            return self._wrong_extraction_perturbation(step, trajectory)
        else:
            # Random choice
            perturbation_func = self.random.choice([
                self._scope_reduction_perturbation,
                self._wrong_source_perturbation,
                self._wrong_query_perturbation,
            ])
            return perturbation_func(step, trajectory)

    def _scope_reduction_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Planning analog: Reduce the scope of the task incorrectly.

        Changes step content to indicate incomplete task understanding.
        """
        perturbed = deepcopy(step)
        original_content = step.content
        task_desc = trajectory.ground_truth.task_description or ""

        # Try pattern matching on step content or task description
        modified = None
        reasoning = None

        for pattern, replacement, reason in self.SCOPE_REDUCTION_PATTERNS:
            # Try on step content first
            match = re.search(pattern, original_content, re.IGNORECASE)
            if match:
                modified = re.sub(pattern, replacement, original_content, count=1, flags=re.IGNORECASE)
                reasoning = reason
                break

            # Try on task description and apply to step
            if not modified and task_desc:
                match = re.search(pattern, task_desc, re.IGNORECASE)
                if match:
                    # Create a modified instruction
                    modified = f"[Focus on partial task] {original_content}"
                    reasoning = reason
                    break

        if not modified or modified == original_content:
            # Fallback: Add explicit scope limitation
            scope_phrases = [
                "Just focusing on the main part:",
                "Simplifying to the core task:",
                "Starting with the basic version:",
            ]
            prefix = self.random.choice(scope_phrases)
            modified = f"{prefix} {original_content}"
            reasoning = "Explicit scope reduction"

        perturbed.content = modified
        perturbed.metadata["perturbation"] = {
            "type": "gaia_planning",
            "error_subtype": "scope_reduction",
            "original_content": original_content[:200],
            "reasoning": reasoning,
        }

        return perturbed

    def _wrong_source_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Tool selection analog: Use wrong source/website for information.
        """
        perturbed = deepcopy(step)
        original_content = step.content

        modified = original_content
        source_change = None

        for correct, wrong in self.WRONG_SOURCES:
            if correct.lower() in original_content.lower():
                modified = re.sub(
                    correct, wrong, original_content, count=1, flags=re.IGNORECASE
                )
                source_change = f"{correct} -> {wrong}"
                break

        if modified == original_content:
            # No direct source mention, add wrong source instruction
            wrong_source_phrases = [
                "Using a quick summary site instead of the original source:",
                "Checking a secondary source rather than the primary one:",
                "Looking at cached/archived version:",
            ]
            prefix = self.random.choice(wrong_source_phrases)
            modified = f"{prefix} {original_content}"
            source_change = "Generic wrong source"

        perturbed.content = modified
        perturbed.metadata["perturbation"] = {
            "type": "gaia_tool_selection",
            "error_subtype": "wrong_source",
            "original_content": original_content[:200],
            "source_change": source_change,
        }

        return perturbed

    def _wrong_query_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Parameter analog: Modify search query incorrectly.
        """
        perturbed = deepcopy(step)
        original_content = step.content

        modified = original_content

        # Try query modification patterns
        for pattern, replacement in self.QUERY_MODIFICATIONS:
            match = re.search(pattern, original_content, re.IGNORECASE)
            if match:
                if callable(replacement):
                    modified = re.sub(pattern, replacement, original_content, count=1)
                else:
                    modified = re.sub(pattern, replacement, original_content, count=1, flags=re.IGNORECASE)
                break

        if modified == original_content:
            # Fallback: Add query limitation
            modified = original_content.replace("search", "briefly search", 1) if "search" in original_content.lower() else original_content
            if modified == original_content:
                # Ultimate fallback: truncate content if it's a query-like step
                words = original_content.split()
                if len(words) > 5:
                    modified = " ".join(words[:len(words)//2 + 1]) + "..."

        perturbed.content = modified
        perturbed.metadata["perturbation"] = {
            "type": "gaia_parameter",
            "error_subtype": "wrong_query",
            "original_content": original_content[:200],
        }

        return perturbed

    def _wrong_extraction_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Data reference analog: Extract wrong data from prior results.
        """
        perturbed = deepcopy(step)
        original_content = step.content

        # Look for numbers, specific values to modify
        modified = original_content

        # Find numbers in parentheses (common in GAIA for expected values)
        num_match = re.search(r'\((\d+)\)', original_content)
        if num_match:
            old_num = num_match.group(1)
            new_num = str(int(old_num) + self.random.randint(1, 10))
            modified = original_content.replace(f"({old_num})", f"({new_num})")
        else:
            # Find any numbers
            num_match = re.search(r'\b(\d+)\b', original_content)
            if num_match:
                old_num = num_match.group(1)
                new_num = str(int(old_num) + self.random.randint(1, 5))
                modified = re.sub(r'\b' + old_num + r'\b', new_num, original_content, count=1)
            else:
                # Add wrong extraction note
                modified = f"[Using approximate value] {original_content}"

        perturbed.content = modified
        perturbed.metadata["perturbation"] = {
            "type": "gaia_data_reference",
            "error_subtype": "wrong_extraction",
            "original_content": original_content[:200],
        }

        return perturbed


class SWEBenchPerturbationStrategy(BasePerturbationStrategy):
    """
    SWE-bench-native perturbation strategies.

    These perturbations are conceptually analogous to ToolBench perturbations
    but semantically appropriate for code editing tasks.

    Perturbation types:
    - wrong_file: Edit a different file than needed (tool selection analog)
    - wrong_location: Edit wrong function/class in correct file (parameter analog - spatial)
    - wrong_value: Change wrong variable/value in code (parameter analog - value)
    - wrong_diagnosis: Planning thinks bug is in wrong component (planning analog)
    - wrong_reference: Use wrong variable name from earlier search (data reference analog)
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize with random seed."""
        super().__init__(random_seed)
        self.subtype = "wrong_file"  # Default

    def perturb_step(
        self,
        step: Step,
        trajectory: Trajectory,
        system_prompt: Optional[str] = None,
        subtype: Optional[str] = None
    ) -> Step:
        """
        Apply SWE-bench-specific perturbation.

        Args:
            step: Step to perturb
            trajectory: Full trajectory for context
            system_prompt: Not used
            subtype: Specific perturbation type to apply

        Returns:
            Perturbed step
        """
        if subtype:
            self.subtype = subtype

        if self.subtype == "wrong_file":
            return self._wrong_file_perturbation(step, trajectory)
        elif self.subtype == "wrong_location":
            return self._wrong_location_perturbation(step, trajectory)
        elif self.subtype == "wrong_value":
            return self._wrong_value_perturbation(step, trajectory)
        elif self.subtype == "wrong_diagnosis":
            return self._wrong_diagnosis_perturbation(step, trajectory)
        elif self.subtype == "wrong_reference":
            return self._wrong_reference_perturbation(step, trajectory)
        else:
            # Random choice
            perturbation_func = self.random.choice([
                self._wrong_file_perturbation,
                self._wrong_location_perturbation,
                self._wrong_value_perturbation,
            ])
            return perturbation_func(step, trajectory)

    def _wrong_file_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Edit wrong file (analogous to wrong tool selection).

        Changes file paths in content to related but wrong paths.
        Works with SWE-bench raw text format.
        """
        perturbed = deepcopy(step)
        content = step.content

        # Find file paths in content (common patterns)
        path_patterns = [
            r'<parameter=path>([^<]+)</parameter>',
            r'"path":\s*"([^"]+)"',
            r'/testbed/([^\s<>"]+\.py)',
        ]

        modified = False
        for pattern in path_patterns:
            match = re.search(pattern, content)
            if match:
                original_path = match.group(1)
                wrong_path = self._generate_wrong_file_path(original_path)
                perturbed.content = content.replace(original_path, wrong_path, 1)
                perturbed.metadata["perturbation"] = {
                    "type": "swebench_error",
                    "error_subtype": "wrong_file",
                    "original_file": original_path,
                    "wrong_file": wrong_path,
                }
                modified = True
                break

        if not modified:
            # Fallback: modify any path-like string
            path_match = re.search(r'(/\w+)+\.py', content)
            if path_match:
                orig = path_match.group(0)
                wrong = orig.replace('.py', '_utils.py')
                perturbed.content = content.replace(orig, wrong, 1)
                perturbed.metadata["perturbation"] = {
                    "type": "swebench_error",
                    "error_subtype": "wrong_file",
                    "original": orig,
                    "modified": wrong,
                }

        return perturbed

    def _wrong_location_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Edit wrong location in file (analogous to wrong parameter - spatial).

        Modifies line numbers, function names, or class names in content.
        """
        perturbed = deepcopy(step)
        content = step.content

        # Try to find and modify line numbers in content
        line_match = re.search(r'line[_\s]*(\d+)', content, re.IGNORECASE)
        if line_match:
            orig_line = line_match.group(1)
            offset = self.random.choice([-20, -10, 10, 20])
            new_line = str(max(1, int(orig_line) + offset))
            perturbed.content = content.replace(
                line_match.group(0),
                line_match.group(0).replace(orig_line, new_line),
                1
            )
            perturbed.metadata["perturbation"] = {
                "type": "swebench_error",
                "error_subtype": "wrong_location",
                "original_line": orig_line,
                "wrong_line": new_line,
            }
            return perturbed

        # Try to modify function/class names
        func_match = re.search(
            r'(def|class|function)\s+(\w+)',
            content,
            re.IGNORECASE
        )
        if func_match:
            orig_name = func_match.group(2)
            wrong_name = orig_name + "_helper"
            perturbed.content = content.replace(orig_name, wrong_name, 1)
            perturbed.metadata["perturbation"] = {
                "type": "swebench_error",
                "error_subtype": "wrong_location",
                "original": orig_name,
                "modified": wrong_name,
            }
            return perturbed

        # Fallback: modify any identifier
        id_match = re.search(r'`(\w{4,})`', content)
        if id_match:
            orig = id_match.group(1)
            wrong = orig + "_v2"
            perturbed.content = content.replace(f"`{orig}`", f"`{wrong}`", 1)
            perturbed.metadata["perturbation"] = {
                "type": "swebench_error",
                "error_subtype": "wrong_location",
                "original": orig,
                "modified": wrong,
            }

        return perturbed

    def _wrong_value_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Wrong code value (analogous to wrong parameter value).

        Modifies values, numbers, or booleans in the content.
        """
        perturbed = deepcopy(step)
        content = step.content

        # Apply value modification to content
        modified_content = self._modify_code_value(content)

        if modified_content != content:
            perturbed.content = modified_content
            perturbed.metadata["perturbation"] = {
                "type": "swebench_error",
                "error_subtype": "wrong_value",
            }

        return perturbed

    def _wrong_diagnosis_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Wrong bug diagnosis (analogous to planning error).

        Injects wrong reasoning about the bug's root cause.
        Works with SWE-bench raw text format (no Thought: prefix).
        """
        perturbed = deepcopy(step)
        content = step.content

        # Wrong diagnoses to inject
        wrong_diagnoses = [
            "The bug is likely in the initialization code",
            "This appears to be a caching issue",
            "The problem is probably in the error handling",
            "This looks like a race condition",
            "The issue is in the data validation logic",
            "I think the bug is in the import statements",
        ]

        # SWE-bench format: plain text with reasoning
        # Look for first sentence to inject after
        first_sentence_match = re.search(r'^([^.!?]+[.!?])', content)

        if first_sentence_match:
            first_sentence = first_sentence_match.group(1)
            diagnosis = self.random.choice(wrong_diagnoses)
            injection = f" However, {diagnosis}."

            perturbed.content = content.replace(
                first_sentence,
                first_sentence + injection,
                1
            )
            perturbed.metadata["perturbation"] = {
                "type": "swebench_error",
                "error_subtype": "wrong_diagnosis",
                "injected": injection,
            }
        else:
            # Fallback: prepend wrong diagnosis
            diagnosis = self.random.choice(wrong_diagnoses)
            perturbed.content = f"[Note: {diagnosis}] {content}"
            perturbed.metadata["perturbation"] = {
                "type": "swebench_error",
                "error_subtype": "wrong_diagnosis",
                "prepended": diagnosis,
            }

        return perturbed

    def _wrong_reference_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Use wrong variable/reference (analogous to data reference error).

        Modifies variable names or identifiers in content.
        """
        perturbed = deepcopy(step)
        content = step.content

        # Find variable-like identifiers in content
        # Look for self.something or module.something patterns
        var_match = re.search(r'(self\.(\w+)|(\w+)\.(\w+))', content)
        if var_match:
            orig = var_match.group(0)
            # Modify the last part
            parts = orig.split('.')
            parts[-1] = parts[-1] + "_old"
            wrong = '.'.join(parts)
            perturbed.content = content.replace(orig, wrong, 1)
            perturbed.metadata["perturbation"] = {
                "type": "swebench_error",
                "error_subtype": "wrong_reference",
                "original": orig,
                "modified": wrong,
            }
            return perturbed

        # Try to find any identifier (snake_case)
        id_match = re.search(r'\b([a-z][a-z0-9_]{3,})\b', content)
        if id_match:
            orig = id_match.group(1)
            wrong = orig + "_v1"
            perturbed.content = content.replace(orig, wrong, 1)
            perturbed.metadata["perturbation"] = {
                "type": "swebench_error",
                "error_subtype": "wrong_reference",
                "original": orig,
                "modified": wrong,
            }

        return perturbed

    def _generate_wrong_file_path(self, original_path: str) -> str:
        """Generate a wrong but plausible file path."""
        # Common patterns for wrong files
        if "/" in original_path:
            parts = original_path.split("/")

            # Strategy 1: Change file name slightly
            if "." in parts[-1]:
                name, ext = parts[-1].rsplit(".", 1)
                # Add _utils, _helper, _old suffix
                suffix = self.random.choice(["_utils", "_helper", "_old", "_base"])
                parts[-1] = f"{name}{suffix}.{ext}"
            else:
                parts[-1] = parts[-1] + "_utils"

            return "/".join(parts)

        # Strategy 2: Change directory
        elif "test" in original_path.lower():
            return original_path.replace("test", "tests")
        elif "src" in original_path:
            return original_path.replace("src", "lib")
        else:
            return original_path.replace(".py", "_utils.py")

    def _modify_code_value(self, code: str) -> str:
        """Modify a value in code."""
        # Find numbers and change them
        def replace_number(match):
            num = int(match.group())
            return str(num + self.random.choice([-1, 1, 10, -10]))

        modified = re.sub(r'\b\d+\b', replace_number, code, count=1)

        if modified != code:
            return modified

        # Find True/False and flip
        if "True" in code:
            return code.replace("True", "False", 1)
        if "False" in code:
            return code.replace("False", "True", 1)

        # Find None and change
        if " None" in code:
            return code.replace(" None", " 0", 1)

        return code
