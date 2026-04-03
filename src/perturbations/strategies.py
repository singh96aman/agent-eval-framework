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

from src.data.schema import Trajectory, Step, StepType
from src.perturbations.tool_similarity import ToolSimilarityMatcher


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

    Inject wrong reasoning that misinterprets the task.

    Examples:
    - User asks for "schedule + details" → Agent only plans to get schedule
    - User asks for "latest data" → Agent plans to get historical data
    - User asks "compare X and Y" → Agent only analyzes X
    """

    PLANNING_ERROR_TEMPLATES = [
        {
            "pattern": r"(schedule|upcoming|future)",
            "replacement": "historical data",
            "reasoning": "Agent confuses future with past"
        },
        {
            "pattern": r"(latest|recent|current)",
            "replacement": "historical trends",
            "reasoning": "Agent ignores recency requirement"
        },
        {
            "pattern": r"(compare|both|and)",
            "replacement": "analyze the first one",
            "reasoning": "Agent misses comparison requirement"
        },
        {
            "pattern": r"(details?|information|data) (?:for|about|on) (.+) and (.+)",
            "replacement": r"information about \2",
            "reasoning": "Agent ignores second requirement"
        }
    ]

    def perturb_step(
        self,
        step: Step,
        trajectory: Trajectory,
        system_prompt: Optional[str] = None
    ) -> Step:
        """
        Inject planning error by modifying the Thought field.

        Strategy:
        1. Parse the step content to extract "Thought" section
        2. Apply error template to inject wrong reasoning
        3. Keep Action/Action Input unchanged (they'll be wrong given the thought)

        Args:
            step: Step to perturb
            trajectory: Full trajectory for context
            system_prompt: Not used for planning errors

        Returns:
            Step with perturbed reasoning
        """
        perturbed = deepcopy(step)

        # Extract Thought from content
        content = step.content
        thought_match = re.search(r"Thought:\s*(.+?)(?:\nAction:|$)", content, re.DOTALL)

        if not thought_match:
            # No clear thought field, inject generic planning error
            perturbed.content = (
                "Thought: I need to handle this task, but I'll focus on a subset "
                "of the requirements for now.\n" + content
            )
            perturbed.metadata["perturbation"] = {
                "type": "planning_error",
                "template": "generic_incomplete_planning",
                "reasoning": "Agent plans to address only part of the task"
            }
            return perturbed

        original_thought = thought_match.group(1).strip()

        # Apply error template
        perturbed_thought = self._apply_error_template(
            original_thought,
            trajectory.ground_truth.task_description
        )

        # Replace thought in content
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

    def _apply_error_template(self, thought: str, task: str) -> str:
        """
        Apply an error template to inject wrong reasoning.

        Args:
            thought: Original thought
            task: Task description for context

        Returns:
            Perturbed thought with planning error
        """
        # Try to match against error templates
        for template in self.PLANNING_ERROR_TEMPLATES:
            pattern = template["pattern"]
            if re.search(pattern, task.lower()) or re.search(pattern, thought.lower()):
                replacement = template["replacement"]

                # Inject the error into the thought
                perturbed = re.sub(
                    pattern,
                    replacement,
                    thought,
                    count=1,
                    flags=re.IGNORECASE
                )

                if perturbed != thought:
                    return perturbed

        # Fallback: Generic incomplete planning
        return (
            f"{thought.split('.')[0]}. "
            "I'll focus on the main requirement and address other aspects if needed."
        )


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

        Changes the file path in a file_edit action to a related but wrong file.
        """
        perturbed = deepcopy(step)

        if not step.tool_input:
            return perturbed

        # Find file path in input
        file_path = step.tool_input.get("file") or step.tool_input.get("path")

        if not file_path:
            return perturbed

        # Generate wrong but plausible file path
        original_path = str(file_path)
        wrong_path = self._generate_wrong_file_path(original_path)

        # Apply perturbation
        perturbed.tool_input = deepcopy(step.tool_input)
        if "file" in perturbed.tool_input:
            perturbed.tool_input["file"] = wrong_path
        if "path" in perturbed.tool_input:
            perturbed.tool_input["path"] = wrong_path

        # Update content
        if original_path in step.content:
            perturbed.content = step.content.replace(original_path, wrong_path)

        perturbed.metadata["perturbation"] = {
            "type": "swebench_error",
            "error_subtype": "wrong_file",
            "original_file": original_path,
            "wrong_file": wrong_path,
        }

        return perturbed

    def _wrong_location_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Edit wrong location in file (analogous to wrong parameter - spatial).

        Changes line numbers or function names in the edit target.
        """
        perturbed = deepcopy(step)

        if not step.tool_input:
            return perturbed

        perturbed.tool_input = deepcopy(step.tool_input)

        # Try to find and modify line numbers
        for key in ["line", "start_line", "line_number", "lineno"]:
            if key in perturbed.tool_input:
                original_line = perturbed.tool_input[key]
                if isinstance(original_line, int):
                    # Shift line number
                    offset = self.random.choice([-20, -10, 10, 20])
                    perturbed.tool_input[key] = max(1, original_line + offset)

                    perturbed.metadata["perturbation"] = {
                        "type": "swebench_error",
                        "error_subtype": "wrong_location",
                        "original_line": original_line,
                        "wrong_line": perturbed.tool_input[key],
                    }
                    return perturbed

        # Try to modify function/class name
        for key in ["function", "method", "class", "target"]:
            if key in perturbed.tool_input:
                original = perturbed.tool_input[key]
                if isinstance(original, str):
                    # Add suffix to name
                    perturbed.tool_input[key] = original + "_helper"

                    perturbed.metadata["perturbation"] = {
                        "type": "swebench_error",
                        "error_subtype": "wrong_location",
                        "original_target": original,
                        "wrong_target": perturbed.tool_input[key],
                    }
                    return perturbed

        return perturbed

    def _wrong_value_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Wrong code value (analogous to wrong parameter value).

        Changes a value in the code patch/edit.
        """
        perturbed = deepcopy(step)

        if not step.tool_input:
            return perturbed

        perturbed.tool_input = deepcopy(step.tool_input)

        # Find code/patch content
        for key in ["patch", "code", "content", "new_content"]:
            if key in perturbed.tool_input:
                original_code = str(perturbed.tool_input[key])

                # Find and modify a value in the code
                modified_code = self._modify_code_value(original_code)

                if modified_code != original_code:
                    perturbed.tool_input[key] = modified_code
                    perturbed.metadata["perturbation"] = {
                        "type": "swebench_error",
                        "error_subtype": "wrong_value",
                        "original_code_snippet": original_code[:200],
                        "modified_code_snippet": modified_code[:200],
                    }
                    return perturbed

        return perturbed

    def _wrong_diagnosis_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Wrong bug diagnosis (analogous to planning error).

        Modifies the reasoning/thought to indicate wrong root cause.
        """
        perturbed = deepcopy(step)

        # Find thought in content
        content = step.content
        thought_match = re.search(
            r"Thought:\s*(.+?)(?:\nAction:|$)", content, re.DOTALL
        )

        if thought_match:
            original_thought = thought_match.group(1).strip()

            # Inject wrong diagnosis
            wrong_diagnoses = [
                "The bug is likely in the initialization code",
                "This appears to be a caching issue",
                "The problem is probably in the error handling",
                "This looks like a race condition",
                "The issue is in the data validation logic",
            ]

            wrong_thought = (
                f"{original_thought.split('.')[0]}. "
                f"However, {self.random.choice(wrong_diagnoses)}."
            )

            perturbed.content = content.replace(
                f"Thought: {original_thought}",
                f"Thought: {wrong_thought}"
            )

            perturbed.metadata["perturbation"] = {
                "type": "swebench_error",
                "error_subtype": "wrong_diagnosis",
                "original_thought": original_thought[:200],
                "wrong_thought": wrong_thought[:200],
            }

        return perturbed

    def _wrong_reference_perturbation(
        self, step: Step, trajectory: Trajectory
    ) -> Step:
        """
        Use wrong variable/reference (analogous to data reference error).

        Changes a variable name that should reference earlier search results.
        """
        perturbed = deepcopy(step)

        if not step.tool_input:
            return perturbed

        # Find variable-like strings in input
        for key, val in step.tool_input.items():
            if isinstance(val, str) and re.match(r'^[a-z_][a-z0-9_]*$', val):
                # Looks like a variable name
                perturbed.tool_input = deepcopy(step.tool_input)
                perturbed.tool_input[key] = val + "_old"

                perturbed.metadata["perturbation"] = {
                    "type": "swebench_error",
                    "error_subtype": "wrong_reference",
                    "original_reference": val,
                    "wrong_reference": val + "_old",
                }
                return perturbed

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
