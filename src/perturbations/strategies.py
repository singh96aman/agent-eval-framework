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

    Examples:
    - {"username": "nike"} → {}
    - {"id_race": "53128"} → {"id_race": "99999"}
    - {"date": "2022-12-01"} → {"date": "12-01-2022"}
    """

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
        system_prompt: Optional[str] = None
    ) -> Step:
        """
        Corrupt tool parameters.

        Strategy:
        1. Parse Action Input JSON
        2. Apply one of: missing param, wrong value, wrong name, wrong type
        3. Replace in content

        Args:
            step: Step to perturb
            trajectory: Full trajectory for context
            system_prompt: Not used for parameter errors

        Returns:
            Step with corrupted parameters
        """
        if not step.tool_input:
            # No parameters to perturb
            return deepcopy(step)

        # Choose error type
        error_type = self.random.choice(self.ERROR_TYPES)

        # Apply error
        perturbed = deepcopy(step)
        original_input = deepcopy(step.tool_input)

        if error_type == "missing_required_param":
            perturbed_input = self._remove_random_param(original_input)
        elif error_type == "wrong_param_value":
            perturbed_input = self._corrupt_param_value(original_input)
        elif error_type == "wrong_param_name":
            perturbed_input = self._rename_random_param(original_input)
        else:  # wrong_param_type
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
            "error_subtype": error_type,
            "original_input": original_input,
            "perturbed_input": perturbed_input
        }

        return perturbed

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
