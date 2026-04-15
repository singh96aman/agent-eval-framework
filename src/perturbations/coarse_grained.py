"""
Coarse-grained perturbation generators for Section 3: Controlled Perturbations.

These generators inject STRUCTURAL DECISION errors that fundamentally alter
the trajectory's approach. They test whether judges recognize high-level failures.

Perturbation types:
- WRONG_PLAN: Generate a plausible but inferior plan via LLM
- FALSE_TERMINAL: Mark an incomplete step as terminal
- PREMATURE_TERMINATION: Truncate trajectory early
- WRONG_TOOL_FAMILY: Swap tool with one from a different family
"""

import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from src.typing.schema import (
    TypedTrajectory,
    TypedStep,
    StepRole,
    DependencyEdge,
)
from src.perturbations.schema import (
    PerturbationClass,
    PerturbationFamily,
    PerturbationType,
    PerturbationRecord,
)
from src.llm import DEFAULT_MODEL_ID

# Note: ToolSimilarityMatcher imported dynamically in WrongToolFamilyGenerator


class BaseCoarseGrainedGenerator(ABC):
    """
    Base class for coarse-grained perturbation generators.

    Coarse-grained perturbations operate at TRAJECTORY level (not just step level)
    and may need to modify multiple steps (indices, terminal flags, etc.).
    """

    def __init__(
        self,
        random_seed: Optional[int] = None,
        llm_client=None,
    ):
        """
        Initialize generator.

        Args:
            random_seed: Random seed for reproducibility
            llm_client: Bedrock client for LLM-based generators (WrongPlanGenerator)
        """
        self.random = random.Random(random_seed)
        self.llm_client = llm_client

    @abstractmethod
    def generate(
        self,
        typed_trajectory: TypedTrajectory,
    ) -> Optional[Tuple[PerturbationRecord, TypedTrajectory]]:
        """
        Generate coarse-grained perturbation.

        Args:
            typed_trajectory: The typed trajectory to perturb

        Returns:
            Tuple of (PerturbationRecord, modified_trajectory) or None if not applicable
        """
        pass

    @property
    @abstractmethod
    def perturbation_family(self) -> PerturbationFamily:
        """Return the perturbation family for this generator."""
        pass

    @property
    @abstractmethod
    def perturbation_type(self) -> PerturbationType:
        """Return the perturbation type for this generator."""
        pass


class WrongPlanGenerator(BaseCoarseGrainedGenerator):
    """
    Uses Claude LLM to generate a plausible but inferior plan.

    Strategy:
    1. Find steps with 'planning' role
    2. Prompt Claude to generate a plausible but inferior plan
    3. Replace the planning step content

    Example: "fix test to pass" instead of "find root cause"
    """

    def __init__(
        self,
        random_seed: Optional[int] = None,
        llm_client=None,
        model_id: str = DEFAULT_MODEL_ID,
        prompt_template: Optional[str] = None,
    ):
        super().__init__(random_seed, llm_client)
        self.model_id = model_id
        self.prompt_template = prompt_template

    @property
    def perturbation_family(self) -> PerturbationFamily:
        return PerturbationFamily.STRUCTURAL

    @property
    def perturbation_type(self) -> PerturbationType:
        return PerturbationType.WRONG_PLAN

    def generate(
        self,
        typed_trajectory: TypedTrajectory,
    ) -> Optional[Tuple[PerturbationRecord, TypedTrajectory]]:
        """
        Replace a planning step with a plausible but inferior plan.

        Returns:
            (record, modified_trajectory) or None if no planning step or no LLM client
        """
        # Find planning steps
        planning_steps = [
            step
            for step in typed_trajectory.steps
            if step.step_role == StepRole.PLANNING.value
        ]

        if not planning_steps:
            return None

        # Choose a planning step to modify
        step_to_modify = self.random.choice(planning_steps)
        step_index = step_to_modify.step_index

        # Generate inferior plan
        original_plan = step_to_modify.raw_text
        inferior_plan = self._generate_inferior_plan(
            original_plan=original_plan,
            task_text=typed_trajectory.task_text,
            benchmark=typed_trajectory.benchmark,
        )

        if not inferior_plan or inferior_plan == original_plan:
            return None

        # Create modified trajectory
        modified_trajectory = deepcopy(typed_trajectory)

        # Find and update the step
        for step in modified_trajectory.steps:
            if step.step_index == step_index:
                step.raw_text = inferior_plan
                break

        # Create perturbation record
        record = PerturbationRecord.create(
            original_trajectory_id=typed_trajectory.trajectory_id,
            perturbation_class=PerturbationClass.COARSE_GRAINED,
            perturbation_family=self.perturbation_family,
            perturbation_type=self.perturbation_type,
            target_step_index=step_index,
            target_slot="raw_text",
            original_value=original_plan,
            perturbed_value=inferior_plan,
            mutation_method=(
                "llm_inferior_plan_generation"
                if self.llm_client
                else "template_inferior_plan"
            ),
            expected_impact=3,  # Critical impact - wrong approach
            notes="Replaced original plan with plausible but inferior strategy",
        )

        return (record, modified_trajectory)

    def _generate_inferior_plan(
        self,
        original_plan: str,
        task_text: str,
        benchmark: str,
    ) -> str:
        """
        Generate a plausible but inferior plan.

        Uses LLM if available, otherwise falls back to template-based approach.
        """
        if self.llm_client:
            return self._generate_inferior_plan_llm(original_plan, task_text)
        else:
            return self._generate_inferior_plan_template(
                original_plan, task_text, benchmark
            )

    def _generate_inferior_plan_llm(
        self,
        original_plan: str,
        task_text: str,
    ) -> str:
        """Generate inferior plan using LLM."""
        # Use prompt from config if provided, otherwise use default
        if self.prompt_template:
            from src.prompts.registry import get_prompt
            prompt_text = get_prompt(self.prompt_template)
            prompt = prompt_text.format(
                task_description=task_text,
                original_plan=original_plan,
            )
        else:
            # Fallback to hardcoded default (V1 behavior)
            prompt = f"""You are helping generate test data for evaluating AI judges.

Given the original task and the agent's good plan, generate a PLAUSIBLE but INFERIOR alternative plan.

The inferior plan should:
1. Be grammatically correct and reasonable-sounding
2. Address the task superficially but miss the core objective
3. Be the kind of mistake a less experienced agent might make
4. NOT be obviously wrong or nonsensical

TASK: {task_text}

ORIGINAL GOOD PLAN:
{original_plan}

Generate ONLY the inferior plan text, nothing else. Keep it similar in length and style to the original.
"""
        try:
            result = self.llm_client.invoke(
                model_id=self.model_id,
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
            )
            return result.get("response", "").strip()
        except Exception:
            # Fallback to template if LLM fails
            return self._generate_inferior_plan_template(original_plan, task_text, "")

    def _generate_inferior_plan_template(
        self,
        original_plan: str,
        task_text: str,
        benchmark: str,
    ) -> str:
        """Generate inferior plan using templates."""
        # Inferior plan templates based on common mistakes
        inferior_templates = [
            # Scope reduction
            (
                ["and", "both", "also", "multiple", "all"],
                lambda p: self._reduce_scope(p),
            ),
            # Surface-level fix
            (
                ["fix", "bug", "error", "issue", "problem"],
                lambda p: p.replace("find", "fix")
                .replace("investigate", "patch")
                .replace("diagnose", "address"),
            ),
            # Skip verification
            (
                ["verify", "check", "confirm", "validate"],
                lambda p: p.replace("verify", "assume")
                .replace("check", "proceed with")
                .replace("confirm", "use"),
            ),
            # Rush to conclusion
            (
                ["search", "find", "look", "research"],
                lambda p: (
                    f"Quickly {p.lower()}" if not p.lower().startswith("quick") else p
                ),
            ),
        ]

        plan_lower = original_plan.lower()

        for trigger_words, transform in inferior_templates:
            if any(word in plan_lower for word in trigger_words):
                transformed = transform(original_plan)
                if transformed != original_plan:
                    return transformed

        # Generic fallback: add "simple" or "quick" approach
        if "simple" not in plan_lower and "quick" not in plan_lower:
            return f"Take a simple approach: {original_plan}"

        return original_plan

    def _reduce_scope(self, plan: str) -> str:
        """Reduce scope of multi-part plan."""
        import re

        # Pattern: "X and Y" -> just "X"
        patterns = [
            (r"(.+?)\s+and\s+(.+?)(?:\.|$)", r"\1."),
            (r"both\s+(.+?)\s+and\s+(.+?)(?:\.|$)", r"\1."),
            (r"(.+?)\s+as well as\s+(.+?)(?:\.|$)", r"\1."),
        ]

        for pattern, replacement in patterns:
            match = re.search(pattern, plan, re.IGNORECASE)
            if match:
                result = re.sub(
                    pattern, replacement, plan, count=1, flags=re.IGNORECASE
                )
                if result != plan:
                    return result

        return plan


class FalseTerminalGenerator(BaseCoarseGrainedGenerator):
    """
    Marks an incomplete step as terminal (false terminal).

    Per Section 3.4.6 constraints, only perturb steps that:
    - Have produced_artifacts (plausible partial answer)
    - Don't end with "..." or incomplete text
    - Are NOT observation steps (just tool output)
    - Are NOT planning steps (intent, not result)

    Valid roles: reasoning, extraction, decision
    """

    VALID_ROLES_FOR_FALSE_TERMINAL = {
        StepRole.REASONING.value,
        StepRole.EXTRACTION.value,
        StepRole.DECISION.value,
    }

    def __init__(
        self,
        random_seed: Optional[int] = None,
        llm_client=None,
    ):
        super().__init__(random_seed, llm_client)

    @property
    def perturbation_family(self) -> PerturbationFamily:
        return PerturbationFamily.TERMINAL_FLAG

    @property
    def perturbation_type(self) -> PerturbationType:
        return PerturbationType.FALSE_TERMINAL

    def generate(
        self,
        typed_trajectory: TypedTrajectory,
    ) -> Optional[Tuple[PerturbationRecord, TypedTrajectory]]:
        """
        Mark a non-terminal step as terminal.

        Returns:
            (record, modified_trajectory) or None if no eligible step found
        """
        # Find eligible steps
        eligible_steps = self._find_eligible_steps(typed_trajectory)

        if not eligible_steps:
            return None

        # Choose a step to make falsely terminal
        step_to_modify = self.random.choice(eligible_steps)
        step_index = step_to_modify.step_index

        # Create modified trajectory
        modified_trajectory = deepcopy(typed_trajectory)

        # Find and update the step
        for step in modified_trajectory.steps:
            if step.step_index == step_index:
                step.is_terminal_step = True
                step.produces_final_answer = True
                break

        # Create perturbation record
        record = PerturbationRecord.create(
            original_trajectory_id=typed_trajectory.trajectory_id,
            perturbation_class=PerturbationClass.COARSE_GRAINED,
            perturbation_family=self.perturbation_family,
            perturbation_type=self.perturbation_type,
            target_step_index=step_index,
            target_slot="is_terminal_step",
            original_value=False,
            perturbed_value=True,
            mutation_method="false_terminal_flag",
            expected_impact=3,  # Critical - trajectory ends prematurely
            notes=f"Marked step {step_index} (role={step_to_modify.step_role}) as falsely terminal",
        )

        return (record, modified_trajectory)

    def _find_eligible_steps(self, trajectory: TypedTrajectory) -> List[TypedStep]:
        """
        Find steps eligible for false terminal perturbation.

        Relaxed criteria (OR instead of AND for main conditions):
        - Has produced_artifacts (plausible partial answer) OR
        - Has valid role (reasoning, extraction, decision)
        - Doesn't end with "..." or incomplete text
        - Is NOT already terminal
        """
        eligible = []

        for step in trajectory.steps:
            # Skip already terminal steps
            if step.is_terminal_step:
                continue

            # Relaxed: Accept valid role OR has artifacts (not requiring both)
            has_valid_role = step.step_role in self.VALID_ROLES_FOR_FALSE_TERMINAL
            has_artifacts = bool(step.produced_artifacts)

            if not (has_valid_role or has_artifacts):
                continue

            # Check content doesn't look incomplete
            raw_text = step.raw_text.strip()
            if raw_text.endswith("..."):
                continue
            if raw_text.endswith(","):
                continue
            if len(raw_text) < 20:
                continue  # Too short to be a plausible answer

            eligible.append(step)

        return eligible


class PrematureTerminationGenerator(BaseCoarseGrainedGenerator):
    """
    Truncates trajectory by removing steps after a chosen point.

    Strategy:
    1. Pick a step before the real terminal
    2. Remove all subsequent steps
    3. Set terminal flags on the new last step
    """

    def __init__(
        self,
        random_seed: Optional[int] = None,
        llm_client=None,
        min_steps_remaining: int = 2,
    ):
        """
        Initialize generator.

        Args:
            random_seed: Random seed
            llm_client: LLM client (not used)
            min_steps_remaining: Minimum steps that must remain after truncation
        """
        super().__init__(random_seed, llm_client)
        self.min_steps_remaining = min_steps_remaining

    @property
    def perturbation_family(self) -> PerturbationFamily:
        return PerturbationFamily.TERMINAL_FLAG

    @property
    def perturbation_type(self) -> PerturbationType:
        return PerturbationType.PREMATURE_TERMINATION

    def generate(
        self,
        typed_trajectory: TypedTrajectory,
    ) -> Optional[Tuple[PerturbationRecord, TypedTrajectory]]:
        """
        Truncate trajectory at a chosen point.

        Returns:
            (record, modified_trajectory) or None if trajectory too short
        """
        num_steps = len(typed_trajectory.steps)

        # Need enough steps to truncate meaningfully
        if num_steps < self.min_steps_remaining + 1:
            return None

        # Find the original terminal step index
        original_terminal_idx = None
        for step in typed_trajectory.steps:
            if step.is_terminal_step:
                original_terminal_idx = step.step_index
                break

        if original_terminal_idx is None:
            # No terminal step found, use last step
            original_terminal_idx = typed_trajectory.steps[-1].step_index

        # Choose truncation point (must leave at least min_steps_remaining)
        # and must be before the original terminal
        max_truncation_idx = min(original_terminal_idx - 1, num_steps - 1)
        min_truncation_idx = self.min_steps_remaining - 1  # 0-indexed

        if max_truncation_idx <= min_truncation_idx:
            return None

        # Find eligible truncation points (prefer steps with artifacts)
        eligible_indices = []
        for step in typed_trajectory.steps:
            idx = step.step_index
            if min_truncation_idx <= idx <= max_truncation_idx:
                # Prefer steps that could plausibly be terminal
                if step.produced_artifacts or step.step_role in {
                    StepRole.REASONING.value,
                    StepRole.EXTRACTION.value,
                    StepRole.DECISION.value,
                }:
                    eligible_indices.append(idx)

        if not eligible_indices:
            # Fallback: any valid index
            eligible_indices = list(range(min_truncation_idx, max_truncation_idx + 1))

        if not eligible_indices:
            return None

        truncation_idx = self.random.choice(eligible_indices)

        # Create modified trajectory
        modified_trajectory = deepcopy(typed_trajectory)

        # Remove steps after truncation point
        modified_trajectory.steps = [
            s for s in modified_trajectory.steps if s.step_index <= truncation_idx
        ]

        # Set terminal flags on new last step
        if modified_trajectory.steps:
            last_step = modified_trajectory.steps[-1]
            last_step.is_terminal_step = True
            last_step.produces_final_answer = True

        # Update num_steps
        modified_trajectory.num_steps = len(modified_trajectory.steps)

        # Count removed steps
        removed_count = num_steps - len(modified_trajectory.steps)

        # Create perturbation record
        record = PerturbationRecord.create(
            original_trajectory_id=typed_trajectory.trajectory_id,
            perturbation_class=PerturbationClass.COARSE_GRAINED,
            perturbation_family=self.perturbation_family,
            perturbation_type=self.perturbation_type,
            target_step_index=truncation_idx,
            target_slot="trajectory_length",
            original_value=num_steps,
            perturbed_value=len(modified_trajectory.steps),
            mutation_method="premature_truncation",
            expected_impact=3,  # Critical - incomplete trajectory
            notes=f"Truncated at step {truncation_idx}, removed {removed_count} subsequent steps",
        )

        return (record, modified_trajectory)


class WrongToolFamilyGenerator(BaseCoarseGrainedGenerator):
    """
    Swaps a tool with one from a completely different family.

    Unlike near-neighbor (fine-grained), this swaps tools across families:
    - file_edit -> web_search (completely wrong category)
    - database_query -> code_execution

    Must still be somewhat plausible for the task.
    """

    # Tool family definitions for cross-family swaps
    TOOL_FAMILIES: Dict[str, Set[str]] = {
        "file_operations": {
            "read_file",
            "write_file",
            "edit_file",
            "list_files",
            "create_file",
            "delete_file",
            "file_search",
            "open_file",
            "str_replace_editor",
            "view",
            "create",
            "insert",
        },
        "web_search": {
            "web_search",
            "google_search",
            "search",
            "browse",
            "fetch_url",
            "http_get",
            "download",
            "web_browse",
            "tavily_search",
            "bing_search",
            "duck_duck_go",
        },
        "code_execution": {
            "execute",
            "run",
            "bash",
            "shell",
            "python",
            "exec",
            "run_python",
            "execute_command",
            "terminal",
            "subprocess",
        },
        "database": {
            "query",
            "sql",
            "database",
            "db_query",
            "select",
            "insert",
            "update",
            "delete",
            "fetch_data",
        },
        "api_calls": {
            "api_call",
            "http_request",
            "rest_api",
            "graphql",
            "post",
            "get",
            "put",
            "patch",
            "request",
        },
        "analysis": {
            "analyze",
            "compute",
            "calculate",
            "summarize",
            "extract",
            "parse",
            "evaluate",
            "measure",
        },
    }

    # Plausible cross-family swaps (from -> list of plausible wrong families)
    PLAUSIBLE_SWAPS: Dict[str, List[str]] = {
        "file_operations": ["web_search", "database", "api_calls"],
        "web_search": ["database", "api_calls", "analysis"],
        "code_execution": ["file_operations", "api_calls"],
        "database": ["file_operations", "api_calls", "web_search"],
        "api_calls": ["database", "web_search", "code_execution"],
        "analysis": ["web_search", "database", "code_execution"],
    }

    def __init__(
        self,
        random_seed: Optional[int] = None,
        llm_client=None,
    ):
        super().__init__(random_seed, llm_client)
        self._build_tool_to_family_map()

    def _build_tool_to_family_map(self):
        """Build reverse mapping from tool to family."""
        self.tool_to_family: Dict[str, str] = {}
        for family, tools in self.TOOL_FAMILIES.items():
            for tool in tools:
                self.tool_to_family[tool.lower()] = family

    @property
    def perturbation_family(self) -> PerturbationFamily:
        return PerturbationFamily.TOOL_SELECTION

    @property
    def perturbation_type(self) -> PerturbationType:
        return PerturbationType.WRONG_TOOL_FAMILY

    def generate(
        self,
        typed_trajectory: TypedTrajectory,
    ) -> Optional[Tuple[PerturbationRecord, TypedTrajectory]]:
        """
        Swap a tool with one from a different family.

        Returns:
            (record, modified_trajectory) or None if no eligible tool found
        """
        # Find tool_call steps
        tool_steps = [
            step
            for step in typed_trajectory.steps
            if step.step_role == StepRole.TOOL_CALL.value and step.tool_name
        ]

        if not tool_steps:
            return None

        # Try to find a step with a tool we can swap
        self.random.shuffle(tool_steps)

        for step in tool_steps:
            swap_result = self._find_cross_family_swap(step.tool_name)
            if swap_result:
                original_tool, wrong_tool, original_family, wrong_family = swap_result

                # Create modified trajectory
                modified_trajectory = deepcopy(typed_trajectory)

                # Find and update the step
                for mod_step in modified_trajectory.steps:
                    if mod_step.step_index == step.step_index:
                        mod_step.tool_name = wrong_tool
                        # Update raw_text if it contains the tool name
                        if original_tool in mod_step.raw_text:
                            mod_step.raw_text = mod_step.raw_text.replace(
                                original_tool, wrong_tool
                            )
                        break

                # Create perturbation record
                record = PerturbationRecord.create(
                    original_trajectory_id=typed_trajectory.trajectory_id,
                    perturbation_class=PerturbationClass.COARSE_GRAINED,
                    perturbation_family=self.perturbation_family,
                    perturbation_type=self.perturbation_type,
                    target_step_index=step.step_index,
                    target_slot="tool_name",
                    original_value=original_tool,
                    perturbed_value=wrong_tool,
                    mutation_method="cross_family_tool_swap",
                    expected_impact=3,  # Critical - completely wrong tool type
                    notes=f"Swapped from {original_family} to {wrong_family}",
                )

                return (record, modified_trajectory)

        return None

    def _find_cross_family_swap(
        self,
        tool_name: str,
    ) -> Optional[Tuple[str, str, str, str]]:
        """
        Find a tool from a different family to swap with.

        Args:
            tool_name: Original tool name

        Returns:
            (original_tool, wrong_tool, original_family, wrong_family) or None
        """
        # Normalize tool name for lookup
        tool_lower = tool_name.lower()

        # Find original family
        original_family = self._get_tool_family(tool_lower)
        if not original_family:
            return None

        # Get plausible wrong families
        wrong_families = self.PLAUSIBLE_SWAPS.get(original_family, [])
        if not wrong_families:
            return None

        # Choose a wrong family and tool
        self.random.shuffle(wrong_families)

        for wrong_family in wrong_families:
            wrong_tools = list(self.TOOL_FAMILIES.get(wrong_family, []))
            if wrong_tools:
                wrong_tool = self.random.choice(wrong_tools)
                return (tool_name, wrong_tool, original_family, wrong_family)

        return None

    def _get_tool_family(self, tool_name: str) -> Optional[str]:
        """Get the family for a tool by name or partial match."""
        # Direct lookup
        if tool_name in self.tool_to_family:
            return self.tool_to_family[tool_name]

        # Partial match
        for family, tools in self.TOOL_FAMILIES.items():
            for tool in tools:
                if tool in tool_name or tool_name in tool:
                    return family

        # Heuristic based on common patterns
        if any(x in tool_name for x in ["file", "read", "write", "edit", "view"]):
            return "file_operations"
        if any(x in tool_name for x in ["search", "web", "browse", "fetch"]):
            return "web_search"
        if any(x in tool_name for x in ["run", "exec", "bash", "shell", "python"]):
            return "code_execution"
        if any(x in tool_name for x in ["query", "sql", "database", "db"]):
            return "database"
        if any(x in tool_name for x in ["api", "http", "request", "post", "get"]):
            return "api_calls"

        return None


# =============================================================================
# Factory Function
# =============================================================================


def get_coarse_grained_generator(
    perturbation_family: PerturbationFamily,
    perturbation_type: PerturbationType,
    random_seed: Optional[int] = None,
    llm_client=None,
    prompt_template: Optional[str] = None,
    **kwargs,
) -> BaseCoarseGrainedGenerator:
    """
    Factory function to get the appropriate coarse-grained generator.

    Args:
        perturbation_family: The perturbation family (STRUCTURAL, TERMINAL_FLAG, TOOL_SELECTION)
        perturbation_type: The specific perturbation type
        random_seed: Random seed for reproducibility
        llm_client: Bedrock client for LLM-based generators
        **kwargs: Additional arguments passed to generator constructor

    Returns:
        Appropriate BaseCoarseGrainedGenerator instance

    Raises:
        ValueError: If invalid combination of family and type
    """
    # Validate family is coarse-grained
    valid_families = {
        PerturbationFamily.STRUCTURAL,
        PerturbationFamily.TERMINAL_FLAG,
        PerturbationFamily.TOOL_SELECTION,
    }

    if perturbation_family not in valid_families:
        raise ValueError(
            f"Invalid perturbation family for coarse-grained: {perturbation_family}. "
            f"Must be one of: {[f.value for f in valid_families]}"
        )

    # Map (family, type) to generator class
    # NOTE: SKIPPED_PREREQUISITE removed - changes trajectory length, trivially detectable
    generator_map = {
        (
            PerturbationFamily.STRUCTURAL,
            PerturbationType.WRONG_PLAN,
        ): WrongPlanGenerator,
        (
            PerturbationFamily.TERMINAL_FLAG,
            PerturbationType.FALSE_TERMINAL,
        ): FalseTerminalGenerator,
        (
            PerturbationFamily.TERMINAL_FLAG,
            PerturbationType.PREMATURE_TERMINATION,
        ): PrematureTerminationGenerator,
        (
            PerturbationFamily.TOOL_SELECTION,
            PerturbationType.WRONG_TOOL_FAMILY,
        ): WrongToolFamilyGenerator,
    }

    key = (perturbation_family, perturbation_type)

    if key not in generator_map:
        raise ValueError(
            f"Invalid combination: family={perturbation_family.value}, "
            f"type={perturbation_type.value}. "
            f"Valid combinations: {[(k[0].value, k[1].value) for k in generator_map.keys()]}"
        )

    generator_class = generator_map[key]
    # Pass prompt_template only to WrongPlanGenerator
    if generator_class == WrongPlanGenerator and prompt_template:
        return generator_class(
            random_seed=random_seed,
            llm_client=llm_client,
            prompt_template=prompt_template,
            **kwargs
        )
    return generator_class(random_seed=random_seed, llm_client=llm_client, **kwargs)


def get_all_coarse_grained_generators(
    random_seed: Optional[int] = None,
    llm_client=None,
) -> List[BaseCoarseGrainedGenerator]:
    """
    Get instances of all coarse-grained generators.

    Args:
        random_seed: Random seed for reproducibility
        llm_client: Bedrock client for LLM-based generators

    Returns:
        List of all coarse-grained generator instances
    """
    return [
        SkippedPrerequisiteGenerator(random_seed=random_seed, llm_client=llm_client),
        WrongPlanGenerator(random_seed=random_seed, llm_client=llm_client),
        FalseTerminalGenerator(random_seed=random_seed, llm_client=llm_client),
        PrematureTerminationGenerator(random_seed=random_seed, llm_client=llm_client),
        WrongToolFamilyGenerator(random_seed=random_seed, llm_client=llm_client),
    ]
