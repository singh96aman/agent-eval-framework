"""
Main perturbation generator that orchestrates all perturbation strategies.

This module provides a unified interface for generating perturbed trajectories
across all perturbation types (planning, tool_selection, parameter) and
positions (early, middle, late).
"""

from typing import Dict, List, Optional, Any
from copy import deepcopy

from src.data.schema import Trajectory, Step, PerturbedTrajectory, StepType
from src.perturbations.strategies import (
    PlanningErrorStrategy,
    ToolSelectionErrorStrategy,
    ParameterErrorStrategy,
)


class PerturbationGenerator:
    """
    Generates perturbed trajectories with realistic errors.

    Usage:
        generator = PerturbationGenerator(random_seed=42)
        perturbed = generator.generate_perturbation(
            trajectory=traj,
            perturbation_type="planning",
            position="early",
            system_prompt=system_prompt
        )
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize perturbation generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed

        # Initialize strategies
        self.strategies = {
            "planning": PlanningErrorStrategy(random_seed),
            "tool_selection": ToolSelectionErrorStrategy(random_seed),
            "parameter": ParameterErrorStrategy(random_seed),
        }

    def generate_perturbation(
        self,
        trajectory: Trajectory,
        perturbation_type: str,
        position: str,
        system_prompt: Optional[str] = None,
        mode: str = "static"
    ) -> Optional[PerturbedTrajectory]:
        """
        Generate a perturbed trajectory.

        Args:
            trajectory: Original trajectory
            perturbation_type: Type of error ("planning", "tool_selection", "parameter")
            position: Position label ("early", "middle", "late")
            system_prompt: System prompt with tool definitions (needed for tool_selection)
            mode: "static" (keep subsequent steps) or "regenerated" (future: use LLM)

        Returns:
            PerturbedTrajectory object or None if perturbation not applicable
        """
        # Validate inputs
        if perturbation_type not in self.strategies:
            raise ValueError(
                f"Unknown perturbation type: {perturbation_type}. "
                f"Must be one of: {list(self.strategies.keys())}"
            )

        if position not in ["early", "middle", "late"]:
            raise ValueError(
                f"Unknown position: {position}. "
                f"Must be one of: early, middle, late"
            )

        # Find step to perturb based on position and perturbation type
        # Parameter perturbations need steps with actual parameters
        require_params = (perturbation_type == "parameter")
        step_to_perturb = self._find_step_for_position(
            trajectory, position, require_params=require_params
        )

        if not step_to_perturb:
            print(f"   Warning: No suitable step found for position={position} in trajectory {trajectory.trajectory_id}")
            return None

        # Get strategy
        strategy = self.strategies[perturbation_type]

        # Apply perturbation to the step
        perturbed_step = strategy.perturb_step(
            step=step_to_perturb,
            trajectory=trajectory,
            system_prompt=system_prompt
        )

        # Check if perturbation was actually applied
        # (content changed OR perturbation metadata exists)
        content_changed = (perturbed_step.content != step_to_perturb.content)
        has_perturbation_metadata = "perturbation" in perturbed_step.metadata

        if not content_changed and not has_perturbation_metadata:
            print(f"   Warning: Perturbation failed to apply for {perturbation_type}/{position} in trajectory {trajectory.trajectory_id}")
            return None

        # Build perturbed trajectory
        perturbed_trajectory = self._build_perturbed_trajectory(
            original_trajectory=trajectory,
            perturbed_step=perturbed_step,
            perturbation_type=perturbation_type,
            perturbation_position=position,
            mode=mode
        )

        # Create PerturbedTrajectory object
        return PerturbedTrajectory(
            original_trajectory=trajectory,
            perturbed_trajectory=perturbed_trajectory,
            perturbation_type=perturbation_type,
            perturbation_position=position,
            perturbed_step_number=perturbed_step.step_number,
            original_step_content=step_to_perturb.content,
            perturbed_step_content=perturbed_step.content,
            perturbation_metadata={
                "mode": mode,
                "step_type": step_to_perturb.step_type.value,
                "tool_name": step_to_perturb.tool_name,
                "perturbation_details": perturbed_step.metadata.get("perturbation", {}),
            }
        )

    def _find_step_for_position(
        self,
        trajectory: Trajectory,
        position: str,
        require_params: bool = False
    ) -> Optional[Step]:
        """
        Find a suitable step to perturb based on position.

        Position mapping (relative to trajectory length):
        - early: first 1/3 of steps (at least step 1-2)
        - middle: middle 1/3 of steps (at least step 2-3)
        - late: last 1/3 of steps (at least last 2 steps)

        We prefer TOOL_EXECUTION steps since they have tools and parameters.

        Args:
            trajectory: Trajectory to search
            position: Position label
            require_params: If True, only return steps with non-empty parameters

        Returns:
            Step to perturb or None
        """
        num_steps = len(trajectory.steps)

        if num_steps == 0:
            return None

        # Calculate position ranges based on trajectory length
        if position == "early":
            # First 1/3, but at least steps 1-2
            end = max(2, num_steps // 3)
            start = 1
        elif position == "middle":
            # Middle 1/3
            if num_steps <= 3:
                # For very short trajectories, use step 2 as middle
                start, end = 2, 2
            else:
                third = num_steps // 3
                start = third + 1
                end = third * 2
        else:  # late
            # Last 1/3, but at least last 2 steps
            third = num_steps // 3
            start = max(num_steps - 1, num_steps - third, 1)
            end = num_steps

        # Find steps in range
        candidates = [
            step for step in trajectory.steps
            if start <= step.step_number <= end
        ]

        if not candidates:
            return None

        # If require_params, filter to steps with non-empty parameters
        if require_params:
            param_steps = [
                s for s in candidates
                if s.tool_input and s.tool_input != {} and s.tool_name != "Finish"
            ]
            if param_steps:
                return param_steps[0]
            # If no steps with params in range, expand search to whole trajectory
            param_steps = [
                s for s in trajectory.steps
                if s.tool_input and s.tool_input != {} and s.tool_name != "Finish"
            ]
            return param_steps[0] if param_steps else None

        # Prefer TOOL_EXECUTION steps
        tool_steps = [s for s in candidates if s.step_type == StepType.TOOL_EXECUTION]

        if tool_steps:
            return tool_steps[0]  # Take first eligible step

        # Fallback to any step
        return candidates[0]

    def _build_perturbed_trajectory(
        self,
        original_trajectory: Trajectory,
        perturbed_step: Step,
        perturbation_type: str,
        perturbation_position: str,
        mode: str
    ) -> Trajectory:
        """
        Build perturbed trajectory by replacing one step.

        For "static" mode: Keep all original steps, just replace the perturbed one.
        For "regenerated" mode (future): Would regenerate steps after perturbation.

        Args:
            original_trajectory: Original trajectory
            perturbed_step: Step with injected error
            perturbation_type: Type of perturbation (planning, tool_selection, parameter)
            perturbation_position: Position of perturbation (early, middle, late)
            mode: "static" or "regenerated"

        Returns:
            Perturbed trajectory
        """
        if mode != "static":
            raise NotImplementedError(f"Mode '{mode}' not yet implemented. Use 'static'.")

        # Deep copy original trajectory
        perturbed_traj = deepcopy(original_trajectory)

        # Replace the perturbed step
        for i, step in enumerate(perturbed_traj.steps):
            if step.step_number == perturbed_step.step_number:
                perturbed_traj.steps[i] = perturbed_step
                break

        # Mark all subsequent steps as "conditioned on error"
        for i, step in enumerate(perturbed_traj.steps):
            if step.step_number > perturbed_step.step_number:
                if "metadata" not in step.metadata:
                    step.metadata["conditioned_on_error"] = True
                step.metadata["conditioned_on_error"] = True

        # Update trajectory ID to be unique for this specific perturbation
        perturbed_traj.trajectory_id = (
            f"{original_trajectory.trajectory_id}_{perturbation_type}_{perturbation_position}_perturbed"
        )

        perturbed_traj.metadata["is_perturbed"] = True
        perturbed_traj.metadata["perturbation_mode"] = mode
        perturbed_traj.metadata["perturbation_type"] = perturbation_type
        perturbed_traj.metadata["perturbation_position"] = perturbation_position

        return perturbed_traj

    def generate_all_perturbations(
        self,
        trajectory: Trajectory,
        system_prompt: Optional[str] = None,
        mode: str = "static"
    ) -> List[PerturbedTrajectory]:
        """
        Generate all 9 perturbation conditions for a trajectory.

        Conditions: 3 types × 3 positions = 9 total

        Args:
            trajectory: Original trajectory
            system_prompt: System prompt with tool definitions
            mode: Perturbation mode

        Returns:
            List of PerturbedTrajectory objects (up to 9)
        """
        perturbations = []

        for ptype in ["planning", "tool_selection", "parameter"]:
            for position in ["early", "middle", "late"]:
                perturbed = self.generate_perturbation(
                    trajectory=trajectory,
                    perturbation_type=ptype,
                    position=position,
                    system_prompt=system_prompt,
                    mode=mode
                )

                if perturbed:
                    perturbations.append(perturbed)

        return perturbations

    def get_perturbation_id(
        self,
        trajectory_id: str,
        perturbation_type: str,
        position: str
    ) -> str:
        """
        Generate standardized perturbation ID.

        Format: {trajectory_id}_{type}_{position}

        Args:
            trajectory_id: Original trajectory ID
            perturbation_type: Type of perturbation
            position: Position label

        Returns:
            Perturbation ID
        """
        return f"{trajectory_id}_{perturbation_type}_{position}"
