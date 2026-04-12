"""
TrajectoryTyper: Main orchestrator for trajectory typing.

Runs all passes to convert raw trajectories to fully typed trajectories:
- Pass 1: Core typing (step_role, terminal flags, extraction fields)
- Pass 1-2: Entity extraction, artifact tracking, dependencies
- Pass 2: Slot typing (perturbable_slots)
- Pass 2-3: Critical path scoring (derived annotations)
"""

import logging
from typing import Any, Dict, List, Optional, Set

from src.typing.schema import TypedStep, TypedTrajectory
from src.typing.step_typer import StepTyper
from src.typing.entity_extractor import EntityExtractor
from src.typing.dependency_analyzer import DependencyAnalyzer
from src.typing.artifact_tracker import ArtifactTracker
from src.typing.slot_typer import SlotTyper
from src.typing.critical_path import CriticalPathScorer

logger = logging.getLogger(__name__)


class TrajectoryTyper:
    """
    Main orchestrator for typing trajectories.

    Converts raw trajectories from the sampled dataset into fully typed
    trajectories with all enrichment fields.
    """

    # Environment type mapping by benchmark
    BENCHMARK_ENVIRONMENT = {
        "toolbench": "tool_use",
        "gaia": "qa",
        "swebench": "code_edit",
    }

    def __init__(self, use_spacy: bool = False):
        """
        Initialize the trajectory typer.

        Args:
            use_spacy: Whether to use spaCy for entity extraction (slower)
        """
        self.step_typer = StepTyper()
        self.entity_extractor = EntityExtractor(use_spacy=use_spacy)
        self.dependency_analyzer = DependencyAnalyzer()
        self.artifact_tracker = ArtifactTracker()
        self.slot_typer = SlotTyper()
        self.critical_path_scorer = CriticalPathScorer()

    def type_trajectory(self, raw_trajectory: Dict[str, Any]) -> TypedTrajectory:
        """
        Convert a raw trajectory to a typed trajectory.

        Args:
            raw_trajectory: Raw trajectory dictionary from sampled data

        Returns:
            TypedTrajectory with all enrichment fields
        """
        trajectory_id = raw_trajectory.get("trajectory_id", "unknown")
        benchmark = raw_trajectory.get("benchmark", "unknown")

        logger.debug(f"Typing trajectory {trajectory_id}")

        # Extract trajectory-level fields
        typed_traj = self._create_trajectory_skeleton(raw_trajectory)

        # Get raw steps
        raw_steps = raw_trajectory.get("steps", [])
        if not raw_steps:
            logger.warning(f"Trajectory {trajectory_id} has no steps")
            return typed_traj

        # Step 0: Segment merged steps into semantic units
        # ToolBench: split Thought/Action/Observation into separate steps
        segmented_steps = self._segment_steps(raw_steps, benchmark)

        # Pass 1: Core typing
        typed_steps = self._pass1_core_typing(segmented_steps, benchmark)

        # Pass 1-2: Entity extraction (use segmented steps, not raw)
        entity_map = self._extract_entities(segmented_steps, typed_steps)

        # Pass 1-2: Artifact tracking
        typed_steps = self.artifact_tracker.track_artifacts(typed_steps)

        # Pass 1-2: Dependency analysis
        typed_steps = self.dependency_analyzer.analyze_dependencies(
            typed_steps, entity_map
        )

        # Pass 2: Slot typing
        for step in typed_steps:
            step["perturbable_slots"] = self.slot_typer.identify_slots(step)

        # Pass 2-3: Critical path scoring
        typed_steps = self.critical_path_scorer.score_trajectory(typed_steps, benchmark)

        # Convert to TypedStep objects
        typed_traj.steps = [self._dict_to_typed_step(s) for s in typed_steps]
        typed_traj.num_steps = len(typed_traj.steps)

        return typed_traj

    def _create_trajectory_skeleton(
        self, raw_trajectory: Dict[str, Any]
    ) -> TypedTrajectory:
        """Create trajectory-level typed structure."""
        benchmark = raw_trajectory.get("benchmark", "unknown")
        ground_truth = raw_trajectory.get("ground_truth", {})

        # Determine task success/outcome
        task_success = ground_truth.get("task_success")
        baseline_outcome = 0.0
        if task_success is True:
            baseline_outcome = 1.0
        elif task_success is False:
            baseline_outcome = 0.0
        elif task_success is None:
            # Unknown - use 0.5 as neutral
            baseline_outcome = 0.5

        # Determine ground truth availability
        expected_answer = ground_truth.get("expected_answer")
        ground_truth_available = expected_answer is not None or task_success is not None

        # Determine environment type
        environment_type = self.BENCHMARK_ENVIRONMENT.get(benchmark, "tool_use")

        # Extract task_id from trajectory_id
        traj_id = raw_trajectory.get("trajectory_id", "")
        task_id = traj_id.replace(f"{benchmark}_", "")

        return TypedTrajectory(
            trajectory_id=traj_id,
            benchmark=benchmark,
            task_id=task_id,
            task_text=ground_truth.get("task_description", ""),
            expected_answer=expected_answer,
            domain=raw_trajectory.get("domain"),
            difficulty=ground_truth.get("difficulty"),
            num_steps=len(raw_trajectory.get("steps", [])),
            environment_type=environment_type,
            ground_truth_available=ground_truth_available,
            baseline_outcome=baseline_outcome,
            has_objective_verifier=benchmark in ("gaia", "swebench"),
            can_replay=benchmark == "swebench",
            can_regenerate_downstream=benchmark == "swebench",
            provenance=raw_trajectory.get("provenance"),
        )

    def _segment_steps(
        self,
        raw_steps: List[Dict[str, Any]],
        benchmark: str,
    ) -> List[Dict[str, Any]]:
        """
        Segment merged steps into semantic units.

        ToolBench steps often contain merged Thought + Action + Observation.
        This method splits them into separate steps for cleaner role assignment:
        - reasoning: Thought content
        - tool_call: Action + Action Input
        - observation: tool_output

        SWE-bench steps are already more naturally separated.

        Args:
            raw_steps: List of raw step dictionaries
            benchmark: Source benchmark name

        Returns:
            List of segmented step dictionaries
        """
        if benchmark != "toolbench":
            # For non-ToolBench, keep steps as-is but re-number them
            return raw_steps

        segmented = []
        new_step_num = 0

        for raw_step in raw_steps:
            metadata = raw_step.get("metadata", {})

            # Check if this is a ToolBench merged step
            thought = metadata.get("thought", "")
            action = metadata.get("action", "")
            action_input = metadata.get("action_input", "")
            tool_output = raw_step.get("tool_output")

            # If no metadata breakdown, keep step as-is
            if not thought and not action:
                new_step_num += 1
                step_copy = dict(raw_step)
                step_copy["step_number"] = new_step_num
                segmented.append(step_copy)
                continue

            # Split into separate semantic steps

            # 1. Reasoning step (if thought exists and is non-trivial)
            if thought and len(thought.strip()) > 10:
                new_step_num += 1
                reasoning_step = {
                    "step_id": f"{raw_step.get('step_id', '')}_reasoning",
                    "step_number": new_step_num,
                    "step_type": "REASONING",
                    "content": thought,
                    "tool_name": None,
                    "tool_input": None,
                    "tool_output": None,
                    "metadata": {
                        "original_step_id": raw_step.get("step_id"),
                        "segmented_from": "thought",
                    },
                }
                segmented.append(reasoning_step)

            # 2. Tool call step (if action exists)
            if action:
                new_step_num += 1
                tool_call_step = {
                    "step_id": f"{raw_step.get('step_id', '')}_tool_call",
                    "step_number": new_step_num,
                    "step_type": "TOOL_EXECUTION",
                    "content": f"Action: {action}\nAction Input: {action_input}",
                    "tool_name": action,
                    "tool_input": raw_step.get("tool_input"),
                    "tool_output": None,  # Observation is separate
                    "metadata": {
                        "original_step_id": raw_step.get("step_id"),
                        "segmented_from": "action",
                        "thought": thought,  # Keep context reference
                    },
                }
                segmented.append(tool_call_step)

            # 3. Observation step (if tool_output exists)
            if tool_output:
                new_step_num += 1
                observation_step = {
                    "step_id": f"{raw_step.get('step_id', '')}_observation",
                    "step_number": new_step_num,
                    "step_type": "OBSERVATION",
                    "content": tool_output[:5000] if tool_output else "",
                    "tool_name": action,  # Reference to the tool that produced this
                    "tool_input": None,
                    "tool_output": tool_output,  # Keep original output
                    "metadata": {
                        "original_step_id": raw_step.get("step_id"),
                        "segmented_from": "observation",
                        "source_tool": action,
                    },
                }
                segmented.append(observation_step)

        logger.debug(
            f"Segmented {len(raw_steps)} steps into {len(segmented)} semantic units"
        )
        return segmented

    def _pass1_core_typing(
        self,
        raw_steps: List[Dict[str, Any]],
        benchmark: str,
    ) -> List[Dict[str, Any]]:
        """
        Pass 1: Core typing of steps.

        Classifies step_role, terminal flags, and extraction fields.
        """
        typed_steps = []
        total_steps = len(raw_steps)

        for i, raw_step in enumerate(raw_steps):
            step_index = raw_step.get("step_number", i + 1)

            # Get previous steps for context
            previous_steps = raw_steps[:i]

            # Type the step
            typed = self.step_typer.type_step(
                step=raw_step,
                step_index=step_index,
                total_steps=total_steps,
                benchmark=benchmark,
                previous_steps=previous_steps,
            )

            typed_steps.append(typed)

        return typed_steps

    def _extract_entities(
        self,
        raw_steps: List[Dict[str, Any]],
        typed_steps: List[Dict[str, Any]],
    ) -> Dict[int, Set[str]]:
        """
        Extract entities for each step and return entity map.

        Also populates the entities field in typed_steps.
        """
        entity_map: Dict[int, Set[str]] = {}

        for raw_step, typed_step in zip(raw_steps, typed_steps):
            step_index = typed_step["step_index"]
            entities = self.entity_extractor.extract_entities(raw_step)
            typed_step["entities"] = entities
            entity_map[step_index] = set(entities)

        return entity_map

    def _dict_to_typed_step(self, step_dict: Dict[str, Any]) -> TypedStep:
        """Convert a typed step dictionary to TypedStep object."""
        return TypedStep.from_dict(step_dict)

    def type_trajectories(
        self,
        raw_trajectories: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None,
    ) -> List[TypedTrajectory]:
        """
        Type multiple trajectories.

        Args:
            raw_trajectories: List of raw trajectory dictionaries
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of TypedTrajectory objects
        """
        typed = []

        for i, raw_traj in enumerate(raw_trajectories):
            try:
                typed_traj = self.type_trajectory(raw_traj)
                typed.append(typed_traj)
            except Exception as e:
                traj_id = raw_traj.get("trajectory_id", f"index_{i}")
                logger.error(f"Failed to type trajectory {traj_id}: {e}")
                continue

            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(i + 1, len(raw_trajectories))

        return typed


def type_trajectory_batch(
    raw_trajectories: List[Dict[str, Any]],
    use_spacy: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience function to type a batch of trajectories.

    Args:
        raw_trajectories: List of raw trajectory dictionaries
        use_spacy: Whether to use spaCy for entity extraction

    Returns:
        List of typed trajectory dictionaries
    """
    typer = TrajectoryTyper(use_spacy=use_spacy)
    typed = typer.type_trajectories(raw_trajectories)
    return [t.to_dict() for t in typed]
