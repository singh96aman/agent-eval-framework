"""
Unified trajectory schema for agent trajectories.

This module defines a common data structure for trajectories from
different benchmarks (ToolBench, GAIA) to enable unified processing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class StepType(Enum):
    """Type of step in a trajectory."""

    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    REASONING = "reasoning"
    VALIDATION = "validation"
    FINAL_ANSWER = "final_answer"
    OTHER = "other"


@dataclass
class Step:
    """
    A single step in an agent trajectory.

    Attributes:
        step_id: Unique identifier for this step (e.g., "step_1")
        step_number: Position in trajectory (1-indexed)
        step_type: Type of step (planning, tool selection, etc.)
        content: Raw content of the step (thought, tool call, observation)
        tool_name: Name of tool used (if applicable)
        tool_input: Arguments passed to tool (if applicable)
        tool_output: Tool execution result (if applicable)
        metadata: Additional benchmark-specific metadata
    """

    step_id: str
    step_number: int
    step_type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        """Create Step from dictionary."""
        data = data.copy()
        data["step_type"] = StepType(data["step_type"])
        return cls(**data)


@dataclass
class GroundTruth:
    """
    Ground truth information for a trajectory.

    Attributes:
        task_description: Original task/question
        expected_answer: Correct answer or outcome
        task_success: Whether trajectory successfully solved the task
        success_criteria: How success is defined (exact match, contains, etc.)
        difficulty: Task difficulty level (if available)
        domain: Task domain/category
    """

    task_description: str
    expected_answer: Optional[str] = None
    task_success: Optional[bool] = None
    success_criteria: str = "exact_match"
    difficulty: Optional[str] = None
    domain: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_description": self.task_description,
            "expected_answer": self.expected_answer,
            "task_success": self.task_success,
            "success_criteria": self.success_criteria,
            "difficulty": self.difficulty,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruth":
        """Create GroundTruth from dictionary."""
        return cls(**data)


@dataclass
class Trajectory:
    """
    A complete agent trajectory (sequence of steps).

    Attributes:
        trajectory_id: Unique identifier
        benchmark: Source benchmark (toolbench, gaia)
        steps: Ordered list of steps
        ground_truth: Ground truth information
        metadata: Additional metadata (timestamps, agent config, etc.)
        domain: Classified domain category for stratified sampling
        complexity: Classified complexity level (simple/medium/complex)
    """

    trajectory_id: str
    benchmark: str
    steps: List[Step]
    ground_truth: GroundTruth
    metadata: Dict[str, Any] = field(default_factory=dict)
    domain: Optional[str] = None
    complexity: Optional[str] = None
    provenance: Optional["SamplingProvenance"] = None

    def __len__(self) -> int:
        """Return number of steps in trajectory."""
        return len(self.steps)

    def get_step_by_number(self, step_number: int) -> Optional[Step]:
        """Get step by its number (1-indexed)."""
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None

    def get_steps_by_type(self, step_type: StepType) -> List[Step]:
        """Get all steps of a specific type."""
        return [step for step in self.steps if step.step_type == step_type]

    def get_position_label(self, step_number: int) -> str:
        """
        Get position label for a step (early, middle, late).

        Args:
            step_number: Step number (1-indexed)

        Returns:
            Position label: "early" (1-2), "middle" (3-5), "late" (6+)
        """
        if step_number <= 2:
            return "early"
        elif step_number <= 5:
            return "middle"
        else:
            return "late"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "trajectory_id": self.trajectory_id,
            "benchmark": self.benchmark,
            "steps": [step.to_dict() for step in self.steps],
            "ground_truth": self.ground_truth.to_dict(),
            "metadata": self.metadata,
            "domain": self.domain,
            "complexity": self.complexity,
        }
        if self.provenance:
            result["provenance"] = self.provenance.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create Trajectory from dictionary."""
        data = data.copy()
        data["steps"] = [Step.from_dict(s) for s in data["steps"]]
        data["ground_truth"] = GroundTruth.from_dict(data["ground_truth"])
        # Handle backward compatibility for old files without domain/complexity/provenance
        data.setdefault("domain", None)
        data.setdefault("complexity", None)
        if "provenance" in data and data["provenance"]:
            data["provenance"] = SamplingProvenance.from_dict(data["provenance"])
        else:
            data.setdefault("provenance", None)
        return cls(**data)

    def get_text_representation(self) -> str:
        """
        Get human-readable text representation of trajectory.

        Returns:
            Formatted string with task and all steps
        """
        lines = [
            f"Task: {self.ground_truth.task_description}",
            f"Trajectory ID: {self.trajectory_id}",
            f"Benchmark: {self.benchmark}",
            f"Steps: {len(self.steps)}",
            "",
        ]

        for step in self.steps:
            lines.append(f"Step {step.step_number} [{step.step_type.value}]:")
            lines.append(f"  {step.content}")

            if step.tool_name:
                lines.append(f"  Tool: {step.tool_name}")
            if step.tool_input:
                lines.append(f"  Input: {step.tool_input}")
            if step.tool_output:
                lines.append(
                    f"  Output: {step.tool_output[:200]}..."
                    if len(step.tool_output) > 200
                    else f"  Output: {step.tool_output}"
                )
            lines.append("")

        if self.ground_truth.expected_answer:
            lines.append(f"Expected Answer: {self.ground_truth.expected_answer}")

        return "\n".join(lines)


@dataclass
class SamplingProvenance:
    """
    Provenance metadata for a sampled trajectory.

    Tracks how and when a trajectory was sampled for reproducibility.

    Attributes:
        sampled_at: ISO timestamp when sampling occurred
        sampling_seed: Random seed used for sampling
        source_dataset: Source dataset identifier (e.g., "toolbench_v1")
        source_index: Original index in source dataset
        filter_criteria: Filters applied during sampling
        loader_version: Version of the loader used
    """

    sampled_at: str  # ISO format timestamp
    sampling_seed: int
    source_dataset: str
    source_index: int
    filter_criteria: Dict[str, Any] = field(default_factory=dict)
    loader_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sampled_at": self.sampled_at,
            "sampling_seed": self.sampling_seed,
            "source_dataset": self.source_dataset,
            "source_index": self.source_index,
            "filter_criteria": self.filter_criteria,
            "loader_version": self.loader_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingProvenance":
        """Create SamplingProvenance from dictionary."""
        return cls(**data)


@dataclass
class SamplingManifest:
    """
    Complete manifest for a sampling run.

    Provides full provenance for an entire sampling operation,
    enabling exact reproduction of the dataset.

    Attributes:
        experiment_id: Associated experiment ID
        created_at: When manifest was created
        seed: Master random seed
        config: Full sampling configuration used
        trajectory_ids: List of sampled trajectory IDs
        counts: Counts by benchmark, domain, complexity
        rejection_log: Why trajectories were filtered out
        checksums: Data integrity checksums
    """

    experiment_id: str
    created_at: str
    seed: int
    config: Dict[str, Any]
    trajectory_ids: List[str] = field(default_factory=list)
    counts: Dict[str, Any] = field(default_factory=dict)
    rejection_log: List[Dict[str, Any]] = field(default_factory=list)
    checksums: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "seed": self.seed,
            "config": self.config,
            "trajectory_ids": self.trajectory_ids,
            "counts": self.counts,
            "rejection_log": self.rejection_log,
            "checksums": self.checksums,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingManifest":
        """Create SamplingManifest from dictionary."""
        return cls(**data)

    def add_rejection(self, trajectory_id: str, reason: str, details: Dict = None):
        """Record a rejected trajectory."""
        self.rejection_log.append(
            {
                "trajectory_id": trajectory_id,
                "reason": reason,
                "details": details or {},
            }
        )


@dataclass
class PerturbedTrajectory:
    """
    A trajectory with an injected perturbation.

    Attributes:
        original_trajectory: The baseline trajectory
        perturbed_trajectory: Trajectory with injected error
        perturbation_type: Type of error (planning, tool_selection, parameter)
        perturbation_position: Position label (early, middle, late)
        perturbed_step_number: Which step was perturbed (1-indexed)
        original_step_content: Original step before perturbation
        perturbed_step_content: Modified step after perturbation
        perturbation_metadata: Additional information about the perturbation
    """

    original_trajectory: Trajectory
    perturbed_trajectory: Trajectory
    perturbation_type: str  # "planning", "tool_selection", "parameter"
    perturbation_position: str  # "early", "middle", "late"
    perturbed_step_number: int
    original_step_content: str
    perturbed_step_content: str
    perturbation_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_trajectory": self.original_trajectory.to_dict(),
            "perturbed_trajectory": self.perturbed_trajectory.to_dict(),
            "perturbation_type": self.perturbation_type,
            "perturbation_position": self.perturbation_position,
            "perturbed_step_number": self.perturbed_step_number,
            "original_step_content": self.original_step_content,
            "perturbed_step_content": self.perturbed_step_content,
            "perturbation_metadata": self.perturbation_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerturbedTrajectory":
        """Create PerturbedTrajectory from dictionary."""
        data = data.copy()
        data["original_trajectory"] = Trajectory.from_dict(data["original_trajectory"])
        data["perturbed_trajectory"] = Trajectory.from_dict(
            data["perturbed_trajectory"]
        )
        return cls(**data)
