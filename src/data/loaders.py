"""
Dataset loaders using Hugging Face datasets.

This module provides functions to load trajectories from ToolBench and GAIA
benchmarks via Hugging Face datasets library.
"""

import os
from typing import List, Optional, Dict, Any
import random
from .schema import Trajectory, Step, StepType, GroundTruth


def load_toolbench_trajectories(
    max_trajectories: Optional[int] = None,
    min_steps: int = 1,
    max_steps: int = 100,
    filter_successful: bool = False,
    random_seed: Optional[int] = 42,
    split: str = "train",
    use_auth_token: Optional[str] = None,
) -> List[Trajectory]:
    """
    Load trajectories from ToolBench dataset via Hugging Face.

    Args:
        max_trajectories: Maximum number of trajectories to load
        min_steps: Minimum number of steps required
        max_steps: Maximum number of steps allowed
        filter_successful: Only load successful trajectories
        random_seed: Random seed for sampling
        split: Dataset split to load (train/validation/test)
        use_auth_token: HuggingFace token (or from env HUGGINGFACE_TOKEN)

    Returns:
        List of Trajectory objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not installed. "
            "Run: pip install datasets"
        )

    # Get HF token from env if not provided
    if use_auth_token is None:
        use_auth_token = os.getenv("HUGGINGFACE_TOKEN")

    try:
        # Load ToolBench dataset from HuggingFace
        # Dataset: "OpenBMB/ToolBench"
        dataset = load_dataset(
            "OpenBMB/ToolBench",
            split=split,
            token=use_auth_token,
        )
    except Exception as e:
        print(f"Warning: Could not load ToolBench from HF: {e}")
        print("Falling back to empty dataset for testing")
        return []

    trajectories = []

    # Convert HF dataset to our Trajectory schema
    for idx, item in enumerate(dataset):
        try:
            traj = _parse_toolbench_item(item, idx)
            if traj:
                # Apply filters
                if len(traj.steps) < min_steps or len(traj.steps) > max_steps:
                    continue
                if (filter_successful and
                        traj.ground_truth.task_success is False):
                    continue

                trajectories.append(traj)
        except Exception as e:
            print(f"Warning: Failed to parse trajectory {idx}: {e}")
            continue

    # Shuffle and sample
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(trajectories)

    if max_trajectories is not None:
        trajectories = trajectories[:max_trajectories]

    return trajectories


def _parse_toolbench_item(
    item: Dict[str, Any],
    idx: int,
) -> Optional[Trajectory]:
    """
    Parse a ToolBench dataset item to Trajectory.

    Expected format varies by ToolBench version, so we handle multiple formats.
    """
    try:
        # Extract task description
        task_desc = item.get("query", item.get("question", "Unknown task"))

        # Extract steps
        raw_steps = item.get("steps", item.get("trajectory", []))
        if not raw_steps and "solution" in item:
            # Parse solution field if steps not directly available
            raw_steps = _parse_solution_string(item["solution"])

        steps = []
        for i, raw_step in enumerate(raw_steps, 1):
            step_type = StepType.OTHER
            content = ""
            tool_name = None
            tool_input = None
            tool_output = None

            if isinstance(raw_step, dict):
                # Dictionary format
                content = raw_step.get("thought", raw_step.get("content", ""))
                tool_name = raw_step.get("action", raw_step.get("tool"))
                tool_input = raw_step.get("action_input", raw_step.get("input"))
                tool_output = raw_step.get(
                    "observation",
                    raw_step.get("output")
                )

                if tool_name:
                    step_type = StepType.TOOL_EXECUTION
                elif content:
                    step_type = StepType.REASONING
            else:
                # String format
                content = str(raw_step)

            step = Step(
                step_id=f"toolbench_{idx}_step_{i}",
                step_number=i,
                step_type=step_type,
                content=content,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                metadata=raw_step if isinstance(raw_step, dict) else {},
            )
            steps.append(step)

        if not steps:
            return None

        # Extract ground truth
        final_answer = item.get("answer", item.get("final_answer"))
        task_success = item.get("success", item.get("correct"))

        ground_truth = GroundTruth(
            task_description=task_desc,
            expected_answer=final_answer,
            task_success=task_success,
            domain=item.get("category", item.get("domain")),
        )

        trajectory = Trajectory(
            trajectory_id=f"toolbench_{idx}",
            benchmark="toolbench",
            steps=steps,
            ground_truth=ground_truth,
            metadata={"hf_index": idx},
        )

        return trajectory

    except Exception as e:
        print(f"Warning: Failed to parse ToolBench item {idx}: {e}")
        return None


def load_gaia_trajectories(
    max_trajectories: Optional[int] = None,
    min_steps: int = 1,
    max_steps: int = 100,
    difficulty: Optional[str] = None,
    random_seed: Optional[int] = 42,
    split: str = "validation",
    use_auth_token: Optional[str] = None,
) -> List[Trajectory]:
    """
    Load trajectories from GAIA benchmark via Hugging Face.

    Args:
        max_trajectories: Maximum number of trajectories to load
        min_steps: Minimum number of steps required
        max_steps: Maximum number of steps allowed
        difficulty: Filter by difficulty level (None = all)
        random_seed: Random seed for sampling
        split: Dataset split (validation/test)
        use_auth_token: HuggingFace token

    Returns:
        List of Trajectory objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not installed. "
            "Run: pip install datasets"
        )

    # Get HF token
    if use_auth_token is None:
        use_auth_token = os.getenv("HUGGINGFACE_TOKEN")

    try:
        # Load GAIA dataset
        # Dataset: "gaia-benchmark/GAIA"
        dataset = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_all",  # version
            split=split,
            token=use_auth_token,
        )
    except Exception as e:
        print(f"Warning: Could not load GAIA from HF: {e}")
        print("Falling back to empty dataset for testing")
        return []

    trajectories = []

    for idx, item in enumerate(dataset):
        try:
            traj = _parse_gaia_item(item, idx)
            if traj:
                # Apply filters
                if len(traj.steps) < min_steps or len(traj.steps) > max_steps:
                    continue
                if (difficulty and
                        traj.ground_truth.difficulty != difficulty):
                    continue

                trajectories.append(traj)
        except Exception as e:
            print(f"Warning: Failed to parse trajectory {idx}: {e}")
            continue

    # Shuffle and sample
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(trajectories)

    if max_trajectories is not None:
        trajectories = trajectories[:max_trajectories]

    return trajectories


def _parse_gaia_item(
    item: Dict[str, Any],
    idx: int,
) -> Optional[Trajectory]:
    """Parse GAIA dataset item to Trajectory."""
    try:
        # GAIA has question and final_answer fields
        question = item.get("Question", item.get("question"))
        final_answer = item.get("Final answer", item.get("final_answer"))
        level = item.get("Level", item.get("level"))

        # GAIA doesn't have explicit trajectory steps in the dataset
        # Create minimal trajectory structure
        steps = []

        # If there's an annotator_metadata or steps field, parse it
        if "Annotator Metadata" in item:
            metadata = item["Annotator Metadata"]
            if "Steps" in metadata:
                raw_steps = metadata["Steps"]
                for i, step_text in enumerate(raw_steps, 1):
                    step = Step(
                        step_id=f"gaia_{idx}_step_{i}",
                        step_number=i,
                        step_type=StepType.REASONING,
                        content=step_text,
                    )
                    steps.append(step)

        # If no steps, create a single reasoning step
        if not steps:
            step = Step(
                step_id=f"gaia_{idx}_step_1",
                step_number=1,
                step_type=StepType.REASONING,
                content=f"Answer the question: {question}",
            )
            steps.append(step)

        ground_truth = GroundTruth(
            task_description=question,
            expected_answer=final_answer,
            difficulty=level,
            domain="qa",
        )

        trajectory = Trajectory(
            trajectory_id=f"gaia_{idx}",
            benchmark="gaia",
            steps=steps,
            ground_truth=ground_truth,
            metadata={"hf_index": idx, "file_name": item.get("file_name")},
        )

        return trajectory

    except Exception as e:
        print(f"Warning: Failed to parse GAIA item {idx}: {e}")
        return None


def _parse_solution_string(solution: str) -> List[Dict[str, Any]]:
    """
    Parse solution string into steps (fallback for ToolBench).

    This is a simple parser for when steps aren't explicitly structured.
    """
    # Simple split by common delimiters
    lines = solution.split("\n")
    steps = []

    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            steps.append({"content": line})

    return steps if steps else [{"content": solution}]


def save_trajectories(trajectories: List[Trajectory], output_path: str) -> None:
    """
    Save trajectories to JSON file.

    Args:
        trajectories: List of Trajectory objects
        output_path: Path to output JSON file
    """
    import json

    data = [traj.to_dict() for traj in trajectories]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_trajectories_from_json(json_path: str) -> List[Trajectory]:
    """
    Load trajectories from JSON file.

    Args:
        json_path: Path to JSON file with saved trajectories

    Returns:
        List of Trajectory objects
    """
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Trajectory.from_dict(item) for item in data]
