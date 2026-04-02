"""
Dataset loaders for ToolBench and GAIA benchmarks.

This module provides functions to load trajectories from different benchmarks
and convert them to the unified Trajectory schema.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import random
from .schema import Trajectory, Step, StepType, GroundTruth


def load_toolbench_trajectories(
    data_dir: str,
    max_trajectories: Optional[int] = None,
    min_steps: int = 5,
    max_steps: int = 10,
    filter_successful: bool = True,
    random_seed: Optional[int] = 42,
) -> List[Trajectory]:
    """
    Load trajectories from ToolBench dataset.

    ToolBench format assumptions:
    - JSON or JSONL files with trajectory data
    - Each trajectory has: task description, steps, final answer
    - Steps include: thought, action (tool name + input), observation

    Args:
        data_dir: Path to ToolBench data directory
        max_trajectories: Maximum number of trajectories to load (None = all)
        min_steps: Minimum number of steps required
        max_steps: Maximum number of steps allowed
        filter_successful: Only load successful trajectories
        random_seed: Random seed for sampling (None = no shuffling)

    Returns:
        List of Trajectory objects

    Raises:
        FileNotFoundError: If data directory doesn't exist
        ValueError: If data format is invalid
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"ToolBench data directory not found: {data_dir}")

    trajectories = []

    # Look for JSON/JSONL files in the directory
    trajectory_files = list(data_path.glob("*.json")) + list(data_path.glob("*.jsonl"))

    if not trajectory_files:
        raise ValueError(f"No JSON/JSONL files found in {data_dir}")

    # Load trajectories from files
    for file_path in trajectory_files:
        try:
            if file_path.suffix == ".jsonl":
                # JSONL format: one trajectory per line
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                traj = _parse_toolbench_trajectory(data, file_path.stem, line_num)
                                if traj:
                                    trajectories.append(traj)
                            except json.JSONDecodeError as e:
                                print(f"Warning: Skipping invalid JSON in {file_path}:{line_num} - {e}")
            else:
                # JSON format: single trajectory or list of trajectories
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        traj = _parse_toolbench_trajectory(item, file_path.stem, idx)
                        if traj:
                            trajectories.append(traj)
                else:
                    traj = _parse_toolbench_trajectory(data, file_path.stem, 0)
                    if traj:
                        trajectories.append(traj)

        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
            continue

    # Filter by criteria
    filtered = []
    for traj in trajectories:
        # Check step count
        if len(traj.steps) < min_steps or len(traj.steps) > max_steps:
            continue

        # Check success if required
        if filter_successful and traj.ground_truth.task_success is False:
            continue

        filtered.append(traj)

    # Shuffle and sample if requested
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(filtered)

    if max_trajectories is not None:
        filtered = filtered[:max_trajectories]

    return filtered


def _parse_toolbench_trajectory(
    data: Dict[str, Any],
    file_id: str,
    trajectory_idx: int,
) -> Optional[Trajectory]:
    """
    Parse a single ToolBench trajectory from JSON data.

    ToolBench format (example):
    {
        "task": "Find the population of Tokyo in 2023",
        "steps": [
            {"thought": "I need to search for this", "action": "Search", "action_input": {...}, "observation": "..."},
            ...
        ],
        "final_answer": "14.09 million",
        "success": true
    }

    Args:
        data: Trajectory data dictionary
        file_id: Source file identifier
        trajectory_idx: Index within file

    Returns:
        Trajectory object or None if parsing fails
    """
    try:
        # Extract task description
        task_desc = data.get("task", data.get("question", data.get("query", "Unknown task")))

        # Extract steps
        raw_steps = data.get("steps", data.get("trajectory", []))
        if not raw_steps:
            return None

        steps = []
        for i, raw_step in enumerate(raw_steps, 1):
            # Determine step type based on content
            step_type = StepType.OTHER
            content = ""
            tool_name = None
            tool_input = None
            tool_output = None

            # Handle different ToolBench formats
            if "thought" in raw_step:
                # Format: thought + action + observation
                content = raw_step.get("thought", "")
                step_type = StepType.REASONING

                if "action" in raw_step:
                    tool_name = raw_step.get("action")
                    tool_input = raw_step.get("action_input", {})
                    tool_output = raw_step.get("observation", "")
                    step_type = StepType.TOOL_EXECUTION

            elif "action" in raw_step:
                # Format: action only
                tool_name = raw_step.get("action")
                tool_input = raw_step.get("action_input", {})
                tool_output = raw_step.get("observation", "")
                content = f"Use {tool_name}"
                step_type = StepType.TOOL_EXECUTION

            else:
                # Generic format
                content = raw_step.get("content", str(raw_step))

            step = Step(
                step_id=f"{file_id}_{trajectory_idx}_step_{i}",
                step_number=i,
                step_type=step_type,
                content=content,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                metadata=raw_step,
            )
            steps.append(step)

        # Extract ground truth
        expected_answer = data.get("final_answer", data.get("answer", None))
        task_success = data.get("success", data.get("correct", None))

        ground_truth = GroundTruth(
            task_description=task_desc,
            expected_answer=expected_answer,
            task_success=task_success,
            success_criteria=data.get("success_criteria", "exact_match"),
            difficulty=data.get("difficulty", None),
            domain=data.get("domain", data.get("category", None)),
        )

        # Create trajectory
        trajectory_id = f"toolbench_{file_id}_{trajectory_idx}"
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            benchmark="toolbench",
            steps=steps,
            ground_truth=ground_truth,
            metadata={"source_file": file_id, "index": trajectory_idx},
        )

        return trajectory

    except Exception as e:
        print(f"Warning: Failed to parse ToolBench trajectory {file_id}_{trajectory_idx}: {e}")
        return None


def load_gaia_trajectories(
    data_dir: str,
    max_trajectories: Optional[int] = None,
    min_steps: int = 4,
    max_steps: int = 8,
    difficulty: Optional[str] = None,
    random_seed: Optional[int] = 42,
) -> List[Trajectory]:
    """
    Load trajectories from GAIA benchmark.

    GAIA format assumptions:
    - JSON or JSONL files with question-answer pairs
    - Trajectories show multi-step reasoning with tool use
    - Includes difficulty levels (Level 1, 2, 3)

    Args:
        data_dir: Path to GAIA data directory
        max_trajectories: Maximum number of trajectories to load (None = all)
        min_steps: Minimum number of steps required
        max_steps: Maximum number of steps allowed
        difficulty: Filter by difficulty level (None = all levels)
        random_seed: Random seed for sampling (None = no shuffling)

    Returns:
        List of Trajectory objects

    Raises:
        FileNotFoundError: If data directory doesn't exist
        ValueError: If data format is invalid
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"GAIA data directory not found: {data_dir}")

    trajectories = []

    # Look for JSON/JSONL files
    trajectory_files = list(data_path.glob("*.json")) + list(data_path.glob("*.jsonl"))

    if not trajectory_files:
        raise ValueError(f"No JSON/JSONL files found in {data_dir}")

    # Load trajectories
    for file_path in trajectory_files:
        try:
            if file_path.suffix == ".jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                traj = _parse_gaia_trajectory(data, file_path.stem, line_num)
                                if traj:
                                    trajectories.append(traj)
                            except json.JSONDecodeError as e:
                                print(f"Warning: Skipping invalid JSON in {file_path}:{line_num} - {e}")
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        traj = _parse_gaia_trajectory(item, file_path.stem, idx)
                        if traj:
                            trajectories.append(traj)
                else:
                    traj = _parse_gaia_trajectory(data, file_path.stem, 0)
                    if traj:
                        trajectories.append(traj)

        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
            continue

    # Filter by criteria
    filtered = []
    for traj in trajectories:
        # Check step count
        if len(traj.steps) < min_steps or len(traj.steps) > max_steps:
            continue

        # Check difficulty if specified
        if difficulty and traj.ground_truth.difficulty != difficulty:
            continue

        filtered.append(traj)

    # Shuffle and sample if requested
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(filtered)

    if max_trajectories is not None:
        filtered = filtered[:max_trajectories]

    return filtered


def _parse_gaia_trajectory(
    data: Dict[str, Any],
    file_id: str,
    trajectory_idx: int,
) -> Optional[Trajectory]:
    """
    Parse a single GAIA trajectory from JSON data.

    GAIA format (example):
    {
        "question": "What is the GDP of France in 2022?",
        "final_answer": "2.78 trillion USD",
        "level": "Level 1",
        "trajectory": [
            {"type": "search", "query": "France GDP 2022", "result": "..."},
            {"type": "extract", "content": "..."},
            ...
        ]
    }

    Args:
        data: Trajectory data dictionary
        file_id: Source file identifier
        trajectory_idx: Index within file

    Returns:
        Trajectory object or None if parsing fails
    """
    try:
        # Extract question
        question = data.get("question", data.get("Question", "Unknown question"))

        # Extract steps
        raw_steps = data.get("trajectory", data.get("steps", []))
        if not raw_steps:
            # If no explicit trajectory, create minimal steps from question/answer
            raw_steps = [{"content": "Answer the question", "type": "reasoning"}]

        steps = []
        for i, raw_step in enumerate(raw_steps, 1):
            # Determine step type
            step_type_str = raw_step.get("type", "other")
            step_type = _map_gaia_step_type(step_type_str)

            # Extract content
            content = raw_step.get("content", raw_step.get("query", raw_step.get("action", "")))

            # Extract tool info
            tool_name = raw_step.get("tool", raw_step.get("action", None))
            tool_input = raw_step.get("input", raw_step.get("query", None))
            tool_output = raw_step.get("result", raw_step.get("observation", None))

            step = Step(
                step_id=f"{file_id}_{trajectory_idx}_step_{i}",
                step_number=i,
                step_type=step_type,
                content=content if content else f"{step_type_str} step",
                tool_name=tool_name,
                tool_input={"query": tool_input} if isinstance(tool_input, str) else tool_input,
                tool_output=tool_output,
                metadata=raw_step,
            )
            steps.append(step)

        # Extract ground truth
        final_answer = data.get("final_answer", data.get("Final answer", None))
        difficulty = data.get("Level", data.get("level", None))

        ground_truth = GroundTruth(
            task_description=question,
            expected_answer=final_answer,
            task_success=None,  # May need manual annotation
            success_criteria="answer_correctness",
            difficulty=difficulty,
            domain=data.get("domain", "qa"),
        )

        # Create trajectory
        trajectory_id = f"gaia_{file_id}_{trajectory_idx}"
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            benchmark="gaia",
            steps=steps,
            ground_truth=ground_truth,
            metadata={"source_file": file_id, "index": trajectory_idx},
        )

        return trajectory

    except Exception as e:
        print(f"Warning: Failed to parse GAIA trajectory {file_id}_{trajectory_idx}: {e}")
        return None


def _map_gaia_step_type(type_str: str) -> StepType:
    """Map GAIA step type string to StepType enum."""
    type_map = {
        "search": StepType.TOOL_EXECUTION,
        "retrieve": StepType.TOOL_EXECUTION,
        "extract": StepType.TOOL_EXECUTION,
        "reasoning": StepType.REASONING,
        "plan": StepType.PLANNING,
        "answer": StepType.FINAL_ANSWER,
        "validation": StepType.VALIDATION,
    }
    return type_map.get(type_str.lower(), StepType.OTHER)


def save_trajectories(trajectories: List[Trajectory], output_path: str) -> None:
    """
    Save trajectories to JSON file.

    Args:
        trajectories: List of Trajectory objects
        output_path: Path to output JSON file
    """
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
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Trajectory.from_dict(item) for item in data]
