"""
Dataset loaders using Hugging Face datasets.

This module provides functions to load trajectories from ToolBench and GAIA
benchmarks via Hugging Face datasets library.
"""

import os
import json
from typing import List, Optional, Dict, Any
import random
from .schema import Trajectory, Step, StepType, GroundTruth


def _parse_tool_input(action_input_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse tool input string to dictionary.

    Args:
        action_input_str: JSON string or raw string from Action Input field

    Returns:
        Parsed dictionary or None if empty/invalid
    """
    if not action_input_str or not action_input_str.strip():
        return None

    # Try to parse as JSON
    try:
        parsed = json.loads(action_input_str)
        # Ensure it's a dict
        if isinstance(parsed, dict):
            return parsed
        else:
            # If it's not a dict (e.g., a string or number), wrap it
            return {"value": parsed}
    except json.JSONDecodeError:
        # If JSON parsing fails, return as raw string value
        return {"raw_input": action_input_str}


def _has_tool_diversity(trajectory: Trajectory) -> bool:
    """
    Check if trajectory has tools with available substitutes in all positions.

    This ensures the trajectory can support tool_selection perturbations
    by verifying that each position has at least one tool with plausible
    alternatives.

    Args:
        trajectory: Trajectory to check

    Returns:
        True if trajectory has sufficient tool diversity
    """
    from src.perturbations.tool_similarity import ToolSimilarityMatcher

    # Get system prompt from metadata
    system_prompt = trajectory.metadata.get("system_prompt")
    if not system_prompt:
        # No system prompt means we can't check tool diversity
        return False

    # Initialize tool matcher
    matcher = ToolSimilarityMatcher()
    matcher.index_tools(system_prompt)

    # Calculate position ranges (same as perturbation generator)
    num_steps = len(trajectory.steps)
    if num_steps == 0:
        return False

    early_end = max(2, num_steps // 3)
    early_range = range(1, early_end + 1)

    if num_steps <= 3:
        middle_range = range(2, 3)
    else:
        third = num_steps // 3
        middle_range = range(third + 1, third * 2 + 1)

    third = num_steps // 3
    late_start = max(num_steps - 1, num_steps - third, 1)
    late_range = range(late_start, num_steps + 1)

    # Check if each position has at least one tool with substitutes
    has_early = False
    has_middle = False
    has_late = False

    for step in trajectory.steps:
        # Skip steps without tools or with Finish tool
        if not step.tool_name or step.tool_name == "Finish":
            continue

        # Check if this tool has substitutes
        substitutes = matcher.find_plausible_substitutes(step.tool_name, max_substitutes=1)
        if not substitutes:
            continue

        # Check which position this step belongs to
        if step.step_number in early_range:
            has_early = True
        if step.step_number in middle_range:
            has_middle = True
        if step.step_number in late_range:
            has_late = True

    return has_early and has_middle and has_late


def _has_parameters_in_all_positions(trajectory: Trajectory) -> bool:
    """
    Check if trajectory has at least one step with non-empty parameters
    in each position (early, middle, late).

    This ensures the trajectory can support parameter perturbations
    in all positions.

    Args:
        trajectory: Trajectory to check

    Returns:
        True if trajectory has parameters in all positions
    """
    num_steps = len(trajectory.steps)
    if num_steps == 0:
        return False

    # Calculate position ranges (same logic as PerturbationGenerator._find_step_for_position)
    # Early: first 1/3, but at least steps 1-2
    early_end = max(2, num_steps // 3)
    early_range = range(1, early_end + 1)

    # Middle: middle 1/3
    if num_steps <= 3:
        middle_range = range(2, 3)
    else:
        third = num_steps // 3
        middle_range = range(third + 1, third * 2 + 1)

    # Late: last 1/3, but at least last 2 steps
    third = num_steps // 3
    late_start = max(num_steps - 1, num_steps - third, 1)
    late_range = range(late_start, num_steps + 1)

    # Check if each position has at least one step with non-empty parameters
    has_early = False
    has_middle = False
    has_late = False

    for step in trajectory.steps:
        # Skip Finish tool
        if step.tool_name == "Finish":
            continue

        # Check if step has non-empty parameters
        has_params = bool(step.tool_input and step.tool_input != {})

        if not has_params:
            continue

        # Check which position this step belongs to
        if step.step_number in early_range:
            has_early = True
        if step.step_number in middle_range:
            has_middle = True
        if step.step_number in late_range:
            has_late = True

    return has_early and has_middle and has_late


def load_toolbench_trajectories(
    max_trajectories: Optional[int] = None,
    min_steps: int = 1,
    max_steps: int = 100,
    filter_successful: bool = False,
    require_parameters_all_positions: bool = False,
    require_tool_diversity: bool = False,
    random_seed: Optional[int] = 42,
    split: str = "train",
    use_auth_token: Optional[str] = None,
    local_path: Optional[str] = None,
) -> List[Trajectory]:
    """
    Load trajectories from ToolBench dataset (local or HuggingFace).

    Args:
        max_trajectories: Maximum number of trajectories to load
        min_steps: Minimum number of steps required
        max_steps: Maximum number of steps allowed
        filter_successful: Only load successful trajectories
        require_parameters_all_positions: Only load trajectories with non-empty
            parameters in early, middle, and late positions (ensures parameter
            perturbations can be applied)
        require_tool_diversity: Only load trajectories where tools in each position
            have plausible substitutes (ensures tool_selection perturbations can be applied)
        random_seed: Random seed for sampling
        split: Dataset split to load (train/eval)
        use_auth_token: HuggingFace token (or from env HUGGINGFACE_TOKEN)
        local_path: Path to local ToolBench data directory (default: data/toolbench/data/)

    Returns:
        List of Trajectory objects
    """
    import json
    from pathlib import Path

    # Try loading from local files first
    if local_path is None:
        # Default to data/toolbench/data/ relative to project root
        project_root = Path(__file__).parent.parent.parent
        local_path = project_root / "data" / "toolbench" / "data"
    else:
        local_path = Path(local_path)

    # Determine which file to load based on split
    if split in ["eval", "validation", "test"]:
        json_file = local_path / "toolllama_G123_dfs_eval.json"
    else:  # train
        json_file = local_path / "toolllama_G123_dfs_train.json"

    dataset = []

    # Try local file first
    if json_file.exists():
        print(f"   Loading from local file: {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            print(f"   Loaded {len(dataset)} examples from local file")
        except Exception as e:
            print(f"   Warning: Failed to load local file: {e}")
            dataset = []

    # Fallback to HuggingFace if local file doesn't exist or failed to load
    if not dataset:
        print(f"   Trying HuggingFace datasets...")
        try:
            from datasets import load_dataset

            # Get HF token from env if not provided
            if use_auth_token is None:
                use_auth_token = os.getenv("HUGGINGFACE_TOKEN")

            # Try various ToolBench alternatives on HuggingFace
            hf_alternatives = [
                "Yhyu13/ToolBench_toolllama_G123_dfs",
                "tuandunghcmut/toolbench-v1",
                "Maurus/ToolBench",
            ]

            for dataset_name in hf_alternatives:
                try:
                    hf_dataset = load_dataset(
                        dataset_name,
                        split=split if split != "eval" else "train",
                        token=use_auth_token,
                    )
                    dataset = list(hf_dataset)
                    print(f"   Loaded from HuggingFace: {dataset_name}")
                    break
                except Exception:
                    continue

        except ImportError:
            print(f"   Warning: datasets library not installed")
        except Exception as e:
            print(f"   Warning: Could not load from HuggingFace: {e}")

    if not dataset:
        print("   Warning: No ToolBench data found (local or HuggingFace)")
        return []

    trajectories = []

    # Convert dataset to our Trajectory schema
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
                if (require_parameters_all_positions and
                        not _has_parameters_in_all_positions(traj)):
                    continue
                if (require_tool_diversity and
                        not _has_tool_diversity(traj)):
                    continue

                trajectories.append(traj)
        except Exception as e:
            # Only print first few errors to avoid spam
            if idx < 5:
                print(f"   Warning: Failed to parse trajectory {idx}: {e}")
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

    Supports two formats:
    1. Local ToolBench format with "conversations" array
    2. HuggingFace alternative formats with "steps" or "trajectory"
    """
    try:
        # Format 1: Local ToolBench with conversations
        if "conversations" in item:
            return _parse_toolbench_conversations(item, idx)

        # Format 2: Other formats with direct steps
        return _parse_toolbench_steps(item, idx)

    except Exception as e:
        # Only print first few errors to avoid spam
        if idx < 5:
            print(f"   Warning: Failed to parse ToolBench item {idx}: {e}")
        return None


def _parse_toolbench_conversations(
    item: Dict[str, Any],
    idx: int,
) -> Optional[Trajectory]:
    """
    Parse local ToolBench format with conversations array.

    Format:
    {
        "id": "task description",
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "user", "value": "task query"},
            {"from": "assistant", "value": "Thought:...\\nAction:...\\nAction Input:..."},
            {"from": "function", "value": "observation result"},
            ...
        ]
    }
    """
    conversations = item.get("conversations", [])
    if not conversations:
        return None

    # Extract system prompt (first message with from="system")
    system_prompt = None
    for conv in conversations:
        if conv.get("from") == "system":
            system_prompt = conv.get("value", "")
            break

    # Extract task description from "id" field or user message
    task_desc = item.get("id", "")
    for conv in conversations:
        if conv.get("from") == "user":
            task_desc = conv.get("value", task_desc)
            break

    steps = []
    step_number = 0

    # Parse conversation turns into steps
    i = 0
    while i < len(conversations):
        conv = conversations[i]
        from_type = conv.get("from")
        value = conv.get("value", "")

        if from_type == "assistant":
            # Parse Thought/Action/Action Input from assistant turn
            step_number += 1
            thought = ""
            action = ""
            action_input = ""

            # Extract Thought, Action, Action Input
            lines = value.split("\n")
            current_field = None
            for line in lines:
                line_lower = line.strip().lower()
                if line_lower.startswith("thought:"):
                    current_field = "thought"
                    thought = line.split(":", 1)[1].strip() if ":" in line else ""
                elif line_lower.startswith("action:"):
                    current_field = "action"
                    action = line.split(":", 1)[1].strip() if ":" in line else ""
                elif line_lower.startswith("action input:"):
                    current_field = "action_input"
                    action_input = line.split(":", 1)[1].strip() if ":" in line else ""
                elif current_field and line.strip():
                    # Continue previous field
                    if current_field == "thought":
                        thought += " " + line.strip()
                    elif current_field == "action":
                        action += " " + line.strip()
                    elif current_field == "action_input":
                        action_input += " " + line.strip()

            # Get function result from next turn if available
            tool_output = None
            if i + 1 < len(conversations) and conversations[i + 1].get("from") == "function":
                tool_output = conversations[i + 1].get("value", "")
                i += 1  # Skip function turn, we've consumed it

            # Parse tool input from JSON string to dict
            parsed_tool_input = _parse_tool_input(action_input) if action_input else None

            # Create step
            # Store the FULL raw value in content (Thought + Action + Action Input)
            # This allows perturbation strategies to modify Action/Action Input
            step = Step(
                step_id=f"toolbench_{idx}_step_{step_number}",
                step_number=step_number,
                step_type=StepType.TOOL_EXECUTION if action else StepType.REASONING,
                content=value,  # Full text, not just thought
                tool_name=action if action else None,
                tool_input=parsed_tool_input,
                tool_output=tool_output,
                metadata={
                    "from": from_type,
                    "raw_value": value,
                    "thought": thought,
                    "action": action,
                    "action_input": action_input
                },
            )
            steps.append(step)

        i += 1

    if not steps:
        return None

    # Extract final answer and success from last assistant message
    final_answer = None
    task_success = None
    for conv in reversed(conversations):
        if conv.get("from") == "assistant":
            val = conv.get("value", "")
            if "Finish" in val or "final answer" in val.lower():
                final_answer = val
                # Assume successful if it says "give_answer", failed if "give_up"
                if "give_answer" in val.lower():
                    task_success = True
                elif "give_up" in val.lower():
                    task_success = False
            break

    ground_truth = GroundTruth(
        task_description=task_desc,
        expected_answer=final_answer,
        task_success=task_success,
        domain="toolbench",
    )

    trajectory = Trajectory(
        trajectory_id=f"toolbench_{idx}",
        benchmark="toolbench",
        steps=steps,
        ground_truth=ground_truth,
        metadata={
            "source": "local_conversations",
            "hf_index": idx,
            "system_prompt": system_prompt
        },
    )

    return trajectory


def _parse_toolbench_steps(
    item: Dict[str, Any],
    idx: int,
) -> Optional[Trajectory]:
    """
    Parse alternative ToolBench formats with direct steps/trajectory.
    """
    # Extract task description
    task_desc = item.get("query", item.get("question", "Unknown task"))

    # Extract steps
    raw_steps = item.get("steps", item.get("trajectory", []))
    if not raw_steps and "solution" in item:
        # Parse solution field if steps not directly available
        raw_steps = _parse_solution_string(item.get("solution", ""))

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
        metadata={"source": "steps_format", "hf_index": idx},
    )

    return trajectory


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


def load_swebench_trajectories(
    max_trajectories: Optional[int] = None,
    min_steps: int = 4,
    max_steps: int = 10,
    filter_successful: bool = True,
    random_seed: Optional[int] = 42,
    split: str = "train",
    use_auth_token: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> List[Trajectory]:
    """
    Load trajectories from SWE-bench SWE-smith dataset via HuggingFace.

    SWE-bench trajectories represent code editing tasks where an agent
    fixes bugs or implements features in real-world repositories.

    Args:
        max_trajectories: Maximum number of trajectories to load
        min_steps: Minimum number of steps required
        max_steps: Maximum number of steps allowed
        filter_successful: Only load successful trajectories (passed tests)
        random_seed: Random seed for sampling
        split: Dataset split (train/test)
        use_auth_token: HuggingFace token
        difficulty: Filter by difficulty (easy/medium/hard)

    Returns:
        List of Trajectory objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not installed. Run: pip install datasets"
        )

    # Get HF token
    if use_auth_token is None:
        use_auth_token = os.getenv("HUGGINGFACE_TOKEN")

    try:
        # Load SWE-smith trajectories dataset
        # Dataset: "SWE-bench/SWE-smith-trajectories"
        dataset = load_dataset(
            "SWE-bench/SWE-smith-trajectories",
            split=split,
            token=use_auth_token,
        )
        print(f"   Loaded {len(dataset)} SWE-bench trajectories from HuggingFace")
    except Exception as e:
        print(f"   Warning: Could not load SWE-bench from HF: {e}")
        print("   Trying alternative dataset...")

        # Try alternative SWE-bench datasets
        alternatives = [
            "princeton-nlp/SWE-bench",
            "bigcode/swe-bench-lite",
        ]

        dataset = None
        for alt in alternatives:
            try:
                dataset = load_dataset(alt, split=split, token=use_auth_token)
                print(f"   Loaded from alternative: {alt}")
                break
            except Exception:
                continue

        if dataset is None:
            print("   Warning: No SWE-bench data available")
            return []

    trajectories = []

    for idx, item in enumerate(dataset):
        try:
            traj = _parse_swebench_item(item, idx)
            if traj:
                # Apply filters
                if len(traj.steps) < min_steps or len(traj.steps) > max_steps:
                    continue
                if filter_successful and traj.ground_truth.task_success is False:
                    continue
                if difficulty and traj.ground_truth.difficulty != difficulty:
                    continue

                trajectories.append(traj)
        except Exception as e:
            if idx < 5:
                print(f"   Warning: Failed to parse SWE-bench item {idx}: {e}")
            continue

    # Shuffle and sample
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(trajectories)

    if max_trajectories is not None:
        trajectories = trajectories[:max_trajectories]

    return trajectories


def _parse_swebench_item(
    item: Dict[str, Any],
    idx: int,
) -> Optional[Trajectory]:
    """
    Parse SWE-bench/SWE-smith dataset item to Trajectory.

    SWE-bench trajectories typically contain:
    - instance_id: Bug/issue identifier
    - problem_statement: Description of the bug
    - trajectory: List of agent actions (file edits, test runs, searches)
    - patch: The final code patch
    - test_result: Whether tests passed after the fix
    """
    try:
        # Extract task description
        task_desc = item.get("problem_statement",
                            item.get("issue_text",
                            item.get("description", "Unknown task")))

        # Get instance/task ID
        instance_id = item.get("instance_id",
                              item.get("task_id", f"swebench_{idx}"))

        # Extract trajectory/actions
        raw_trajectory = item.get("trajectory",
                                  item.get("actions",
                                  item.get("model_patch", [])))

        steps = []

        # Parse trajectory based on format
        if isinstance(raw_trajectory, list):
            # List of actions format
            for i, action in enumerate(raw_trajectory, 1):
                step = _parse_swebench_action(action, idx, i)
                if step:
                    steps.append(step)
        elif isinstance(raw_trajectory, str):
            # String format (e.g., model_patch) - parse as single edit step
            if raw_trajectory.strip():
                step = Step(
                    step_id=f"swebench_{idx}_step_1",
                    step_number=1,
                    step_type=StepType.TOOL_EXECUTION,
                    content=f"Apply patch:\n{raw_trajectory[:500]}...",
                    tool_name="file_edit",
                    tool_input={"patch": raw_trajectory},
                    metadata={"raw_patch": raw_trajectory}
                )
                steps.append(step)
        elif isinstance(raw_trajectory, dict):
            # Dictionary format - extract steps
            if "steps" in raw_trajectory:
                for i, step_data in enumerate(raw_trajectory["steps"], 1):
                    step = _parse_swebench_action(step_data, idx, i)
                    if step:
                        steps.append(step)
            elif "actions" in raw_trajectory:
                for i, action in enumerate(raw_trajectory["actions"], 1):
                    step = _parse_swebench_action(action, idx, i)
                    if step:
                        steps.append(step)

        # If no steps parsed, try to create from patch/model_output
        if not steps:
            patch = item.get("patch", item.get("model_patch", ""))
            if patch:
                step = Step(
                    step_id=f"swebench_{idx}_step_1",
                    step_number=1,
                    step_type=StepType.TOOL_EXECUTION,
                    content=f"Apply code fix",
                    tool_name="file_edit",
                    tool_input={"patch": str(patch)[:1000]},
                    metadata={}
                )
                steps.append(step)

        if not steps:
            return None

        # Determine success from test results
        test_result = item.get("resolved",
                               item.get("test_result",
                               item.get("passed", None)))
        task_success = None
        if test_result is not None:
            if isinstance(test_result, bool):
                task_success = test_result
            elif isinstance(test_result, str):
                task_success = test_result.lower() in ["pass", "passed", "true", "success"]

        # Classify difficulty based on heuristics
        difficulty = _classify_swebench_difficulty(item, len(steps))

        ground_truth = GroundTruth(
            task_description=task_desc[:2000] if task_desc else "Unknown task",
            expected_answer=item.get("patch", item.get("gold_patch")),
            task_success=task_success,
            difficulty=difficulty,
            domain="code",
        )

        trajectory = Trajectory(
            trajectory_id=f"swebench_{instance_id}",
            benchmark="swebench",
            steps=steps,
            ground_truth=ground_truth,
            metadata={
                "instance_id": instance_id,
                "repo": item.get("repo", ""),
                "base_commit": item.get("base_commit", ""),
                "hf_index": idx,
            },
        )

        return trajectory

    except Exception as e:
        if idx < 5:
            print(f"   Warning: Failed to parse SWE-bench item {idx}: {e}")
        return None


def _parse_swebench_action(
    action: Any,
    traj_idx: int,
    step_num: int
) -> Optional[Step]:
    """
    Parse a single SWE-bench action into a Step.

    SWE-bench actions can be:
    - file_edit: Edit a file
    - search_code: Search for code patterns
    - run_tests: Execute test suite
    - view_file: Read a file
    """
    if action is None:
        return None

    step_type = StepType.TOOL_EXECUTION
    tool_name = None
    tool_input = {}
    tool_output = None
    content = ""

    if isinstance(action, dict):
        # Extract action type
        action_type = action.get("action",
                                 action.get("type",
                                 action.get("tool", "unknown")))

        # Normalize action type to tool name
        action_type_lower = str(action_type).lower()
        if "edit" in action_type_lower or "write" in action_type_lower:
            tool_name = "file_edit"
        elif "search" in action_type_lower or "find" in action_type_lower:
            tool_name = "search_code"
        elif "test" in action_type_lower or "run" in action_type_lower:
            tool_name = "run_tests"
        elif "view" in action_type_lower or "read" in action_type_lower:
            tool_name = "view_file"
        else:
            tool_name = action_type_lower.replace(" ", "_")[:50]

        # Extract input/arguments
        tool_input = action.get("input",
                               action.get("args",
                               action.get("arguments", {})))
        if not isinstance(tool_input, dict):
            tool_input = {"value": tool_input}

        # Add file path if present
        if "file" in action:
            tool_input["file"] = action["file"]
        if "path" in action:
            tool_input["path"] = action["path"]

        # Extract output/observation
        tool_output = action.get("output",
                                action.get("observation",
                                action.get("result")))
        if tool_output and not isinstance(tool_output, str):
            tool_output = str(tool_output)[:500]

        # Extract thought/reasoning
        thought = action.get("thought", action.get("reasoning", ""))
        content = f"Thought: {thought}\nAction: {tool_name}" if thought else f"Action: {tool_name}"

    elif isinstance(action, str):
        # String action - try to parse
        content = action
        if "edit" in action.lower():
            tool_name = "file_edit"
        elif "search" in action.lower():
            tool_name = "search_code"
        elif "test" in action.lower():
            tool_name = "run_tests"
        else:
            tool_name = "unknown_action"
        tool_input = {"raw": action}
    else:
        return None

    return Step(
        step_id=f"swebench_{traj_idx}_step_{step_num}",
        step_number=step_num,
        step_type=step_type,
        content=content,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        metadata={"raw_action": action if isinstance(action, dict) else {"raw": action}},
    )


def _classify_swebench_difficulty(item: Dict[str, Any], num_steps: int) -> str:
    """
    Classify SWE-bench task difficulty based on heuristics.

    Args:
        item: Raw dataset item
        num_steps: Number of steps in trajectory

    Returns:
        Difficulty label: "easy", "medium", or "hard"
    """
    # Check if difficulty is already provided
    if "difficulty" in item:
        return item["difficulty"]

    # Heuristics based on task complexity
    patch = item.get("patch", item.get("model_patch", ""))
    patch_len = len(patch) if patch else 0

    # Count files changed (from patch diff headers)
    files_changed = patch.count("diff --git") if patch else 1

    # Classify based on complexity indicators
    if num_steps <= 4 and files_changed == 1 and patch_len < 500:
        return "easy"
    elif num_steps <= 7 and files_changed <= 2 and patch_len < 2000:
        return "medium"
    else:
        return "hard"


def classify_trajectory_domain(trajectory: Trajectory) -> str:
    """
    Classify a trajectory's domain category for stratified sampling.

    For ToolBench: Uses tool names to infer domain
    For SWE-bench: Uses repo/file info
    For GAIA: Uses task type

    Args:
        trajectory: Trajectory to classify

    Returns:
        Domain category string
    """
    benchmark = trajectory.benchmark.lower()

    if benchmark == "toolbench":
        return _classify_toolbench_domain(trajectory)
    elif benchmark == "swebench":
        return _classify_swebench_domain(trajectory)
    elif benchmark == "gaia":
        return _classify_gaia_domain(trajectory)
    else:
        return "unknown"


def _classify_toolbench_domain(trajectory: Trajectory) -> str:
    """Classify ToolBench trajectory by API domain."""
    # Collect all tool names
    tools = [s.tool_name.lower() for s in trajectory.steps if s.tool_name]
    tools_str = " ".join(tools)

    # Domain patterns
    patterns = {
        "data_information": ["weather", "forecast", "news", "wiki", "data"],
        "media_entertainment": ["movie", "music", "video", "image", "photo", "spotify"],
        "ecommerce_shopping": ["product", "shop", "coupon", "price", "amazon", "ebay"],
        "travel_logistics": ["flight", "hotel", "travel", "booking", "track", "colis"],
        "finance_business": ["stock", "currency", "finance", "bank", "crypto", "exchange"],
        "social_communication": ["social", "tweet", "message", "review", "comment"],
        "utilities_tools": ["convert", "calculate", "validate", "translate", "qr"],
        "sports_gaming": ["sport", "game", "score", "race", "team", "player"],
    }

    for domain, keywords in patterns.items():
        if any(kw in tools_str for kw in keywords):
            return domain

    return "other"


def _classify_swebench_domain(trajectory: Trajectory) -> str:
    """Classify SWE-bench trajectory by repository type."""
    repo = trajectory.metadata.get("repo", "").lower()

    if "django" in repo or "flask" in repo or "fastapi" in repo:
        return "web_framework"
    elif "numpy" in repo or "pandas" in repo or "scipy" in repo:
        return "data_science"
    elif "test" in repo or "pytest" in repo:
        return "testing"
    elif "requests" in repo or "http" in repo or "api" in repo:
        return "networking"
    else:
        return "general_python"


def _classify_gaia_domain(trajectory: Trajectory) -> str:
    """Classify GAIA trajectory by task type."""
    task = trajectory.ground_truth.task_description.lower()

    if "search" in task or "find" in task or "web" in task:
        return "web_search"
    elif "document" in task or "pdf" in task or "file" in task:
        return "document_analysis"
    elif "calculate" in task or "math" in task or "compute" in task:
        return "calculation"
    elif "compare" in task or "multiple" in task:
        return "multi_source"
    else:
        return "general_qa"


def classify_trajectory_complexity(trajectory: Trajectory) -> str:
    """
    Classify trajectory complexity based on step count.

    Args:
        trajectory: Trajectory to classify

    Returns:
        Complexity level: "simple", "medium", or "complex"
    """
    n = len(trajectory.steps)
    if n <= 4:
        return "simple"
    elif n <= 6:
        return "medium"
    else:
        return "complex"
