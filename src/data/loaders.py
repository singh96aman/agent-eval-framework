"""
Dataset loaders using Hugging Face datasets.

This module provides functions to load trajectories from ToolBench and GAIA
benchmarks via Hugging Face datasets library.
"""

import os
import json
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import SamplingManifest
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


import re

# Compiled patterns for SWE-bench tool parsing
_SWEBENCH_FUNCTION_RE = re.compile(r'<function=(\w+)>')
_SWEBENCH_PARAM_RE = re.compile(r'<parameter=(\w+)>(.*?)</parameter>', re.DOTALL)


def _parse_swebench_tool_args(content: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Parse SWE-bench content to extract structured tool arguments.

    SWE-bench uses XML-like syntax:
    - <function=NAME> for the function/tool name
    - <parameter=PARAM_NAME>VALUE</parameter> for parameters

    Common tools:
    - str_replace_editor: command, path, old_str, new_str, insert_line, view_range
    - bash: command
    - submit: (terminal action)

    Args:
        content: Raw assistant message content

    Returns:
        Tuple of (tool_name, structured_args)
    """
    if not content:
        return None, None

    # Extract function name
    func_match = _SWEBENCH_FUNCTION_RE.search(content)
    if not func_match:
        return None, None

    tool_name = func_match.group(1)

    # Extract all parameters
    params = {}
    for param_match in _SWEBENCH_PARAM_RE.finditer(content):
        param_name = param_match.group(1)
        param_value = param_match.group(2).strip()
        params[param_name] = param_value

    # Build structured args based on tool type
    structured_args = {}

    if tool_name == "str_replace_editor":
        # str_replace_editor has: command, path, old_str, new_str, insert_line, view_range
        if "command" in params:
            structured_args["command"] = params["command"]
        if "path" in params:
            structured_args["path"] = params["path"]
        if "old_str" in params:
            structured_args["old_str"] = params["old_str"]
        if "new_str" in params:
            structured_args["new_str"] = params["new_str"]
        if "insert_line" in params:
            try:
                structured_args["insert_line"] = int(params["insert_line"])
            except ValueError:
                structured_args["insert_line"] = params["insert_line"]
        if "view_range" in params:
            structured_args["view_range"] = params["view_range"]

    elif tool_name == "bash":
        # bash has: command
        if "command" in params:
            structured_args["bash_command"] = params["command"]

    elif tool_name == "submit":
        # submit is a terminal action
        structured_args["submit"] = True

    else:
        # For unknown tools, store all params
        structured_args = params.copy() if params else {}

    # Always preserve the raw content for reference (truncated)
    if not structured_args:
        structured_args["raw"] = content[:1000]

    return tool_name, structured_args


def _determine_swebench_tool_name(
    content: str,
    parsed_tool: Optional[str],
    tool_input: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], str]:
    """
    Determine the canonical tool name and step type for SWE-bench content.

    Checks the command parameter to distinguish read vs write operations.
    str_replace_editor with command=view is a read operation (view_file),
    not an edit operation.

    Args:
        content: Raw content
        parsed_tool: Tool name parsed from XML-like tags
        tool_input: Parsed tool arguments (may contain 'command' field)

    Returns:
        Tuple of (tool_name, step_type)
    """
    # Check command parameter FIRST to distinguish read vs write
    command = ""
    if tool_input and isinstance(tool_input, dict):
        command = tool_input.get("command", "")

    # If we parsed a tool name, use that with command-based refinement
    if parsed_tool:
        if parsed_tool == "str_replace_editor":
            # View command = read operation, not edit
            if command == "view":
                return "view_file", "TOOL_EXECUTION"
            # str_replace/create/insert = write operations
            elif command in ("str_replace", "create", "insert"):
                return "str_replace_editor", "TOOL_EXECUTION"
            # Default to str_replace_editor if command unclear
            return "str_replace_editor", "TOOL_EXECUTION"
        elif parsed_tool == "bash":
            return "bash", "TOOL_EXECUTION"
        elif parsed_tool == "submit":
            return "submit", "TOOL_EXECUTION"
        else:
            return parsed_tool, "TOOL_EXECUTION"

    # Fallback to keyword heuristics
    content_lower = content.lower() if content else ""
    if any(kw in content_lower for kw in ["edit", "write", "create file", "modify"]):
        return "file_edit", "TOOL_EXECUTION"
    elif any(kw in content_lower for kw in ["search", "find", "grep", "locate"]):
        return "search_code", "TOOL_EXECUTION"
    elif any(kw in content_lower for kw in ["test", "run test", "pytest"]):
        return "run_tests", "TOOL_EXECUTION"
    elif any(kw in content_lower for kw in ["view", "read", "cat ", "open file"]):
        return "view_file", "TOOL_EXECUTION"

    return None, "REASONING"


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
        substitutes = matcher.find_plausible_substitutes(
            step.tool_name, max_substitutes=1
        )
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
            with open(json_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"   Loaded {len(dataset)} examples from local file")
        except Exception as e:
            print(f"   Warning: Failed to load local file: {e}")
            dataset = []

    # Fallback to HuggingFace if local file doesn't exist or failed to load
    if not dataset:
        print("   Trying HuggingFace datasets...")
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
            print("   Warning: datasets library not installed")
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
                if filter_successful and traj.ground_truth.task_success is False:
                    continue
                if (
                    require_parameters_all_positions
                    and not _has_parameters_in_all_positions(traj)
                ):
                    continue
                if require_tool_diversity and not _has_tool_diversity(traj):
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
            if (
                i + 1 < len(conversations)
                and conversations[i + 1].get("from") == "function"
            ):
                tool_output = conversations[i + 1].get("value", "")
                i += 1  # Skip function turn, we've consumed it

            # Parse tool input from JSON string to dict
            parsed_tool_input = (
                _parse_tool_input(action_input) if action_input else None
            )

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
                    "action_input": action_input,
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
            "system_prompt": system_prompt,
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
            tool_output = raw_step.get("observation", raw_step.get("output"))

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
            "datasets library not installed. " "Run: pip install datasets"
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
                if difficulty and traj.ground_truth.difficulty != difficulty:
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


def _parse_gaia_steps_string(steps_str: str) -> List[str]:
    """
    Parse GAIA Steps string into list of individual steps.

    GAIA stores steps as a numbered string like:
    "1. Go to arxiv.org...
     2. Enter search query...
     3. Filter by date..."
    """
    import re

    # Split by numbered patterns (1. 2. 3. etc)
    # Match pattern: newline or start, followed by number and period
    parts = re.split(r"(?:^|\n)\s*(\d+)\.\s*", steps_str.strip())

    steps = []
    # parts will be: ['', '1', 'step1 text', '2', 'step2 text', ...]
    i = 1
    while i < len(parts) - 1:
        # step_num = parts[i]  # Not used - we just need the text
        step_text = parts[i + 1].strip()
        if step_text:
            steps.append(step_text)
        i += 2

    # If regex didn't work, try simple newline split
    if not steps:
        for line in steps_str.split("\n"):
            line = line.strip()
            # Remove leading number and period
            line = re.sub(r"^\d+\.\s*", "", line)
            if line:
                steps.append(line)

    return steps


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

        steps = []

        # Parse Annotator Metadata Steps field
        if "Annotator Metadata" in item:
            metadata = item["Annotator Metadata"]
            if isinstance(metadata, dict) and "Steps" in metadata:
                raw_steps = metadata["Steps"]

                # Handle string format (numbered list)
                if isinstance(raw_steps, str):
                    parsed_steps = _parse_gaia_steps_string(raw_steps)
                    for i, step_text in enumerate(parsed_steps, 1):
                        # Infer step type from content
                        step_type = StepType.REASONING
                        tool_name = None
                        if any(
                            kw in step_text.lower()
                            for kw in ["search", "go to", "navigate", "click", "open"]
                        ):
                            step_type = StepType.TOOL_EXECUTION
                            if "search" in step_text.lower():
                                tool_name = "web_search"
                            elif any(
                                kw in step_text.lower()
                                for kw in ["go to", "navigate", "open"]
                            ):
                                tool_name = "web_browse"

                        step = Step(
                            step_id=f"gaia_{idx}_step_{i}",
                            step_number=i,
                            step_type=step_type,
                            content=step_text,
                            tool_name=tool_name,
                        )
                        steps.append(step)

                # Handle list format (original code path)
                elif isinstance(raw_steps, list):
                    for i, step_text in enumerate(raw_steps, 1):
                        step = Step(
                            step_id=f"gaia_{idx}_step_{i}",
                            step_number=i,
                            step_type=StepType.REASONING,
                            content=str(step_text),
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
    max_steps: int = 30,
    filter_successful: bool = True,
    random_seed: Optional[int] = 42,
    split: str = "tool",
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
        split: Dataset split ('tool', 'xml', or 'ticks')
        use_auth_token: HuggingFace token
        difficulty: Filter by difficulty (easy/medium/hard)

    Returns:
        List of Trajectory objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library not installed. Run: pip install datasets")

    # Get HF token
    if use_auth_token is None:
        use_auth_token = os.getenv("HUGGINGFACE_TOKEN")

    try:
        # Load SWE-smith trajectories dataset
        # Dataset has splits: 'tool', 'xml', 'ticks'
        dataset = load_dataset(
            "SWE-bench/SWE-smith-trajectories",
            split=split,
            token=use_auth_token,
        )
        print(
            f"   Loaded {len(dataset)} SWE-bench trajectories from HuggingFace (split={split})"
        )
    except Exception as e:
        print(f"   Warning: Could not load SWE-bench from HF: {e}")
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

    SWE-smith format:
    - messages: JSON string containing list of {role, content} messages
    - instance_id: Bug/issue identifier
    - resolved: Boolean success indicator
    - model: Model used for generation
    - traj_id: Trajectory identifier
    - patch: The final code patch
    """
    try:
        # Get instance/task ID
        instance_id = item.get("instance_id", f"swebench_{idx}")
        traj_id = item.get("traj_id", instance_id)

        # Parse messages JSON string
        messages_str = item.get("messages", "")
        if not messages_str:
            return None

        messages = (
            json.loads(messages_str) if isinstance(messages_str, str) else messages_str
        )

        # Extract task description from first user message
        task_desc = "Unknown task"
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Content may be list of content blocks
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            task_desc = block.get("text", "")[:2000]
                            break
                        elif isinstance(block, str):
                            task_desc = block[:2000]
                            break
                elif isinstance(content, str):
                    task_desc = content[:2000]
                break

        # Parse assistant messages into steps
        steps = []
        step_num = 0

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            step_num += 1
            content = msg.get("content", "")

            # Parse structured tool arguments from XML-like tags
            parsed_tool, tool_input = _parse_swebench_tool_args(content)

            # Determine canonical tool name and step type
            # Pass tool_input to check command (view vs str_replace)
            tool_name, step_type_str = _determine_swebench_tool_name(content, parsed_tool, tool_input)
            step_type = (
                StepType.TOOL_EXECUTION
                if step_type_str == "TOOL_EXECUTION"
                else StepType.REASONING
            )

            step = Step(
                step_id=f"swebench_{idx}_step_{step_num}",
                step_number=step_num,
                step_type=step_type,
                content=content[:5000] if content else "",
                tool_name=tool_name,
                tool_input=tool_input if tool_name else None,
                metadata={"role": "assistant"},
            )
            steps.append(step)

        if not steps:
            return None

        # Determine success from resolved field
        resolved = item.get("resolved")
        task_success = resolved if isinstance(resolved, bool) else None

        # Classify difficulty based on step count
        difficulty = _classify_swebench_difficulty(item, len(steps))

        ground_truth = GroundTruth(
            task_description=task_desc,
            expected_answer=item.get("patch", ""),
            task_success=task_success,
            difficulty=difficulty,
            domain="code",
        )

        trajectory = Trajectory(
            trajectory_id=f"swebench_{traj_id}",
            benchmark="swebench",
            steps=steps,
            ground_truth=ground_truth,
            metadata={
                "instance_id": instance_id,
                "traj_id": traj_id,
                "model": item.get("model", ""),
                "hf_index": idx,
            },
        )

        return trajectory

    except Exception as e:
        if idx < 5:
            print(f"   Warning: Failed to parse SWE-bench item {idx}: {e}")
        return None


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
        "finance_business": [
            "stock",
            "currency",
            "finance",
            "bank",
            "crypto",
            "exchange",
        ],
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


# =============================================================================
# QUALITY FILTERS
# =============================================================================


def compute_quality_metrics(trajectory: Trajectory) -> Dict[str, Any]:
    """
    Compute quality metrics for a trajectory.

    Returns dict with:
        - http_error_rate: Fraction of steps with HTTP errors (403/500/503)
        - thought_none_count: Number of "Thought: None" steps
        - empty_content_rate: Fraction of steps with empty/short content
        - missing_tool_input_rate: Fraction of tool steps without inputs
        - has_placeholder_answer: Whether expected_answer has placeholders
    """
    total_steps = len(trajectory.steps)
    if total_steps == 0:
        return {
            "http_error_rate": 0.0,
            "thought_none_count": 0,
            "empty_content_rate": 0.0,
            "missing_tool_input_rate": 0.0,
            "has_placeholder_answer": False,
            "has_null_outcome": True,  # Empty trajectory has no outcome
            "has_finish_step": False,
            "grounding_score": 0.0,
            "is_graceful_failure": False,
        }

    http_errors = 0
    thought_none = 0
    empty_content = 0
    tool_steps = 0
    missing_inputs = 0
    grounded_tool_steps = 0  # Steps with actual payloads

    for step in trajectory.steps:
        # Check for HTTP errors in tool output
        if step.tool_output:
            output_str = str(step.tool_output)
            if "403" in output_str or "500" in output_str or "503" in output_str:
                http_errors += 1

        # Check for "Thought: None"
        if step.content and "Thought: None" in step.content:
            thought_none += 1

        # Check for empty/short content
        if not step.content or len(step.content.strip()) < 5:
            empty_content += 1

        # Check for tool steps without inputs
        if step.tool_name:
            tool_steps += 1
            if not step.tool_input:
                missing_inputs += 1
            # Check for grounded tool steps (have actual payloads)
            # Expert feedback: GAIA steps are label-only with null payloads
            if step.tool_input or step.tool_output:
                grounded_tool_steps += 1

    # Check for placeholder answers
    answer = trajectory.ground_truth.expected_answer or ""
    has_placeholder = any(
        p in answer for p in ["Country 1", "Capital 1", "placeholder", "N/A", "TBD"]
    )

    # Check for null/empty outcomes (expert feedback: toolbench_23203 has null outcomes)
    expected_answer = trajectory.ground_truth.expected_answer
    task_success = trajectory.ground_truth.task_success
    has_null_outcome = (
        expected_answer is None
        or (isinstance(expected_answer, str) and not expected_answer.strip())
        or task_success is None
    )

    # Check for Finish step (expert feedback: toolbench_23203 has no Finish step)
    has_finish_step = any(step.tool_name == "Finish" for step in trajectory.steps)

    # Calculate grounding score (expert feedback: GAIA tool steps are label-only)
    grounding_score = grounded_tool_steps / tool_steps if tool_steps > 0 else 0.0

    # Check for graceful failure (expert feedback: toolbench_87039 has apology answer
    # but task_success=true - valid under exact-match but weak baseline)
    apology_phrases = [
        "please try again",
        "try again later",
        "unable to",
        "could not",
        "sorry",
        "apologize",
        "cannot provide",
        "no results",
        "failed to",
        "error occurred",
        "not available",
        "unavailable",
    ]
    is_graceful_failure = False
    if task_success is True and expected_answer:
        answer_lower = expected_answer.lower()
        is_graceful_failure = any(phrase in answer_lower for phrase in apology_phrases)

    return {
        "http_error_rate": http_errors / total_steps,
        "thought_none_count": thought_none,
        "empty_content_rate": empty_content / total_steps,
        "missing_tool_input_rate": (
            missing_inputs / tool_steps if tool_steps > 0 else 0.0
        ),
        "has_placeholder_answer": has_placeholder,
        "has_null_outcome": has_null_outcome,
        "has_finish_step": has_finish_step,
        "grounding_score": grounding_score,
        "is_graceful_failure": is_graceful_failure,
    }


def check_toolbench_strict(trajectory: Trajectory) -> Tuple[bool, Optional[str]]:
    """
    Expert-provided strict filter for ToolBench trajectories.

    Checks:
    1. task_success must be True
    2. Must have steps
    3. Final step must be "Finish" tool
    4. tool_input.return_type must be "give_answer"
    5. Must have non-empty final_answer
    6. Must have at least one real tool output (not Finish, not empty)
    7. Final answer must not contain bad patterns

    Returns:
        Tuple of (passes, rejection_reason)
    """
    gt = trajectory.ground_truth
    if gt.task_success is not True:
        return False, "toolbench_strict:task_success_not_true"

    steps = trajectory.steps
    if not steps:
        return False, "toolbench_strict:no_steps"

    final = steps[-1]
    if final.tool_name != "Finish":
        return False, "toolbench_strict:final_not_finish"

    ti = final.tool_input or {}
    if ti.get("return_type") != "give_answer":
        return (
            False,
            f"toolbench_strict:return_type_not_give_answer:{ti.get('return_type')}",
        )

    final_answer = (ti.get("final_answer") or "").strip()
    if not final_answer:
        return False, "toolbench_strict:empty_final_answer"

    # Need at least one observed tool result before Finish
    has_real_output = any(
        s.tool_name != "Finish" and s.tool_output not in (None, "", "{}", "[]")
        for s in steps
    )
    if not has_real_output:
        return False, "toolbench_strict:no_real_tool_output"

    # Check for bad patterns in final answer
    text = final_answer.lower()
    bad_patterns = [
        "[news articles]",
        "[image link]",
        "[link]",
        "country 1",
        "capital 1",
        "currency 1",
        "time zone 1",
        "please try again later",
        "i couldn't retrieve",
        "i could not retrieve",
        "unable to retrieve",
        "couldn't find",
        "could not find",
        "no data associated",
        "sorry! no data",
        "currently unavailable",
        "not available at the moment",
    ]
    for pattern in bad_patterns:
        if pattern in text:
            return False, f"toolbench_strict:bad_pattern:{pattern}"

    return True, None


def apply_quality_filters(
    trajectory: Trajectory,
    quality_config: Dict[str, Any],
    benchmark: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Check if trajectory passes quality filters.

    Args:
        trajectory: Trajectory to check
        quality_config: Quality filter thresholds
        benchmark: Optional benchmark name for per-benchmark overrides

    Returns:
        Tuple of (passes, rejection_reason)
    """
    if not quality_config.get("enabled", False):
        return True, None

    # Apply expert strict filter for ToolBench
    if benchmark == "toolbench":
        thresholds = quality_config.get("thresholds", {}).copy()
        overrides = quality_config.get("benchmark_overrides", {}).get("toolbench", {})
        thresholds.update(overrides)

        if thresholds.get("use_strict_filter", True):
            passes, reason = check_toolbench_strict(trajectory)
            if not passes:
                return False, reason

    metrics = compute_quality_metrics(trajectory)

    # Get base thresholds and apply benchmark-specific overrides
    thresholds = quality_config.get("thresholds", {}).copy()
    if benchmark:
        overrides = quality_config.get("benchmark_overrides", {}).get(benchmark, {})
        thresholds.update(overrides)

    # Check HTTP error rate
    max_error_rate = thresholds.get("max_http_error_rate", 0.5)
    if metrics["http_error_rate"] > max_error_rate:
        return (
            False,
            f"http_error_rate:{metrics['http_error_rate']:.2f}>{max_error_rate}",
        )

    # Check thought_none count
    max_thought_none = thresholds.get("max_thought_none", 2)
    if metrics["thought_none_count"] >= max_thought_none:
        return (
            False,
            f"thought_none_count:{metrics['thought_none_count']}>={max_thought_none}",
        )

    # Check empty content rate (especially for SWE-bench)
    max_empty_rate = thresholds.get("max_empty_content_rate", 0.2)
    if metrics["empty_content_rate"] > max_empty_rate:
        return (
            False,
            f"empty_content_rate:{metrics['empty_content_rate']:.2f}>{max_empty_rate}",
        )

    # Check missing tool input rate (primarily for GAIA browser actions)
    max_missing_input = thresholds.get("max_missing_tool_input_rate", 0.5)
    if metrics["missing_tool_input_rate"] > max_missing_input:
        return (
            False,
            f"missing_tool_input_rate:{metrics['missing_tool_input_rate']:.2f}>{max_missing_input}",
        )

    # Check placeholder answers
    if (
        thresholds.get("reject_placeholder_answers", True)
        and metrics["has_placeholder_answer"]
    ):
        return False, "placeholder_answer"

    # Check null outcomes (expert feedback: toolbench_23203 has expected_answer=null, task_success=null)
    if thresholds.get("reject_null_outcomes", True) and metrics["has_null_outcome"]:
        return False, "null_outcome"

    # Check for Finish step (expert feedback: toolbench_23203 has no Finish step)
    # Only apply to ToolBench by default
    if benchmark == "toolbench" and thresholds.get("require_finish_step", True):
        if not metrics["has_finish_step"]:
            return False, "missing_finish_step"

    # Combined filter: reject trajectories with BOTH HTTP errors AND Thought:None
    # Expert feedback: toolbench_19354 has 403, 503, AND "Thought: None" - each alone
    # might pass thresholds, but combination indicates fundamentally broken trajectory
    if thresholds.get("reject_combined_errors", True):
        has_http_errors = metrics["http_error_rate"] > 0
        has_thought_none = metrics["thought_none_count"] > 0
        if has_http_errors and has_thought_none:
            return False, "combined_errors:http_errors+thought_none"

    # Check grounding score (expert feedback: GAIA steps are label-only with null payloads)
    # Only apply to GAIA - require minimum percentage of tool steps to have actual payloads
    if benchmark == "gaia":
        min_grounding = thresholds.get("min_grounding_score", 0.3)
        if metrics["grounding_score"] < min_grounding:
            return (
                False,
                f"low_grounding:{metrics['grounding_score']:.2f}<{min_grounding}",
            )

    # Check for graceful failure (expert feedback: toolbench_87039 has apology answer
    # but task_success=true - valid under exact-match but weak baseline for Step 2)
    if (
        thresholds.get("reject_graceful_failures", True)
        and metrics["is_graceful_failure"]
    ):
        return False, "graceful_failure:apology_answer_with_success"

    return True, None


def clean_trajectory_steps(
    trajectory: Trajectory,
    cleaning_config: Dict[str, Any],
) -> Trajectory:
    """
    Clean individual steps within a trajectory.

    Expert feedback:
    - 30 ToolBench steps with "Thought: None"
    - 57 empty SWE-bench steps
    - 53 GAIA tool_execution steps with null tool names

    Args:
        trajectory: Trajectory to clean
        cleaning_config: Configuration for step cleaning

    Returns:
        Trajectory with cleaned steps (modifies in place and returns)
    """
    if not cleaning_config.get("enabled", False):
        return trajectory

    cleaned_steps = []
    removed_count = 0
    normalized_count = 0

    for step in trajectory.steps:
        # Remove empty steps (expert feedback: 57 empty SWE-bench steps)
        if cleaning_config.get("remove_empty_steps", True):
            if not step.content or len(step.content.strip()) < 3:
                removed_count += 1
                continue

        # Remove "Thought: None" steps (expert feedback: 30 ToolBench steps)
        if cleaning_config.get("remove_thought_none_steps", True):
            if step.content and step.content.strip() == "Thought: None":
                removed_count += 1
                continue

        # Normalize GAIA tool_execution with null tool_name to REASONING
        # Expert feedback: 53 GAIA tool_execution steps with null tool names
        if cleaning_config.get("normalize_null_tool_steps", True):
            if step.step_type == StepType.TOOL_EXECUTION and not step.tool_name:
                step.step_type = StepType.REASONING
                normalized_count += 1

        cleaned_steps.append(step)

    # Renumber steps
    for i, step in enumerate(cleaned_steps, 1):
        step.step_number = i
        step.step_id = f"step_{i}"

    trajectory.steps = cleaned_steps

    # Store cleaning info in metadata
    if removed_count > 0 or normalized_count > 0:
        trajectory.metadata["step_cleaning"] = {
            "removed": removed_count,
            "normalized": normalized_count,
        }

    return trajectory


def deduplicate_trajectories(
    trajectories: List[Trajectory],
    manifest: "SamplingManifest" = None,
) -> List[Trajectory]:
    """
    Remove duplicate trajectories based on task description.

    Expert feedback: SWE-bench has duplicate task descriptions for some PRs,
    which can bias perturbation analysis or human-annotation sampling.

    Args:
        trajectories: List of trajectories to deduplicate
        manifest: Optional manifest to log rejections

    Returns:
        List of unique trajectories (first occurrence kept)
    """
    seen_tasks = {}  # task_description -> trajectory_id
    unique_trajectories = []

    for traj in trajectories:
        task_desc = traj.ground_truth.task_description
        # Normalize whitespace for comparison
        normalized_task = " ".join(task_desc.split()) if task_desc else ""

        if normalized_task in seen_tasks:
            # Duplicate found
            if manifest:
                manifest.add_rejection(
                    traj.trajectory_id,
                    "duplicate_task",
                    {
                        "duplicate_of": seen_tasks[normalized_task],
                        "task_preview": normalized_task[:100],
                    },
                )
        else:
            seen_tasks[normalized_task] = traj.trajectory_id
            unique_trajectories.append(traj)

    return unique_trajectories


# =============================================================================
# STRATIFIED SAMPLING WITH PROVENANCE
# =============================================================================


def load_stratified_sample(
    config: Dict[str, Any],
    experiment_id: str,
) -> Tuple[List[Trajectory], "SamplingManifest"]:
    """
    Load trajectories with stratified sampling and full provenance tracking.

    This is the main entry point for loading trajectories in the load phase.
    It supports config-driven sampling with reproducibility guarantees.

    Args:
        config: Load phase configuration with datasets, sampling, validation
        experiment_id: Experiment ID for provenance tracking

    Returns:
        Tuple of (trajectories, manifest) where manifest contains full provenance

    Config structure:
        {
            "datasets": {
                "toolbench": {"enabled": true, "source": "json"|"huggingface", ...},
                "gaia": {...},
                "swebench": {...}
            },
            "sampling": {
                "enabled": true,
                "seed": 42,
                "targets": {"toolbench": 400, "gaia": 100, "swebench": 100},
                "stratify_by": {"complexity": {...}, "domain": {...}}
            },
            "validation": {
                "step_requirements": {"min_steps": 2, "max_steps": 30},
                "coverage_requirements": {"min_per_benchmark": 50}
            },
            "provenance": {"enabled": true, "seed": 42}
        }
    """
    from datetime import datetime, timezone
    from collections import defaultdict
    from .schema import SamplingManifest

    # Extract config sections
    datasets_config = config.get("datasets", {})
    sampling_config = config.get("sampling", {})
    validation_config = config.get("validation", {})
    provenance_config = config.get("provenance", {})
    quality_config = config.get("quality_filters", {})

    # Get master seed (check both sampling and provenance for backward compat)
    seed = provenance_config.get("seed", sampling_config.get("seed", 42))
    random.seed(seed)

    timestamp = datetime.now(timezone.utc).isoformat()

    # Initialize manifest
    manifest = SamplingManifest(
        experiment_id=experiment_id,
        created_at=timestamp,
        seed=seed,
        config=config,
    )

    # Load from each dataset
    all_trajectories = []
    counts_by_benchmark = defaultdict(int)

    for dataset_name, ds_config in datasets_config.items():
        if not ds_config.get("enabled", True):
            print(f"   Skipping {dataset_name} (disabled)")
            continue

        trajs = _load_dataset_with_filters(
            dataset_name=dataset_name,
            ds_config=ds_config,
            validation_config=validation_config,
            quality_config=quality_config,
            manifest=manifest,
            timestamp=timestamp,
            seed=seed,
        )

        # Add domain and complexity classification
        for traj in trajs:
            traj.domain = classify_trajectory_domain(traj)
            traj.complexity = classify_trajectory_complexity(traj)

        all_trajectories.extend(trajs)
        counts_by_benchmark[dataset_name] = len(trajs)
        print(f"   {dataset_name}: loaded {len(trajs)} trajectories")

    # Apply deduplication if enabled (expert feedback: SWE-bench has duplicate tasks)
    dedup_config = config.get("deduplication", {})
    if dedup_config.get("enabled", False):
        before_count = len(all_trajectories)
        # Can dedupe specific benchmarks or all
        benchmarks_to_dedupe = dedup_config.get("benchmarks", ["swebench"])

        # Separate trajectories by whether they need deduplication
        to_dedupe = [t for t in all_trajectories if t.benchmark in benchmarks_to_dedupe]
        keep_as_is = [
            t for t in all_trajectories if t.benchmark not in benchmarks_to_dedupe
        ]

        # Deduplicate
        deduped = deduplicate_trajectories(to_dedupe, manifest)

        all_trajectories = keep_as_is + deduped
        after_count = len(all_trajectories)
        print(
            f"   Deduplication: {before_count} -> {after_count} ({before_count - after_count} duplicates removed)"
        )

    # Apply stratified sampling if enabled
    if sampling_config.get("enabled", True):
        targets = sampling_config.get("targets", {})
        all_trajectories = _apply_stratified_sampling(
            trajectories=all_trajectories,
            targets=targets,
            sampling_config=sampling_config,
            manifest=manifest,
            seed=seed,
        )

    # Apply step-level cleaning if enabled (expert feedback: empty steps, Thought:None, null tools)
    cleaning_config = config.get("step_cleaning", {})
    if cleaning_config.get("enabled", False):
        total_removed = 0
        total_normalized = 0
        for traj in all_trajectories:
            clean_trajectory_steps(traj, cleaning_config)
            if "step_cleaning" in traj.metadata:
                total_removed += traj.metadata["step_cleaning"].get("removed", 0)
                total_normalized += traj.metadata["step_cleaning"].get("normalized", 0)
        print(
            f"   Step cleaning: {total_removed} steps removed, {total_normalized} steps normalized"
        )

    # Update manifest counts
    manifest.counts = {
        "total": len(all_trajectories),
        "by_benchmark": dict(defaultdict(int)),
        "by_domain": dict(defaultdict(int)),
        "by_complexity": dict(defaultdict(int)),
    }

    for traj in all_trajectories:
        manifest.counts["by_benchmark"][traj.benchmark] = (
            manifest.counts["by_benchmark"].get(traj.benchmark, 0) + 1
        )
        manifest.counts["by_domain"][traj.domain] = (
            manifest.counts["by_domain"].get(traj.domain, 0) + 1
        )
        manifest.counts["by_complexity"][traj.complexity] = (
            manifest.counts["by_complexity"].get(traj.complexity, 0) + 1
        )

    manifest.trajectory_ids = [t.trajectory_id for t in all_trajectories]

    # Validate coverage if enabled
    if validation_config.get("enabled", True):
        _validate_coverage(all_trajectories, validation_config, manifest)

    print(f"\n   Total: {len(all_trajectories)} trajectories")
    print(f"   Rejections: {len(manifest.rejection_log)}")

    return all_trajectories, manifest


def _load_dataset_with_filters(
    dataset_name: str,
    ds_config: Dict[str, Any],
    validation_config: Dict[str, Any],
    quality_config: Dict[str, Any],
    manifest: "SamplingManifest",
    timestamp: str,
    seed: int,
) -> List[Trajectory]:
    """
    Load a single dataset with config-driven filters.

    Applies step count filters, quality filters, and adds provenance to each trajectory.
    """
    from .schema import SamplingProvenance

    source = ds_config.get("source", "huggingface")
    limit = ds_config.get("limit")
    filters = ds_config.get("filters", {})

    # Get step requirements (merge dataset-specific with global)
    step_req = validation_config.get("step_requirements", {})
    min_steps = filters.get("min_steps", step_req.get("min_steps", 2))
    max_steps = filters.get("max_steps", step_req.get("max_steps", 100))

    trajectories = []

    # Load based on source type
    if source == "json":
        path = ds_config.get("path")
        if not path:
            print(f"   WARNING: No path specified for {dataset_name} JSON source")
            return []
        raw_trajs = load_trajectories_from_json(path)
    elif dataset_name == "toolbench":
        raw_trajs = load_toolbench_trajectories(
            max_trajectories=None,  # Load all, filter later
            min_steps=min_steps,
            max_steps=max_steps,
            random_seed=seed,
        )
    elif dataset_name == "gaia":
        raw_trajs = load_gaia_trajectories(
            max_trajectories=None,
            min_steps=min_steps,
            max_steps=max_steps,
            random_seed=seed,
        )
    elif dataset_name == "swebench":
        raw_trajs = load_swebench_trajectories(
            max_trajectories=None,
            min_steps=min_steps,
            max_steps=max_steps,
            random_seed=seed,
        )
    else:
        print(f"   WARNING: Unknown dataset: {dataset_name}")
        return []

    # Apply additional filters and add provenance
    filter_criteria = {
        "min_steps": min_steps,
        "max_steps": max_steps,
        **filters,
    }

    for idx, traj in enumerate(raw_trajs):
        # Check step count (may have already been filtered by loader)
        n_steps = len(traj.steps)
        if n_steps < min_steps:
            manifest.add_rejection(
                traj.trajectory_id,
                "step_count_too_low",
                {"steps": n_steps, "min_required": min_steps},
            )
            continue
        if n_steps > max_steps:
            manifest.add_rejection(
                traj.trajectory_id,
                "step_count_too_high",
                {"steps": n_steps, "max_allowed": max_steps},
            )
            continue

        # Apply quality filters (with benchmark-specific overrides)
        passes_quality, rejection_reason = apply_quality_filters(
            traj, quality_config, dataset_name
        )
        if not passes_quality:
            manifest.add_rejection(
                traj.trajectory_id,
                "quality_filter",
                {"reason": rejection_reason, "benchmark": dataset_name},
            )
            continue

        # Add provenance
        traj.provenance = SamplingProvenance(
            sampled_at=timestamp,
            sampling_seed=seed,
            source_dataset=f"{dataset_name}_{source}",
            source_index=traj.metadata.get("hf_index", idx),
            filter_criteria=filter_criteria,
        )

        trajectories.append(traj)

        # Check limit
        if limit and len(trajectories) >= limit:
            break

    return trajectories


def _apply_stratified_sampling(
    trajectories: List[Trajectory],
    targets: Dict[str, int],
    sampling_config: Dict[str, Any],
    manifest: "SamplingManifest",
    seed: int,
) -> List[Trajectory]:
    """
    Apply stratified sampling to balance by benchmark and complexity.

    Strategy:
    1. Group by benchmark
    2. Sample to hit benchmark target counts
    3. Apply GLOBAL complexity rebalancing to hit target distribution
    """
    from collections import defaultdict

    random.seed(seed)

    # Group by benchmark
    by_benchmark = defaultdict(list)
    for traj in trajectories:
        by_benchmark[traj.benchmark].append(traj)

    # Get domain config for pre-filtering during over-sampling
    domain_config = sampling_config.get("stratify_by", {}).get("domain", {})
    domain_caps = (
        domain_config.get("caps", {}) if domain_config.get("enabled", False) else {}
    )

    # Step 1: Sample from each benchmark with domain-aware selection
    per_benchmark_samples = {}

    for benchmark, target_count in targets.items():
        pool = by_benchmark.get(benchmark, [])
        if not pool:
            print(f"   WARNING: No trajectories for {benchmark}")
            continue

        bench_caps = domain_caps.get(benchmark, {})

        if bench_caps:
            # Apply domain caps during sampling to ensure diversity
            from collections import defaultdict

            by_domain = defaultdict(list)
            for t in pool:
                by_domain[t.domain or "unknown"].append(t)

            # Shuffle each domain pool
            for domain in by_domain:
                random.shuffle(by_domain[domain])

            # Calculate how many to take from each domain
            # Capped domains: take up to cap * target * 1.5 (over-sample for complexity)
            # Uncapped domains: take all available
            selected = []

            for domain, domain_pool in by_domain.items():
                cap = bench_caps.get(domain)
                if cap is not None and not domain.startswith("_"):
                    # Capped domain - over-sample but respect cap
                    max_take = int(target_count * cap * 1.5)
                    selected.extend(domain_pool[:max_take])
                else:
                    # Uncapped - take up to 2x fair share for this domain
                    fair_share = target_count // len(by_domain)
                    max_take = min(len(domain_pool), fair_share * 2)
                    selected.extend(domain_pool[:max_take])

            random.shuffle(selected)
            per_benchmark_samples[benchmark] = selected
        else:
            # No domain caps - simple over-sampling
            oversample_count = min(len(pool), int(target_count * 1.5))
            random.shuffle(pool)
            per_benchmark_samples[benchmark] = pool[:oversample_count]

        if len(pool) < target_count:
            manifest.add_rejection(
                f"{benchmark}_target_shortfall",
                "insufficient_trajectories",
                {"target": target_count, "actual": len(pool)},
            )

    # Step 2: Global complexity rebalancing
    stratify_config = sampling_config.get("stratify_by", {}).get("complexity", {})
    if stratify_config.get("enabled", True):
        result = _global_complexity_rebalance(
            per_benchmark_samples,
            targets,
            stratify_config,
            seed,
        )
    else:
        # No rebalancing - just take target count from each benchmark
        result = []
        for benchmark, target_count in targets.items():
            samples = per_benchmark_samples.get(benchmark, [])
            result.extend(samples[:target_count])

    # Step 3: Apply domain diversity caps
    domain_config = sampling_config.get("stratify_by", {}).get("domain", {})
    if domain_config.get("enabled", True):
        result = _apply_domain_diversity(result, domain_config, seed, targets)

    return result


def _global_complexity_rebalance(
    per_benchmark_samples: Dict[str, List[Trajectory]],
    targets: Dict[str, int],
    stratify_config: Dict[str, Any],
    seed: int,
) -> List[Trajectory]:
    """
    Rebalance samples globally to hit target complexity distribution.

    Strategy: For each benchmark, sample according to the target complexity
    distribution. This ensures both benchmark counts AND global complexity
    distribution are respected.
    """
    from collections import defaultdict

    random.seed(seed)

    # Get target distribution
    distribution = stratify_config.get(
        "distribution",
        {
            "simple": 0.20,
            "medium": 0.50,
            "complex": 0.30,
        },
    )

    result = []

    for benchmark, bench_target in targets.items():
        samples = per_benchmark_samples.get(benchmark, [])
        if not samples:
            continue

        # Group by complexity
        by_complexity = defaultdict(list)
        for traj in samples:
            comp = traj.complexity or "medium"
            by_complexity[comp].append(traj)

        # Shuffle each pool
        for comp in by_complexity:
            random.shuffle(by_complexity[comp])

        # Calculate per-benchmark complexity targets
        bench_complexity_targets = {
            comp: int(bench_target * pct) for comp, pct in distribution.items()
        }

        # Adjust for rounding
        diff = bench_target - sum(bench_complexity_targets.values())
        if diff > 0:
            bench_complexity_targets["medium"] += diff

        # Sample from each complexity bucket
        bench_result = []
        used_ids = set()

        for comp, comp_target in bench_complexity_targets.items():
            pool = by_complexity.get(comp, [])
            taken = 0
            for traj in pool:
                if taken >= comp_target:
                    break
                if traj.trajectory_id not in used_ids:
                    bench_result.append(traj)
                    used_ids.add(traj.trajectory_id)
                    taken += 1

        # Fill shortfall from any complexity (prioritize medium)
        shortfall = bench_target - len(bench_result)
        if shortfall > 0:
            for comp in ["medium", "complex", "simple"]:
                pool = by_complexity.get(comp, [])
                for traj in pool:
                    if shortfall <= 0:
                        break
                    if traj.trajectory_id not in used_ids:
                        bench_result.append(traj)
                        used_ids.add(traj.trajectory_id)
                        shortfall -= 1

        result.extend(bench_result)

    return result


def _apply_domain_diversity(
    trajectories: List[Trajectory],
    domain_config: Dict[str, Any],
    seed: int,
    targets: Optional[Dict[str, int]] = None,
) -> List[Trajectory]:
    """
    Apply domain diversity constraints by capping over-represented domains.

    Strategy:
    - For each benchmark, cap domains that exceed their max_fraction
    - Take all trajectories from uncapped domains
    - Return at least `targets[benchmark]` trajectories if available

    Args:
        trajectories: Trajectories to filter (already balanced by complexity)
        domain_config: Config with domain caps per benchmark
        seed: Random seed
        targets: Optional target counts per benchmark

    Example config:
        {
            "enabled": true,
            "caps": {
                "toolbench": {"other": 0.25},
                "gaia": {"general_qa": 0.50}
            }
        }
    """
    if not domain_config.get("enabled", False):
        return trajectories

    from collections import defaultdict

    random.seed(seed)

    caps = domain_config.get("caps", {})
    if not caps:
        return trajectories

    # Group by benchmark
    by_benchmark = defaultdict(list)
    for traj in trajectories:
        by_benchmark[traj.benchmark].append(traj)

    result = []

    for benchmark, bench_trajs in by_benchmark.items():
        bench_caps = caps.get(benchmark, {})
        if not bench_caps:
            # No caps for this benchmark
            if targets and benchmark in targets:
                result.extend(bench_trajs[: targets[benchmark]])
            else:
                result.extend(bench_trajs)
            continue

        # Get target count for this benchmark
        bench_target = (
            targets.get(benchmark, len(bench_trajs)) if targets else len(bench_trajs)
        )

        # Group by domain
        by_domain = defaultdict(list)
        for traj in bench_trajs:
            by_domain[traj.domain or "unknown"].append(traj)

        # Shuffle within each domain
        for domain in by_domain:
            random.shuffle(by_domain[domain])

        # Calculate how many we can take from each domain
        capped_samples = []
        uncapped_samples = []

        for domain, domain_trajs in by_domain.items():
            max_fraction = bench_caps.get(domain)
            if max_fraction is not None and not domain.startswith("_"):
                # This domain is capped - take up to max_fraction of TARGET
                max_count = int(bench_target * max_fraction)
                capped_samples.extend(domain_trajs[:max_count])
            else:
                # Uncapped domain - take all available
                uncapped_samples.extend(domain_trajs)

        # Combine capped + uncapped
        combined = capped_samples + uncapped_samples
        random.shuffle(combined)

        # Take up to target count
        result.extend(combined[:bench_target])

    return result


def _validate_coverage(
    trajectories: List[Trajectory],
    validation_config: Dict[str, Any],
    manifest: "SamplingManifest",
) -> bool:
    """
    Validate that sampled trajectories meet coverage requirements.

    Returns True if all requirements met, False otherwise.
    """
    from collections import defaultdict

    coverage_req = validation_config.get("coverage_requirements", {})
    min_per_benchmark = coverage_req.get("min_per_benchmark", 50)
    min_per_complexity = coverage_req.get("min_per_complexity", 30)
    # min_per_domain not yet implemented in validation

    # Count by each dimension
    by_benchmark = defaultdict(int)
    by_complexity = defaultdict(int)
    by_domain = defaultdict(int)

    for traj in trajectories:
        by_benchmark[traj.benchmark] += 1
        by_complexity[traj.complexity or "unknown"] += 1
        by_domain[traj.domain or "unknown"] += 1

    all_valid = True

    # Check benchmark coverage
    for bench, count in by_benchmark.items():
        if count < min_per_benchmark:
            print(
                f"   WARNING: {bench} has {count} trajectories (min: {min_per_benchmark})"
            )
            all_valid = False

    # Check complexity coverage
    for comp, count in by_complexity.items():
        if count < min_per_complexity:
            print(
                f"   WARNING: {comp} complexity has {count} trajectories (min: {min_per_complexity})"
            )
            all_valid = False

    return all_valid


def save_sampling_manifest(manifest: "SamplingManifest", output_path: str) -> None:
    """
    Save sampling manifest to JSON file.

    Args:
        manifest: SamplingManifest to save
        output_path: Path to output JSON file
    """
    import json
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"   Saved manifest to {output_path}")
