"""
Core assembly logic for creating evaluation units.

This module provides functions to assemble EvaluationUnit objects from
perturbation data. It handles:
- Step identity migration (canonical_step_id, display_step_index)
- Baseline reconstruction from perturbed trajectories
- Complete EvaluationUnit assembly with all derived fields

Key Insight:
The perturbation data file contains:
- perturbed_trajectory: The full trajectory WITH the perturbation applied
- original_trajectory: Just a reference (trajectory_id, benchmark)
- perturbation_record: The perturbation metadata including original_value and perturbed_value

We reconstruct the baseline by reverting the perturbation.
"""

import copy
import json
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.evaluation.capabilities import compute_capabilities
from src.evaluation.ids import (
    generate_canonical_step_id,
    generate_trajectory_variant_id,
)
from src.utils import generate_evaluation_unit_id
from src.evaluation.schema import (
    BaselineData,
    BlindingAssignment,
    DerivedCache,
    EvaluationCapabilities,
    EvaluationUnit,
    PerturbedData,
)
from src.evaluation.tier_assignment import assign_replay_tier


def migrate_step_identity(
    trajectory: Dict[str, Any],
    source_trajectory_id: str,
    is_deletion: bool = False,
    deleted_step_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Add canonical_step_id and display_step_index to each step.

    For baseline trajectories and non-deletion perturbations:
    - canonical_step_id format: {source_trajectory_id}::step::{step_index}
    - display_step_index = step_index

    For deletion perturbations in the perturbed trajectory:
    - Maintain canonical_step_id from original step positions
    - display_step_index is renumbered sequentially (1, 2, 3, ...)

    Args:
        trajectory: The trajectory dict containing a 'steps' list.
        source_trajectory_id: The source trajectory ID for canonical IDs.
        is_deletion: Whether this is a deletion perturbation (affects display indexing).
        deleted_step_index: For perturbed trajectories with deletion, the index
            of the step that was deleted (used to preserve canonical IDs).

    Returns:
        A new trajectory dict with step identity fields added.
    """
    trajectory = copy.deepcopy(trajectory)
    steps = trajectory.get("steps", [])

    for display_idx, step in enumerate(steps, start=1):
        original_step_index = step.get("step_index", display_idx)

        # Generate canonical step ID based on the original step index
        step["canonical_step_id"] = generate_canonical_step_id(
            source_trajectory_id, original_step_index
        )

        # For deletions in perturbed trajectory, renumber display indices
        if is_deletion and deleted_step_index is not None:
            # Display index is sequential 1, 2, 3... regardless of original indices
            step["display_step_index"] = display_idx
        else:
            # For baseline or non-deletion, display matches original
            step["display_step_index"] = original_step_index

    trajectory["steps"] = steps
    return trajectory


def _get_nested_value(obj: Dict[str, Any], path: str) -> Any:
    """
    Get a value from a nested dict using dot notation.

    Args:
        obj: The dictionary to traverse.
        path: Dot-separated path (e.g., "tool_arguments.return_type").

    Returns:
        The value at the path, or None if not found.
    """
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _set_nested_value(obj: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a value in a nested dict using dot notation.

    Args:
        obj: The dictionary to modify (in place).
        path: Dot-separated path (e.g., "tool_arguments.return_type").
        value: The value to set.
    """
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def reconstruct_baseline(
    perturbed_trajectory: Dict[str, Any],
    perturbation_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create baseline by reverting the perturbation.

    This function takes the perturbed trajectory and the perturbation record,
    then reverses the perturbation to reconstruct the original baseline.

    Handles three types of perturbations:
    1. Field modifications (raw_text, tool_arguments.*, etc.):
       - Find target step and revert target_slot from perturbed_value to original_value
    2. Step deletions (target_slot="step"):
       - Re-insert the deleted step at target_step_index
    3. Step insertions (perturbed_value is a step, original_value is null):
       - Remove the inserted step

    Args:
        perturbed_trajectory: The full trajectory with perturbation applied.
        perturbation_record: The perturbation metadata including original_value,
            perturbed_value, target_step_index, and target_slot.

    Returns:
        The reconstructed baseline trajectory.
    """
    baseline = copy.deepcopy(perturbed_trajectory)
    steps = baseline.get("steps", [])

    target_step_index = perturbation_record.get("target_step_index")
    target_slot = perturbation_record.get("target_slot", "")
    original_value = perturbation_record.get("original_value")
    perturbed_value = perturbation_record.get("perturbed_value")

    # Case 1: Step deletion (target_slot="step", perturbed_value=null)
    # To revert: re-insert the original step and restore original indices
    if target_slot == "step" and perturbed_value is None and original_value is not None:
        # The perturbed trajectory may have renumbered steps after deletion.
        # We need to shift indices of steps >= target_step_index to make room.
        for step in steps:
            current_idx = step.get("step_index", 0)
            if current_idx >= target_step_index:
                step["step_index"] = current_idx + 1

        # Find insertion position based on step_index ordering (now shifted)
        inserted = False
        for i, step in enumerate(steps):
            if step.get("step_index", 0) > target_step_index:
                steps.insert(i, copy.deepcopy(original_value))
                inserted = True
                break
        if not inserted:
            # Append at the end if all existing steps have lower indices
            steps.append(copy.deepcopy(original_value))
        baseline["steps"] = steps
        # Update num_steps
        baseline["num_steps"] = len(steps)
        return baseline

    # Case 2: Step insertion (original_value=null, perturbed_value is a step)
    # To revert: remove the inserted step
    if target_slot == "step" and original_value is None and perturbed_value is not None:
        # Find and remove the step at target_step_index
        steps = [s for s in steps if s.get("step_index") != target_step_index]
        baseline["steps"] = steps
        baseline["num_steps"] = len(steps)
        return baseline

    # Case 3: Field modification - find the step and revert the field
    target_step = None
    for step in steps:
        if step.get("step_index") == target_step_index:
            target_step = step
            break

    if target_step is None:
        # Step not found, return as-is (shouldn't happen with valid data)
        return baseline

    # Revert the field value
    if target_slot == "raw_text":
        target_step["raw_text"] = original_value
    elif target_slot.startswith("tool_arguments."):
        # Handle nested tool_arguments fields
        _set_nested_value(target_step, target_slot, original_value)
    elif target_slot == "tool_name":
        target_step["tool_name"] = original_value
    elif target_slot == "observation":
        target_step["observation"] = original_value
    else:
        # Generic field handling
        _set_nested_value(target_step, target_slot, original_value)

    baseline["steps"] = steps
    return baseline


def _build_derived_cache(
    baseline: Dict[str, Any],
    perturbed: Dict[str, Any],
    perturbation_record: Dict[str, Any],
    source_trajectory_id: str,
    target_step_index: int,
) -> DerivedCache:
    """
    Build the DerivedCache from canonical objects.

    Args:
        baseline: The baseline trajectory dict.
        perturbed: The perturbed trajectory dict.
        perturbation_record: The perturbation record dict.
        source_trajectory_id: The source trajectory ID.
        target_step_index: The target step index from perturbation record.

    Returns:
        A DerivedCache instance.
    """
    return DerivedCache(
        baseline_outcome=baseline.get("baseline_outcome", 0.0),
        baseline_num_steps=len(baseline.get("steps", [])),
        perturbed_num_steps=len(perturbed.get("steps", [])),
        perturbation_class=perturbation_record.get("perturbation_class", ""),
        perturbation_family=perturbation_record.get("perturbation_family", ""),
        perturbation_type=perturbation_record.get("perturbation_type", ""),
        target_step_canonical_id=generate_canonical_step_id(
            source_trajectory_id, target_step_index
        ),
        expected_impact=perturbation_record.get("expected_impact", 0),
        expected_detectability=perturbation_record.get("expected_detectability", 0),
    )


def _generate_blinding_assignment(
    evaluation_unit_id: str,
    baseline_variant_id: str,
    perturbed_variant_id: str,
) -> BlindingAssignment:
    """
    Generate a random A/B blinding assignment.

    Randomly assigns baseline and perturbed trajectories to positions A and B
    for blinded evaluation.

    Args:
        evaluation_unit_id: The evaluation unit ID.
        baseline_variant_id: The baseline trajectory variant ID.
        perturbed_variant_id: The perturbed trajectory variant ID.

    Returns:
        A BlindingAssignment instance.
    """
    is_a_baseline = random.choice([True, False])

    if is_a_baseline:
        trajectory_a_variant_id = baseline_variant_id
        trajectory_b_variant_id = perturbed_variant_id
    else:
        trajectory_a_variant_id = perturbed_variant_id
        trajectory_b_variant_id = baseline_variant_id

    return BlindingAssignment(
        evaluation_unit_id=evaluation_unit_id,
        trajectory_a_variant_id=trajectory_a_variant_id,
        trajectory_b_variant_id=trajectory_b_variant_id,
        is_a_baseline=is_a_baseline,
        is_a_perturbed=not is_a_baseline,
    )


def assemble_evaluation_unit(
    perturbation_data: Dict[str, Any],
    experiment_id: str,
    perturbation_index: int,
) -> Dict[str, Any]:
    """
    Assemble a complete EvaluationUnit from perturbation data.

    Takes a single perturbation object from the data file and creates
    a complete EvaluationUnit with:
    - Generated IDs (evaluation_unit_id, trajectory_variant_ids)
    - Baseline and perturbed data with migrated step identity
    - Computed capabilities and replay_tier
    - Built derived_cache
    - Generated blinding assignment

    Args:
        perturbation_data: A single perturbation object from the perturbations file.
            Expected structure:
            {
                "perturbation_id": "pert_...",
                "original_trajectory_id": "toolbench_...",
                "original_trajectory": {"trajectory_id": "...", "benchmark": "..."},
                "perturbation_record": {...},
                "perturbed_trajectory": {...}
            }
        experiment_id: The experiment ID this unit belongs to.
        perturbation_index: The index of this perturbation (0-999) for ID generation.

    Returns:
        The EvaluationUnit as a dictionary (via to_dict()).
    """
    # Extract components from perturbation data
    perturbation_record = perturbation_data["perturbation_record"]
    perturbed_trajectory = perturbation_data["perturbed_trajectory"]
    original_trajectory_ref = perturbation_data["original_trajectory"]

    # Determine source trajectory ID
    source_trajectory_id = original_trajectory_ref.get(
        "trajectory_id", perturbation_data.get("original_trajectory_id", "unknown")
    )
    benchmark = perturbed_trajectory.get(
        "benchmark", original_trajectory_ref.get("benchmark", "unknown")
    )

    # Reconstruct baseline from perturbed trajectory
    baseline_trajectory = reconstruct_baseline(
        perturbed_trajectory, perturbation_record
    )

    # Determine if this is a deletion perturbation
    target_slot = perturbation_record.get("target_slot", "")
    is_deletion = (
        target_slot == "step" and perturbation_record.get("perturbed_value") is None
    )
    target_step_index = perturbation_record.get("target_step_index", 0)

    # Migrate step identity to both trajectories
    baseline_with_identity = migrate_step_identity(
        baseline_trajectory,
        source_trajectory_id,
        is_deletion=False,
        deleted_step_index=None,
    )
    perturbed_with_identity = migrate_step_identity(
        perturbed_trajectory,
        source_trajectory_id,
        is_deletion=is_deletion,
        deleted_step_index=target_step_index if is_deletion else None,
    )

    # Generate IDs
    # Get perturbation_id from record or perturbation_data
    perturbation_id = perturbation_record.get(
        "perturbation_id", perturbation_data.get("perturbation_id", f"pert_{perturbation_index:03d}")
    )
    # Use the trajectory_id from perturbation record (may have config encoding)
    trajectory_id = perturbation_record.get(
        "original_trajectory_id", source_trajectory_id
    )
    evaluation_unit_id = generate_evaluation_unit_id(trajectory_id, perturbation_id)
    baseline_variant_id = generate_trajectory_variant_id(source_trajectory_id, "base")
    perturbed_variant_id = generate_trajectory_variant_id(
        source_trajectory_id, "pert", perturbation_index
    )

    # Compute capabilities
    capabilities_dict = compute_capabilities(baseline_with_identity, benchmark)
    evaluation_capabilities = EvaluationCapabilities(
        has_objective_verifier=capabilities_dict["has_objective_verifier"],
        can_replay=capabilities_dict["can_replay"],
        can_regenerate_downstream=capabilities_dict["can_regenerate_downstream"],
        environment_accessible=capabilities_dict["environment_accessible"],
        ground_truth_available=capabilities_dict["ground_truth_available"],
    )

    # Assign replay tier
    replay_tier = assign_replay_tier(capabilities_dict)

    # Build derived cache
    derived_cache = _build_derived_cache(
        baseline_with_identity,
        perturbed_with_identity,
        perturbation_record,
        source_trajectory_id,
        target_step_index,
    )

    # Generate blinding assignment
    blinding = _generate_blinding_assignment(
        evaluation_unit_id,
        baseline_variant_id,
        perturbed_variant_id,
    )

    # Create data wrappers
    baseline_data = BaselineData(
        trajectory_variant_id=baseline_variant_id,
        trajectory=baseline_with_identity,
    )

    perturbed_data = PerturbedData(
        trajectory_variant_id=perturbed_variant_id,
        trajectory=perturbed_with_identity,
        perturbation_record=perturbation_record,
    )

    # Extract task info from trajectory
    task_id = perturbed_trajectory.get("task_id", "")
    task_text = perturbed_trajectory.get("task_text", "")

    # Create timestamp
    created_at = datetime.now(timezone.utc).isoformat()

    # Assemble EvaluationUnit
    evaluation_unit = EvaluationUnit(
        evaluation_unit_id=evaluation_unit_id,
        experiment_id=experiment_id,
        created_at=created_at,
        source_trajectory_id=source_trajectory_id,
        benchmark=benchmark,
        task_id=task_id,
        task_text=task_text,
        baseline=baseline_data,
        perturbed=perturbed_data,
        derived_cache=derived_cache,
        evaluation_capabilities=evaluation_capabilities,
        replay_tier=replay_tier,
        blinding=blinding,
    )

    return evaluation_unit.to_dict()


def assemble_all_units(
    perturbations_file: str,
    experiment_id: str,
) -> List[Dict[str, Any]]:
    """
    Load perturbations from JSON file and assemble all evaluation units.

    Args:
        perturbations_file: Path to the perturbations JSON file.
        experiment_id: The experiment ID for all units.

    Returns:
        List of EvaluationUnit dictionaries.
    """
    with open(perturbations_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    perturbations = data.get("perturbations", [])
    return _assemble_units_from_list(perturbations, experiment_id)


def assemble_all_units_from_mongodb(
    storage,
    experiment_id: str,
) -> List[Dict[str, Any]]:
    """
    Load perturbations from MongoDB and assemble all evaluation units.

    Args:
        storage: MongoDBStorage instance.
        experiment_id: The experiment ID to query perturbations for.

    Returns:
        List of EvaluationUnit dictionaries.
    """
    # Query perturbations collection for this experiment
    perturbations = list(
        storage.db.perturbed_trajectories.find({"experiment_id": experiment_id})
    )
    return _assemble_units_from_list(perturbations, experiment_id)


def _assemble_units_from_list(
    perturbations: List[Dict[str, Any]],
    experiment_id: str,
) -> List[Dict[str, Any]]:
    """
    Assemble evaluation units from a list of perturbation records.

    Args:
        perturbations: List of perturbation data dicts.
        experiment_id: The experiment ID for all units.

    Returns:
        List of EvaluationUnit dictionaries.
    """
    evaluation_units = []

    for idx, perturbation_data in enumerate(perturbations):
        try:
            unit = assemble_evaluation_unit(
                perturbation_data,
                experiment_id,
                perturbation_index=idx,
            )
            evaluation_units.append(unit)
        except Exception as e:
            # Log error but continue processing
            perturbation_id = perturbation_data.get("perturbation_id", f"index_{idx}")
            print(f"Warning: Failed to assemble unit for {perturbation_id}: {e}")
            continue

    return evaluation_units
