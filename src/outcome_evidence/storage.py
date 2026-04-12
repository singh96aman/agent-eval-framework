"""
Storage functions for Section 5C: Outcome Evidence.

Provides functions to save and load outcome evidence,
and create the directory structure per 5C.10.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import OutcomeRecord

# Default data location per 5C.10
DEFAULT_BASE_DIR = "data/outcome_evidence"


def create_outcome_directories(base_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Create directory structure for outcome evidence per 5C.10.

    Args:
        base_dir: Base directory (defaults to data/outcome_evidence)

    Returns:
        Dict mapping directory name to path
    """
    base = Path(base_dir or DEFAULT_BASE_DIR)

    directories = {
        "base": base,
        "tier_1": base / "tier_1",
        "tier_1_logs": base / "tier_1" / "logs",
        "tier_1_step_outputs": base / "tier_1" / "step_outputs",
        "tier_2": base / "tier_2",
        "tier_2_regenerated": base / "tier_2" / "regenerated_trajectories",
        "tier_3": base / "tier_3",
        "aggregated": base / "aggregated",
        "metadata": base / "metadata",
    }

    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)

    return {name: str(path) for name, path in directories.items()}


def save_outcome_evidence(
    outcomes: List[OutcomeRecord],
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Save outcome evidence to JSON file.

    Args:
        outcomes: List of OutcomeRecord objects
        output_dir: Output directory (defaults to aggregated/)
        filename: Output filename (defaults to outcome_evidence_all.json)

    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_BASE_DIR, "aggregated")

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = "outcome_evidence_all.json"

    output_path = os.path.join(output_dir, filename)

    data = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_outcomes": len(outcomes),
        "outcomes": [o.to_dict() for o in outcomes],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def load_outcome_evidence(
    input_path: Optional[str] = None,
) -> List[OutcomeRecord]:
    """
    Load outcome evidence from JSON file.

    Args:
        input_path: Path to JSON file (defaults to aggregated/outcome_evidence_all.json)

    Returns:
        List of OutcomeRecord objects
    """
    if input_path is None:
        input_path = os.path.join(
            DEFAULT_BASE_DIR, "aggregated", "outcome_evidence_all.json"
        )

    with open(input_path, "r") as f:
        data = json.load(f)

    outcomes = []
    for outcome_dict in data.get("outcomes", []):
        outcomes.append(OutcomeRecord.from_dict(outcome_dict))

    return outcomes


def save_tier_results(
    tier: int,
    results: List[Dict[str, Any]],
    base_dir: Optional[str] = None,
) -> str:
    """
    Save tier-specific results.

    Args:
        tier: Replay tier (1, 2, or 3)
        results: List of result dicts
        base_dir: Base directory (defaults to data/outcome_evidence)

    Returns:
        Path to saved file
    """
    base = Path(base_dir or DEFAULT_BASE_DIR)

    tier_dir = base / f"tier_{tier}"
    tier_dir.mkdir(parents=True, exist_ok=True)

    if tier == 1:
        filename = "replay_results.json"
    elif tier == 2:
        filename = "regeneration_results.json"
    else:
        filename = "grading_results.json"

    output_path = tier_dir / filename

    data = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tier": tier,
        "total_results": len(results),
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return str(output_path)


def load_tier_results(
    tier: int,
    base_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load tier-specific results.

    Args:
        tier: Replay tier (1, 2, or 3)
        base_dir: Base directory (defaults to data/outcome_evidence)

    Returns:
        List of result dicts
    """
    base = Path(base_dir or DEFAULT_BASE_DIR)

    if tier == 1:
        filename = "replay_results.json"
    elif tier == 2:
        filename = "regeneration_results.json"
    else:
        filename = "grading_results.json"

    input_path = base / f"tier_{tier}" / filename

    with open(input_path, "r") as f:
        data = json.load(f)

    return data.get("results", [])


def save_tier_assignments(
    assignments: List[Dict[str, Any]],
    base_dir: Optional[str] = None,
) -> str:
    """
    Save tier assignments to metadata directory.

    Args:
        assignments: List of tier assignment dicts
        base_dir: Base directory (defaults to data/outcome_evidence)

    Returns:
        Path to saved file
    """
    base = Path(base_dir or DEFAULT_BASE_DIR)
    metadata_dir = base / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    output_path = metadata_dir / "tier_assignments.json"

    data = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_assignments": len(assignments),
        "assignments": assignments,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return str(output_path)


def load_tier_assignments(
    base_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load tier assignments from metadata directory.

    Args:
        base_dir: Base directory (defaults to data/outcome_evidence)

    Returns:
        List of tier assignment dicts
    """
    base = Path(base_dir or DEFAULT_BASE_DIR)
    input_path = base / "metadata" / "tier_assignments.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    return data.get("assignments", [])


def save_execution_log(
    log_entries: List[Dict[str, Any]],
    base_dir: Optional[str] = None,
) -> str:
    """
    Save execution log to metadata directory.

    Args:
        log_entries: List of log entry dicts
        base_dir: Base directory (defaults to data/outcome_evidence)

    Returns:
        Path to saved file
    """
    base = Path(base_dir or DEFAULT_BASE_DIR)
    metadata_dir = base / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    output_path = metadata_dir / "execution_log.json"

    data = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_entries": len(log_entries),
        "entries": log_entries,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return str(output_path)


def save_logs_for_unit(
    evaluation_unit_id: str,
    baseline_log: str,
    perturbed_log: str,
    base_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Save execution logs for an evaluation unit (Tier 1).

    Args:
        evaluation_unit_id: The evaluation unit ID
        baseline_log: Baseline execution log
        perturbed_log: Perturbed execution log
        base_dir: Base directory (defaults to data/outcome_evidence)

    Returns:
        Dict with paths to saved log files
    """
    base = Path(base_dir or DEFAULT_BASE_DIR)
    logs_dir = base / "tier_1" / "logs" / evaluation_unit_id
    logs_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = logs_dir / "baseline_log.txt"
    perturbed_path = logs_dir / "perturbed_log.txt"

    with open(baseline_path, "w") as f:
        f.write(baseline_log)

    with open(perturbed_path, "w") as f:
        f.write(perturbed_log)

    return {
        "baseline_log": str(baseline_path),
        "perturbed_log": str(perturbed_path),
    }


def save_step_outputs_for_unit(
    evaluation_unit_id: str,
    baseline_steps: List[Dict[str, Any]],
    perturbed_steps: List[Dict[str, Any]],
    base_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Save step outputs for an evaluation unit (Tier 1).

    Args:
        evaluation_unit_id: The evaluation unit ID
        baseline_steps: Baseline step outputs
        perturbed_steps: Perturbed step outputs
        base_dir: Base directory (defaults to data/outcome_evidence)

    Returns:
        Dict with paths to saved step output files
    """
    base = Path(base_dir or DEFAULT_BASE_DIR)
    outputs_dir = base / "tier_1" / "step_outputs" / evaluation_unit_id
    outputs_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = outputs_dir / "baseline_steps.json"
    perturbed_path = outputs_dir / "perturbed_steps.json"

    with open(baseline_path, "w") as f:
        json.dump(baseline_steps, f, indent=2)

    with open(perturbed_path, "w") as f:
        json.dump(perturbed_steps, f, indent=2)

    return {
        "baseline_steps": str(baseline_path),
        "perturbed_steps": str(perturbed_path),
    }


def save_outcome_evidence_to_mongodb(
    outcomes: List[OutcomeRecord],
    storage,
    experiment_id: str,
) -> int:
    """
    Save outcome evidence to MongoDB.

    Args:
        outcomes: List of OutcomeRecord objects
        storage: MongoDBStorage instance
        experiment_id: Experiment identifier

    Returns:
        Number of outcomes saved
    """
    collection = storage.db["outcome_evidence"]

    # Create index if not exists
    collection.create_index("evaluation_unit_id")
    collection.create_index("experiment_id")

    saved = 0
    for outcome in outcomes:
        doc = outcome.to_dict()
        doc["experiment_id"] = experiment_id

        # Upsert by evaluation_unit_id and experiment_id
        collection.update_one(
            {
                "evaluation_unit_id": outcome.evaluation_unit_id,
                "experiment_id": experiment_id,
            },
            {"$set": doc},
            upsert=True,
        )
        saved += 1

    return saved


def load_outcome_evidence_from_mongodb(
    experiment_id: str,
    storage,
    filters: Optional[Dict[str, Any]] = None,
) -> List[OutcomeRecord]:
    """
    Load outcome evidence from MongoDB.

    Args:
        experiment_id: Experiment identifier
        storage: MongoDBStorage instance
        filters: Optional filter dict

    Returns:
        List of OutcomeRecord objects
    """
    collection = storage.db["outcome_evidence"]

    query = {"experiment_id": experiment_id}
    if filters:
        query.update(filters)

    docs = list(collection.find(query))

    outcomes = []
    for doc in docs:
        # Remove MongoDB-specific fields
        doc.pop("_id", None)
        doc.pop("experiment_id", None)
        outcomes.append(OutcomeRecord.from_dict(doc))

    return outcomes
