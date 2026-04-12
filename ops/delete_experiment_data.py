#!/usr/bin/env python3
"""
Delete experiment data from MongoDB.

Supports deleting specific data types, phases, or all data for an experiment.

Usage:
    # Delete by phase (recommended - deletes all related collections)
    python ops/delete_experiment_data.py exp_id --phase perturb
    python ops/delete_experiment_data.py exp_id --phase typing perturb

    # Delete specific data types
    python ops/delete_experiment_data.py exp_id --data perturbations
    python ops/delete_experiment_data.py exp_id --data evaluation_units

    # Delete all data for an experiment
    python ops/delete_experiment_data.py exp_id --all

    # Skip confirmation
    python ops/delete_experiment_data.py exp_id --phase perturb --force

    # Check status
    python ops/delete_experiment_data.py exp_id --status
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.storage.mongodb import MongoDBStorage

# Mapping of data type names to MongoDB collections
DATA_TYPES = {
    "trajectories": {
        "collection": "trajectories",
        "description": "Raw trajectories (load phase)",
    },
    "typed": {
        "collection": "typed_trajectories",
        "description": "Typed trajectories (typing phase)",
    },
    "perturbations": {
        "collection": "perturbed_trajectories",
        "description": "Perturbed trajectories (perturb phase)",
    },
    "perturbation_index": {
        "collection": "perturbation_index",
        "description": "Perturbation index/summary (perturb phase)",
    },
    "evaluation_units": {
        "collection": "evaluation_units",
        "description": "Evaluation units (perturb phase)",
    },
    "judge_outputs": {
        "collection": "judge_outputs",
        "description": "Judge evaluation outputs (judge phase)",
    },
    "judge_eval_outputs": {
        "collection": "judge_eval_outputs",
        "description": "Judge eval outputs on evaluation units (compute phase)",
    },
    "outcome_evidence": {
        "collection": "outcome_evidence",
        "description": "Outcome evidence records (compute phase)",
    },
    "annotations": {
        "collection": "annotations",
        "description": "Human annotations (annotate phase)",
    },
}

# Mapping of phases to data types (for --phase option)
PHASE_DATA = {
    "load": ["trajectories"],
    "typing": ["typed"],
    "perturb": ["perturbations", "perturbation_index", "evaluation_units"],
    "judge": ["judge_outputs"],
    "annotate": ["annotations"],
    "compute": ["judge_eval_outputs", "outcome_evidence"],
}


def get_counts(storage: MongoDBStorage, experiment_id: str) -> Dict[str, int]:
    """Get document counts for each data type."""
    counts = {}
    for name, info in DATA_TYPES.items():
        collection = storage.db[info["collection"]]
        count = collection.count_documents({"experiment_id": experiment_id})
        counts[name] = count
    return counts


def delete_data(
    experiment_id: str,
    data_types: List[str],
    force: bool = False,
    delete_all: bool = False,
):
    """Delete specified data types for an experiment."""
    print("Connecting to MongoDB...")
    storage = MongoDBStorage()

    # Get current counts
    counts = get_counts(storage, experiment_id)

    # Determine which types to delete
    if delete_all:
        types_to_delete = list(DATA_TYPES.keys())
    else:
        # Validate data types
        invalid = set(data_types) - set(DATA_TYPES.keys())
        if invalid:
            print(f"Error: Invalid data types: {invalid}")
            print(f"Valid types: {list(DATA_TYPES.keys())}")
            storage.close()
            return

        types_to_delete = data_types

    # Show what will be deleted
    print(f"\nExperiment: {experiment_id}")
    print("-" * 50)

    total = 0
    for dtype in types_to_delete:
        count = counts[dtype]
        desc = DATA_TYPES[dtype]["description"]
        if count > 0:
            print(f"  {dtype}: {count} documents ({desc})")
            total += count
        else:
            print(f"  {dtype}: 0 documents")

    if total == 0:
        print("\nNo data found to delete.")
        storage.close()
        return

    print("-" * 50)
    print(f"Total: {total} documents")

    # Confirmation prompt
    if not force:
        response = input("\nDelete this data? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("Aborted.")
            storage.close()
            return

    # Delete
    print("\nDeleting...")
    for dtype in types_to_delete:
        if counts[dtype] > 0:
            collection = storage.db[DATA_TYPES[dtype]["collection"]]
            result = collection.delete_many({"experiment_id": experiment_id})
            print(f"  {dtype}: deleted {result.deleted_count}")

    print("\nDone.")
    storage.close()


def list_experiments(storage: MongoDBStorage):
    """List all experiment IDs in the database."""
    experiments = set()

    for info in DATA_TYPES.values():
        collection = storage.db[info["collection"]]
        ids = collection.distinct("experiment_id")
        experiments.update(ids)

    return sorted(experiments)


def show_status(experiment_id: str = None):
    """Show status of experiments in database."""
    print("Connecting to MongoDB...")
    storage = MongoDBStorage()

    if experiment_id:
        # Show status for specific experiment
        counts = get_counts(storage, experiment_id)
        print(f"\nExperiment: {experiment_id}")
        print("-" * 50)
        for dtype, count in counts.items():
            desc = DATA_TYPES[dtype]["description"]
            print(f"  {dtype}: {count} ({desc})")
    else:
        # List all experiments
        experiments = list_experiments(storage)
        print(f"\nExperiments in database ({len(experiments)}):")
        print("-" * 50)
        for exp_id in experiments:
            counts = get_counts(storage, exp_id)
            total = sum(counts.values())
            print(f"  {exp_id}: {total} total documents")

    storage.close()


def main():
    parser = argparse.ArgumentParser(
        description="Delete experiment data from MongoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data types:
  trajectories        Raw trajectories (load phase)
  typed               Typed trajectories (typing phase)
  perturbations       Perturbed trajectories (perturb phase)
  perturbation_index  Perturbation index/summary (perturb phase)
  evaluation_units    Evaluation units (perturb phase)
  judge_outputs       Judge evaluation outputs (judge phase)
  judge_eval_outputs  Judge eval outputs on evaluation units (compute phase)
  outcome_evidence    Outcome evidence records (compute phase)
  annotations         Human annotations (annotate phase)

Phases (deletes all related data types):
  load      -> trajectories
  typing    -> typed
  perturb   -> perturbations, perturbation_index, evaluation_units
  judge     -> judge_outputs
  annotate  -> annotations
  compute   -> judge_eval_outputs, outcome_evidence

Examples:
  %(prog)s exp_id --phase perturb          # Delete all perturb phase data
  %(prog)s exp_id --data perturbations     # Delete only perturbations
  %(prog)s exp_id --all --force
  %(prog)s --status
        """,
    )

    parser.add_argument(
        "experiment_id",
        nargs="?",
        help="Experiment ID",
    )
    parser.add_argument(
        "--data",
        "-d",
        nargs="+",
        choices=list(DATA_TYPES.keys()),
        help="Data types to delete",
    )
    parser.add_argument(
        "--phase",
        "-p",
        nargs="+",
        choices=list(PHASE_DATA.keys()),
        help="Delete all data for specified phase(s)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Delete all data for the experiment",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--status",
        "-s",
        action="store_true",
        help="Show status instead of deleting",
    )

    args = parser.parse_args()

    # Status mode
    if args.status:
        show_status(args.experiment_id)
        return

    # Delete mode requires experiment_id
    if not args.experiment_id:
        parser.error("experiment_id is required (or use --status)")

    # Need either --data, --phase, or --all
    if not args.data and not args.phase and not args.all:
        parser.error("Specify --data, --phase, or --all")

    # Expand phases to data types
    data_types = list(args.data or [])
    if args.phase:
        for phase in args.phase:
            data_types.extend(PHASE_DATA[phase])
        # Remove duplicates while preserving order
        data_types = list(dict.fromkeys(data_types))

    delete_data(
        experiment_id=args.experiment_id,
        data_types=data_types,
        force=args.force,
        delete_all=args.all,
    )


if __name__ == "__main__":
    main()
