#!/usr/bin/env python3
"""
Dump experiment data from MongoDB to JSON files.

Supports dumping trajectories, typed representations, and perturbations
with consistent naming conventions.

Usage:
    # Dump specific data types
    python ops/dump_experiment_data.py exp_id --data trajectories
    python ops/dump_experiment_data.py exp_id --data typed
    python ops/dump_experiment_data.py exp_id --data perturbations

    # Dump multiple types
    python ops/dump_experiment_data.py exp_id --data typed perturbations

    # Dump all data
    python ops/dump_experiment_data.py exp_id --all

    # Limit results
    python ops/dump_experiment_data.py exp_id --data trajectories --limit 100

    # Custom output directory
    python ops/dump_experiment_data.py exp_id --data typed --output data/exports
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.storage.mongodb import MongoDBStorage

# Data type configurations
DATA_TYPES = {
    "trajectories": {
        "collection": "trajectories",
        "description": "Raw trajectories (load phase)",
        "output_dir": "data/raw",
        "filename_template": "{experiment_id}_trajectories.json",
        "wrapper_key": "trajectories",
    },
    "typed": {
        "collection": "typed_trajectories",
        "description": "Typed trajectories (typing phase)",
        "output_dir": "data/typed",
        "filename_template": "{experiment_id}_typed.json",
        "wrapper_key": "typed_trajectories",
    },
    "perturbations": {
        "collection": "perturbed_trajectories",
        "description": "Perturbed trajectories (perturb phase)",
        "output_dir": "data/perturbed",
        "filename_template": "{experiment_id}_perturbations.json",
        "wrapper_key": "perturbations",
    },
    "perturbation_index": {
        "collection": "perturbation_index",
        "description": "Perturbation index/summary",
        "output_dir": "data/perturbed",
        "filename_template": "{experiment_id}_perturbation_index.json",
        "wrapper_key": "index",
    },
    "judge_outputs": {
        "collection": "judge_outputs",
        "description": "Judge evaluation outputs",
        "output_dir": "data/judge_outputs",
        "filename_template": "{experiment_id}_judge_outputs.json",
        "wrapper_key": "judge_outputs",
    },
}


def clean_for_json(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Clean MongoDB document for JSON serialization."""
    doc.pop("_id", None)

    # Convert datetime fields to strings
    for key, value in doc.items():
        if hasattr(value, "isoformat"):
            doc[key] = value.isoformat()
        elif isinstance(value, dict):
            clean_for_json(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    clean_for_json(item)

    return doc


def dump_data_type(
    storage: MongoDBStorage,
    experiment_id: str,
    data_type: str,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Optional[Path]:
    """Dump a single data type to JSON."""

    config = DATA_TYPES[data_type]
    collection = storage.db[config["collection"]]

    # Query
    query = {"experiment_id": experiment_id}
    cursor = collection.find(query)
    if limit:
        cursor = cursor.limit(limit)

    documents = list(cursor)

    if not documents:
        print(f"  {data_type}: No documents found")
        return None

    # Clean for JSON
    for doc in documents:
        clean_for_json(doc)

    # Determine output path
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / config["output_dir"]
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = config["filename_template"].format(experiment_id=experiment_id)
    output_path = output_dir / filename

    # Build output structure
    output_data = {
        "experiment_id": experiment_id,
        "data_type": data_type,
        "exported_at": datetime.now().isoformat(),
        "count": len(documents),
        config["wrapper_key"]: documents,
    }

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"  {data_type}: {len(documents)} documents -> {output_path}")
    return output_path


def dump_experiment_data(
    experiment_id: str,
    data_types: List[str],
    output_dir: Optional[str] = None,
    limit: Optional[int] = None,
    dump_all: bool = False,
):
    """Dump specified data types for an experiment."""

    print("Connecting to MongoDB...")
    storage = MongoDBStorage()

    # Determine which types to dump
    if dump_all:
        types_to_dump = list(DATA_TYPES.keys())
    else:
        # Validate data types
        invalid = set(data_types) - set(DATA_TYPES.keys())
        if invalid:
            print(f"Error: Invalid data types: {invalid}")
            print(f"Valid types: {list(DATA_TYPES.keys())}")
            storage.close()
            return
        types_to_dump = data_types

    print(f"\nDumping data for experiment: {experiment_id}")
    print("-" * 50)

    output_path = Path(output_dir) if output_dir else None
    exported_files = []

    for dtype in types_to_dump:
        path = dump_data_type(
            storage=storage,
            experiment_id=experiment_id,
            data_type=dtype,
            output_dir=output_path,
            limit=limit,
        )
        if path:
            exported_files.append(path)

    print("-" * 50)
    print(f"Exported {len(exported_files)} files")

    storage.close()
    return exported_files


def show_status(experiment_id: str):
    """Show what data exists for an experiment."""
    print("Connecting to MongoDB...")
    storage = MongoDBStorage()

    print(f"\nExperiment: {experiment_id}")
    print("-" * 50)

    total = 0
    for dtype, config in DATA_TYPES.items():
        collection = storage.db[config["collection"]]
        count = collection.count_documents({"experiment_id": experiment_id})
        desc = config["description"]
        print(f"  {dtype}: {count} documents ({desc})")
        total += count

    print("-" * 50)
    print(f"Total: {total} documents")

    storage.close()


def main():
    parser = argparse.ArgumentParser(
        description="Dump experiment data from MongoDB to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data types:
  trajectories       Raw trajectories (load phase)
  typed              Typed trajectories (typing phase)
  perturbations      Perturbed trajectories (perturb phase)
  perturbation_index Perturbation index/summary
  judge_outputs      Judge evaluation outputs

Output locations (default):
  trajectories       -> data/raw/{exp_id}_trajectories.json
  typed              -> data/typed/{exp_id}_typed.json
  perturbations      -> data/perturbed/{exp_id}_perturbations.json
  perturbation_index -> data/perturbed/{exp_id}_perturbation_index.json
  judge_outputs      -> data/judge_outputs/{exp_id}_judge_outputs.json

Examples:
  %(prog)s exp_id --data typed
  %(prog)s exp_id --data typed perturbations
  %(prog)s exp_id --all
  %(prog)s exp_id --data trajectories --limit 100
  %(prog)s exp_id --status
        """,
    )

    parser.add_argument(
        "experiment_id",
        help="Experiment ID to dump",
    )
    parser.add_argument(
        "--data",
        "-d",
        nargs="+",
        choices=list(DATA_TYPES.keys()),
        help="Data types to dump",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Dump all data types",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Custom output directory (overrides defaults)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Max documents per data type",
    )
    parser.add_argument(
        "--status",
        "-s",
        action="store_true",
        help="Show status instead of dumping",
    )

    args = parser.parse_args()

    # Status mode
    if args.status:
        show_status(args.experiment_id)
        return

    # Need either --data or --all
    if not args.data and not args.all:
        parser.error("Specify --data or --all")

    dump_experiment_data(
        experiment_id=args.experiment_id,
        data_types=args.data or [],
        output_dir=args.output,
        limit=args.limit,
        dump_all=args.all,
    )


if __name__ == "__main__":
    main()
