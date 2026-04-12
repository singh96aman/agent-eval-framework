#!/usr/bin/env python3
"""
Dump experiment data from MongoDB to JSON files.

Supports dumping by phase (recommended) or specific data types.

Usage:
    # Dump by phase (recommended - dumps all related collections)
    python ops/dump_experiment_data.py exp_id --phase perturb
    python ops/dump_experiment_data.py exp_id --phase typing perturb

    # Dump specific data types
    python ops/dump_experiment_data.py exp_id --data perturbations
    python ops/dump_experiment_data.py exp_id --data evaluation_units

    # Dump all data
    python ops/dump_experiment_data.py exp_id --all

    # Limit results
    python ops/dump_experiment_data.py exp_id --phase perturb --limit 100

    # Check status
    python ops/dump_experiment_data.py exp_id --status
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
        "description": "Perturbation index/summary (perturb phase)",
        "output_dir": "data/perturbed",
        "filename_template": "{experiment_id}_perturbation_index.json",
        "wrapper_key": "index",
    },
    "evaluation_units": {
        "collection": "evaluation_units",
        "description": "Evaluation units (perturb phase)",
        "output_dir": "data/evaluation_units",
        "filename_template": "{experiment_id}_evaluation_units.json",
        "wrapper_key": "evaluation_units",
    },
    "judge_outputs": {
        "collection": "judge_outputs",
        "description": "Judge evaluation outputs (judge phase)",
        "output_dir": "data/judge_outputs",
        "filename_template": "{experiment_id}_judge_outputs.json",
        "wrapper_key": "judge_outputs",
    },
    "judge_eval_outputs": {
        "collection": "judge_eval_outputs",
        "description": "Judge eval outputs on evaluation units (compute phase)",
        "output_dir": "data/judge_outputs/eval_units",
        "filename_template": "{experiment_id}_judge_eval_outputs.json",
        "wrapper_key": "judge_eval_outputs",
    },
    "outcome_evidence": {
        "collection": "outcome_evidence",
        "description": "Outcome evidence records (compute phase)",
        "output_dir": "data/outcome_evidence",
        "filename_template": "{experiment_id}_outcome_evidence.json",
        "wrapper_key": "outcome_evidence",
    },
}

# Mapping of phases to data types (for --phase option)
PHASE_DATA = {
    "load": ["trajectories"],
    "typing": ["typed"],
    "perturb": ["perturbations", "perturbation_index", "evaluation_units"],
    "judge": ["judge_outputs"],
    "compute": ["judge_eval_outputs", "outcome_evidence"],
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
  perturbation_index Perturbation index/summary (perturb phase)
  evaluation_units   Evaluation units (perturb phase)
  judge_outputs      Judge evaluation outputs

Phases (dumps all related data types):
  load      -> trajectories
  typing    -> typed
  perturb   -> perturbations, perturbation_index, evaluation_units
  judge     -> judge_outputs

Output locations (default):
  trajectories       -> data/raw/{exp_id}_trajectories.json
  typed              -> data/typed/{exp_id}_typed.json
  perturbations      -> data/perturbed/{exp_id}_perturbations.json
  perturbation_index -> data/perturbed/{exp_id}_perturbation_index.json
  evaluation_units   -> data/evaluation_units/{exp_id}_evaluation_units.json
  judge_outputs      -> data/judge_outputs/{exp_id}_judge_outputs.json

Examples:
  %(prog)s exp_id --phase perturb          # Dump all perturb phase data
  %(prog)s exp_id --data perturbations     # Dump only perturbations
  %(prog)s exp_id --all
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
        "--phase",
        "-p",
        nargs="+",
        choices=list(PHASE_DATA.keys()),
        help="Dump all data for specified phase(s)",
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

    dump_experiment_data(
        experiment_id=args.experiment_id,
        data_types=data_types,
        output_dir=args.output,
        limit=args.limit,
        dump_all=args.all,
    )


if __name__ == "__main__":
    main()
