"""
Storage utilities for evaluation units.

Handles:
- Saving evaluation units to MongoDB with indexes
- Exporting evaluation units to JSON files (by benchmark)
- Loading evaluation units with filters
- Creating data directory structure
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pymongo import ASCENDING


def save_evaluation_units_to_mongodb(
    units: List[Dict],
    experiment_id: str,
    storage,
) -> int:
    """
    Save evaluation units to MongoDB with appropriate indexes.

    Saves to the 'evaluation_units' collection with indexes on:
    - evaluation_unit_id (unique)
    - experiment_id
    - benchmark
    - perturbation_class

    Args:
        units: List of evaluation unit dicts (from EvaluationUnit.to_dict())
        experiment_id: Experiment identifier for tagging
        storage: MongoDBStorage instance

    Returns:
        Number of units saved
    """
    if not units:
        return 0

    # Get or create evaluation_units collection
    collection = storage.db["evaluation_units"]

    # Create indexes (idempotent - won't duplicate if exists)
    collection.create_index(
        [("evaluation_unit_id", ASCENDING)],
        unique=True,
    )
    collection.create_index([("experiment_id", ASCENDING)])
    collection.create_index([("benchmark", ASCENDING)])
    collection.create_index(
        [
            ("derived_cache.perturbation_class", ASCENDING),
        ],
        name="perturbation_class_idx",
    )
    # Compound index for common query patterns
    collection.create_index(
        [
            ("experiment_id", ASCENDING),
            ("benchmark", ASCENDING),
        ]
    )

    # Prepare documents with storage metadata
    timestamp = datetime.utcnow()
    docs = []
    for unit in units:
        doc = unit.copy()
        doc["experiment_id"] = experiment_id
        doc["stored_at"] = timestamp
        docs.append(doc)

    # Bulk insert with ordered=False to continue on duplicates
    saved_count = 0
    try:
        result = collection.insert_many(docs, ordered=False)
        saved_count = len(result.inserted_ids)
    except Exception as e:
        # Some might be duplicates - count actual inserts
        # BulkWriteError contains nInserted
        if hasattr(e, "details") and "nInserted" in e.details:
            saved_count = e.details["nInserted"]
        else:
            # Fallback: count documents with this experiment_id
            saved_count = collection.count_documents({"experiment_id": experiment_id})

    return saved_count


def export_evaluation_units_to_json(
    units: List[Dict],
    output_dir: str,
    by_benchmark: bool = True,
) -> Dict[str, str]:
    """
    Export evaluation units to JSON files.

    Args:
        units: List of evaluation unit dicts
        output_dir: Directory for output files
        by_benchmark: If True, save separate files per benchmark
                      (toolbench_eval_units.json, swebench_eval_units.json, etc.)
                      If False, save all to all_eval_units.json

    Returns:
        Dict mapping benchmark (or 'all') to file path
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build metadata header
    metadata = {
        "_metadata": {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "total_units": len(units),
            "format_version": "1.0",
        }
    }

    paths = {}

    if by_benchmark:
        # Group units by benchmark
        by_bench: Dict[str, List[Dict]] = {}
        for unit in units:
            benchmark = unit.get("benchmark", "unknown")
            if benchmark not in by_bench:
                by_bench[benchmark] = []
            by_bench[benchmark].append(unit)

        # Save each benchmark to separate file
        for benchmark, bench_units in by_bench.items():
            filename = f"{benchmark}_eval_units.json"
            filepath = output_path / filename

            # Add benchmark-specific metadata
            export_data = metadata.copy()
            export_data["_metadata"]["benchmark"] = benchmark
            export_data["_metadata"]["unit_count"] = len(bench_units)
            export_data["units"] = bench_units

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            paths[benchmark] = str(filepath)
    else:
        # Save all to single file
        filename = "all_eval_units.json"
        filepath = output_path / filename

        export_data = metadata.copy()
        export_data["units"] = units

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        paths["all"] = str(filepath)

    return paths


def load_evaluation_units_from_mongodb(
    experiment_id: str,
    storage,
    filters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Load evaluation units from MongoDB with optional filters.

    Args:
        experiment_id: Experiment identifier
        storage: MongoDBStorage instance
        filters: Optional filter dict with supported keys:
                 - benchmark: str (e.g., "toolbench")
                 - perturbation_class: str (e.g., "placebo", "fine_grained")
                 - perturbation_family: str
                 - replay_tier: int (1, 2, or 3)
                 - expected_impact: int (0-3)
                 - expected_detectability: int (0-2)

    Returns:
        List of evaluation unit dicts (with storage metadata stripped)
    """
    collection = storage.db["evaluation_units"]

    # Build query
    query = {"experiment_id": experiment_id}

    if filters:
        # Direct field filters
        if "benchmark" in filters:
            query["benchmark"] = filters["benchmark"]
        if "replay_tier" in filters:
            query["replay_tier"] = filters["replay_tier"]

        # Derived cache field filters
        if "perturbation_class" in filters:
            query["derived_cache.perturbation_class"] = filters["perturbation_class"]
        if "perturbation_family" in filters:
            query["derived_cache.perturbation_family"] = filters["perturbation_family"]
        if "expected_impact" in filters:
            query["derived_cache.expected_impact"] = filters["expected_impact"]
        if "expected_detectability" in filters:
            query["derived_cache.expected_detectability"] = filters[
                "expected_detectability"
            ]

    # Execute query
    docs = list(collection.find(query))

    # Strip storage metadata
    units = []
    storage_fields = {"_id", "experiment_id", "stored_at"}
    for doc in docs:
        unit = {k: v for k, v in doc.items() if k not in storage_fields}
        units.append(unit)

    return units


def create_data_directories(base_dir: str = "data/evaluation_units") -> Dict[str, str]:
    """
    Create directory structure for evaluation unit data.

    Creates:
    - canonical/         - Source of truth evaluation units
    - private/           - Blinding keys and sensitive data
    - views/human/       - Human evaluator views (blinded)
    - views/llm_judge/   - LLM judge views (blinded)

    Args:
        base_dir: Base directory for evaluation unit data

    Returns:
        Dict mapping directory type to absolute path
    """
    base_path = Path(base_dir)

    # Define directory structure
    dirs = {
        "canonical": base_path / "canonical",
        "private": base_path / "private",
        "views_human": base_path / "views" / "human",
        "views_llm_judge": base_path / "views" / "llm_judge",
    }

    # Create all directories
    paths = {}
    for name, dir_path in dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        paths[name] = str(dir_path.absolute())

    return paths
