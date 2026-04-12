"""
Storage utilities for human labels.

Handles:
- Saving annotations to JSON files
- Loading annotations from JSON files
- Exporting aggregated labels
- Creating data directory structure

Per Section 5A.10, data location:
data/evaluation_units/
    human_labels/
        raw/                    # Individual annotation batches
        aggregated/             # Post-aggregation labels
        gold/                   # Calibration gold set
        metadata/               # Assignment and QC reports
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.human_labels.schema import AggregatedLabel, AnnotationRecord


def save_annotations_to_json(
    annotations: List[AnnotationRecord],
    output_dir: str,
    batch_id: Optional[str] = None,
) -> str:
    """
    Save annotations to JSON file.

    Args:
        annotations: List of AnnotationRecord objects
        output_dir: Directory for output files
        batch_id: Optional batch identifier for filename. If None, uses timestamp.

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if batch_id is None:
        batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    filename = f"annotations_batch_{batch_id}.json"
    filepath = output_path / filename

    # Build export data
    export_data = {
        "_metadata": {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "batch_id": batch_id,
            "annotation_count": len(annotations),
            "format_version": "1.0",
        },
        "annotations": [ann.to_dict() for ann in annotations],
    }

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    return str(filepath)


def load_annotations_from_json(
    input_path: str,
) -> List[AnnotationRecord]:
    """
    Load annotations from JSON file or directory.

    Args:
        input_path: Path to JSON file or directory containing JSON files

    Returns:
        List of AnnotationRecord objects
    """
    path = Path(input_path)
    annotations = []

    if path.is_file():
        # Single file
        annotations.extend(_load_annotations_file(path))
    elif path.is_dir():
        # Directory - load all JSON files
        for json_file in sorted(path.glob("*.json")):
            annotations.extend(_load_annotations_file(json_file))

    return annotations


def _load_annotations_file(filepath: Path) -> List[AnnotationRecord]:
    """Load annotations from a single JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    # Handle both formats: with _metadata wrapper or raw list
    if "annotations" in data:
        annotation_dicts = data["annotations"]
    elif isinstance(data, list):
        annotation_dicts = data
    else:
        annotation_dicts = []

    return [AnnotationRecord.from_dict(d) for d in annotation_dicts]


def export_aggregated_labels(
    labels: List[AggregatedLabel],
    output_path: str,
) -> str:
    """
    Export aggregated labels to JSON file.

    Args:
        labels: List of AggregatedLabel objects
        output_path: Full path for output file

    Returns:
        Path to saved file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "_metadata": {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "label_count": len(labels),
            "format_version": "1.0",
        },
        "labels": [label.to_dict() for label in labels],
    }

    with open(path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    return str(path)


def load_aggregated_labels(
    input_path: str,
) -> List[AggregatedLabel]:
    """
    Load aggregated labels from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        List of AggregatedLabel objects
    """
    with open(input_path) as f:
        data = json.load(f)

    if "labels" in data:
        label_dicts = data["labels"]
    elif isinstance(data, list):
        label_dicts = data
    else:
        label_dicts = []

    return [AggregatedLabel.from_dict(d) for d in label_dicts]


def save_gold_set(
    gold_labels: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """
    Save calibration gold set.

    Args:
        gold_labels: List of gold standard labels with:
            - evaluation_unit_id: str
            - gold_error_detected: bool
            - gold_error_trajectory: str
            - gold_error_step_id: str (optional)
            - gold_error_type: str (optional)
            - adjudication_notes: str (optional)
        output_path: Full path for output file

    Returns:
        Path to saved file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "_metadata": {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "gold_label_count": len(gold_labels),
            "purpose": "calibration_and_drift_monitoring",
            "format_version": "1.0",
        },
        "gold_labels": gold_labels,
    }

    with open(path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    return str(path)


def load_gold_set(input_path: str) -> List[Dict[str, Any]]:
    """
    Load calibration gold set.

    Args:
        input_path: Path to gold set JSON file

    Returns:
        List of gold standard label dicts
    """
    with open(input_path) as f:
        data = json.load(f)

    if "gold_labels" in data:
        return data["gold_labels"]
    elif isinstance(data, list):
        return data
    return []


def save_annotator_assignments(
    assignments: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Save annotator assignment configuration.

    Args:
        assignments: Output from assign_to_annotators()
        output_path: Full path for output file

    Returns:
        Path to saved file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "_metadata": {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "format_version": "1.0",
        },
        "assignments": assignments,
    }

    with open(path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    return str(path)


def load_annotator_assignments(input_path: str) -> Dict[str, Any]:
    """
    Load annotator assignment configuration.

    Args:
        input_path: Path to assignments JSON file

    Returns:
        Assignment configuration dict
    """
    with open(input_path) as f:
        data = json.load(f)

    if "assignments" in data:
        return data["assignments"]
    return data


def save_qc_report(
    report: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Save QC report.

    Args:
        report: Output from run_all_qc_checks()
        output_path: Full path for output file

    Returns:
        Path to saved file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return str(path)


def save_agreement_report(
    report: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Save agreement report.

    Args:
        report: Output from compute_agreement_report()
        output_path: Full path for output file

    Returns:
        Path to saved file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "_metadata": {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "format_version": "1.0",
        },
        "agreement": report,
    }

    with open(path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    return str(path)


def create_human_labels_directories(
    base_dir: str = "data/evaluation_units",
) -> Dict[str, str]:
    """
    Create directory structure for human label data.

    Creates:
    - human_labels/raw/           - Individual annotation batches
    - human_labels/aggregated/    - Post-aggregation labels
    - human_labels/gold/          - Calibration gold set
    - human_labels/metadata/      - Assignment and QC reports

    Args:
        base_dir: Base directory for evaluation unit data

    Returns:
        Dict mapping directory type to absolute path
    """
    base_path = Path(base_dir)
    human_labels_path = base_path / "human_labels"

    dirs = {
        "raw": human_labels_path / "raw",
        "aggregated": human_labels_path / "aggregated",
        "gold": human_labels_path / "gold",
        "metadata": human_labels_path / "metadata",
    }

    paths = {}
    for name, dir_path in dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        paths[name] = str(dir_path.absolute())

    return paths


# =============================================================================
# MongoDB Storage Functions
# =============================================================================


def save_annotations_to_mongodb(
    annotations: List[AnnotationRecord],
    storage,
    experiment_id: str,
) -> int:
    """
    Save annotations to MongoDB with upsert.

    Args:
        annotations: List of AnnotationRecord objects
        storage: MongoDBStorage instance
        experiment_id: Experiment identifier

    Returns:
        Number of annotations saved
    """
    collection = storage.db["human_labels"]

    # Create indexes if not exist
    collection.create_index("annotation_id")
    collection.create_index("evaluation_unit_id")
    collection.create_index("experiment_id")
    collection.create_index("annotator_id")

    saved = 0
    for annotation in annotations:
        doc = annotation.to_dict()
        doc["experiment_id"] = experiment_id

        # Upsert by annotation_id (unique per annotation)
        collection.update_one(
            {"annotation_id": annotation.annotation_id},
            {"$set": doc},
            upsert=True,
        )
        saved += 1

    return saved


def load_annotations_from_mongodb(
    experiment_id: str,
    storage,
    filters: Optional[Dict[str, Any]] = None,
) -> List[AnnotationRecord]:
    """
    Load annotations from MongoDB.

    Args:
        experiment_id: Experiment identifier
        storage: MongoDBStorage instance
        filters: Optional filter dict (e.g., annotator_id, evaluation_unit_id)

    Returns:
        List of AnnotationRecord objects
    """
    collection = storage.db["human_labels"]

    query = {"experiment_id": experiment_id}
    if filters:
        query.update(filters)

    docs = list(collection.find(query))

    annotations = []
    for doc in docs:
        # Remove MongoDB-specific fields
        doc.pop("_id", None)
        doc.pop("experiment_id", None)
        annotations.append(AnnotationRecord.from_dict(doc))

    return annotations


def get_completed_annotation_unit_ids(
    experiment_id: str,
    storage,
    annotator_id: Optional[str] = None,
) -> Set[str]:
    """
    Get set of evaluation_unit_ids that have been annotated.
    Used for resume support.

    Args:
        experiment_id: Experiment identifier
        storage: MongoDBStorage instance
        annotator_id: Optional filter by annotator

    Returns:
        Set of evaluation_unit_id strings
    """
    collection = storage.db["human_labels"]

    query = {"experiment_id": experiment_id}
    if annotator_id:
        query["annotator_id"] = annotator_id

    docs = collection.find(query, {"evaluation_unit_id": 1})
    return {doc["evaluation_unit_id"] for doc in docs}


def save_aggregated_labels_to_mongodb(
    labels: List[AggregatedLabel],
    storage,
    experiment_id: str,
) -> int:
    """
    Save aggregated labels to MongoDB.

    Args:
        labels: List of AggregatedLabel objects
        storage: MongoDBStorage instance
        experiment_id: Experiment identifier

    Returns:
        Number of labels saved
    """
    collection = storage.db["aggregated_human_labels"]

    # Create indexes if not exist
    collection.create_index("evaluation_unit_id")
    collection.create_index("experiment_id")
    collection.create_index(
        [
            ("evaluation_unit_id", 1),
            ("experiment_id", 1),
        ],
        unique=True,
    )

    saved = 0
    for label in labels:
        doc = label.to_dict()
        doc["experiment_id"] = experiment_id

        # Upsert by composite key
        collection.update_one(
            {
                "evaluation_unit_id": label.evaluation_unit_id,
                "experiment_id": experiment_id,
            },
            {"$set": doc},
            upsert=True,
        )
        saved += 1

    return saved


def load_aggregated_labels_from_mongodb(
    experiment_id: str,
    storage,
    filters: Optional[Dict[str, Any]] = None,
) -> List[AggregatedLabel]:
    """
    Load aggregated labels from MongoDB.

    Args:
        experiment_id: Experiment identifier
        storage: MongoDBStorage instance
        filters: Optional filter dict

    Returns:
        List of AggregatedLabel objects
    """
    collection = storage.db["aggregated_human_labels"]

    query = {"experiment_id": experiment_id}
    if filters:
        query.update(filters)

    docs = list(collection.find(query))

    labels = []
    for doc in docs:
        # Remove MongoDB-specific fields
        doc.pop("_id", None)
        doc.pop("experiment_id", None)
        labels.append(AggregatedLabel.from_dict(doc))

    return labels
