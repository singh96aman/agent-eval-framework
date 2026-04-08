"""
Storage and export utilities for perturbed trajectories.

Handles:
- Saving to MongoDB
- Exporting to JSON files
- Building perturbation index
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.perturbations.schema import (
    PerturbationRecord,
    PerturbationIndex,
)


class PerturbationStorage:
    """
    Storage handler for perturbed trajectories and perturbation records.
    """

    def __init__(
        self,
        mongodb_client=None,
        database_name: str = "agent_judge_experiment",
        collection_name: str = "perturbed_trajectories",
        index_collection_name: str = "perturbation_index",
    ):
        """
        Initialize storage handler.

        Args:
            mongodb_client: MongoDB client (optional, for DB storage)
            database_name: MongoDB database name
            collection_name: Collection for perturbed trajectories
            index_collection_name: Collection for perturbation index
        """
        self.mongodb_client = mongodb_client
        self.database_name = database_name
        self.collection_name = collection_name
        self.index_collection_name = index_collection_name

        self._db = None
        self._collection = None
        self._index_collection = None

    def _get_db(self):
        """Get MongoDB database connection."""
        if self._db is None and self.mongodb_client is not None:
            self._db = self.mongodb_client[self.database_name]
        return self._db

    def _get_collection(self):
        """Get perturbed trajectories collection."""
        if self._collection is None:
            db = self._get_db()
            if db is not None:
                self._collection = db[self.collection_name]
        return self._collection

    def _get_index_collection(self):
        """Get perturbation index collection."""
        if self._index_collection is None:
            db = self._get_db()
            if db is not None:
                self._index_collection = db[self.index_collection_name]
        return self._index_collection

    def save_perturbation(
        self,
        perturbed_trajectory: Dict[str, Any],
        experiment_id: str,
    ) -> Optional[str]:
        """
        Save a single perturbed trajectory to MongoDB.

        Args:
            perturbed_trajectory: Perturbed trajectory dict
            experiment_id: Experiment identifier

        Returns:
            Inserted document ID or None
        """
        collection = self._get_collection()
        if collection is None:
            return None

        # Add experiment metadata
        doc = perturbed_trajectory.copy()
        doc["experiment_id"] = experiment_id
        doc["stored_at"] = datetime.utcnow().isoformat() + "Z"

        result = collection.insert_one(doc)
        return str(result.inserted_id)

    def save_perturbations_batch(
        self,
        perturbed_trajectories: List[Dict[str, Any]],
        experiment_id: str,
    ) -> int:
        """
        Save multiple perturbed trajectories to MongoDB.

        Args:
            perturbed_trajectories: List of perturbed trajectory dicts
            experiment_id: Experiment identifier

        Returns:
            Number of documents inserted
        """
        collection = self._get_collection()
        if collection is None:
            return 0

        if not perturbed_trajectories:
            return 0

        # Add experiment metadata to each
        docs = []
        timestamp = datetime.utcnow().isoformat() + "Z"
        for pt in perturbed_trajectories:
            doc = pt.copy()
            doc["experiment_id"] = experiment_id
            doc["stored_at"] = timestamp
            docs.append(doc)

        result = collection.insert_many(docs)
        return len(result.inserted_ids)

    def save_index(
        self,
        index: PerturbationIndex,
        experiment_id: str,
    ) -> Optional[str]:
        """
        Save perturbation index to MongoDB.

        Args:
            index: PerturbationIndex object
            experiment_id: Experiment identifier

        Returns:
            Inserted document ID or None
        """
        collection = self._get_index_collection()
        if collection is None:
            return None

        doc = index.to_dict()
        doc["experiment_id"] = experiment_id
        doc["created_at"] = datetime.utcnow().isoformat() + "Z"

        # Upsert - replace if exists
        collection.replace_one({"experiment_id": experiment_id}, doc, upsert=True)
        return experiment_id

    def load_perturbations(
        self,
        experiment_id: str,
        benchmark: Optional[str] = None,
        generation_status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load perturbed trajectories from MongoDB.

        Args:
            experiment_id: Experiment identifier
            benchmark: Optional benchmark filter
            generation_status: Optional status filter (valid/invalid/borderline)

        Returns:
            List of perturbed trajectory dicts
        """
        collection = self._get_collection()
        if collection is None:
            return []

        query = {"experiment_id": experiment_id}
        if benchmark:
            query["original_trajectory.benchmark"] = benchmark
        if generation_status:
            query["perturbation_record.generation_status"] = generation_status

        return list(collection.find(query))

    def load_index(self, experiment_id: str) -> Optional[PerturbationIndex]:
        """
        Load perturbation index from MongoDB.

        Args:
            experiment_id: Experiment identifier

        Returns:
            PerturbationIndex or None
        """
        collection = self._get_index_collection()
        if collection is None:
            return None

        doc = collection.find_one({"experiment_id": experiment_id})
        if doc:
            # Remove MongoDB fields
            doc.pop("_id", None)
            doc.pop("experiment_id", None)
            doc.pop("created_at", None)
            return PerturbationIndex.from_dict(doc)
        return None


class PerturbationExporter:
    """
    Export perturbed trajectories and index to JSON files.
    """

    def __init__(self, output_dir: str = "data/perturbed"):
        """
        Initialize exporter.

        Args:
            output_dir: Directory for output JSON files
        """
        self.output_dir = Path(output_dir)

    def ensure_dir(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_by_benchmark(
        self,
        perturbed_trajectories: List[Dict[str, Any]],
        benchmark: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        Export perturbed trajectories for a benchmark to JSON.

        Args:
            perturbed_trajectories: List of perturbed trajectory dicts
            benchmark: Benchmark name
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        self.ensure_dir()

        if filename is None:
            filename = f"{benchmark}_perturbed.json"

        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(perturbed_trajectories, f, indent=2, default=str)

        return str(filepath)

    def export_index(
        self,
        index: PerturbationIndex,
        filename: str = "perturbation_index.json",
    ) -> str:
        """
        Export perturbation index to JSON.

        Args:
            index: PerturbationIndex object
            filename: Output filename

        Returns:
            Path to exported file
        """
        self.ensure_dir()

        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(index.to_dict(), f, indent=2)

        return str(filepath)

    def export_all(
        self,
        perturbed_by_benchmark: Dict[str, List[Dict[str, Any]]],
        index: PerturbationIndex,
    ) -> Dict[str, str]:
        """
        Export all perturbed trajectories and index.

        Args:
            perturbed_by_benchmark: Dict mapping benchmark to perturbations
            index: PerturbationIndex

        Returns:
            Dict mapping output type to file path
        """
        self.ensure_dir()

        paths = {}

        # Export each benchmark
        for benchmark, perturbations in perturbed_by_benchmark.items():
            path = self.export_by_benchmark(perturbations, benchmark)
            paths[benchmark] = path

        # Export index
        paths["index"] = self.export_index(index)

        return paths


def build_index_from_perturbations(
    perturbed_trajectories: List[Dict[str, Any]],
    output_dir: str = "data/perturbed",
) -> PerturbationIndex:
    """
    Build a PerturbationIndex from a list of perturbed trajectories.

    Args:
        perturbed_trajectories: List of perturbed trajectory dicts
        output_dir: Directory where files will be stored (for file paths)

    Returns:
        PerturbationIndex
    """
    index = PerturbationIndex()

    for pt in perturbed_trajectories:
        record_dict = pt.get("perturbation_record")
        if not record_dict:
            continue

        record = PerturbationRecord.from_dict(record_dict)

        # Get benchmark from original trajectory
        orig_traj = pt.get("original_trajectory", {})
        benchmark = orig_traj.get("benchmark", "unknown")

        # Determine file path
        file_path = f"{output_dir}/{benchmark}_perturbed.json"

        index.add_perturbation(record, benchmark, file_path)

    return index


def group_by_benchmark(
    perturbed_trajectories: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group perturbed trajectories by benchmark.

    Args:
        perturbed_trajectories: List of perturbed trajectory dicts

    Returns:
        Dict mapping benchmark name to list of perturbations
    """
    grouped = {}

    for pt in perturbed_trajectories:
        orig_traj = pt.get("original_trajectory", {})
        benchmark = orig_traj.get("benchmark", "unknown")

        if benchmark not in grouped:
            grouped[benchmark] = []
        grouped[benchmark].append(pt)

    return grouped


def filter_valid_perturbations(
    perturbed_trajectories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Filter to only valid perturbations (generation_status == 'valid').

    Args:
        perturbed_trajectories: List of perturbed trajectory dicts

    Returns:
        Filtered list
    """
    valid = []

    for pt in perturbed_trajectories:
        record = pt.get("perturbation_record", {})
        status = record.get("generation_status", "valid")
        if status == "valid":
            valid.append(pt)

    return valid
