"""
MongoDB storage backend for experiment results.

This module handles storing and retrieving all experiment data:
- Trajectories (original and perturbed)
- Annotations (human judgments)
- Judge evaluations (LLM ratings)
- CCG scores and analysis results
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError


class MongoDBStorage:
    """
    MongoDB storage for experiment data.

    Collections:
    - trajectories: Original and perturbed trajectories
    - annotations: Human annotations (TSD, SER)
    - judge_evaluations: LLM judge ratings
    - ccg_scores: Computed CCG scores
    - experiments: Experiment metadata and configurations
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Initialize MongoDB connection.

        Args:
            uri: MongoDB connection URI (default: from env MONGODB_URI)
            database: Database name (default: from env MONGODB_DATABASE)
        """
        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.database_name = database or os.getenv(
            "MONGODB_DATABASE",
            "agent_judge_experiment"
        )

        self.client = MongoClient(self.uri)
        self.db = self.client[self.database_name]

        # Collection references
        self.trajectories = self.db["trajectories"]
        self.annotations = self.db["annotations"]
        self.judge_evaluations = self.db["judge_evaluations"]
        self.ccg_scores = self.db["ccg_scores"]
        self.experiments = self.db["experiments"]

        # Create indexes for efficient queries
        self._create_indexes()

    def _create_indexes(self):
        """Create indexes on collections for efficient queries."""
        # Trajectories: indexed by ID and benchmark
        self.trajectories.create_index([("trajectory_id", ASCENDING)], unique=True)
        self.trajectories.create_index([("benchmark", ASCENDING)])
        self.trajectories.create_index([("is_perturbed", ASCENDING)])

        # Annotations: indexed by trajectory ID
        self.annotations.create_index([("trajectory_id", ASCENDING)])
        self.annotations.create_index([("annotator", ASCENDING)])

        # Judge evaluations: indexed by trajectory and judge
        self.judge_evaluations.create_index([
            ("trajectory_id", ASCENDING),
            ("judge_model", ASCENDING)
        ])

        # CCG scores: indexed by experiment and condition
        self.ccg_scores.create_index([("experiment_id", ASCENDING)])
        self.ccg_scores.create_index([
            ("perturbation_type", ASCENDING),
            ("perturbation_position", ASCENDING)
        ])

        # Experiments: indexed by ID and timestamp
        self.experiments.create_index([("experiment_id", ASCENDING)], unique=True)
        self.experiments.create_index([("created_at", DESCENDING)])

    def test_connection(self) -> bool:
        """
        Test MongoDB connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client.admin.command("ping")
            return True
        except ConnectionFailure:
            return False

    # === Trajectory Storage ===

    def save_trajectory(self, trajectory: Dict[str, Any]) -> str:
        """
        Save a trajectory (original or perturbed).

        Args:
            trajectory: Trajectory dict (from Trajectory.to_dict())

        Returns:
            Trajectory ID
        """
        trajectory["stored_at"] = datetime.utcnow()

        try:
            result = self.trajectories.insert_one(trajectory)
            return str(result.inserted_id)
        except DuplicateKeyError:
            # Update existing
            self.trajectories.replace_one(
                {"trajectory_id": trajectory["trajectory_id"]},
                trajectory
            )
            return trajectory["trajectory_id"]

    def save_trajectories_bulk(self, trajectories: List[Dict[str, Any]]) -> int:
        """
        Save multiple trajectories in bulk.

        Args:
            trajectories: List of trajectory dicts

        Returns:
            Number of trajectories saved
        """
        for traj in trajectories:
            traj["stored_at"] = datetime.utcnow()

        if not trajectories:
            return 0

        try:
            result = self.trajectories.insert_many(trajectories, ordered=False)
            return len(result.inserted_ids)
        except Exception as e:
            # Some might be duplicates, count successful inserts
            print(f"Warning: Bulk insert partially failed: {e}")
            return 0

    def get_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """Get trajectory by ID."""
        return self.trajectories.find_one({"trajectory_id": trajectory_id})

    def get_trajectories_by_benchmark(
        self,
        benchmark: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all trajectories from a specific benchmark."""
        cursor = self.trajectories.find({"benchmark": benchmark})
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def get_perturbed_trajectories(
        self,
        experiment_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all perturbed trajectories, optionally filtered by experiment."""
        query = {"is_perturbed": True}
        if experiment_id:
            query["experiment_id"] = experiment_id
        return list(self.trajectories.find(query))

    # === Annotation Storage ===

    def save_annotation(self, annotation: Dict[str, Any]) -> str:
        """
        Save a human annotation.

        Expected fields:
        - trajectory_id: str
        - annotator: str
        - task_success_degradation: float
        - subsequent_error_rate: float
        - subsequent_errors: List[Dict]
        - notes: str (optional)
        """
        annotation["annotated_at"] = datetime.utcnow()
        result = self.annotations.insert_one(annotation)
        return str(result.inserted_id)

    def get_annotation(
        self,
        trajectory_id: str,
        annotator: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get annotation for a trajectory."""
        query = {"trajectory_id": trajectory_id}
        if annotator:
            query["annotator"] = annotator
        return self.annotations.find_one(query)

    def get_all_annotations(
        self,
        experiment_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all annotations, optionally filtered by experiment."""
        query = {}
        if experiment_id:
            query["experiment_id"] = experiment_id
        return list(self.annotations.find(query))

    # === Judge Evaluation Storage ===

    def save_judge_evaluation(self, evaluation: Dict[str, Any]) -> str:
        """
        Save judge evaluation.

        Expected fields:
        - trajectory_id: str
        - judge_model: str (e.g., "claude-3.5-sonnet")
        - overall_score: float (0-100)
        - errors: List[Dict] (step errors with severity)
        - raw_response: str
        """
        evaluation["evaluated_at"] = datetime.utcnow()
        result = self.judge_evaluations.insert_one(evaluation)
        return str(result.inserted_id)

    def get_judge_evaluations(
        self,
        trajectory_id: str,
        judge_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get judge evaluations for a trajectory."""
        query = {"trajectory_id": trajectory_id}
        if judge_model:
            query["judge_model"] = judge_model
        return list(self.judge_evaluations.find(query))

    def get_all_evaluations(
        self,
        experiment_id: Optional[str] = None,
        judge_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all judge evaluations."""
        query = {}
        if experiment_id:
            query["experiment_id"] = experiment_id
        if judge_model:
            query["judge_model"] = judge_model
        return list(self.judge_evaluations.find(query))

    # === CCG Score Storage ===

    def save_ccg_score(self, ccg_data: Dict[str, Any]) -> str:
        """
        Save computed CCG score.

        Expected fields:
        - trajectory_id: str
        - experiment_id: str
        - perturbation_type: str
        - perturbation_position: str
        - tcs: float (True Criticality Score)
        - jps: float (Judge Penalty Score)
        - ccg: float (Criticality-Calibration Gap)
        - judge_model: str
        """
        ccg_data["computed_at"] = datetime.utcnow()
        result = self.ccg_scores.insert_one(ccg_data)
        return str(result.inserted_id)

    def get_ccg_scores(
        self,
        experiment_id: str,
        perturbation_type: Optional[str] = None,
        perturbation_position: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get CCG scores filtered by experiment and conditions."""
        query = {"experiment_id": experiment_id}
        if perturbation_type:
            query["perturbation_type"] = perturbation_type
        if perturbation_position:
            query["perturbation_position"] = perturbation_position
        return list(self.ccg_scores.find(query))

    def get_ccg_aggregates(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Compute aggregate CCG statistics by condition.

        Returns dict with structure:
        {
            "planning_early": {"mean_ccg": ..., "count": ...},
            "planning_middle": {...},
            ...
        }
        """
        pipeline = [
            {"$match": {"experiment_id": experiment_id}},
            {
                "$group": {
                    "_id": {
                        "type": "$perturbation_type",
                        "position": "$perturbation_position"
                    },
                    "mean_ccg": {"$avg": "$ccg"},
                    "std_ccg": {"$stdDevPop": "$ccg"},
                    "count": {"$sum": 1},
                    "mean_tcs": {"$avg": "$tcs"},
                    "mean_jps": {"$avg": "$jps"},
                }
            }
        ]

        results = list(self.ccg_scores.aggregate(pipeline))

        # Reformat to dict
        aggregates = {}
        for result in results:
            key = f"{result['_id']['type']}_{result['_id']['position']}"
            aggregates[key] = {
                "mean_ccg": result["mean_ccg"],
                "std_ccg": result.get("std_ccg", 0),
                "count": result["count"],
                "mean_tcs": result["mean_tcs"],
                "mean_jps": result["mean_jps"],
            }

        return aggregates

    # === Experiment Metadata ===

    def create_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """
        Create new experiment record.

        Args:
            experiment_config: Configuration dict with:
                - experiment_id: str
                - name: str
                - description: str
                - config: Dict (perturbations, judges, etc.)

        Returns:
            Experiment ID
        """
        experiment_config["created_at"] = datetime.utcnow()
        experiment_config["status"] = "created"

        try:
            result = self.experiments.insert_one(experiment_config)
            return experiment_config["experiment_id"]
        except DuplicateKeyError:
            raise ValueError(
                f"Experiment {experiment_config['experiment_id']} "
                "already exists"
            )

    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        progress: Optional[Dict[str, Any]] = None
    ):
        """Update experiment status and progress."""
        update = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        if progress:
            update["progress"] = progress

        self.experiments.update_one(
            {"experiment_id": experiment_id},
            {"$set": update}
        )

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        return self.experiments.find_one({"experiment_id": experiment_id})

    def list_experiments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent experiments."""
        return list(
            self.experiments.find()
            .sort("created_at", DESCENDING)
            .limit(limit)
        )

    def close(self):
        """Close MongoDB connection."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
