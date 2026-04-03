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
        test_connection: bool = True,
    ):
        """
        Initialize MongoDB connection.

        Args:
            uri: MongoDB connection URI (default: from env MONGODB_URI)
            database: Database name (default: from env MONGODB_DATABASE)
            test_connection: Whether to test connection on init (default: True)

        Raises:
            ConnectionFailure: If test_connection=True and connection fails
        """
        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.database_name = database or os.getenv(
            "MONGODB_DATABASE",
            "agent_judge_experiment"
        )

        # Detect if using MongoDB Atlas
        self.is_atlas = "mongodb+srv://" in self.uri

        print(f"🔌 Connecting to MongoDB {'Atlas' if self.is_atlas else 'local'}...")
        print(f"   Database: {self.database_name}")

        self.client = MongoClient(self.uri)
        self.db = self.client[self.database_name]

        # Test connection
        if test_connection:
            if not self.test_connection():
                raise ConnectionFailure(
                    f"Failed to connect to MongoDB at {self.uri}"
                )
            print(f"✅ Connected to MongoDB {'Atlas' if self.is_atlas else 'local'}")

        # Collection references
        self.trajectories = self.db["trajectories"]
        self.perturbations = self.db["perturbations"]  # NEW!
        self.annotations = self.db["annotations"]
        self.judge_evaluations = self.db["judge_evaluations"]
        self.ccg_scores = self.db["ccg_scores"]
        self.experiments = self.db["experiments"]

        # Create indexes for efficient queries
        self._create_indexes()

    def _create_indexes(self):
        """
        Create indexes on collections for efficient queries.

        CRITICAL: Foreign key indexes for O(1) lookups and pagination.

        ARCHITECTURE:
        - Trajectories: Tagged with experiment_id for cleanup
        - Perturbations: Links experiment → original → perturbed
        - All downstream entities: Have experiment_id
        """
        # Trajectories: Tagged with experiment_id
        self.trajectories.create_index([
            ("trajectory_id", ASCENDING),
            ("experiment_id", ASCENDING)
        ], unique=True)
        self.trajectories.create_index([("experiment_id", ASCENDING)])
        self.trajectories.create_index([("benchmark", ASCENDING)])
        self.trajectories.create_index([("is_perturbed", ASCENDING)])
        self.trajectories.create_index([
            ("experiment_id", ASCENDING),
            ("is_perturbed", ASCENDING)
        ])

        # Perturbations: Experiment-scoped (NEW collection!)
        self.perturbations.create_index([("perturbation_id", ASCENDING)], unique=True)
        self.perturbations.create_index([("experiment_id", ASCENDING)])  # FK to experiments
        self.perturbations.create_index([("original_trajectory_id", ASCENDING)])  # FK to trajectories
        self.perturbations.create_index([("perturbed_trajectory_id", ASCENDING)])  # FK to trajectories
        self.perturbations.create_index([
            ("perturbation_type", ASCENDING),
            ("perturbation_position", ASCENDING)
        ])  # Compound index for analysis queries

        # Annotations: indexed by trajectory ID and experiment FK
        self.annotations.create_index([("annotation_id", ASCENDING)], unique=True)
        self.annotations.create_index([("experiment_id", ASCENDING)])  # FK to experiments
        self.annotations.create_index([("trajectory_id", ASCENDING)])  # FK to trajectories
        self.annotations.create_index([("annotator", ASCENDING)])
        self.annotations.create_index([
            ("experiment_id", ASCENDING),
            ("trajectory_id", ASCENDING)
        ])  # Compound index

        # Judge evaluations: indexed by trajectory, judge, and experiment FK
        self.judge_evaluations.create_index([("evaluation_id", ASCENDING)], unique=True)
        self.judge_evaluations.create_index([("experiment_id", ASCENDING)])  # FK to experiments
        self.judge_evaluations.create_index([("trajectory_id", ASCENDING)])  # FK to trajectories
        self.judge_evaluations.create_index([
            ("trajectory_id", ASCENDING),
            ("judge_model", ASCENDING)
        ])
        self.judge_evaluations.create_index([
            ("experiment_id", ASCENDING),
            ("judge_model", ASCENDING)
        ])  # Filter by experiment and judge

        # CCG scores: indexed by experiment and condition with FKs
        self.ccg_scores.create_index([("ccg_id", ASCENDING)], unique=True)
        self.ccg_scores.create_index([("experiment_id", ASCENDING)])  # FK to experiments
        self.ccg_scores.create_index([("trajectory_id", ASCENDING)])  # FK to trajectories
        self.ccg_scores.create_index([("annotation_id", ASCENDING)])  # FK to annotations
        self.ccg_scores.create_index([("evaluation_id", ASCENDING)])  # FK to evaluations
        self.ccg_scores.create_index([
            ("perturbation_type", ASCENDING),
            ("perturbation_position", ASCENDING)
        ])
        self.ccg_scores.create_index([("judge_model", ASCENDING)])

        # Experiments: indexed by ID and timestamp
        self.experiments.create_index([("experiment_id", ASCENDING)], unique=True)
        self.experiments.create_index([("created_at", DESCENDING)])
        self.experiments.create_index([("status", ASCENDING)])

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

    def get_trajectory_by_experiment(
        self,
        trajectory_id: str,
        experiment_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get trajectory by ID for a specific experiment."""
        return self.trajectories.find_one({
            "trajectory_id": trajectory_id,
            "experiment_id": experiment_id
        })

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
        benchmark: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all perturbed trajectories from cache.

        NOTE: To get perturbed trajectories for a specific experiment,
        use get_perturbations_by_experiment() instead.

        Args:
            benchmark: Filter by benchmark (optional)

        Returns:
            List of perturbed trajectory dicts
        """
        query = {"is_perturbed": True}
        if benchmark:
            query["benchmark"] = benchmark
        return list(self.trajectories.find(query))

    def get_trajectories_by_experiment(
        self,
        experiment_id: str,
        skip: int = 0,
        limit: Optional[int] = None,
        is_perturbed: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get trajectories for a specific experiment with pagination.

        NEW ARCHITECTURE: Queries via perturbations collection to find
        trajectories used in this experiment.

        Args:
            experiment_id: Experiment ID
            skip: Number of records to skip (for pagination)
            limit: Max number of records to return
            is_perturbed: Filter by perturbed status (optional)

        Returns:
            List of trajectory dicts
        """
        # Get perturbations for this experiment
        perturbations = list(self.perturbations.find({"experiment_id": experiment_id}))

        if not perturbations:
            return []

        # Extract trajectory IDs
        if is_perturbed is True:
            # Only perturbed
            trajectory_ids = [p["perturbed_trajectory_id"] for p in perturbations]
        elif is_perturbed is False:
            # Only originals (unique)
            trajectory_ids = list(set([p["original_trajectory_id"] for p in perturbations]))
        else:
            # Both originals and perturbed
            original_ids = [p["original_trajectory_id"] for p in perturbations]
            perturbed_ids = [p["perturbed_trajectory_id"] for p in perturbations]
            trajectory_ids = list(set(original_ids + perturbed_ids))

        # Get trajectories by IDs with pagination
        cursor = self.trajectories.find({
            "trajectory_id": {"$in": trajectory_ids}
        }).skip(skip)

        if limit:
            cursor = cursor.limit(limit)

        return list(cursor)

    def count_trajectories(
        self,
        experiment_id: Optional[str] = None,
        is_perturbed: Optional[bool] = None
    ) -> int:
        """
        Count trajectories matching criteria.

        NEW ARCHITECTURE: If experiment_id is provided, queries via
        perturbations collection.

        Args:
            experiment_id: Filter by experiment (optional)
            is_perturbed: Filter by perturbed status (optional)

        Returns:
            Count of matching trajectories
        """
        if experiment_id:
            # Query via perturbations
            perturbations = list(self.perturbations.find({"experiment_id": experiment_id}))

            if not perturbations:
                return 0

            # Extract trajectory IDs
            if is_perturbed is True:
                trajectory_ids = [p["perturbed_trajectory_id"] for p in perturbations]
            elif is_perturbed is False:
                trajectory_ids = list(set([p["original_trajectory_id"] for p in perturbations]))
            else:
                original_ids = [p["original_trajectory_id"] for p in perturbations]
                perturbed_ids = [p["perturbed_trajectory_id"] for p in perturbations]
                trajectory_ids = list(set(original_ids + perturbed_ids))

            return len(trajectory_ids)
        else:
            # Direct query on trajectories
            query = {}
            if is_perturbed is not None:
                query["is_perturbed"] = is_perturbed
            return self.trajectories.count_documents(query)

    # === Perturbation Storage (NEW!) ===

    def save_perturbation(self, perturbation: Dict[str, Any]) -> str:
        """
        Save a perturbation record linking experiment to trajectories.

        Expected fields:
        - perturbation_id: str (unique)
        - experiment_id: str (FK to experiments)
        - original_trajectory_id: str (FK to trajectories)
        - perturbed_trajectory_id: str (FK to trajectories)
        - perturbation_type: str (planning, tool_selection, parameter)
        - perturbation_position: str (early, middle, late)
        - perturbation_config: Dict (method, step_number, etc.)

        Returns:
            Perturbation ID
        """
        perturbation["created_at"] = datetime.utcnow()

        try:
            result = self.perturbations.insert_one(perturbation)
            return perturbation["perturbation_id"]
        except DuplicateKeyError:
            # Update existing
            self.perturbations.replace_one(
                {"perturbation_id": perturbation["perturbation_id"]},
                perturbation
            )
            return perturbation["perturbation_id"]

    def save_perturbations_bulk(self, perturbations: List[Dict[str, Any]]) -> int:
        """
        Save multiple perturbation records in bulk.

        Args:
            perturbations: List of perturbation dicts

        Returns:
            Number of perturbations saved
        """
        for pert in perturbations:
            pert["created_at"] = datetime.utcnow()

        if not perturbations:
            return 0

        try:
            result = self.perturbations.insert_many(perturbations, ordered=False)
            return len(result.inserted_ids)
        except Exception as e:
            print(f"Warning: Bulk perturbation insert partially failed: {e}")
            return 0

    def get_perturbation(self, perturbation_id: str) -> Optional[Dict[str, Any]]:
        """Get perturbation by ID."""
        return self.perturbations.find_one({"perturbation_id": perturbation_id})

    def get_perturbations_by_experiment(
        self,
        experiment_id: str,
        skip: int = 0,
        limit: Optional[int] = None,
        perturbation_type: Optional[str] = None,
        perturbation_position: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all perturbations for a specific experiment.

        Args:
            experiment_id: Experiment ID
            skip: Number of records to skip (for pagination)
            limit: Max number of records to return
            perturbation_type: Filter by type (optional)
            perturbation_position: Filter by position (optional)

        Returns:
            List of perturbation dicts
        """
        query = {"experiment_id": experiment_id}
        if perturbation_type:
            query["perturbation_type"] = perturbation_type
        if perturbation_position:
            query["perturbation_position"] = perturbation_position

        cursor = self.perturbations.find(query).skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def count_perturbations(
        self,
        experiment_id: Optional[str] = None,
        perturbation_type: Optional[str] = None
    ) -> int:
        """
        Count perturbations matching criteria.

        Args:
            experiment_id: Filter by experiment (optional)
            perturbation_type: Filter by type (optional)

        Returns:
            Count of matching perturbations
        """
        query = {}
        if experiment_id:
            query["experiment_id"] = experiment_id
        if perturbation_type:
            query["perturbation_type"] = perturbation_type
        return self.perturbations.count_documents(query)

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

    def get_annotations_by_experiment(
        self,
        experiment_id: str,
        skip: int = 0,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get annotations for a specific experiment with pagination.

        Uses foreign key index for O(1) lookup.

        Args:
            experiment_id: Experiment ID
            skip: Number of records to skip
            limit: Max number of records to return

        Returns:
            List of annotation dicts
        """
        cursor = self.annotations.find(
            {"experiment_id": experiment_id}
        ).skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def count_annotations(
        self,
        experiment_id: Optional[str] = None,
        trajectory_id: Optional[str] = None
    ) -> int:
        """
        Count annotations matching criteria.

        Args:
            experiment_id: Filter by experiment (optional)
            trajectory_id: Filter by trajectory (optional)

        Returns:
            Count of matching annotations
        """
        query = {}
        if experiment_id:
            query["experiment_id"] = experiment_id
        if trajectory_id:
            query["trajectory_id"] = trajectory_id
        return self.annotations.count_documents(query)

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

    def get_evaluations_by_experiment(
        self,
        experiment_id: str,
        skip: int = 0,
        limit: Optional[int] = None,
        judge_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get judge evaluations for a specific experiment with pagination.

        Uses foreign key index for O(1) lookup.

        Args:
            experiment_id: Experiment ID
            skip: Number of records to skip
            limit: Max number of records to return
            judge_model: Filter by judge model (optional)

        Returns:
            List of evaluation dicts
        """
        query = {"experiment_id": experiment_id}
        if judge_model:
            query["judge_model"] = judge_model

        cursor = self.judge_evaluations.find(query).skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def count_evaluations(
        self,
        experiment_id: Optional[str] = None,
        trajectory_id: Optional[str] = None,
        judge_model: Optional[str] = None
    ) -> int:
        """
        Count judge evaluations matching criteria.

        Args:
            experiment_id: Filter by experiment (optional)
            trajectory_id: Filter by trajectory (optional)
            judge_model: Filter by judge model (optional)

        Returns:
            Count of matching evaluations
        """
        query = {}
        if experiment_id:
            query["experiment_id"] = experiment_id
        if trajectory_id:
            query["trajectory_id"] = trajectory_id
        if judge_model:
            query["judge_model"] = judge_model
        return self.judge_evaluations.count_documents(query)

    def check_evaluation_cache(
        self,
        trajectory_id: str,
        judge_model: str,
        sample_number: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Check if judge evaluation exists in cache.

        This is the MOST IMPORTANT cache check - saves API costs!

        Args:
            trajectory_id: Trajectory ID
            judge_model: Judge model name
            sample_number: Sample number (for multiple samples)

        Returns:
            Existing evaluation dict or None if cache miss
        """
        return self.judge_evaluations.find_one({
            "trajectory_id": trajectory_id,
            "judge_model": judge_model,
            "sample_number": sample_number
        })

    # === Judge Output Storage (New Schema) ===

    def store_judge_output(
        self,
        judge_output: Any,  # JudgeOutput from src/judges/schema
        experiment_id: str
    ) -> str:
        """
        Store a JudgeOutput object from the new judge system.

        Args:
            judge_output: JudgeOutput object
            experiment_id: Experiment ID

        Returns:
            Inserted document ID
        """
        # Convert JudgeOutput to dict
        output_dict = judge_output.to_dict()
        output_dict["experiment_id"] = experiment_id

        result = self.judge_evaluations.insert_one(output_dict)
        return str(result.inserted_id)

    def get_judge_outputs(
        self,
        experiment_id: str,
        judge_name: Optional[str] = None,
        trajectory_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get judge outputs from the new judge system.

        Args:
            experiment_id: Experiment ID
            judge_name: Filter by judge name (optional)
            trajectory_id: Filter by trajectory (optional)

        Returns:
            List of judge output dicts
        """
        query = {"experiment_id": experiment_id}
        if judge_name:
            query["judge_name"] = judge_name
        if trajectory_id:
            query["trajectory_id"] = trajectory_id

        return list(self.judge_evaluations.find(query))

    def count_judge_outputs(
        self,
        experiment_id: str,
        trajectory_id: Optional[str] = None,
        judge_name: Optional[str] = None
    ) -> int:
        """
        Count judge outputs matching criteria.

        Args:
            experiment_id: Experiment ID
            trajectory_id: Filter by trajectory (optional)
            judge_name: Filter by judge name (optional)

        Returns:
            Count of matching outputs
        """
        query = {"experiment_id": experiment_id}
        if trajectory_id:
            query["trajectory_id"] = trajectory_id
        if judge_name:
            query["judge_name"] = judge_name

        return self.judge_evaluations.count_documents(query)

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
        perturbation_position: Optional[str] = None,
        skip: int = 0,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get CCG scores filtered by experiment and conditions with pagination.

        Args:
            experiment_id: Experiment ID
            perturbation_type: Filter by type (optional)
            perturbation_position: Filter by position (optional)
            skip: Number of records to skip
            limit: Max number of records to return

        Returns:
            List of CCG score dicts
        """
        query = {"experiment_id": experiment_id}
        if perturbation_type:
            query["perturbation_type"] = perturbation_type
        if perturbation_position:
            query["perturbation_position"] = perturbation_position

        cursor = self.ccg_scores.find(query).skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def count_ccg_scores(
        self,
        experiment_id: Optional[str] = None,
        perturbation_type: Optional[str] = None,
        perturbation_position: Optional[str] = None
    ) -> int:
        """
        Count CCG scores matching criteria.

        Args:
            experiment_id: Filter by experiment (optional)
            perturbation_type: Filter by type (optional)
            perturbation_position: Filter by position (optional)

        Returns:
            Count of matching CCG scores
        """
        query = {}
        if experiment_id:
            query["experiment_id"] = experiment_id
        if perturbation_type:
            query["perturbation_type"] = perturbation_type
        if perturbation_position:
            query["perturbation_position"] = perturbation_position
        return self.ccg_scores.count_documents(query)

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
        """
        Update experiment status and progress.

        Note: Progress dict should contain COUNTS only, not arrays of IDs!
        Example:
            {
                "trajectories_loaded": 50,
                "annotations_completed": 25,
                "evaluations_completed": 150
            }
        """
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

    def update_experiment_progress(
        self,
        experiment_id: str,
        **counts: int
    ):
        """
        Update experiment progress counts.

        Uses MongoDB $inc to atomically increment counts.
        Avoids race conditions in concurrent updates.

        Args:
            experiment_id: Experiment ID
            **counts: Keyword arguments with count names and increments
                Example: trajectories_loaded=5, annotations_completed=1
        """
        if not counts:
            return

        update = {
            f"progress.{key}": value
            for key, value in counts.items()
        }
        update["updated_at"] = datetime.utcnow()

        self.experiments.update_one(
            {"experiment_id": experiment_id},
            {
                "$set": {"updated_at": datetime.utcnow()},
                "$inc": {f"progress.{key}": value for key, value in counts.items()}
            }
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
