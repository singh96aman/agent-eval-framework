"""
Storage module for Section 6 Analysis Results.

Provides MongoDB operations for storing and retrieving per-unit analysis results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING
from pymongo.collection import Collection

from src.analysis.section6.schema import AnalysisResult
from src.storage.mongodb import MongoDBStorage


class Section6Storage:
    """
    MongoDB storage for Section 6 analysis results.

    Stores one document per (evaluation_unit_id, judge_model) pair in the
    'analysis_results' collection.
    """

    COLLECTION_NAME = "analysis_results"

    def __init__(self, storage: MongoDBStorage):
        """
        Initialize storage with existing MongoDB connection.

        Args:
            storage: MongoDBStorage instance
        """
        self.storage = storage
        self.collection: Collection = storage.db[self.COLLECTION_NAME]
        self._create_indexes()

    def _create_indexes(self):
        """Create indexes for efficient queries."""
        # Unique index on (evaluation_unit_id, judge_model) to prevent duplicates
        self.collection.create_index(
            [
                ("evaluation_unit_id", ASCENDING),
                ("judge_model", ASCENDING),
            ],
            unique=True,
        )

        # Experiment index for loading all results
        self.collection.create_index([("experiment_id", ASCENDING)])

        # Analysis ID index for direct lookups
        self.collection.create_index([("analysis_id", ASCENDING)], unique=True)

        # Compound indexes for stratified queries
        self.collection.create_index(
            [
                ("experiment_id", ASCENDING),
                ("judge_model", ASCENDING),
            ]
        )
        self.collection.create_index(
            [
                ("experiment_id", ASCENDING),
                ("ground_truth.perturbation_class", ASCENDING),
            ]
        )
        self.collection.create_index(
            [
                ("experiment_id", ASCENDING),
                ("ground_truth.benchmark", ASCENDING),
            ]
        )

    def save_result(self, result: AnalysisResult) -> str:
        """
        Save or update an analysis result.

        Uses upsert on (evaluation_unit_id, judge_model) to allow re-running.

        Args:
            result: AnalysisResult to save

        Returns:
            The analysis_id of the saved result
        """
        doc = result.to_dict()
        doc["stored_at"] = datetime.utcnow()

        self.collection.update_one(
            {
                "evaluation_unit_id": result.evaluation_unit_id,
                "judge_model": result.judge_model,
            },
            {"$set": doc},
            upsert=True,
        )

        return result.analysis_id

    def save_results(self, results: List[AnalysisResult]) -> int:
        """
        Save multiple analysis results.

        Args:
            results: List of AnalysisResult objects

        Returns:
            Number of results saved
        """
        count = 0
        for result in results:
            self.save_result(result)
            count += 1
        return count

    def load_result(
        self, evaluation_unit_id: str, judge_model: str
    ) -> Optional[AnalysisResult]:
        """
        Load a specific analysis result.

        Args:
            evaluation_unit_id: The evaluation unit ID
            judge_model: The judge model name

        Returns:
            AnalysisResult or None if not found
        """
        doc = self.collection.find_one(
            {
                "evaluation_unit_id": evaluation_unit_id,
                "judge_model": judge_model,
            }
        )

        if doc:
            return AnalysisResult.from_dict(doc)
        return None

    def load_results_by_experiment(
        self,
        experiment_id: str,
        judge_model: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AnalysisResult]:
        """
        Load all analysis results for an experiment.

        Args:
            experiment_id: The experiment ID
            judge_model: Optional filter by judge model
            limit: Optional limit on number of results

        Returns:
            List of AnalysisResult objects
        """
        query = {"experiment_id": experiment_id}
        if judge_model:
            query["judge_model"] = judge_model

        cursor = self.collection.find(query)
        if limit:
            cursor = cursor.limit(limit)

        return [AnalysisResult.from_dict(doc) for doc in cursor]

    def load_results_as_dicts(
        self,
        experiment_id: str,
        judge_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load all analysis results as dictionaries (for aggregation).

        Args:
            experiment_id: The experiment ID
            judge_model: Optional filter by judge model

        Returns:
            List of result dictionaries
        """
        query = {"experiment_id": experiment_id}
        if judge_model:
            query["judge_model"] = judge_model

        return list(self.collection.find(query))

    def count_results(
        self,
        experiment_id: str,
        judge_model: Optional[str] = None,
    ) -> int:
        """
        Count analysis results for an experiment.

        Args:
            experiment_id: The experiment ID
            judge_model: Optional filter by judge model

        Returns:
            Number of results
        """
        query = {"experiment_id": experiment_id}
        if judge_model:
            query["judge_model"] = judge_model

        return self.collection.count_documents(query)

    def get_judge_models(self, experiment_id: str) -> List[str]:
        """
        Get list of unique judge models for an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            List of judge model names
        """
        return self.collection.distinct("judge_model", {"experiment_id": experiment_id})

    def delete_results(
        self,
        experiment_id: str,
        judge_model: Optional[str] = None,
    ) -> int:
        """
        Delete analysis results for an experiment.

        Args:
            experiment_id: The experiment ID
            judge_model: Optional filter by judge model

        Returns:
            Number of results deleted
        """
        query = {"experiment_id": experiment_id}
        if judge_model:
            query["judge_model"] = judge_model

        result = self.collection.delete_many(query)
        return result.deleted_count

    def exists(self, evaluation_unit_id: str, judge_model: str) -> bool:
        """
        Check if an analysis result exists.

        Args:
            evaluation_unit_id: The evaluation unit ID
            judge_model: The judge model name

        Returns:
            True if result exists
        """
        return (
            self.collection.count_documents(
                {
                    "evaluation_unit_id": evaluation_unit_id,
                    "judge_model": judge_model,
                }
            )
            > 0
        )


def get_storage(database: str = "agent_judge_experiment") -> Section6Storage:
    """
    Get a Section6Storage instance.

    Args:
        database: Database name

    Returns:
        Section6Storage instance
    """
    mongo_storage = MongoDBStorage(database=database)
    return Section6Storage(mongo_storage)
