"""
Tests for MongoDB storage backend.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.storage.mongodb import MongoDBStorage


@pytest.fixture
def mock_mongo_client():
    """Mock MongoDB client."""
    with patch('src.storage.mongodb.MongoClient') as mock_client:
        # Mock the client and database
        mock_db = MagicMock()
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_client.return_value.admin.command.return_value = {"ok": 1}

        yield mock_client, mock_db


class TestMongoDBStorage:
    """Tests for MongoDBStorage class."""

    def test_initialization(self, mock_mongo_client):
        """Test storage initialization."""
        mock_client, mock_db = mock_mongo_client

        storage = MongoDBStorage(
            uri="mongodb://localhost:27017",
            database="test_db"
        )

        assert storage.uri == "mongodb://localhost:27017"
        assert storage.database_name == "test_db"

    def test_test_connection_success(self, mock_mongo_client):
        """Test successful connection test."""
        mock_client, mock_db = mock_mongo_client

        storage = MongoDBStorage()
        result = storage.test_connection()

        assert result is True

    def test_test_connection_failure(self, mock_mongo_client):
        """Test failed connection."""
        mock_client, mock_db = mock_mongo_client

        # Make ping command raise exception
        from pymongo.errors import ConnectionFailure
        mock_client.return_value.admin.command.side_effect = ConnectionFailure()

        storage = MongoDBStorage()
        result = storage.test_connection()

        assert result is False

    def test_save_trajectory(self, mock_mongo_client):
        """Test saving a trajectory."""
        mock_client, mock_db = mock_mongo_client

        storage = MongoDBStorage()

        trajectory = {
            "trajectory_id": "test_traj_1",
            "benchmark": "toolbench",
            "steps": [],
            "ground_truth": {}
        }

        # Mock insert result
        mock_result = MagicMock()
        mock_result.inserted_id = "mock_id_123"
        storage.trajectories.insert_one = MagicMock(return_value=mock_result)

        result = storage.save_trajectory(trajectory)

        assert result == "mock_id_123"
        storage.trajectories.insert_one.assert_called_once()

        # Check that stored_at was added
        call_args = storage.trajectories.insert_one.call_args[0][0]
        assert "stored_at" in call_args

    def test_get_trajectory(self, mock_mongo_client):
        """Test retrieving a trajectory."""
        mock_client, mock_db = mock_mongo_client

        storage = MongoDBStorage()

        expected_traj = {
            "trajectory_id": "test_traj_1",
            "benchmark": "toolbench"
        }

        storage.trajectories.find_one = MagicMock(return_value=expected_traj)

        result = storage.get_trajectory("test_traj_1")

        assert result == expected_traj
        storage.trajectories.find_one.assert_called_once_with(
            {"trajectory_id": "test_traj_1"}
        )

    def test_save_annotation(self, mock_mongo_client):
        """Test saving an annotation."""
        mock_client, mock_db = mock_mongo_client

        storage = MongoDBStorage()

        annotation = {
            "trajectory_id": "test_traj_1",
            "annotator": "human_1",
            "task_success_degradation": 1.0,
            "subsequent_error_rate": 0.5
        }

        mock_result = MagicMock()
        mock_result.inserted_id = "annotation_id_123"
        storage.annotations.insert_one = MagicMock(return_value=mock_result)

        result = storage.save_annotation(annotation)

        assert result == "annotation_id_123"

    def test_save_judge_evaluation(self, mock_mongo_client):
        """Test saving a judge evaluation."""
        mock_client, mock_db = mock_mongo_client

        storage = MongoDBStorage()

        evaluation = {
            "trajectory_id": "test_traj_1",
            "judge_model": "claude-3.5-sonnet",
            "overall_score": 85.0,
            "errors": []
        }

        mock_result = MagicMock()
        mock_result.inserted_id = "eval_id_123"
        storage.judge_evaluations.insert_one = MagicMock(return_value=mock_result)

        result = storage.save_judge_evaluation(evaluation)

        assert result == "eval_id_123"

    def test_save_ccg_score(self, mock_mongo_client):
        """Test saving CCG score."""
        mock_client, mock_db = mock_mongo_client

        storage = MongoDBStorage()

        ccg_data = {
            "trajectory_id": "test_traj_1",
            "experiment_id": "exp_1",
            "perturbation_type": "planning",
            "perturbation_position": "early",
            "tcs": 100.0,
            "jps": 30.0,
            "ccg": -0.7,
            "judge_model": "claude-3.5-sonnet"
        }

        mock_result = MagicMock()
        mock_result.inserted_id = "ccg_id_123"
        storage.ccg_scores.insert_one = MagicMock(return_value=mock_result)

        result = storage.save_ccg_score(ccg_data)

        assert result == "ccg_id_123"

    def test_create_experiment(self, mock_mongo_client):
        """Test creating an experiment."""
        mock_client, mock_db = mock_mongo_client

        storage = MongoDBStorage()

        experiment = {
            "experiment_id": "exp_poc_1",
            "name": "POC Experiment",
            "description": "Initial POC with 50 trajectories",
            "config": {}
        }

        mock_result = MagicMock()
        mock_result.inserted_id = "exp_mongo_id"
        storage.experiments.insert_one = MagicMock(return_value=mock_result)

        result = storage.create_experiment(experiment)

        assert result == "exp_poc_1"

    def test_context_manager(self, mock_mongo_client):
        """Test using storage as context manager."""
        mock_client, mock_db = mock_mongo_client

        with MongoDBStorage() as storage:
            assert storage is not None

        # Verify close was called
        mock_client.return_value.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
