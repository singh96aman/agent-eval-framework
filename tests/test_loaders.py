"""
Tests for dataset loaders (ToolBench and GAIA).
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from src.data.loaders import (
    load_toolbench_trajectories,
    load_gaia_trajectories,
    save_trajectories,
    load_trajectories_from_json,
)
from src.data.schema import Trajectory


@pytest.fixture
def sample_toolbench_data():
    """Sample ToolBench trajectory data."""
    return {
        "query": "Find the population of Tokyo in 2023",
        "steps": [
            {
                "thought": "I need to search for Tokyo population data",
                "action": "Search",
                "action_input": {"query": "Tokyo population 2023"},
                "observation": "Tokyo has a population of approximately 14.09 million in 2023"
            },
            {
                "thought": "I found the answer",
                "action": "Finish",
                "action_input": {"answer": "14.09 million"},
                "observation": "Task completed"
            }
        ],
        "answer": "14.09 million",
        "success": True,
        "domain": "geography"
    }


@pytest.fixture
def sample_gaia_data():
    """Sample GAIA trajectory data."""
    return {
        "Question": "What is the capital of France?",
        "Final answer": "Paris",
        "Level": "Level 1"
    }


@pytest.fixture
def mock_toolbench_dataset(sample_toolbench_data):
    """Mock HuggingFace dataset for ToolBench."""
    dataset = Mock()
    dataset.__iter__ = Mock(return_value=iter([sample_toolbench_data] * 5))
    return dataset


@pytest.fixture
def mock_gaia_dataset(sample_gaia_data):
    """Mock HuggingFace dataset for GAIA."""
    dataset = Mock()
    dataset.__iter__ = Mock(return_value=iter([sample_gaia_data] * 5))
    return dataset


class TestToolBenchLoader:
    """Tests for ToolBench dataset loader."""

    @patch('datasets.load_dataset')
    def test_load_from_json(self, mock_load_dataset, mock_toolbench_dataset):
        """Test loading from HuggingFace dataset."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        trajectories = load_toolbench_trajectories(max_trajectories=10)

        assert len(trajectories) > 0
        traj = trajectories[0]
        assert isinstance(traj, Trajectory)
        assert traj.benchmark == "toolbench"
        assert len(traj.steps) == 2
        assert traj.ground_truth.task_description == "Find the population of Tokyo in 2023"
        assert traj.ground_truth.expected_answer == "14.09 million"
        assert traj.ground_truth.task_success is True

    @patch('datasets.load_dataset')
    def test_load_from_jsonl(self, mock_load_dataset, mock_toolbench_dataset):
        """Test loading from HuggingFace dataset."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        trajectories = load_toolbench_trajectories(max_trajectories=10)
        # Should load successfully
        assert len(trajectories) >= 1

    @patch('datasets.load_dataset')
    def test_filter_by_step_count(self, mock_load_dataset, mock_toolbench_dataset):
        """Test filtering by min/max steps."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        # Load with restrictive filter
        trajectories = load_toolbench_trajectories(
            min_steps=5,
            max_steps=10
        )
        # Sample has 2 steps, should be filtered out
        for traj in trajectories:
            assert 5 <= len(traj.steps) <= 10

    @patch('datasets.load_dataset')
    def test_filter_successful_only(self, mock_load_dataset, mock_toolbench_dataset):
        """Test filtering for successful trajectories only."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        trajectories = load_toolbench_trajectories(
            filter_successful=True
        )
        for traj in trajectories:
            assert traj.ground_truth.task_success is not False

    @patch('datasets.load_dataset')
    def test_max_trajectories_limit(self, mock_load_dataset, mock_toolbench_dataset):
        """Test limiting number of loaded trajectories."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        trajectories = load_toolbench_trajectories(
            max_trajectories=1
        )
        assert len(trajectories) <= 1

    @patch('datasets.load_dataset')
    def test_missing_directory(self, mock_load_dataset):
        """Test handling of dataset loading failure."""
        mock_load_dataset.side_effect = Exception("Dataset not found")

        # Should return empty list on failure
        trajectories = load_toolbench_trajectories()
        assert trajectories == []

    @patch('datasets.load_dataset')
    def test_empty_directory(self, mock_load_dataset):
        """Test handling of empty dataset."""
        empty_dataset = Mock()
        empty_dataset.__iter__ = Mock(return_value=iter([]))
        mock_load_dataset.return_value = empty_dataset

        trajectories = load_toolbench_trajectories()
        assert len(trajectories) == 0


class TestGAIALoader:
    """Tests for GAIA dataset loader."""

    @patch('datasets.load_dataset')
    def test_load_from_json(self, mock_load_dataset, mock_gaia_dataset):
        """Test loading from HuggingFace dataset."""
        mock_load_dataset.return_value = mock_gaia_dataset

        trajectories = load_gaia_trajectories(max_trajectories=10)

        assert len(trajectories) > 0
        traj = trajectories[0]
        assert isinstance(traj, Trajectory)
        assert traj.benchmark == "gaia"
        assert len(traj.steps) >= 1
        assert "capital of France" in traj.ground_truth.task_description
        assert traj.ground_truth.expected_answer == "Paris"
        assert traj.ground_truth.difficulty == "Level 1"

    @patch('datasets.load_dataset')
    def test_filter_by_difficulty(self, mock_load_dataset, mock_gaia_dataset):
        """Test filtering by difficulty level."""
        mock_load_dataset.return_value = mock_gaia_dataset

        trajectories = load_gaia_trajectories(
            difficulty="Level 1"
        )
        for traj in trajectories:
            assert traj.ground_truth.difficulty == "Level 1"

    @patch('datasets.load_dataset')
    def test_missing_directory(self, mock_load_dataset):
        """Test handling of dataset loading failure."""
        mock_load_dataset.side_effect = Exception("Dataset not found")

        # Should return empty list on failure
        trajectories = load_gaia_trajectories()
        assert trajectories == []


class TestTrajectorySchema:
    """Tests for Trajectory schema methods."""

    @patch('datasets.load_dataset')
    def test_trajectory_length(self, mock_load_dataset, mock_toolbench_dataset):
        """Test trajectory length."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        trajectories = load_toolbench_trajectories(max_trajectories=1)
        traj = trajectories[0]
        assert len(traj) == len(traj.steps)

    @patch('datasets.load_dataset')
    def test_get_step_by_number(
        self, mock_load_dataset, mock_toolbench_dataset
    ):
        """Test retrieving step by number."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        trajectories = load_toolbench_trajectories(max_trajectories=1)
        traj = trajectories[0]

        step = traj.get_step_by_number(1)
        assert step is not None
        assert step.step_number == 1

        # Non-existent step
        step = traj.get_step_by_number(999)
        assert step is None

    @patch('datasets.load_dataset')
    def test_get_position_label(
        self, mock_load_dataset, mock_toolbench_dataset
    ):
        """Test position labeling (early/middle/late)."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        trajectories = load_toolbench_trajectories(max_trajectories=1)
        traj = trajectories[0]

        assert traj.get_position_label(1) == "early"
        assert traj.get_position_label(2) == "early"
        assert traj.get_position_label(3) == "middle"
        assert traj.get_position_label(5) == "middle"
        assert traj.get_position_label(6) == "late"
        assert traj.get_position_label(10) == "late"

    @patch('datasets.load_dataset')
    def test_get_text_representation(
        self, mock_load_dataset, mock_toolbench_dataset
    ):
        """Test text representation generation."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        trajectories = load_toolbench_trajectories(max_trajectories=1)
        traj = trajectories[0]

        text = traj.get_text_representation()
        assert "Task:" in text
        assert "Tokyo" in text
        assert "Step 1" in text

    @patch('datasets.load_dataset')
    def test_to_dict_and_from_dict(
        self, mock_load_dataset, mock_toolbench_dataset
    ):
        """Test serialization and deserialization."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        trajectories = load_toolbench_trajectories(max_trajectories=1)
        original = trajectories[0]

        # Convert to dict
        data = original.to_dict()
        assert isinstance(data, dict)
        assert "trajectory_id" in data
        assert "steps" in data

        # Convert back to Trajectory
        restored = Trajectory.from_dict(data)
        assert restored.trajectory_id == original.trajectory_id
        assert len(restored.steps) == len(original.steps)
        assert restored.benchmark == original.benchmark


class TestSaveLoad:
    """Tests for saving and loading trajectories."""

    @patch('datasets.load_dataset')
    def test_save_and_load(self, mock_load_dataset, mock_toolbench_dataset):
        """Test saving trajectories to JSON and loading back."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        # Load original trajectories
        original_trajectories = load_toolbench_trajectories(
            max_trajectories=2
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            save_trajectories(original_trajectories, tmp_path)

            # Load back
            loaded_trajectories = load_trajectories_from_json(tmp_path)

            # Verify
            assert len(loaded_trajectories) == len(original_trajectories)
            for orig, loaded in zip(
                original_trajectories, loaded_trajectories
            ):
                assert orig.trajectory_id == loaded.trajectory_id
                assert len(orig.steps) == len(loaded.steps)

        finally:
            Path(tmp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
