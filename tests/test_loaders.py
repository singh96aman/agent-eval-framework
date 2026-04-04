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
    load_swebench_trajectories,
    save_trajectories,
    load_trajectories_from_json,
    classify_trajectory_domain,
    classify_trajectory_complexity,
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

    def test_load_from_local_file(self):
        """Test loading from local file (if available)."""
        # This tests actual local file loading (uses real data if available)
        trajectories = load_toolbench_trajectories(max_trajectories=5)

        if len(trajectories) > 0:
            traj = trajectories[0]
            assert isinstance(traj, Trajectory)
            assert traj.benchmark == "toolbench"
            assert len(traj.steps) >= 1
            assert traj.ground_truth is not None

    @patch('src.data.loaders.json.load')
    @patch('builtins.open', create=True)
    @patch('pathlib.Path.exists')
    def test_load_from_json_mock(self, mock_exists, mock_open, mock_json_load, sample_toolbench_data):
        """Test loading from local JSON file with mocked data."""
        # Mock local file existence and content
        mock_exists.return_value = True
        mock_json_load.return_value = [sample_toolbench_data] * 5

        trajectories = load_toolbench_trajectories(max_trajectories=10)

        assert len(trajectories) > 0
        traj = trajectories[0]
        assert isinstance(traj, Trajectory)
        assert traj.benchmark == "toolbench"

    def test_load_trajectories_basic(self):
        """Test basic trajectory loading functionality."""
        trajectories = load_toolbench_trajectories(max_trajectories=3)
        # Should load at least some trajectories
        assert isinstance(trajectories, list)
        if len(trajectories) > 0:
            assert all(isinstance(t, Trajectory) for t in trajectories)

    def test_filter_by_step_count(self):
        """Test filtering by min/max steps."""
        # Load with restrictive filter
        trajectories = load_toolbench_trajectories(
            max_trajectories=100,
            min_steps=5,
            max_steps=10
        )
        # All returned trajectories should have 5-10 steps
        for traj in trajectories:
            assert 5 <= len(traj.steps) <= 10

    def test_filter_successful_only(self):
        """Test filtering for successful trajectories only."""
        trajectories = load_toolbench_trajectories(
            max_trajectories=50,
            filter_successful=True
        )
        for traj in trajectories:
            assert traj.ground_truth.task_success is not False

    def test_max_trajectories_limit(self):
        """Test limiting number of loaded trajectories."""
        trajectories = load_toolbench_trajectories(max_trajectories=3)
        assert len(trajectories) <= 3

    @patch('src.data.loaders.json.load')
    @patch('builtins.open', create=True)
    @patch('pathlib.Path.exists')
    def test_missing_directory(self, mock_exists, mock_open, mock_json_load):
        """Test handling when local file doesn't exist and HF fails."""
        # Mock local file doesn't exist
        mock_exists.return_value = False

        # Mock HuggingFace to raise exception
        with patch('datasets.load_dataset', side_effect=Exception("Dataset not found")):
            trajectories = load_toolbench_trajectories(local_path="/nonexistent")
            # Should return empty list on failure
            assert trajectories == []

    @patch('src.data.loaders.json.load')
    @patch('builtins.open', create=True)
    @patch('pathlib.Path.exists')
    def test_empty_dataset(self, mock_exists, mock_open, mock_json_load):
        """Test handling of empty dataset."""
        mock_exists.return_value = True
        mock_json_load.return_value = []

        trajectories = load_toolbench_trajectories(max_trajectories=10)
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

    def test_trajectory_length(self):
        """Test trajectory length."""
        trajectories = load_toolbench_trajectories(max_trajectories=1)
        if len(trajectories) > 0:
            traj = trajectories[0]
            assert len(traj) == len(traj.steps)

    def test_get_step_by_number(self):
        """Test retrieving step by number."""
        trajectories = load_toolbench_trajectories(max_trajectories=1)
        if len(trajectories) > 0:
            traj = trajectories[0]

            step = traj.get_step_by_number(1)
            assert step is not None
            assert step.step_number == 1

            # Non-existent step
            step = traj.get_step_by_number(999)
            assert step is None

    def test_get_position_label(self):
        """Test position labeling (early/middle/late)."""
        trajectories = load_toolbench_trajectories(max_trajectories=1)
        if len(trajectories) > 0:
            traj = trajectories[0]

            # Position labels are relative to trajectory length
            assert traj.get_position_label(1) == "early"
            # Additional position tests depend on trajectory length
            if len(traj.steps) >= 6:
                assert traj.get_position_label(6) in ["middle", "late"]

    def test_get_text_representation(self):
        """Test text representation generation."""
        trajectories = load_toolbench_trajectories(max_trajectories=1)
        if len(trajectories) > 0:
            traj = trajectories[0]

            text = traj.get_text_representation()
            assert "Task:" in text
            assert "Step 1" in text
            # Should contain step content
            assert len(text) > 50

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        trajectories = load_toolbench_trajectories(max_trajectories=1)
        if len(trajectories) > 0:
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


class TestSWEBenchLoader:
    """Tests for SWE-bench dataset loader.

    NOTE: These tests use mocks to avoid downloading actual datasets.
    The mocks patch 'datasets.load_dataset' to prevent any network calls.
    """

    @pytest.fixture
    def sample_swebench_data(self):
        """Sample SWE-bench/SWE-smith trajectory data (messages format)."""
        import json
        # SWE-smith format uses 'messages' field with role/content structure
        messages = [
            {"role": "user", "content": "Fix the bug in utils.py"},
            {"role": "assistant", "content": "I'll search for the bug location in the codebase."},
            {"role": "assistant", "content": "I need to edit the file to fix the issue."},
        ]
        return {
            "instance_id": "test_repo__12345",
            "messages": json.dumps(messages),
            "patch": "diff --git a/utils.py...",
            "resolved": True,
            "traj_id": "traj_12345",
            "repo": "test/repo"
        }

    def test_swebench_missing_dataset_graceful(self):
        """Test that missing dataset returns empty list gracefully."""
        # This test doesn't need mocking - it tests the error handling
        # by using an invalid token that will cause auth failure
        with patch('datasets.load_dataset') as mock_load:
            mock_load.side_effect = Exception("Dataset not found")
            trajectories = load_swebench_trajectories()
            assert trajectories == []

    def test_parse_swebench_item_directly(self, sample_swebench_data):
        """Test parsing SWE-bench item without loading dataset."""
        from src.data.loaders import _parse_swebench_item

        traj = _parse_swebench_item(sample_swebench_data, idx=0)

        assert traj is not None
        assert traj.benchmark == "swebench"
        assert "swebench" in traj.trajectory_id or "traj_" in traj.trajectory_id
        assert len(traj.steps) >= 1
        assert traj.ground_truth.task_success is True

    def test_parse_swebench_messages_format(self, sample_swebench_data):
        """Test parsing SWE-bench messages format."""
        from src.data.loaders import _parse_swebench_item

        traj = _parse_swebench_item(sample_swebench_data, idx=0)

        assert traj is not None
        # Should have steps from assistant messages
        assert len(traj.steps) == 2  # Two assistant messages
        assert traj.ground_truth.task_description == "Fix the bug in utils.py"


class TestDomainClassification:
    """Tests for trajectory domain classification."""

    def test_classify_toolbench_domain(self):
        """Test classifying ToolBench trajectory domains."""
        from src.data.schema import Step, StepType, GroundTruth

        # Create trajectory with weather-related tool
        traj = Trajectory(
            trajectory_id="test_1",
            benchmark="toolbench",
            steps=[
                Step(
                    step_id="s1",
                    step_number=1,
                    step_type=StepType.TOOL_EXECUTION,
                    content="Get weather",
                    tool_name="get_weather_forecast",
                    tool_input={}
                )
            ],
            ground_truth=GroundTruth(task_description="Get weather")
        )

        domain = classify_trajectory_domain(traj)
        assert domain == "data_information"

    def test_classify_complexity(self):
        """Test classifying trajectory complexity."""
        from src.data.schema import Step, StepType, GroundTruth

        # Simple trajectory (<=4 steps)
        simple_traj = Trajectory(
            trajectory_id="simple",
            benchmark="toolbench",
            steps=[
                Step(
                    step_id=f"s{i}",
                    step_number=i,
                    step_type=StepType.TOOL_EXECUTION,
                    content="step"
                ) for i in range(1, 4)
            ],
            ground_truth=GroundTruth(task_description="Simple task")
        )
        assert classify_trajectory_complexity(simple_traj) == "simple"

        # Medium trajectory (5-6 steps)
        medium_traj = Trajectory(
            trajectory_id="medium",
            benchmark="toolbench",
            steps=[
                Step(
                    step_id=f"s{i}",
                    step_number=i,
                    step_type=StepType.TOOL_EXECUTION,
                    content="step"
                ) for i in range(1, 6)
            ],
            ground_truth=GroundTruth(task_description="Medium task")
        )
        assert classify_trajectory_complexity(medium_traj) == "medium"

        # Complex trajectory (7+ steps)
        complex_traj = Trajectory(
            trajectory_id="complex",
            benchmark="toolbench",
            steps=[
                Step(
                    step_id=f"s{i}",
                    step_number=i,
                    step_type=StepType.TOOL_EXECUTION,
                    content="step"
                ) for i in range(1, 9)
            ],
            ground_truth=GroundTruth(task_description="Complex task")
        )
        assert classify_trajectory_complexity(complex_traj) == "complex"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
