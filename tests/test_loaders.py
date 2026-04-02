"""
Tests for dataset loaders (ToolBench and GAIA).
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.data.loaders import (
    load_toolbench_trajectories,
    load_gaia_trajectories,
    save_trajectories,
    load_trajectories_from_json,
)
from src.data.schema import Trajectory, Step, StepType, GroundTruth


@pytest.fixture
def sample_toolbench_data():
    """Sample ToolBench trajectory data."""
    return {
        "task": "Find the population of Tokyo in 2023",
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
        "final_answer": "14.09 million",
        "success": True,
        "domain": "geography"
    }


@pytest.fixture
def sample_gaia_data():
    """Sample GAIA trajectory data."""
    return {
        "question": "What is the capital of France?",
        "final_answer": "Paris",
        "Level": "Level 1",
        "trajectory": [
            {
                "type": "search",
                "query": "capital of France",
                "result": "Paris is the capital of France"
            },
            {
                "type": "answer",
                "content": "Paris"
            }
        ]
    }


@pytest.fixture
def temp_toolbench_dir(sample_toolbench_data):
    """Create temporary directory with ToolBench data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create JSON file
        json_file = tmpdir / "sample.json"
        with open(json_file, "w") as f:
            json.dump([sample_toolbench_data], f)

        # Create JSONL file
        jsonl_file = tmpdir / "sample.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps(sample_toolbench_data) + "\n")

        yield str(tmpdir)


@pytest.fixture
def temp_gaia_dir(sample_gaia_data):
    """Create temporary directory with GAIA data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create JSON file
        json_file = tmpdir / "sample.json"
        with open(json_file, "w") as f:
            json.dump([sample_gaia_data], f)

        yield str(tmpdir)


class TestToolBenchLoader:
    """Tests for ToolBench dataset loader."""

    def test_load_from_json(self, temp_toolbench_dir):
        """Test loading from JSON file."""
        trajectories = load_toolbench_trajectories(temp_toolbench_dir, max_trajectories=10)

        assert len(trajectories) > 0
        traj = trajectories[0]
        assert isinstance(traj, Trajectory)
        assert traj.benchmark == "toolbench"
        assert len(traj.steps) == 2
        assert traj.ground_truth.task_description == "Find the population of Tokyo in 2023"
        assert traj.ground_truth.expected_answer == "14.09 million"
        assert traj.ground_truth.task_success is True

    def test_load_from_jsonl(self, temp_toolbench_dir):
        """Test loading from JSONL file."""
        trajectories = load_toolbench_trajectories(temp_toolbench_dir, max_trajectories=10)
        # Should load from both JSON and JSONL files
        assert len(trajectories) >= 1

    def test_filter_by_step_count(self, temp_toolbench_dir):
        """Test filtering by min/max steps."""
        # Load with restrictive filter
        trajectories = load_toolbench_trajectories(
            temp_toolbench_dir,
            min_steps=5,
            max_steps=10
        )
        # Sample has 2 steps, should be filtered out
        for traj in trajectories:
            assert 5 <= len(traj.steps) <= 10

    def test_filter_successful_only(self, temp_toolbench_dir):
        """Test filtering for successful trajectories only."""
        trajectories = load_toolbench_trajectories(
            temp_toolbench_dir,
            filter_successful=True
        )
        for traj in trajectories:
            assert traj.ground_truth.task_success is not False

    def test_max_trajectories_limit(self, temp_toolbench_dir):
        """Test limiting number of loaded trajectories."""
        trajectories = load_toolbench_trajectories(
            temp_toolbench_dir,
            max_trajectories=1
        )
        assert len(trajectories) <= 1

    def test_missing_directory(self):
        """Test handling of missing directory."""
        with pytest.raises(FileNotFoundError):
            load_toolbench_trajectories("/nonexistent/path")

    def test_empty_directory(self):
        """Test handling of empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No JSON/JSONL files"):
                load_toolbench_trajectories(tmpdir)


class TestGAIALoader:
    """Tests for GAIA dataset loader."""

    def test_load_from_json(self, temp_gaia_dir):
        """Test loading from JSON file."""
        trajectories = load_gaia_trajectories(temp_gaia_dir, max_trajectories=10)

        assert len(trajectories) > 0
        traj = trajectories[0]
        assert isinstance(traj, Trajectory)
        assert traj.benchmark == "gaia"
        assert len(traj.steps) >= 1
        assert traj.ground_truth.task_description == "What is the capital of France?"
        assert traj.ground_truth.expected_answer == "Paris"
        assert traj.ground_truth.difficulty == "Level 1"

    def test_filter_by_difficulty(self, temp_gaia_dir):
        """Test filtering by difficulty level."""
        trajectories = load_gaia_trajectories(
            temp_gaia_dir,
            difficulty="Level 1"
        )
        for traj in trajectories:
            assert traj.ground_truth.difficulty == "Level 1"

    def test_missing_directory(self):
        """Test handling of missing directory."""
        with pytest.raises(FileNotFoundError):
            load_gaia_trajectories("/nonexistent/path")


class TestTrajectorySchema:
    """Tests for Trajectory schema methods."""

    def test_trajectory_length(self, temp_toolbench_dir):
        """Test trajectory length."""
        trajectories = load_toolbench_trajectories(temp_toolbench_dir, max_trajectories=1)
        traj = trajectories[0]
        assert len(traj) == len(traj.steps)

    def test_get_step_by_number(self, temp_toolbench_dir):
        """Test retrieving step by number."""
        trajectories = load_toolbench_trajectories(temp_toolbench_dir, max_trajectories=1)
        traj = trajectories[0]

        step = traj.get_step_by_number(1)
        assert step is not None
        assert step.step_number == 1

        # Non-existent step
        step = traj.get_step_by_number(999)
        assert step is None

    def test_get_position_label(self, temp_toolbench_dir):
        """Test position labeling (early/middle/late)."""
        trajectories = load_toolbench_trajectories(temp_toolbench_dir, max_trajectories=1)
        traj = trajectories[0]

        assert traj.get_position_label(1) == "early"
        assert traj.get_position_label(2) == "early"
        assert traj.get_position_label(3) == "middle"
        assert traj.get_position_label(5) == "middle"
        assert traj.get_position_label(6) == "late"
        assert traj.get_position_label(10) == "late"

    def test_get_text_representation(self, temp_toolbench_dir):
        """Test text representation generation."""
        trajectories = load_toolbench_trajectories(temp_toolbench_dir, max_trajectories=1)
        traj = trajectories[0]

        text = traj.get_text_representation()
        assert "Task:" in text
        assert "Tokyo" in text
        assert "Step 1" in text

    def test_to_dict_and_from_dict(self, temp_toolbench_dir):
        """Test serialization and deserialization."""
        trajectories = load_toolbench_trajectories(temp_toolbench_dir, max_trajectories=1)
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

    def test_save_and_load(self, temp_toolbench_dir):
        """Test saving trajectories to JSON and loading back."""
        # Load original trajectories
        original_trajectories = load_toolbench_trajectories(
            temp_toolbench_dir,
            max_trajectories=2
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_trajectories(original_trajectories, tmp_path)

            # Load back
            loaded_trajectories = load_trajectories_from_json(tmp_path)

            # Verify
            assert len(loaded_trajectories) == len(original_trajectories)
            for orig, loaded in zip(original_trajectories, loaded_trajectories):
                assert orig.trajectory_id == loaded.trajectory_id
                assert len(orig.steps) == len(loaded.steps)

        finally:
            Path(tmp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
