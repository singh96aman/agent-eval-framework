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
    compute_quality_metrics,
    apply_quality_filters,
    deduplicate_trajectories,
    clean_trajectory_steps,
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
                "observation": "Tokyo has a population of approximately 14.09 million in 2023",
            },
            {
                "thought": "I found the answer",
                "action": "Finish",
                "action_input": {"answer": "14.09 million"},
                "observation": "Task completed",
            },
        ],
        "answer": "14.09 million",
        "success": True,
        "domain": "geography",
    }


@pytest.fixture
def sample_gaia_data():
    """Sample GAIA trajectory data."""
    return {
        "Question": "What is the capital of France?",
        "Final answer": "Paris",
        "Level": "Level 1",
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

    @patch("src.data.loaders.json.load")
    @patch("builtins.open", create=True)
    @patch("pathlib.Path.exists")
    def test_load_from_json_mock(
        self, mock_exists, mock_open, mock_json_load, sample_toolbench_data
    ):
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
            max_trajectories=100, min_steps=5, max_steps=10
        )
        # All returned trajectories should have 5-10 steps
        for traj in trajectories:
            assert 5 <= len(traj.steps) <= 10

    def test_filter_successful_only(self):
        """Test filtering for successful trajectories only."""
        trajectories = load_toolbench_trajectories(
            max_trajectories=50, filter_successful=True
        )
        for traj in trajectories:
            assert traj.ground_truth.task_success is not False

    def test_max_trajectories_limit(self):
        """Test limiting number of loaded trajectories."""
        trajectories = load_toolbench_trajectories(max_trajectories=3)
        assert len(trajectories) <= 3

    @patch("src.data.loaders.json.load")
    @patch("builtins.open", create=True)
    @patch("pathlib.Path.exists")
    def test_missing_directory(self, mock_exists, mock_open, mock_json_load):
        """Test handling when local file doesn't exist and HF fails."""
        # Mock local file doesn't exist
        mock_exists.return_value = False

        # Mock HuggingFace to raise exception
        with patch("datasets.load_dataset", side_effect=Exception("Dataset not found")):
            trajectories = load_toolbench_trajectories(local_path="/nonexistent")
            # Should return empty list on failure
            assert trajectories == []

    @patch("src.data.loaders.json.load")
    @patch("builtins.open", create=True)
    @patch("pathlib.Path.exists")
    def test_empty_dataset(self, mock_exists, mock_open, mock_json_load):
        """Test handling of empty dataset."""
        mock_exists.return_value = True
        mock_json_load.return_value = []

        trajectories = load_toolbench_trajectories(max_trajectories=10)
        assert len(trajectories) == 0


class TestGAIALoader:
    """Tests for GAIA dataset loader."""

    @patch("datasets.load_dataset")
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

    @patch("datasets.load_dataset")
    def test_filter_by_difficulty(self, mock_load_dataset, mock_gaia_dataset):
        """Test filtering by difficulty level."""
        mock_load_dataset.return_value = mock_gaia_dataset

        trajectories = load_gaia_trajectories(difficulty="Level 1")
        for traj in trajectories:
            assert traj.ground_truth.difficulty == "Level 1"

    @patch("datasets.load_dataset")
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

    @patch("datasets.load_dataset")
    def test_save_and_load(self, mock_load_dataset, mock_toolbench_dataset):
        """Test saving trajectories to JSON and loading back."""
        mock_load_dataset.return_value = mock_toolbench_dataset

        # Load original trajectories
        original_trajectories = load_toolbench_trajectories(max_trajectories=2)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
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
            {
                "role": "assistant",
                "content": "I'll search for the bug location in the codebase.",
            },
            {
                "role": "assistant",
                "content": "I need to edit the file to fix the issue.",
            },
        ]
        return {
            "instance_id": "test_repo__12345",
            "messages": json.dumps(messages),
            "patch": "diff --git a/utils.py...",
            "resolved": True,
            "traj_id": "traj_12345",
            "repo": "test/repo",
        }

    def test_swebench_missing_dataset_graceful(self):
        """Test that missing dataset returns empty list gracefully."""
        # This test doesn't need mocking - it tests the error handling
        # by using an invalid token that will cause auth failure
        with patch("datasets.load_dataset") as mock_load:
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
                    tool_input={},
                )
            ],
            ground_truth=GroundTruth(task_description="Get weather"),
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
                    content="step",
                )
                for i in range(1, 4)
            ],
            ground_truth=GroundTruth(task_description="Simple task"),
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
                    content="step",
                )
                for i in range(1, 6)
            ],
            ground_truth=GroundTruth(task_description="Medium task"),
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
                    content="step",
                )
                for i in range(1, 9)
            ],
            ground_truth=GroundTruth(task_description="Complex task"),
        )
        assert classify_trajectory_complexity(complex_traj) == "complex"


class TestSamplingProvenance:
    """Tests for SamplingProvenance dataclass."""

    def test_provenance_creation(self):
        """Test creating SamplingProvenance."""
        from src.data.schema import SamplingProvenance

        provenance = SamplingProvenance(
            sampled_at="2026-04-07T12:00:00Z",
            sampling_seed=42,
            source_dataset="toolbench_huggingface",
            source_index=100,
            filter_criteria={"min_steps": 3, "max_steps": 15},
        )

        assert provenance.sampled_at == "2026-04-07T12:00:00Z"
        assert provenance.sampling_seed == 42
        assert provenance.source_dataset == "toolbench_huggingface"
        assert provenance.source_index == 100
        assert provenance.filter_criteria["min_steps"] == 3

    def test_provenance_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        from src.data.schema import SamplingProvenance

        original = SamplingProvenance(
            sampled_at="2026-04-07T12:00:00Z",
            sampling_seed=42,
            source_dataset="gaia_huggingface",
            source_index=50,
            filter_criteria={"min_steps": 2},
            loader_version="1.0.0",
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = SamplingProvenance.from_dict(data)

        assert restored.sampled_at == original.sampled_at
        assert restored.sampling_seed == original.sampling_seed
        assert restored.source_dataset == original.source_dataset
        assert restored.loader_version == original.loader_version


class TestSamplingManifest:
    """Tests for SamplingManifest dataclass."""

    def test_manifest_creation(self):
        """Test creating SamplingManifest."""
        from src.data.schema import SamplingManifest

        manifest = SamplingManifest(
            experiment_id="exp_test_001",
            created_at="2026-04-07T12:00:00Z",
            seed=42,
            config={"datasets": {"toolbench": {"enabled": True}}},
        )

        assert manifest.experiment_id == "exp_test_001"
        assert manifest.seed == 42
        assert manifest.trajectory_ids == []
        assert manifest.rejection_log == []

    def test_manifest_add_rejection(self):
        """Test adding rejections to manifest."""
        from src.data.schema import SamplingManifest

        manifest = SamplingManifest(
            experiment_id="exp_test_001",
            created_at="2026-04-07T12:00:00Z",
            seed=42,
            config={},
        )

        manifest.add_rejection(
            "traj_001", "step_count_too_low", {"steps": 1, "min_required": 3}
        )

        assert len(manifest.rejection_log) == 1
        assert manifest.rejection_log[0]["trajectory_id"] == "traj_001"
        assert manifest.rejection_log[0]["reason"] == "step_count_too_low"

    def test_manifest_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        from src.data.schema import SamplingManifest

        original = SamplingManifest(
            experiment_id="exp_test_001",
            created_at="2026-04-07T12:00:00Z",
            seed=42,
            config={"key": "value"},
            trajectory_ids=["t1", "t2", "t3"],
            counts={"total": 3, "by_benchmark": {"toolbench": 3}},
        )

        data = original.to_dict()
        restored = SamplingManifest.from_dict(data)

        assert restored.experiment_id == original.experiment_id
        assert restored.seed == original.seed
        assert restored.trajectory_ids == original.trajectory_ids
        assert restored.counts == original.counts


class TestTrajectoryWithProvenance:
    """Tests for Trajectory with provenance field."""

    def test_trajectory_with_provenance(self):
        """Test Trajectory with provenance field."""
        from src.data.schema import SamplingProvenance, Step, StepType, GroundTruth

        provenance = SamplingProvenance(
            sampled_at="2026-04-07T12:00:00Z",
            sampling_seed=42,
            source_dataset="toolbench_huggingface",
            source_index=100,
        )

        traj = Trajectory(
            trajectory_id="test_001",
            benchmark="toolbench",
            steps=[
                Step(
                    step_id="s1",
                    step_number=1,
                    step_type=StepType.TOOL_EXECUTION,
                    content="Test step",
                )
            ],
            ground_truth=GroundTruth(task_description="Test task"),
            provenance=provenance,
        )

        assert traj.provenance is not None
        assert traj.provenance.sampling_seed == 42

    def test_trajectory_to_dict_with_provenance(self):
        """Test Trajectory serialization includes provenance."""
        from src.data.schema import SamplingProvenance, Step, StepType, GroundTruth

        provenance = SamplingProvenance(
            sampled_at="2026-04-07T12:00:00Z",
            sampling_seed=42,
            source_dataset="gaia_huggingface",
            source_index=50,
        )

        traj = Trajectory(
            trajectory_id="test_002",
            benchmark="gaia",
            steps=[
                Step(
                    step_id="s1",
                    step_number=1,
                    step_type=StepType.REASONING,
                    content="Reasoning step",
                )
            ],
            ground_truth=GroundTruth(task_description="GAIA task"),
            provenance=provenance,
        )

        data = traj.to_dict()
        assert "provenance" in data
        assert data["provenance"]["sampling_seed"] == 42

        # Restore and verify
        restored = Trajectory.from_dict(data)
        assert restored.provenance is not None
        assert restored.provenance.source_dataset == "gaia_huggingface"

    def test_trajectory_without_provenance_backward_compat(self):
        """Test backward compatibility for trajectories without provenance."""
        # Old format data without provenance
        data = {
            "trajectory_id": "old_001",
            "benchmark": "toolbench",
            "steps": [
                {
                    "step_id": "s1",
                    "step_number": 1,
                    "step_type": "tool_execution",
                    "content": "Old step",
                    "tool_name": None,
                    "tool_input": None,
                    "tool_output": None,
                    "metadata": {},
                }
            ],
            "ground_truth": {
                "task_description": "Old task",
                "expected_answer": None,
                "task_success": None,
                "success_criteria": "exact_match",
                "difficulty": None,
                "domain": None,
            },
            "metadata": {},
        }

        traj = Trajectory.from_dict(data)
        assert traj.provenance is None
        assert traj.trajectory_id == "old_001"


class TestStratifiedSampling:
    """Tests for load_stratified_sample function."""

    def test_stratified_sample_basic_config(self):
        """Test stratified sampling with basic config."""
        from src.data.loaders import load_stratified_sample

        config = {
            "datasets": {
                "toolbench": {
                    "enabled": True,
                    "source": "huggingface",
                    "limit": 10,
                    "filters": {"min_steps": 3, "max_steps": 15},
                },
            },
            "sampling": {
                "enabled": True,
                "seed": 42,
                "targets": {"toolbench": 5},
            },
            "validation": {"enabled": False},
            "provenance": {"seed": 42},
        }

        trajectories, manifest = load_stratified_sample(
            config=config,
            experiment_id="test_stratified_001",
        )

        assert len(trajectories) <= 5
        assert manifest.experiment_id == "test_stratified_001"
        assert manifest.seed == 42
        assert len(manifest.trajectory_ids) == len(trajectories)

    def test_stratified_sample_adds_provenance(self):
        """Test that stratified sampling adds provenance to trajectories."""
        from src.data.loaders import load_stratified_sample

        config = {
            "datasets": {
                "toolbench": {
                    "enabled": True,
                    "source": "huggingface",
                    "limit": 5,
                    "filters": {"min_steps": 3, "max_steps": 15},
                },
            },
            "sampling": {"enabled": False},
            "validation": {"enabled": False},
            "provenance": {"seed": 123},
        }

        trajectories, manifest = load_stratified_sample(
            config=config,
            experiment_id="test_provenance_001",
        )

        if len(trajectories) > 0:
            # Check that provenance was added
            traj = trajectories[0]
            assert traj.provenance is not None
            assert traj.provenance.sampling_seed == 123
            assert "toolbench" in traj.provenance.source_dataset


class TestQualityFiltersV2:
    """Tests for v2 quality filters (null outcomes, Finish step, deduplication)."""

    def _make_trajectory(
        self,
        trajectory_id: str,
        steps: list,
        expected_answer=None,
        task_success=None,
        task_description="Test task",
    ) -> Trajectory:
        """Helper to create a test trajectory."""
        from src.data.schema import Step, StepType, GroundTruth

        parsed_steps = []
        for i, step in enumerate(steps, 1):
            parsed_steps.append(
                Step(
                    step_id=f"step_{i}",
                    step_number=i,
                    step_type=StepType.TOOL_EXECUTION,
                    content=step.get("content", ""),
                    tool_name=step.get("tool_name"),
                    tool_input=step.get("tool_input"),
                    tool_output=step.get("tool_output"),
                )
            )

        return Trajectory(
            trajectory_id=trajectory_id,
            benchmark="toolbench",
            steps=parsed_steps,
            ground_truth=GroundTruth(
                task_description=task_description,
                expected_answer=expected_answer,
                task_success=task_success,
            ),
        )

    def test_null_outcome_detection_expected_answer_null(self):
        """Test that null expected_answer is detected."""
        traj = self._make_trajectory(
            "test_1",
            [{"content": "Step 1", "tool_name": "Finish"}],
            expected_answer=None,
            task_success=True,
        )
        metrics = compute_quality_metrics(traj)
        assert metrics["has_null_outcome"] is True

    def test_null_outcome_detection_task_success_null(self):
        """Test that null task_success is detected."""
        traj = self._make_trajectory(
            "test_2",
            [{"content": "Step 1", "tool_name": "Finish"}],
            expected_answer="Valid answer",
            task_success=None,
        )
        metrics = compute_quality_metrics(traj)
        assert metrics["has_null_outcome"] is True

    def test_null_outcome_detection_empty_answer(self):
        """Test that empty expected_answer is detected."""
        traj = self._make_trajectory(
            "test_3",
            [{"content": "Step 1", "tool_name": "Finish"}],
            expected_answer="   ",  # Whitespace only
            task_success=True,
        )
        metrics = compute_quality_metrics(traj)
        assert metrics["has_null_outcome"] is True

    def test_valid_outcome(self):
        """Test that valid outcome is not flagged."""
        traj = self._make_trajectory(
            "test_4",
            [{"content": "Step 1", "tool_name": "Finish"}],
            expected_answer="Valid answer",
            task_success=True,
        )
        metrics = compute_quality_metrics(traj)
        assert metrics["has_null_outcome"] is False

    def test_finish_step_detection(self):
        """Test that Finish step presence is detected."""
        traj_with_finish = self._make_trajectory(
            "test_5",
            [
                {"content": "Step 1", "tool_name": "Search"},
                {"content": "Step 2", "tool_name": "Finish"},
            ],
            expected_answer="Answer",
            task_success=True,
        )
        metrics = compute_quality_metrics(traj_with_finish)
        assert metrics["has_finish_step"] is True

    def test_missing_finish_step_detection(self):
        """Test that missing Finish step is detected."""
        traj_no_finish = self._make_trajectory(
            "test_6",
            [
                {"content": "Step 1", "tool_name": "Search"},
                {"content": "Step 2", "tool_name": "Analyze"},
            ],
            expected_answer="Answer",
            task_success=True,
        )
        metrics = compute_quality_metrics(traj_no_finish)
        assert metrics["has_finish_step"] is False

    def test_apply_filters_rejects_null_outcome(self):
        """Test that apply_quality_filters rejects null outcomes."""
        traj = self._make_trajectory(
            "test_7",
            [
                {
                    "content": "Step 1",
                    "tool_name": "Finish",
                    "tool_input": {"answer": "test"},
                }
            ],
            expected_answer=None,
            task_success=True,
        )
        quality_config = {
            "enabled": True,
            "thresholds": {
                "reject_null_outcomes": True,
                "max_missing_tool_input_rate": 1.0,  # Disable this filter for test
                "use_strict_filter": False,  # Disable strict filter to test null_outcome
            },
        }
        passes, reason = apply_quality_filters(traj, quality_config, "toolbench")
        assert passes is False
        assert "null_outcome" in reason

    def test_apply_filters_rejects_missing_finish(self):
        """Test that apply_quality_filters rejects missing Finish for ToolBench."""
        traj = self._make_trajectory(
            "test_8",
            [
                {
                    "content": "Step 1",
                    "tool_name": "Search",
                    "tool_input": {"query": "test"},
                }
            ],
            expected_answer="Answer",
            task_success=True,
        )
        quality_config = {
            "enabled": True,
            "thresholds": {
                "require_finish_step": True,
                "reject_null_outcomes": False,
                "max_missing_tool_input_rate": 1.0,  # Disable this filter for test
                "use_strict_filter": False,  # Disable strict filter to test missing_finish_step
            },
        }
        passes, reason = apply_quality_filters(traj, quality_config, "toolbench")
        assert passes is False
        assert "missing_finish_step" in reason

    def test_apply_filters_allows_missing_finish_for_gaia(self):
        """Test that missing Finish is allowed for GAIA (not required)."""
        traj = self._make_trajectory(
            "test_9",
            [
                {
                    "content": "Step 1",
                    "tool_name": "Browse",
                    "tool_input": {"url": "test.com"},
                }
            ],
            expected_answer="Answer",
            task_success=True,
        )
        traj.benchmark = "gaia"
        quality_config = {
            "enabled": True,
            "thresholds": {
                "require_finish_step": True,
                "reject_null_outcomes": False,
                "max_missing_tool_input_rate": 1.0,  # Disable this filter for test
            },
        }
        # Should pass because require_finish_step only applies to toolbench
        passes, reason = apply_quality_filters(traj, quality_config, "gaia")
        assert passes is True

    def test_deduplication_removes_duplicates(self):
        """Test that deduplication removes trajectories with same task."""
        traj1 = self._make_trajectory(
            "traj_1",
            [{"content": "S1"}],
            task_description="Same task",
            expected_answer="A",
            task_success=True,
        )
        traj2 = self._make_trajectory(
            "traj_2",
            [{"content": "S2"}],
            task_description="Same task",
            expected_answer="B",
            task_success=True,
        )
        traj3 = self._make_trajectory(
            "traj_3",
            [{"content": "S3"}],
            task_description="Different task",
            expected_answer="C",
            task_success=True,
        )

        trajectories = [traj1, traj2, traj3]
        deduped = deduplicate_trajectories(trajectories)

        assert len(deduped) == 2
        assert deduped[0].trajectory_id == "traj_1"  # First one kept
        assert deduped[1].trajectory_id == "traj_3"

    def test_deduplication_normalizes_whitespace(self):
        """Test that deduplication normalizes whitespace."""
        traj1 = self._make_trajectory(
            "traj_1",
            [{"content": "S1"}],
            task_description="Same  task",
            expected_answer="A",
            task_success=True,
        )
        traj2 = self._make_trajectory(
            "traj_2",
            [{"content": "S2"}],
            task_description="Same task",
            expected_answer="B",
            task_success=True,
        )

        trajectories = [traj1, traj2]
        deduped = deduplicate_trajectories(trajectories)

        assert len(deduped) == 1

    def test_deduplication_logs_to_manifest(self):
        """Test that deduplication logs rejections to manifest."""
        from src.data.schema import SamplingManifest

        traj1 = self._make_trajectory(
            "traj_1",
            [{"content": "S1"}],
            task_description="Same task",
            expected_answer="A",
            task_success=True,
        )
        traj2 = self._make_trajectory(
            "traj_2",
            [{"content": "S2"}],
            task_description="Same task",
            expected_answer="B",
            task_success=True,
        )

        manifest = SamplingManifest(
            experiment_id="test",
            created_at="2026-04-07T00:00:00Z",
            seed=42,
            config={},
        )
        deduplicate_trajectories([traj1, traj2], manifest)

        assert len(manifest.rejection_log) == 1
        assert manifest.rejection_log[0]["reason"] == "duplicate_task"
        assert manifest.rejection_log[0]["trajectory_id"] == "traj_2"

    def test_combined_errors_filter(self):
        """Test combined HTTP errors + Thought:None filter (toolbench_19354)."""
        # Trajectory with BOTH http error AND thought_none
        traj = self._make_trajectory(
            "test_combined",
            [
                {
                    "content": "Step 1",
                    "tool_name": "api_call",
                    "tool_input": {"q": "test"},
                    "tool_output": "Error 403 Forbidden",
                },
                {
                    "content": "Thought: None",
                    "tool_name": "Finish",
                    "tool_input": {"answer": "done"},
                },
            ],
            expected_answer="Answer",
            task_success=True,
        )
        quality_config = {
            "enabled": True,
            "thresholds": {
                "reject_combined_errors": True,
                "max_missing_tool_input_rate": 1.0,
                "reject_null_outcomes": False,
                "max_http_error_rate": 1.0,  # Would pass alone
                "max_thought_none": 10,  # Would pass alone
                "use_strict_filter": False,  # Disable strict filter to test combined_errors
            },
        }
        passes, reason = apply_quality_filters(traj, quality_config, "toolbench")
        assert passes is False
        assert "combined_errors" in reason

    def test_step_cleaning_removes_empty_steps(self):
        """Test that step cleaning removes empty steps."""
        traj = self._make_trajectory(
            "test_clean",
            [
                {"content": "Valid step 1"},
                {"content": ""},  # Empty
                {"content": "  "},  # Whitespace only
                {"content": "Valid step 2"},
            ],
            expected_answer="Answer",
            task_success=True,
        )
        cleaning_config = {
            "enabled": True,
            "remove_empty_steps": True,
            "remove_thought_none_steps": False,
            "normalize_null_tool_steps": False,
        }
        clean_trajectory_steps(traj, cleaning_config)

        assert len(traj.steps) == 2
        assert traj.steps[0].content == "Valid step 1"
        assert traj.steps[1].content == "Valid step 2"
        # Check renumbering
        assert traj.steps[0].step_number == 1
        assert traj.steps[1].step_number == 2

    def test_step_cleaning_removes_thought_none(self):
        """Test that step cleaning removes Thought: None steps."""
        traj = self._make_trajectory(
            "test_clean_thought",
            [
                {"content": "Valid step"},
                {"content": "Thought: None"},
                {"content": "Another valid step"},
            ],
            expected_answer="Answer",
            task_success=True,
        )
        cleaning_config = {
            "enabled": True,
            "remove_empty_steps": False,
            "remove_thought_none_steps": True,
            "normalize_null_tool_steps": False,
        }
        clean_trajectory_steps(traj, cleaning_config)

        assert len(traj.steps) == 2
        assert "Thought: None" not in [s.content for s in traj.steps]

    def test_step_cleaning_normalizes_null_tool_steps(self):
        """Test that null tool_name steps are normalized to REASONING."""
        from src.data.schema import StepType

        traj = self._make_trajectory(
            "test_normalize",
            [
                {"content": "Click on result", "tool_name": None},
                {"content": "Valid tool step", "tool_name": "search"},
            ],
            expected_answer="Answer",
            task_success=True,
        )
        # Set first step to TOOL_EXECUTION (simulating GAIA pseudo-step)
        traj.steps[0].step_type = StepType.TOOL_EXECUTION

        cleaning_config = {
            "enabled": True,
            "remove_empty_steps": False,
            "remove_thought_none_steps": False,
            "normalize_null_tool_steps": True,
        }
        clean_trajectory_steps(traj, cleaning_config)

        # First step should be normalized to REASONING
        assert traj.steps[0].step_type == StepType.REASONING
        # Second step should remain TOOL_EXECUTION
        assert traj.steps[1].step_type == StepType.TOOL_EXECUTION

    def test_grounding_score_calculation(self):
        """Test grounding score calculation for GAIA trajectories."""
        # Trajectory with 2 tool steps, only 1 grounded (has payload)
        traj = self._make_trajectory(
            "test_grounding",
            [
                {
                    "content": "Search",
                    "tool_name": "web_search",
                    "tool_input": {"query": "test"},
                    "tool_output": "results",
                },
                {
                    "content": "Click result",
                    "tool_name": "click",
                    "tool_input": None,
                    "tool_output": None,
                },
            ],
            expected_answer="Answer",
            task_success=True,
        )
        metrics = compute_quality_metrics(traj)
        # 1 grounded out of 2 tool steps = 0.5
        assert metrics["grounding_score"] == 0.5

    def test_grounding_filter_rejects_low_grounding(self):
        """Test that GAIA trajectories with low grounding are rejected."""
        traj = self._make_trajectory(
            "test_low_grounding",
            [
                {"content": "Click", "tool_name": "click"},  # No payload
                {"content": "Navigate", "tool_name": "navigate"},  # No payload
            ],
            expected_answer="Answer",
            task_success=True,
        )
        traj.benchmark = "gaia"
        quality_config = {
            "enabled": True,
            "thresholds": {
                "min_grounding_score": 0.3,
                "reject_null_outcomes": False,
                "max_missing_tool_input_rate": 1.0,
            },
        }
        passes, reason = apply_quality_filters(traj, quality_config, "gaia")
        assert passes is False
        assert "low_grounding" in reason

    def test_graceful_failure_detection(self):
        """Test detection of graceful failure (apology answer with success)."""
        traj = self._make_trajectory(
            "test_graceful",
            [
                {"content": "Step 1", "tool_name": "api", "tool_input": {"q": "test"}},
                {
                    "content": "Done",
                    "tool_name": "Finish",
                    "tool_input": {"answer": "done"},
                },
            ],
            expected_answer="Please try again later",
            task_success=True,
        )
        metrics = compute_quality_metrics(traj)
        assert metrics["is_graceful_failure"] is True

    def test_graceful_failure_filter_rejects(self):
        """Test that graceful failures are rejected (toolbench_87039 style)."""
        traj = self._make_trajectory(
            "test_graceful_reject",
            [
                {
                    "content": "I will search for the requested data using the API",
                    "tool_name": "api",
                    "tool_input": {"q": "test"},
                },
                {
                    "content": "Based on the API response, I will provide the answer",
                    "tool_name": "Finish",
                    "tool_input": {"answer": "done"},
                },
            ],
            expected_answer="Sorry, I could not retrieve the data",
            task_success=True,
        )
        quality_config = {
            "enabled": True,
            "thresholds": {
                "reject_graceful_failures": True,
                "reject_null_outcomes": False,
                "max_missing_tool_input_rate": 1.0,
                "max_empty_content_rate": 1.0,  # Disable this filter for test
                "use_strict_filter": False,  # Disable strict filter to test graceful_failure
            },
        }
        passes, reason = apply_quality_filters(traj, quality_config, "toolbench")
        assert passes is False
        assert "graceful_failure" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
