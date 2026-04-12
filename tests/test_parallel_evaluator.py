"""
Tests for ParallelJudgeEvaluator.

Tests:
- Parallelization runs concurrent evaluations
- Checkpoint saves periodically
- Resume skips already-evaluated perturbations
- Error handling continues batch on failure
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time
from concurrent.futures import Future

from src.judges.parallel_evaluator import ParallelJudgeEvaluator
from src.judges.schema import JudgeOutput
from src.data.schema import Trajectory, Step, GroundTruth, StepType


@pytest.fixture
def mock_judge():
    """Create a mock judge that returns fake evaluations."""
    judge = Mock()
    judge.name = "test-judge"
    judge.model_id = "test-model"

    def mock_evaluate(trajectory, **kwargs):
        output = Mock(spec=JudgeOutput)
        output.trajectory_id = trajectory.trajectory_id
        output.judge_name = "test-judge"
        output.model_id = "test-model"
        output.overall_score = 75.0
        output.task_success = 1
        output.completeness = 80.0
        output.efficiency_errors = 0
        output.hallucination = 0
        output.sycophancy = 0
        output.step_errors = []
        output.reasoning = "Test reasoning"
        output.tokens_used = 100
        output.evaluation_time_ms = 500
        output.to_dict = Mock(
            return_value={
                "trajectory_id": trajectory.trajectory_id,
                "judge_name": "test-judge",
                "model_id": "test-model",
                "overall_score": 75.0,
                "task_success": 1,
                "completeness": 80.0,
                "efficiency_errors": 0,
                "hallucination": 0,
                "sycophancy": 0,
                "step_errors": [],
                "reasoning": "Test reasoning",
            }
        )
        return output

    judge.evaluate = mock_evaluate
    return judge


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = Mock()

    # Track stored outputs
    storage._stored_outputs = []
    storage._trajectories = {}

    def mock_get_trajectory(traj_id, exp_id):
        return storage._trajectories.get(traj_id)

    def mock_count_outputs(experiment_id, trajectory_id=None, judge_name=None):
        return 0  # Nothing evaluated yet

    def mock_store_output(output, exp_id, sample_number=1):
        storage._stored_outputs.append(output)
        return "stored_id"

    storage.get_trajectory_by_experiment = mock_get_trajectory
    storage.count_judge_outputs = mock_count_outputs
    storage.store_judge_output = mock_store_output

    return storage


@pytest.fixture
def sample_perturbations():
    """Create sample perturbation records."""
    return [
        {
            "perturbation_id": f"pert_{i}",
            "original_trajectory_id": f"orig_{i}",
            "perturbed_trajectory_id": f"perturbed_{i}",
            "perturbation_type": "planning",
            "perturbation_position": "early",
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_trajectories(sample_perturbations):
    """Create sample trajectory dicts."""
    trajectories = {}
    for pert in sample_perturbations:
        traj_id = pert["perturbed_trajectory_id"]
        trajectories[traj_id] = {
            "trajectory_id": traj_id,
            "benchmark": "toolbench",
            "steps": [
                {
                    "step_id": "step_1",
                    "step_number": 1,
                    "step_type": "planning",
                    "content": "Test step",
                }
            ],
            "ground_truth": {"task_description": "Test task"},
            "metadata": {},
        }
    return trajectories


class TestParallelJudgeEvaluator:
    """Tests for ParallelJudgeEvaluator."""

    def test_init_with_config(self, mock_judge, mock_storage):
        """Test initialization with config."""
        config = {
            "judge_parallelization": 3,
            "checkpoint_batch_size": 10,
            "rate_limit_delay_seconds": 0.1,
        }

        evaluator = ParallelJudgeEvaluator(mock_judge, mock_storage, config)

        assert evaluator.parallelization == 3
        assert evaluator.checkpoint_size == 10
        assert evaluator.rate_limit_delay == 0.1

    def test_init_with_defaults(self, mock_judge, mock_storage):
        """Test initialization with default values."""
        evaluator = ParallelJudgeEvaluator(mock_judge, mock_storage, {})

        assert evaluator.parallelization == 2  # Default
        assert evaluator.checkpoint_size == 20  # Default
        assert evaluator.rate_limit_delay == 0.5  # Default

    def test_evaluate_all_processes_all_perturbations(
        self, mock_judge, mock_storage, sample_perturbations, sample_trajectories
    ):
        """Test that all perturbations are processed."""
        # Setup storage with trajectories
        mock_storage._trajectories = sample_trajectories

        config = {
            "judge_parallelization": 2,
            "checkpoint_batch_size": 5,
            "rate_limit_delay_seconds": 0.01,
        }

        evaluator = ParallelJudgeEvaluator(mock_judge, mock_storage, config)

        results = evaluator.evaluate_all(
            sample_perturbations, experiment_id="test_exp", resume=False
        )

        # Should have results for all perturbations
        assert len(results) == len(sample_perturbations)

        # All should be successful (trajectories exist)
        successful = [r for r in results if r.get("status") == "success"]
        assert len(successful) == len(sample_perturbations)

    def test_resume_skips_evaluated(
        self, mock_judge, mock_storage, sample_perturbations, sample_trajectories
    ):
        """Test that resume mode skips already-evaluated perturbations."""
        mock_storage._trajectories = sample_trajectories

        # Mock that first 5 perturbations are already evaluated
        def mock_count(experiment_id, trajectory_id=None, judge_name=None):
            if trajectory_id in [
                "perturbed_0",
                "perturbed_1",
                "perturbed_2",
                "perturbed_3",
                "perturbed_4",
            ]:
                return 1
            return 0

        mock_storage.count_judge_outputs = mock_count

        config = {"judge_parallelization": 2, "rate_limit_delay_seconds": 0.01}
        evaluator = ParallelJudgeEvaluator(mock_judge, mock_storage, config)

        results = evaluator.evaluate_all(
            sample_perturbations, experiment_id="test_exp", resume=True
        )

        # Should only process the 5 that weren't evaluated
        assert len(results) == 5

    def test_error_handling_continues_batch(
        self, mock_judge, mock_storage, sample_perturbations, sample_trajectories
    ):
        """Test that failed evaluations don't block the batch."""
        # Only some trajectories exist
        mock_storage._trajectories = {
            k: v
            for k, v in sample_trajectories.items()
            if k
            in [
                "perturbed_0",
                "perturbed_2",
                "perturbed_4",
                "perturbed_6",
                "perturbed_8",
            ]
        }

        config = {"judge_parallelization": 2, "rate_limit_delay_seconds": 0.01}
        evaluator = ParallelJudgeEvaluator(mock_judge, mock_storage, config)

        results = evaluator.evaluate_all(
            sample_perturbations, experiment_id="test_exp", resume=False
        )

        # Should have results for all (some failed)
        assert len(results) == len(sample_perturbations)

        # 5 should succeed, 5 should fail
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]
        assert len(successful) == 5
        assert len(failed) == 5

    def test_checkpoint_saves_results(
        self, mock_judge, mock_storage, sample_perturbations, sample_trajectories
    ):
        """Test that checkpointing saves results to storage."""
        mock_storage._trajectories = sample_trajectories

        config = {
            "judge_parallelization": 2,
            "checkpoint_batch_size": 5,
            "rate_limit_delay_seconds": 0.01,
        }

        evaluator = ParallelJudgeEvaluator(mock_judge, mock_storage, config)

        results = evaluator.evaluate_all(
            sample_perturbations, experiment_id="test_exp", resume=False
        )

        # Should have stored all successful evaluations
        assert len(mock_storage._stored_outputs) == 10

    def test_jps_calculation(
        self, mock_judge, mock_storage, sample_perturbations, sample_trajectories
    ):
        """Test that JPS is correctly calculated as 100 - overall_score."""
        mock_storage._trajectories = sample_trajectories

        config = {"judge_parallelization": 2, "rate_limit_delay_seconds": 0.01}
        evaluator = ParallelJudgeEvaluator(mock_judge, mock_storage, config)

        results = evaluator.evaluate_all(
            sample_perturbations[:1], experiment_id="test_exp", resume=False  # Just one
        )

        # overall_score is 75, so JPS should be 25
        assert results[0]["jps"] == 25.0
        assert results[0]["overall_score"] == 75.0


class TestParallelizationBehavior:
    """Tests for parallelization-specific behavior."""

    def test_parallelization_respects_limit(self, mock_storage):
        """Test that parallelization doesn't exceed configured limit."""
        # Track concurrent execution count
        max_concurrent = [0]
        current_concurrent = [0]

        def slow_evaluate(trajectory, **kwargs):
            current_concurrent[0] += 1
            max_concurrent[0] = max(max_concurrent[0], current_concurrent[0])
            time.sleep(0.1)  # Simulate API call
            current_concurrent[0] -= 1

            output = Mock()
            output.overall_score = 75.0
            output.trajectory_id = trajectory.trajectory_id
            output.judge_name = "test-judge"
            output.model_id = "test-model"
            output.task_success = 1
            output.completeness = 80.0
            output.efficiency_errors = 0
            output.hallucination = 0
            output.sycophancy = 0
            output.step_errors = []
            output.reasoning = "Test"
            output.tokens_used = 100
            output.evaluation_time_ms = 100
            output.to_dict = Mock(
                return_value={
                    "trajectory_id": trajectory.trajectory_id,
                    "judge_name": "test-judge",
                    "model_id": "test-model",
                    "overall_score": 75.0,
                    "task_success": 1,
                    "completeness": 80.0,
                    "efficiency_errors": 0,
                    "hallucination": 0,
                    "sycophancy": 0,
                    "step_errors": [],
                    "reasoning": "Test",
                }
            )
            return output

        mock_judge = Mock()
        mock_judge.name = "test-judge"
        mock_judge.evaluate = slow_evaluate

        # Setup
        perturbations = [
            {"perturbation_id": f"p{i}", "perturbed_trajectory_id": f"t{i}"}
            for i in range(6)
        ]

        trajectories = {
            f"t{i}": {
                "trajectory_id": f"t{i}",
                "benchmark": "test",
                "steps": [
                    {
                        "step_id": "s1",
                        "step_number": 1,
                        "step_type": "planning",
                        "content": "test",
                    }
                ],
                "ground_truth": {"task_description": "test"},
            }
            for i in range(6)
        }
        mock_storage._trajectories = trajectories
        mock_storage.count_judge_outputs = Mock(return_value=0)
        mock_storage.store_judge_output = Mock()

        config = {"judge_parallelization": 2, "rate_limit_delay_seconds": 0.01}

        evaluator = ParallelJudgeEvaluator(mock_judge, mock_storage, config)
        evaluator.evaluate_all(perturbations, "test_exp", resume=False)

        # Max concurrent should be <= parallelization
        assert max_concurrent[0] <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
