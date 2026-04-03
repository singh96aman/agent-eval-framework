"""
Tests for stratified sampling and assignment logic.
"""

import pytest
from src.data.schema import Trajectory, Step, StepType, GroundTruth
from src.data.sampling import (
    SamplingConfig,
    StratifiedSample,
    stratified_sample_trajectories,
    generate_sample_report,
    validate_sample_coverage,
)


def create_test_trajectory(
    traj_id: str,
    benchmark: str,
    num_steps: int = 5,
    tool_name: str = "test_tool"
) -> Trajectory:
    """Create a test trajectory with specified parameters."""
    steps = [
        Step(
            step_id=f"{traj_id}_step_{i}",
            step_number=i,
            step_type=StepType.TOOL_EXECUTION,
            content=f"Step {i} content",
            tool_name=tool_name,
            tool_input={"param": str(i)}
        )
        for i in range(1, num_steps + 1)
    ]

    return Trajectory(
        trajectory_id=traj_id,
        benchmark=benchmark,
        steps=steps,
        ground_truth=GroundTruth(
            task_description=f"Task for {traj_id}",
            task_success=True
        ),
        metadata={"repo": "test/repo"} if benchmark == "swebench" else {}
    )


class TestSamplingConfig:
    """Tests for SamplingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SamplingConfig()

        assert config.toolbench_count == 400
        assert config.gaia_count == 100
        assert config.swebench_count == 100
        assert config.control_group_size == 100
        assert config.total_trajectories == 600
        assert config.main_sample_size == 500

    def test_custom_config(self):
        """Test custom configuration."""
        config = SamplingConfig(
            toolbench_count=200,
            gaia_count=50,
            swebench_count=50,
            control_group_size=50
        )

        assert config.total_trajectories == 300
        assert config.main_sample_size == 250


class TestStratifiedSampling:
    """Tests for stratified sampling."""

    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectories for each benchmark."""
        toolbench = [
            create_test_trajectory(
                f"tb_{i}",
                "toolbench",
                num_steps=i % 7 + 3,  # 3-9 steps
                tool_name=f"weather_tool_{i}" if i < 5 else f"finance_tool_{i}"
            )
            for i in range(20)
        ]

        gaia = [
            create_test_trajectory(
                f"gaia_{i}",
                "gaia",
                num_steps=i % 5 + 3
            )
            for i in range(10)
        ]

        swebench = [
            create_test_trajectory(
                f"swe_{i}",
                "swebench",
                num_steps=i % 6 + 4
            )
            for i in range(10)
        ]

        return toolbench, gaia, swebench

    def test_stratified_sample_basic(self, sample_trajectories):
        """Test basic stratified sampling."""
        toolbench, gaia, swebench = sample_trajectories

        config = SamplingConfig(
            toolbench_count=15,
            gaia_count=5,
            swebench_count=5,
            control_group_size=5,
            random_seed=42
        )

        sample = stratified_sample_trajectories(
            toolbench, gaia, swebench, config
        )

        assert isinstance(sample, StratifiedSample)
        assert len(sample.main_sample) > 0
        assert len(sample.control_group) == 5

    def test_perturbation_assignment(self, sample_trajectories):
        """Test that perturbation conditions are assigned."""
        toolbench, gaia, swebench = sample_trajectories

        config = SamplingConfig(
            toolbench_count=10,
            gaia_count=5,
            swebench_count=5,
            control_group_size=5,
            random_seed=42
        )

        sample = stratified_sample_trajectories(
            toolbench, gaia, swebench, config
        )

        # Each item in main_sample should have (trajectory, type, position)
        for item in sample.main_sample:
            traj, ptype, position = item
            assert isinstance(traj, Trajectory)
            assert ptype in ["planning", "tool_selection", "parameter",
                           "data_reference"]
            assert position in ["early", "middle", "late"]

    def test_data_reference_not_assigned_to_early(self, sample_trajectories):
        """Test that data_reference is not assigned to early position."""
        toolbench, gaia, swebench = sample_trajectories

        config = SamplingConfig(
            toolbench_count=15,
            gaia_count=5,
            swebench_count=5,
            control_group_size=3,
            random_seed=42
        )

        sample = stratified_sample_trajectories(
            toolbench, gaia, swebench, config
        )

        # Check that data_reference × early combination doesn't exist
        for _, ptype, position in sample.main_sample:
            if ptype == "data_reference":
                assert position != "early"

    def test_stats_calculation(self, sample_trajectories):
        """Test that statistics are calculated correctly."""
        toolbench, gaia, swebench = sample_trajectories

        config = SamplingConfig(
            toolbench_count=10,
            gaia_count=5,
            swebench_count=5,
            control_group_size=5,
            random_seed=42
        )

        sample = stratified_sample_trajectories(
            toolbench, gaia, swebench, config
        )

        assert "total_trajectories" in sample.stats
        assert "by_benchmark" in sample.stats
        assert "by_condition" in sample.stats
        assert "by_complexity" in sample.stats


class TestSampleReport:
    """Tests for sample report generation."""

    def test_generate_report(self):
        """Test generating human-readable report."""
        sample = StratifiedSample(
            main_sample=[],
            control_group=[],
            stats={
                "total_trajectories": 100,
                "main_sample_size": 80,
                "control_group_size": 20,
                "by_benchmark": {"toolbench": 60, "gaia": 20, "swebench": 20},
                "by_condition": {"planning_early": 10, "parameter_middle": 15},
                "by_domain": {"data_information": 30, "finance": 30},
                "by_complexity": {"simple": 20, "medium": 50, "complex": 30},
            }
        )

        report = generate_sample_report(sample)

        assert "STRATIFIED SAMPLE REPORT" in report
        assert "Total trajectories: 100" in report
        assert "toolbench: 60" in report
        assert "planning_early: 10" in report


class TestSampleValidation:
    """Tests for sample validation."""

    def test_validate_adequate_coverage(self):
        """Test validation passes with adequate coverage."""
        sample = StratifiedSample(
            main_sample=[],
            control_group=[],
            stats={
                "by_condition": {
                    "planning_early": 35,
                    "planning_middle": 35,
                    "parameter_early": 35,
                },
                "by_benchmark": {
                    "toolbench": 350,
                    "gaia": 60,
                    "swebench": 60
                },
                "by_complexity": {
                    "simple": 100,
                    "medium": 250,
                    "complex": 120
                }
            }
        )

        is_valid, issues = validate_sample_coverage(sample, min_per_condition=30)
        assert is_valid
        assert len(issues) == 0

    def test_validate_insufficient_condition(self):
        """Test validation fails with insufficient condition coverage."""
        sample = StratifiedSample(
            main_sample=[],
            control_group=[],
            stats={
                "by_condition": {
                    "planning_early": 10,  # Below minimum
                    "planning_middle": 35,
                },
                "by_benchmark": {"toolbench": 350, "gaia": 60, "swebench": 60},
                "by_complexity": {"simple": 100, "medium": 250, "complex": 120}
            }
        )

        is_valid, issues = validate_sample_coverage(sample, min_per_condition=30)
        assert not is_valid
        assert any("planning_early" in issue for issue in issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
