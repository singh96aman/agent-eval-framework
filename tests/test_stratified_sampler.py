"""
Tests for StratifiedAnnotationSampler.

Tests:
- Samples exactly 100 perturbations
- All 11 conditions are covered
- All 3 benchmarks are represented
- Quality tier distribution is reasonable
"""

import pytest
from collections import Counter

from src.annotation.stratified_sampler import (
    StratifiedAnnotationSampler,
    validate_annotations,
)


@pytest.fixture
def sample_perturbations():
    """
    Create sample perturbations covering all conditions and benchmarks.

    Creates ~600 perturbations with realistic distribution.
    """
    perturbations = []
    idx = 0

    conditions = [
        ("planning", "early"),
        ("planning", "middle"),
        ("planning", "late"),
        ("tool_selection", "early"),
        ("tool_selection", "middle"),
        ("tool_selection", "late"),
        ("parameter", "early"),
        ("parameter", "middle"),
        ("parameter", "late"),
        ("data_reference", "middle"),
        ("data_reference", "late"),
    ]

    benchmarks = ["toolbench", "gaia", "swebench"]
    tiers = ["high", "medium", "low"]

    # Create ~55 perturbations per condition (600 total / 11 conditions)
    for ptype, pos in conditions:
        for benchmark in benchmarks:
            for tier in tiers:
                # Create 5-6 perturbations per (condition, benchmark, tier)
                for i in range(6):
                    perturbations.append(
                        {
                            "perturbation_id": f"pert_{idx}",
                            "original_trajectory_id": f"{benchmark}_traj_{idx}",
                            "perturbed_trajectory_id": f"{benchmark}_perturbed_{idx}",
                            "perturbation_type": ptype,
                            "perturbation_position": pos,
                            "quality_tier": tier,
                            "is_primary_for_experiment": True,
                            "original_step_content": f"Original content {idx}",
                            "perturbed_step_content": f"Perturbed content {idx}",
                        }
                    )
                    idx += 1

    return perturbations


class TestStratifiedAnnotationSampler:
    """Tests for StratifiedAnnotationSampler."""

    def test_samples_exact_count(self, sample_perturbations):
        """Verify exactly 100 samples are returned."""
        sampler = StratifiedAnnotationSampler(sample_perturbations, random_seed=42)
        selected = sampler.sample(total=100)

        assert len(selected) == 100

    def test_all_conditions_covered(self, sample_perturbations):
        """Verify all 11 conditions have samples."""
        sampler = StratifiedAnnotationSampler(sample_perturbations, random_seed=42)
        selected = sampler.sample(total=100)

        # Count by condition
        condition_counts = Counter(
            f"{p['perturbation_type']}_{p['perturbation_position']}" for p in selected
        )

        expected_conditions = {
            "planning_early",
            "planning_middle",
            "planning_late",
            "tool_selection_early",
            "tool_selection_middle",
            "tool_selection_late",
            "parameter_early",
            "parameter_middle",
            "parameter_late",
            "data_reference_middle",
            "data_reference_late",
        }

        # All conditions should be present
        assert set(condition_counts.keys()) == expected_conditions

        # Each condition should have samples (roughly 9 each = 100/11)
        for condition, count in condition_counts.items():
            assert count >= 5, f"Condition {condition} has only {count} samples"

    def test_all_benchmarks_represented(self, sample_perturbations):
        """Verify all 3 benchmarks have samples."""
        sampler = StratifiedAnnotationSampler(sample_perturbations, random_seed=42)
        selected = sampler.sample(total=100)

        # Get benchmark distribution
        report = sampler.get_distribution_report(selected)

        benchmarks = report["by_benchmark"]
        assert "toolbench" in benchmarks
        assert "gaia" in benchmarks
        assert "swebench" in benchmarks

        # Each benchmark should have reasonable representation
        for benchmark, count in benchmarks.items():
            assert count >= 20, f"Benchmark {benchmark} has only {count} samples"

    def test_quality_tier_distribution(self, sample_perturbations):
        """Verify mix of high/medium/low quality tiers."""
        sampler = StratifiedAnnotationSampler(sample_perturbations, random_seed=42)
        selected = sampler.sample(total=100)

        # Get tier distribution
        report = sampler.get_distribution_report(selected)
        tiers = report["by_quality_tier"]

        # All tiers should be present
        assert "high" in tiers
        assert "medium" in tiers
        assert "low" in tiers

        # Each tier should have some representation
        for tier, count in tiers.items():
            assert count >= 10, f"Tier {tier} has only {count} samples"

    def test_reproducibility_with_seed(self, sample_perturbations):
        """Verify same seed produces same samples."""
        sampler1 = StratifiedAnnotationSampler(sample_perturbations, random_seed=42)
        selected1 = sampler1.sample(total=100)

        sampler2 = StratifiedAnnotationSampler(sample_perturbations, random_seed=42)
        selected2 = sampler2.sample(total=100)

        # Should be the same
        ids1 = {p["perturbation_id"] for p in selected1}
        ids2 = {p["perturbation_id"] for p in selected2}
        assert ids1 == ids2

    def test_different_seeds_produce_different_samples(self, sample_perturbations):
        """Verify different seeds produce different samples."""
        sampler1 = StratifiedAnnotationSampler(sample_perturbations, random_seed=42)
        selected1 = sampler1.sample(total=100)

        sampler2 = StratifiedAnnotationSampler(sample_perturbations, random_seed=123)
        selected2 = sampler2.sample(total=100)

        ids1 = {p["perturbation_id"] for p in selected1}
        ids2 = {p["perturbation_id"] for p in selected2}

        # Should not be exactly the same (may have some overlap)
        assert ids1 != ids2

    def test_filters_non_primary_perturbations(self):
        """Verify only primary perturbations are sampled."""
        perturbations = []
        for i in range(200):
            perturbations.append(
                {
                    "perturbation_id": f"pert_{i}",
                    "original_trajectory_id": f"toolbench_traj_{i}",
                    "perturbation_type": "planning",
                    "perturbation_position": "early",
                    "quality_tier": "high",
                    "is_primary_for_experiment": i < 100,  # Only first 100 are primary
                }
            )

        sampler = StratifiedAnnotationSampler(perturbations, random_seed=42)

        # Sampler should only see 100 perturbations
        assert len(sampler.perturbations) == 100

    def test_distribution_report_format(self, sample_perturbations):
        """Verify distribution report has correct format."""
        sampler = StratifiedAnnotationSampler(sample_perturbations, random_seed=42)
        selected = sampler.sample(total=100)
        report = sampler.get_distribution_report(selected)

        assert "total" in report
        assert "by_condition" in report
        assert "by_benchmark" in report
        assert "by_quality_tier" in report
        assert "by_type" in report
        assert "by_position" in report

        assert report["total"] == 100


class TestValidateAnnotations:
    """Tests for annotation validation."""

    def test_validate_complete_annotations(self):
        """Test validation of complete annotations."""
        annotations = [
            {
                "perturbation_id": "p1",
                "annotation": {
                    "task_success_degradation": 0,
                    "subsequent_error_rate": 1,
                    "criticality_rating": 3,
                },
            },
            {
                "perturbation_id": "p2",
                "annotation": {
                    "task_success_degradation": 1,
                    "subsequent_error_rate": 2,
                    "criticality_rating": 4,
                },
            },
        ]

        result = validate_annotations(annotations)

        assert result["total"] == 2
        assert result["complete"] == 2
        assert result["incomplete"] == 0
        assert result["invalid"] == 0

    def test_validate_incomplete_annotations(self):
        """Test validation catches incomplete annotations."""
        annotations = [
            {
                "perturbation_id": "p1",
                "annotation": {
                    "task_success_degradation": 0
                    # Missing subsequent_error_rate and criticality_rating
                },
            },
            {"perturbation_id": "p2", "annotation": {}},  # Empty
        ]

        result = validate_annotations(annotations)

        assert result["incomplete"] == 2

    def test_validate_invalid_ranges(self):
        """Test validation catches out-of-range values."""
        annotations = [
            {
                "perturbation_id": "p1",
                "annotation": {
                    "task_success_degradation": 2,  # Invalid: should be 0 or 1
                    "subsequent_error_rate": 1,
                    "criticality_rating": 3,
                },
            },
            {
                "perturbation_id": "p2",
                "annotation": {
                    "task_success_degradation": 0,
                    "subsequent_error_rate": 5,  # Invalid: should be 0-3
                    "criticality_rating": 3,
                },
            },
            {
                "perturbation_id": "p3",
                "annotation": {
                    "task_success_degradation": 0,
                    "subsequent_error_rate": 1,
                    "criticality_rating": 6,  # Invalid: should be 1-5
                },
            },
        ]

        result = validate_annotations(annotations)

        assert result["invalid"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
