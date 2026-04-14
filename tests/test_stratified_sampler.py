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


def _get_class_for_type(ptype: str) -> str:
    """Map perturbation type to class."""
    if ptype in ("planning", "tool_selection", "structural"):
        return "coarse_grained"
    elif ptype in ("parameter", "data_reference", "near_neighbor_tool"):
        return "fine_grained"
    elif ptype in ("paraphrase", "formatting", "synonym", "reorder_args"):
        return "placebo"
    return "fine_grained"  # default


@pytest.fixture
def sample_perturbations():
    """
    Create sample perturbations covering all conditions, benchmarks, and classes.

    Creates ~800 perturbations with realistic distribution:
    - placebo: ~20%
    - fine_grained: ~50%
    - coarse_grained: ~30%
    """
    perturbations = []
    idx = 0

    # Coarse-grained conditions (~30%)
    coarse_conditions = [
        ("planning", "early"),
        ("planning", "middle"),
        ("planning", "late"),
        ("tool_selection", "early"),
        ("tool_selection", "middle"),
        ("tool_selection", "late"),
    ]

    # Fine-grained conditions (~50%)
    fine_conditions = [
        ("parameter", "early"),
        ("parameter", "middle"),
        ("parameter", "late"),
        ("data_reference", "early"),
        ("data_reference", "middle"),
        ("data_reference", "late"),
    ]

    # Placebo conditions (~20%)
    placebo_conditions = [
        ("paraphrase", "early"),
        ("paraphrase", "middle"),
        ("paraphrase", "late"),
        ("formatting", "middle"),
    ]

    benchmarks = ["toolbench", "gaia", "swebench"]
    tiers = ["high", "medium", "low"]

    # Create coarse-grained (~30%): 4 per combination = 4*6*3*3 = 216
    for ptype, pos in coarse_conditions:
        for benchmark in benchmarks:
            for tier in tiers:
                for i in range(4):
                    perturbations.append({
                        "perturbation_id": f"pert_{idx}",
                        "original_trajectory_id": f"{benchmark}_traj_{idx}",
                        "perturbed_trajectory_id": f"{benchmark}_perturbed_{idx}",
                        "perturbation_type": ptype,
                        "perturbation_position": pos,
                        "perturbation_class": "coarse_grained",
                        "quality_tier": tier,
                        "is_primary_for_experiment": True,
                        "original_step_content": f"Original content {idx}",
                        "perturbed_step_content": f"Perturbed content {idx}",
                    })
                    idx += 1

    # Create fine-grained (~50%): 7 per combination = 7*6*3*3 = 378
    for ptype, pos in fine_conditions:
        for benchmark in benchmarks:
            for tier in tiers:
                for i in range(7):
                    perturbations.append({
                        "perturbation_id": f"pert_{idx}",
                        "original_trajectory_id": f"{benchmark}_traj_{idx}",
                        "perturbed_trajectory_id": f"{benchmark}_perturbed_{idx}",
                        "perturbation_type": ptype,
                        "perturbation_position": pos,
                        "perturbation_class": "fine_grained",
                        "quality_tier": tier,
                        "is_primary_for_experiment": True,
                        "original_step_content": f"Original content {idx}",
                        "perturbed_step_content": f"Perturbed content {idx}",
                    })
                    idx += 1

    # Create placebo (~20%): 5 per combination = 5*4*3*3 = 180
    for ptype, pos in placebo_conditions:
        for benchmark in benchmarks:
            for tier in tiers:
                for i in range(5):
                    perturbations.append({
                        "perturbation_id": f"pert_{idx}",
                        "original_trajectory_id": f"{benchmark}_traj_{idx}",
                        "perturbed_trajectory_id": f"{benchmark}_perturbed_{idx}",
                        "perturbation_type": ptype,
                        "perturbation_position": pos,
                        "perturbation_class": "placebo",
                        "quality_tier": tier,
                        "is_primary_for_experiment": True,
                        "original_step_content": f"Original content {idx}",
                        "perturbed_step_content": f"Perturbed content {idx}",
                    })
                    idx += 1

    return perturbations


class TestStratifiedAnnotationSampler:
    """Tests for StratifiedAnnotationSampler."""

    def test_samples_exact_count(self, sample_perturbations):
        """Verify exactly 100 samples are returned."""
        sampler = StratifiedAnnotationSampler(sample_perturbations, random_seed=42)
        selected = sampler.sample(total=100)

        assert len(selected) == 100

    def test_all_classes_covered(self, sample_perturbations):
        """Verify all 3 classes have samples with correct distribution."""
        sampler = StratifiedAnnotationSampler(sample_perturbations, random_seed=42)
        selected = sampler.sample(total=100)

        # Count by class
        class_counts = Counter(p['perturbation_class'] for p in selected)

        # All classes should be present
        assert "placebo" in class_counts
        assert "fine_grained" in class_counts
        assert "coarse_grained" in class_counts

        # Check distribution is roughly correct (within tolerance)
        total = len(selected)
        placebo_pct = class_counts["placebo"] / total
        fine_pct = class_counts["fine_grained"] / total
        coarse_pct = class_counts["coarse_grained"] / total

        # Placebo ~20% (15-25%)
        assert 0.15 <= placebo_pct <= 0.25, f"Placebo {placebo_pct:.1%} not in range"
        # Fine-grained ~50% (45-55%)
        assert 0.45 <= fine_pct <= 0.55, f"Fine {fine_pct:.1%} not in range"
        # Coarse-grained ~30% (25-35%)
        assert 0.25 <= coarse_pct <= 0.35, f"Coarse {coarse_pct:.1%} not in range"

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
