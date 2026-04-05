"""
Tests for CriticalityScorer.

Tests:
- Heuristic TCS computation
- Human TCS computation from annotations
- Hybrid mode (human + heuristic fallback)
- TCS formula validation
- Heuristic validation against human annotations
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.metrics.criticality_scorer import CriticalityScorer


@pytest.fixture
def basic_config():
    """Basic configuration for heuristic mode."""
    return {"mode": "heuristic"}


@pytest.fixture
def annotations_file():
    """Create a temporary annotations file."""
    annotations = [
        {
            "perturbation_id": "pert_1",
            "perturbation_type": "planning",
            "perturbation_position": "early",
            "annotation": {
                "task_success_degradation": 1,
                "subsequent_error_rate": 2,
                "criticality_rating": 4
            }
        },
        {
            "perturbation_id": "pert_2",
            "perturbation_type": "parameter",
            "perturbation_position": "late",
            "annotation": {
                "task_success_degradation": 0,
                "subsequent_error_rate": 1,
                "criticality_rating": 2
            }
        },
        {
            "perturbation_id": "pert_3",
            "perturbation_type": "tool_selection",
            "perturbation_position": "middle",
            "annotation": {
                "task_success_degradation": 1,
                "subsequent_error_rate": 3,
                "criticality_rating": 5
            }
        },
        # Incomplete annotation (should be skipped)
        {
            "perturbation_id": "pert_4",
            "perturbation_type": "planning",
            "perturbation_position": "middle",
            "annotation": {
                "task_success_degradation": 1
                # Missing other fields
            }
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(annotations, f)
        return f.name


class TestHeuristicTCS:
    """Tests for heuristic TCS computation."""

    def test_planning_early_high_tcs(self, basic_config):
        """Early planning errors should have high TCS."""
        scorer = CriticalityScorer(basic_config)

        perturbation = {
            "perturbation_id": "test",
            "perturbation_type": "planning",
            "perturbation_position": "early"
        }

        tcs = scorer.compute_tcs(perturbation)
        assert tcs == 90

    def test_parameter_late_low_tcs(self, basic_config):
        """Late parameter errors should have low TCS."""
        scorer = CriticalityScorer(basic_config)

        perturbation = {
            "perturbation_id": "test",
            "perturbation_type": "parameter",
            "perturbation_position": "late"
        }

        tcs = scorer.compute_tcs(perturbation)
        assert tcs == 20

    def test_tool_selection_middle(self, basic_config):
        """Middle tool selection errors should have moderate TCS."""
        scorer = CriticalityScorer(basic_config)

        perturbation = {
            "perturbation_id": "test",
            "perturbation_type": "tool_selection",
            "perturbation_position": "middle"
        }

        tcs = scorer.compute_tcs(perturbation)
        assert tcs == 60

    def test_unknown_condition_default(self, basic_config):
        """Unknown condition should return default TCS of 50."""
        scorer = CriticalityScorer(basic_config)

        perturbation = {
            "perturbation_id": "test",
            "perturbation_type": "unknown",
            "perturbation_position": "unknown"
        }

        tcs = scorer.compute_tcs(perturbation)
        assert tcs == 50

    def test_custom_heuristic_values(self):
        """Test custom heuristic values from config."""
        config = {
            "mode": "heuristic",
            "heuristic_tcs": {
                "custom_type_early": 95
            }
        }

        scorer = CriticalityScorer(config)

        perturbation = {
            "perturbation_id": "test",
            "perturbation_type": "custom_type",
            "perturbation_position": "early"
        }

        tcs = scorer.compute_tcs(perturbation)
        assert tcs == 95

    def test_early_gt_late_for_same_type(self, basic_config):
        """TCS for early position should be greater than late for same type."""
        scorer = CriticalityScorer(basic_config)

        for ptype in ["planning", "tool_selection", "parameter"]:
            early = {"perturbation_type": ptype, "perturbation_position": "early"}
            late = {"perturbation_type": ptype, "perturbation_position": "late"}

            tcs_early = scorer.compute_tcs(early)
            tcs_late = scorer.compute_tcs(late)

            assert tcs_early > tcs_late, f"Failed for {ptype}"


class TestHumanTCS:
    """Tests for human TCS computation from annotations."""

    def test_human_tcs_formula(self, annotations_file):
        """Test that TCS formula is correctly applied."""
        config = {
            "mode": "human",
            "human_annotation_path": annotations_file
        }

        scorer = CriticalityScorer(config)

        # pert_1: TSD=1, SER=2, crit=4
        # TCS = (1*50) + (2*10) + (4*8) = 50 + 20 + 32 = 102 -> capped at 100
        pert_1 = {"perturbation_id": "pert_1"}
        tcs_1 = scorer.compute_tcs(pert_1)
        assert tcs_1 == 100  # Capped at 100

        # pert_2: TSD=0, SER=1, crit=2
        # TCS = (0*50) + (1*10) + (2*8) = 0 + 10 + 16 = 26
        pert_2 = {"perturbation_id": "pert_2"}
        tcs_2 = scorer.compute_tcs(pert_2)
        assert tcs_2 == 26

    def test_human_tcs_caps_ser_at_3(self, annotations_file):
        """Test that SER is capped at 3."""
        config = {
            "mode": "human",
            "human_annotation_path": annotations_file
        }

        scorer = CriticalityScorer(config)

        # pert_3: TSD=1, SER=3, crit=5
        # TCS = (1*50) + (3*10) + (5*8) = 50 + 30 + 40 = 120 -> capped at 100
        pert_3 = {"perturbation_id": "pert_3"}
        tcs_3 = scorer.compute_tcs(pert_3)
        assert tcs_3 == 100

    def test_human_mode_raises_for_missing(self, annotations_file):
        """Human mode should raise for perturbation without annotation."""
        config = {
            "mode": "human",
            "human_annotation_path": annotations_file
        }

        scorer = CriticalityScorer(config)

        # pert_4 has incomplete annotation (should not be loaded)
        # pert_999 doesn't exist at all
        with pytest.raises(ValueError):
            scorer.compute_tcs({"perturbation_id": "pert_999"})

    def test_incomplete_annotations_skipped(self, annotations_file):
        """Incomplete annotations should not be loaded."""
        config = {
            "mode": "human",
            "human_annotation_path": annotations_file
        }

        scorer = CriticalityScorer(config)

        # Should have loaded 3 complete annotations (pert_1, pert_2, pert_3)
        assert len(scorer.human_annotations) == 3
        assert "pert_4" not in scorer.human_annotations


class TestHybridMode:
    """Tests for hybrid TCS mode."""

    def test_hybrid_uses_human_when_available(self, annotations_file):
        """Hybrid mode should use human TCS when annotation exists."""
        config = {
            "mode": "hybrid",
            "human_annotation_path": annotations_file
        }

        scorer = CriticalityScorer(config)

        # pert_2 has annotation
        pert_2 = {
            "perturbation_id": "pert_2",
            "perturbation_type": "parameter",
            "perturbation_position": "late"
        }

        tcs = scorer.compute_tcs(pert_2)

        # Should use human TCS (26), not heuristic (20)
        assert tcs == 26

    def test_hybrid_falls_back_to_heuristic(self, annotations_file):
        """Hybrid mode should fall back to heuristic when no annotation."""
        config = {
            "mode": "hybrid",
            "human_annotation_path": annotations_file
        }

        scorer = CriticalityScorer(config)

        # No annotation for this perturbation
        pert = {
            "perturbation_id": "pert_no_annotation",
            "perturbation_type": "planning",
            "perturbation_position": "early"
        }

        tcs = scorer.compute_tcs(pert)

        # Should use heuristic TCS (90)
        assert tcs == 90


class TestBatchComputation:
    """Tests for batch TCS computation."""

    def test_compute_batch(self, basic_config):
        """Test batch TCS computation."""
        scorer = CriticalityScorer(basic_config)

        perturbations = [
            {"perturbation_id": "p1", "perturbation_type": "planning", "perturbation_position": "early"},
            {"perturbation_id": "p2", "perturbation_type": "parameter", "perturbation_position": "late"},
            {"perturbation_id": "p3", "perturbation_type": "tool_selection", "perturbation_position": "middle"}
        ]

        tcs_values = scorer.compute_batch(perturbations)

        assert len(tcs_values) == 3
        assert tcs_values[0] == 90  # planning_early
        assert tcs_values[1] == 20  # parameter_late
        assert tcs_values[2] == 60  # tool_selection_middle

    def test_compute_batch_with_ids(self, basic_config):
        """Test batch TCS computation returning dict."""
        scorer = CriticalityScorer(basic_config)

        perturbations = [
            {"perturbation_id": "p1", "perturbation_type": "planning", "perturbation_position": "early"},
            {"perturbation_id": "p2", "perturbation_type": "parameter", "perturbation_position": "late"}
        ]

        tcs_dict = scorer.compute_batch_with_ids(perturbations)

        assert tcs_dict["p1"] == 90
        assert tcs_dict["p2"] == 20


class TestHeuristicValidation:
    """Tests for heuristic validation against human annotations."""

    def test_validation_returns_correlation(self, annotations_file):
        """Test that validation returns correlation metrics."""
        config = {
            "mode": "hybrid",
            "human_annotation_path": annotations_file
        }

        scorer = CriticalityScorer(config)
        validation = scorer.validate_heuristic()

        assert "n_samples" in validation
        assert "pearson_r" in validation
        assert "spearman_r" in validation
        assert "correlation_strength" in validation
        assert "heuristic_valid" in validation

    def test_validation_requires_annotations(self, basic_config):
        """Validation should fail without human annotations."""
        scorer = CriticalityScorer(basic_config)
        validation = scorer.validate_heuristic()

        assert "error" in validation

    def test_validation_correlation_strength_classification(self, annotations_file):
        """Test correlation strength classification."""
        config = {
            "mode": "hybrid",
            "human_annotation_path": annotations_file
        }

        scorer = CriticalityScorer(config)
        validation = scorer.validate_heuristic()

        strength = validation.get("correlation_strength")
        assert strength in ["strong", "moderate", "weak"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
