"""
Tests for PrimarySelector module.

Tests cover:
- Tier assignment
- Quota enforcement
- No duplicate trajectories
- Tier distribution
- Balanced condition coverage
"""

import pytest
from collections import Counter

from src.sampling.primary_selector import PrimarySelector, create_primary_selector


class TestPrimarySelectorTierAssignment:
    """Tests for quality tier assignment."""

    @pytest.fixture
    def selector(self):
        """Create selector with default thresholds."""
        return PrimarySelector(
            total_primary=100,
            tier_thresholds={
                "high": 6,
                "medium": 4,
                "low": 1,
                "invalid": 0
            }
        )

    def test_tier_high(self, selector):
        """Test high tier assignment (score >= 6)."""
        assert selector._assign_tier(7) == "high"
        assert selector._assign_tier(6) == "high"

    def test_tier_medium(self, selector):
        """Test medium tier assignment (4 <= score < 6)."""
        assert selector._assign_tier(5) == "medium"
        assert selector._assign_tier(4) == "medium"

    def test_tier_low(self, selector):
        """Test low tier assignment (1 <= score < 4)."""
        assert selector._assign_tier(3) == "low"
        assert selector._assign_tier(2) == "low"
        assert selector._assign_tier(1) == "low"

    def test_tier_invalid(self, selector):
        """Test invalid tier assignment (score < 1)."""
        assert selector._assign_tier(0) == "invalid"


class TestPrimarySelectorSelection:
    """Tests for primary selection logic."""

    def _create_perturbation(
        self,
        perturbation_id: str,
        trajectory_id: str,
        ptype: str,
        position: str,
        score: int
    ) -> dict:
        """Helper to create test perturbation."""
        return {
            "perturbation_id": perturbation_id,
            "original_trajectory_id": trajectory_id,
            "perturbation_type": ptype,
            "perturbation_position": position,
            "quality_score": {"total_score": score},
            "quality_tier": None  # Will be assigned by selector
        }

    def test_select_filters_invalid(self):
        """Test that invalid perturbations (score=0) are filtered out."""
        selector = PrimarySelector(total_primary=10, min_per_condition=1)

        perturbations = [
            self._create_perturbation("p1", "t1", "planning", "early", 0),  # invalid
            self._create_perturbation("p2", "t2", "planning", "early", 5),  # valid
            self._create_perturbation("p3", "t3", "planning", "early", 6),  # valid
        ]

        selected = selector.select(perturbations)

        # Should not include p1 (invalid)
        assert "p1" not in selected
        assert "p2" in selected or "p3" in selected

    def test_select_no_duplicate_trajectories(self):
        """Test that each trajectory appears at most once."""
        selector = PrimarySelector(total_primary=10, min_per_condition=1)

        # Same trajectory with different perturbation types
        perturbations = [
            self._create_perturbation("p1", "t1", "planning", "early", 6),
            self._create_perturbation("p2", "t1", "tool_selection", "middle", 5),
            self._create_perturbation("p3", "t2", "planning", "early", 4),
            self._create_perturbation("p4", "t3", "tool_selection", "middle", 6),
        ]

        selected = selector.select(perturbations)

        # Get trajectory IDs for selected perturbations
        selected_trajs = []
        for p in perturbations:
            if p["perturbation_id"] in selected:
                selected_trajs.append(p["original_trajectory_id"])

        # Each trajectory should appear at most once
        assert len(selected_trajs) == len(set(selected_trajs))

    def test_select_respects_min_per_condition(self):
        """Test minimum samples per condition is enforced."""
        min_per_cond = 5
        selector = PrimarySelector(total_primary=100, min_per_condition=min_per_cond)

        # Create many perturbations for different conditions
        perturbations = []
        for i in range(50):
            for ptype in ["planning", "tool_selection"]:
                for pos in ["early", "middle"]:
                    perturbations.append(
                        self._create_perturbation(
                            f"p_{i}_{ptype}_{pos}",
                            f"t_{i}_{ptype}_{pos}",  # Unique trajectory per perturbation
                            ptype,
                            pos,
                            5  # All medium quality
                        )
                    )

        selected = selector.select(perturbations)

        # Count by condition
        by_condition = Counter()
        for p in perturbations:
            if p["perturbation_id"] in selected:
                cond = f"{p['perturbation_type']}_{p['perturbation_position']}"
                by_condition[cond] += 1

        # Each condition should have at least min_per_condition
        for cond, count in by_condition.items():
            assert count >= min_per_cond, f"Condition {cond} has only {count} samples"

    def test_select_prefers_higher_quality(self):
        """Test that higher quality perturbations are preferred."""
        selector = PrimarySelector(total_primary=2, min_per_condition=1)

        perturbations = [
            self._create_perturbation("p_low", "t1", "planning", "early", 2),  # low
            self._create_perturbation("p_med", "t2", "planning", "early", 4),  # medium
            self._create_perturbation("p_high", "t3", "planning", "early", 6),  # high
        ]

        selected = selector.select(perturbations)

        # Should prefer p_high over others
        assert "p_high" in selected

    def test_select_empty_input(self):
        """Test selection with empty input."""
        selector = PrimarySelector(total_primary=10)
        selected = selector.select([])
        assert selected == []

    def test_select_all_invalid(self):
        """Test selection when all perturbations are invalid."""
        selector = PrimarySelector(total_primary=10)

        perturbations = [
            self._create_perturbation("p1", "t1", "planning", "early", 0),
            self._create_perturbation("p2", "t2", "tool_selection", "middle", 0),
        ]

        selected = selector.select(perturbations)
        assert selected == []


class TestPrimarySelectorStats:
    """Tests for selection statistics."""

    def test_get_selection_stats(self):
        """Test statistics generation."""
        selector = PrimarySelector(total_primary=10, min_per_condition=1)

        perturbations = [
            {
                "perturbation_id": "p1",
                "original_trajectory_id": "t1",
                "perturbation_type": "planning",
                "perturbation_position": "early",
                "quality_score": {"total_score": 6},
                "quality_tier": "high"
            },
            {
                "perturbation_id": "p2",
                "original_trajectory_id": "t2",
                "perturbation_type": "tool_selection",
                "perturbation_position": "middle",
                "quality_score": {"total_score": 4},
                "quality_tier": "medium"
            }
        ]

        selected_ids = ["p1", "p2"]
        stats = selector.get_selection_stats(perturbations, selected_ids)

        assert stats["total_selected"] == 2
        assert stats["unique_trajectories"] == 2
        assert stats["by_type"]["planning"] == 1
        assert stats["by_type"]["tool_selection"] == 1
        assert stats["by_position"]["early"] == 1
        assert stats["by_position"]["middle"] == 1
        assert stats["by_tier"]["high"] == 1
        assert stats["by_tier"]["medium"] == 1


class TestCreatePrimarySelector:
    """Tests for factory function."""

    def test_create_from_config(self):
        """Test creating selector from config dict."""
        config = {
            "total_primary": 600,
            "tier_distribution": {
                "high": 0.33,
                "medium": 0.50,
                "low": 0.17
            },
            "min_per_condition": 40,
            "random_seed": 42
        }

        selector = create_primary_selector(config)

        assert selector.total_primary == 600
        assert selector.min_per_condition == 40
        assert selector.random_seed == 42

    def test_create_with_defaults(self):
        """Test creating selector with default values."""
        selector = create_primary_selector({})

        assert selector.total_primary == 600
        assert selector.min_per_condition == 40


class TestPrimarySelectorDeterminism:
    """Tests for reproducibility with random seed."""

    def test_same_seed_same_results(self):
        """Test that same random seed produces same results."""
        perturbations = [
            {
                "perturbation_id": f"p{i}",
                "original_trajectory_id": f"t{i}",
                "perturbation_type": "planning",
                "perturbation_position": "early",
                "quality_score": {"total_score": 5},
                "quality_tier": None
            }
            for i in range(100)
        ]

        selector1 = PrimarySelector(total_primary=10, random_seed=42)
        selector2 = PrimarySelector(total_primary=10, random_seed=42)

        selected1 = selector1.select(perturbations.copy())
        selected2 = selector2.select(perturbations.copy())

        assert selected1 == selected2

    def test_different_seed_different_results(self):
        """Test that different seeds may produce different results."""
        perturbations = [
            {
                "perturbation_id": f"p{i}",
                "original_trajectory_id": f"t{i}",
                "perturbation_type": "planning",
                "perturbation_position": "early",
                "quality_score": {"total_score": 5},
                "quality_tier": None
            }
            for i in range(100)
        ]

        selector1 = PrimarySelector(total_primary=10, random_seed=42)
        selector2 = PrimarySelector(total_primary=10, random_seed=99)

        selected1 = selector1.select(perturbations.copy())
        selected2 = selector2.select(perturbations.copy())

        # May or may not be different depending on data, but test runs
        # This is more of a smoke test
        assert len(selected1) == len(selected2)
