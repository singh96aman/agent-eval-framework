"""
Tests for CCGCalculator.

Tests:
- CCG formula: CCG = (JPS - TCS) / TCS
- CCG interpretation (under-penalize vs over-penalize)
- Aggregation by condition/type/position
- ANOVA statistical tests
- Effect size calculations
- Hypothesis assessment
"""

import pytest
import pandas as pd
import numpy as np

from src.metrics.ccg_calculator import CCGCalculator


@pytest.fixture
def calculator():
    """Create a CCGCalculator instance."""
    return CCGCalculator()


@pytest.fixture
def sample_evaluations():
    """
    Create sample evaluation data with known CCG patterns.

    Creates evaluations that demonstrate:
    - Early structural: under-penalization (CCG < 0)
    - Late surface: over-penalization (CCG > 0)
    """
    evaluations = []

    # Early planning errors - under-penalized
    # JPS=30, TCS=90 -> CCG = (30-90)/90 = -0.67
    for i in range(20):
        evaluations.append({
            "perturbation_id": f"planning_early_{i}",
            "jps": 30,
            "perturbation_type": "planning",
            "perturbation_position": "early",
            "benchmark": "toolbench"
        })

    # Middle tool selection errors
    # JPS=40, TCS=60 -> CCG = (40-60)/60 = -0.33
    for i in range(20):
        evaluations.append({
            "perturbation_id": f"tool_selection_middle_{i}",
            "jps": 40,
            "perturbation_type": "tool_selection",
            "perturbation_position": "middle",
            "benchmark": "gaia"
        })

    # Late parameter errors - over-penalized
    # JPS=30, TCS=20 -> CCG = (30-20)/20 = 0.5
    for i in range(20):
        evaluations.append({
            "perturbation_id": f"parameter_late_{i}",
            "jps": 30,
            "perturbation_type": "parameter",
            "perturbation_position": "late",
            "benchmark": "swebench"
        })

    return evaluations


@pytest.fixture
def sample_tcs_values():
    """Create TCS lookup for sample evaluations."""
    tcs = {}

    for i in range(20):
        tcs[f"planning_early_{i}"] = 90
        tcs[f"tool_selection_middle_{i}"] = 60
        tcs[f"parameter_late_{i}"] = 20

    return tcs


class TestCCGFormula:
    """Tests for basic CCG computation."""

    def test_ccg_formula_basic(self, calculator):
        """Test CCG = (JPS - TCS) / TCS."""
        # JPS=30, TCS=90 -> (30-90)/90 = -0.667
        ccg = calculator.compute_ccg(jps=30, tcs=90)
        assert abs(ccg - (-0.667)) < 0.01

        # JPS=60, TCS=60 -> (60-60)/60 = 0 (perfect calibration)
        ccg = calculator.compute_ccg(jps=60, tcs=60)
        assert ccg == 0.0

        # JPS=30, TCS=20 -> (30-20)/20 = 0.5
        ccg = calculator.compute_ccg(jps=30, tcs=20)
        assert ccg == 0.5

    def test_ccg_zero_tcs(self, calculator):
        """Test CCG returns None when TCS is 0 (BUG-002 fix)."""
        ccg = calculator.compute_ccg(jps=30, tcs=0)
        assert ccg is None  # Changed from 0.0 - samples excluded from analysis

    def test_ccg_tcs_capped_at_100(self, calculator):
        """Test TCS values > 100 are capped at 100 (BUG-001 fix)."""
        # TCS=140 should be capped to 100
        # JPS=75, TCS=100 -> CCG = (75-100)/100 = -0.25
        ccg = calculator.compute_ccg(jps=75, tcs=140)
        assert ccg == -0.25

        # Without capping: CCG = (75-140)/140 = -0.46
        # With capping: CCG = (75-100)/100 = -0.25
        ccg_if_not_capped = (75 - 140) / 140
        assert ccg != ccg_if_not_capped  # Verify capping made a difference

    def test_ccg_negative_means_under_penalize(self, calculator):
        """Negative CCG means judge under-penalized."""
        # JPS < TCS means judge didn't penalize enough
        ccg = calculator.compute_ccg(jps=20, tcs=80)
        assert ccg < 0

    def test_ccg_positive_means_over_penalize(self, calculator):
        """Positive CCG means judge over-penalized."""
        # JPS > TCS means judge penalized too much
        ccg = calculator.compute_ccg(jps=40, tcs=20)
        assert ccg > 0


class TestComputeAll:
    """Tests for batch CCG computation."""

    def test_compute_all_creates_dataframe(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test compute_all returns proper DataFrame and stats."""
        df, stats = calculator.compute_all(sample_evaluations, sample_tcs_values)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 60
        assert "ccg" in df.columns
        assert "jps" in df.columns
        assert "tcs" in df.columns
        assert "perturbation_type" in df.columns
        assert "position" in df.columns
        assert "condition" in df.columns

        # Check exclusion stats structure
        assert "total_evaluated" in stats
        assert "tcs_zero_excluded" in stats
        assert "tcs_capped" in stats
        assert "final_sample_size" in stats

    def test_compute_all_ccg_values(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test that CCG values are computed correctly."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)

        # Check planning_early CCG values
        planning_early = df[df["condition"] == "planning_early"]
        assert len(planning_early) == 20
        assert abs(planning_early["ccg"].mean() - (-0.667)) < 0.01

        # Check parameter_late CCG values
        parameter_late = df[df["condition"] == "parameter_late"]
        assert len(parameter_late) == 20
        assert abs(parameter_late["ccg"].mean() - 0.5) < 0.01

    def test_compute_all_uses_overall_score_if_no_jps(self, calculator, sample_tcs_values):
        """Test that overall_score is converted to JPS if jps not provided."""
        evaluations = [
            {
                "perturbation_id": "planning_early_0",
                "overall_score": 70,  # JPS = 100 - 70 = 30
                "perturbation_type": "planning",
                "perturbation_position": "early"
            }
        ]

        df, _ = calculator.compute_all(evaluations, sample_tcs_values)

        assert df["jps"].iloc[0] == 30
        assert abs(df["ccg"].iloc[0] - (-0.667)) < 0.01

    def test_compute_all_excludes_tcs_zero(self, calculator):
        """Test that TCS=0 samples are excluded (BUG-002 fix)."""
        evaluations = [
            {
                "perturbation_id": "tcs_zero_sample",
                "jps": 50,
                "perturbation_type": "planning",
                "perturbation_position": "early"
            },
            {
                "perturbation_id": "tcs_nonzero_sample",
                "jps": 50,
                "perturbation_type": "planning",
                "perturbation_position": "early"
            }
        ]
        tcs_values = {
            "tcs_zero_sample": 0,      # Should be excluded
            "tcs_nonzero_sample": 50   # Should be included
        }

        df, stats = calculator.compute_all(evaluations, tcs_values)

        assert len(df) == 1  # Only non-zero TCS sample
        assert stats["tcs_zero_excluded"] == 1
        assert stats["final_sample_size"] == 1

    def test_compute_all_caps_tcs_over_100(self, calculator):
        """Test that TCS > 100 is capped and tracked (BUG-001 fix)."""
        evaluations = [
            {
                "perturbation_id": "high_tcs_sample",
                "jps": 75,
                "perturbation_type": "tool_selection",
                "perturbation_position": "middle"
            }
        ]
        tcs_values = {"high_tcs_sample": 140}  # Should be capped to 100

        df, stats = calculator.compute_all(evaluations, tcs_values)

        assert df["tcs"].iloc[0] == 100  # Capped value
        assert df["tcs_raw"].iloc[0] == 140  # Original value preserved
        assert stats["tcs_capped"] == 1
        # CCG = (75-100)/100 = -0.25
        assert df["ccg"].iloc[0] == -0.25


class TestAggregation:
    """Tests for CCG aggregation."""

    def test_aggregate_by_condition(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test aggregation by condition."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)
        agg = calculator.aggregate(df, "condition")

        # Should have 3 conditions
        assert len(agg) == 3

    def test_aggregate_by_position(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test aggregation by position."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)
        agg = calculator.aggregate(df, "position")

        # Should have 3 positions
        assert len(agg) == 3


class TestStatisticalTests:
    """Tests for statistical analyses."""

    def test_anova_returns_results(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test ANOVA returns proper results."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)
        results = calculator.run_anova(df, ["perturbation_type", "position"])

        assert "perturbation_type" in results
        assert "position" in results

        # Check result structure
        for factor in ["perturbation_type", "position"]:
            assert "f_statistic" in results[factor]
            assert "p_value" in results[factor]
            assert "eta_squared" in results[factor]
            assert "significant" in results[factor]

    def test_anova_detects_significant_difference(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test ANOVA detects significant differences in our sample data."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)
        results = calculator.run_anova(df, ["position"])

        # Our data has large differences between positions
        # Should be statistically significant
        assert results["position"]["significant"]

    def test_tukey_hsd_returns_dataframe(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test Tukey HSD returns pairwise comparisons."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)
        tukey_df = calculator.run_tukey_hsd(df, "position")

        # Should return DataFrame with pairwise comparisons
        if tukey_df is not None:  # May be None if statsmodels not installed
            assert isinstance(tukey_df, pd.DataFrame)
            assert len(tukey_df) > 0


class TestEffectSizes:
    """Tests for effect size calculations."""

    def test_compute_effect_sizes(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test effect size computation."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)
        effects = calculator.compute_effect_sizes(df)

        assert "early_vs_late" in effects
        assert "cohens_d" in effects["early_vs_late"]

    def test_cohens_d_large_for_clear_difference(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test that Cohen's d is large when groups differ substantially."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)
        effects = calculator.compute_effect_sizes(df)

        # Our early (CCG=-0.67) vs late (CCG=0.5) should have large d
        d = effects["early_vs_late"]["cohens_d"]
        assert abs(d) > 0.8  # Large effect size


class TestHypothesisAssessment:
    """Tests for hypothesis assessment."""

    def test_assess_hypothesis_structure(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test hypothesis assessment returns proper structure."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)
        assessment = calculator.assess_hypothesis(df)

        assert "early_structural_mean_ccg" in assessment
        assert "late_surface_mean_ccg" in assessment
        assert "criteria" in assessment
        assert "support_level" in assessment

    def test_hypothesis_early_under_penalized(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test that early structural errors show under-penalization."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)
        assessment = calculator.assess_hypothesis(df)

        # Early structural CCG should be negative
        assert assessment["early_structural_mean_ccg"] < 0
        assert assessment["criteria"]["early_under_penalized"]

    def test_hypothesis_late_over_penalized(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test that late surface errors show over-penalization."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)
        assessment = calculator.assess_hypothesis(df)

        # Late surface CCG should be positive
        assert assessment["late_surface_mean_ccg"] > 0
        assert assessment["criteria"]["late_over_penalized"]

    def test_hypothesis_direction_correct(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test that direction of difference is correct."""
        df, _ = calculator.compute_all(sample_evaluations, sample_tcs_values)
        assessment = calculator.assess_hypothesis(df)

        # Early < Late
        assert assessment["criteria"]["direction_correct"]


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report_structure(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test that report has expected structure."""
        df, stats = calculator.compute_all(sample_evaluations, sample_tcs_values)
        report = calculator.generate_report(df, stats)

        assert "overall" in report
        assert "by_condition" in report
        assert "by_position" in report
        assert "by_type" in report
        assert "anova" in report
        assert "effect_sizes" in report
        assert "hypothesis_support" in report

    def test_report_overall_statistics(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test that overall statistics are computed."""
        df, stats = calculator.compute_all(sample_evaluations, sample_tcs_values)
        report = calculator.generate_report(df, stats)

        overall = report["overall"]
        assert overall["n_samples"] == 60
        assert "mean_ccg" in overall
        assert "std_ccg" in overall
        assert "mean_jps" in overall
        assert "mean_tcs" in overall

    def test_report_includes_data_quality(
        self, calculator, sample_evaluations, sample_tcs_values
    ):
        """Test that report includes data quality stats when provided."""
        df, stats = calculator.compute_all(sample_evaluations, sample_tcs_values)
        report = calculator.generate_report(df, stats)

        assert "data_quality" in report
        assert "total_evaluated" in report["data_quality"]
        assert "tcs_zero_excluded" in report["data_quality"]
        assert "tcs_capped_count" in report["data_quality"]
        assert "final_sample_size" in report["data_quality"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
