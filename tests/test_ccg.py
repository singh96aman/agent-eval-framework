"""
Tests for CCG (Criticality-Calibration Gap) metrics.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.metrics.ccg import (
    compute_tcs,
    compute_jps,
    compute_ccg,
    CCGResult,
    CCGAnalysis,
    aggregate_by_condition,
    statistical_analysis
)


class TestCCGFormulas:
    """Test basic CCG formula computations."""

    def test_compute_tcs(self):
        """Test True Criticality Score computation."""
        # Task failed, 3 downstream errors
        assert compute_tcs(1, 3) == 130.0  # (1 * 100) + (3 * 10)

        # Task succeeded, 2 downstream errors
        assert compute_tcs(0, 2) == 20.0  # (0 * 100) + (2 * 10)

        # No errors
        assert compute_tcs(0, 0) == 0.0

        # Task failed, no downstream errors
        assert compute_tcs(1, 0) == 100.0

    def test_compute_jps(self):
        """Test Judge Penalty Score computation."""
        # High judge score (trajectory was good)
        assert compute_jps(85.0) == 15.0  # 100 - 85

        # Low judge score (trajectory was bad)
        assert compute_jps(40.0) == 60.0  # 100 - 40

        # Perfect score
        assert compute_jps(100.0) == 0.0

        # Zero score
        assert compute_jps(0.0) == 100.0

    def test_compute_ccg(self):
        """Test CCG computation."""
        # Judge under-penalized (JPS < TCS)
        ccg = compute_ccg(60.0, 130.0)
        assert ccg is not None
        assert ccg < 0  # Negative CCG
        assert abs(ccg - (-0.538)) < 0.01  # (60-130)/130 ≈ -0.538

        # Judge over-penalized (JPS > TCS)
        ccg = compute_ccg(30.0, 20.0)
        assert ccg is not None
        assert ccg > 0  # Positive CCG
        assert ccg == 0.5  # (30-20)/20 = 0.5

        # Perfect calibration
        ccg = compute_ccg(50.0, 50.0)
        assert ccg is not None
        assert ccg == 0.0

        # TCS is zero (undefined CCG)
        ccg = compute_ccg(10.0, 0.0)
        assert ccg is None

    def test_ccg_interpretation(self):
        """Test CCG interpretation scenarios."""
        # Scenario 1: Early planning error (high criticality)
        # Human: Task failed, 5 downstream errors -> TCS = 150
        # Judge: Gave 40/100 -> JPS = 60
        # CCG = (60-150)/150 = -0.6 (judge under-penalized)
        tcs = compute_tcs(1, 5)
        jps = compute_jps(40.0)
        ccg = compute_ccg(jps, tcs)
        assert tcs == 150.0
        assert jps == 60.0
        assert ccg < 0  # Judge under-penalized

        # Scenario 2: Late parameter error (low criticality)
        # Human: Task succeeded, 1 downstream error -> TCS = 10
        # Judge: Gave 70/100 -> JPS = 30
        # CCG = (30-10)/10 = 2.0 (judge over-penalized)
        tcs = compute_tcs(0, 1)
        jps = compute_jps(70.0)
        ccg = compute_ccg(jps, tcs)
        assert tcs == 10.0
        assert jps == 30.0
        assert ccg == 2.0  # Judge over-penalized


class TestCCGResult:
    """Test CCGResult dataclass."""

    def test_create_ccg_result(self):
        """Test creating CCG result."""
        result = CCGResult(
            perturbation_id="test_pert_1",
            experiment_id="exp_001",
            judge_name="claude-sonnet-4.5",
            perturbation_type="planning",
            perturbation_position="early",
            tsd=1,
            ser=3,
            tcs=130.0,
            judge_overall_score=60.0,
            jps=40.0,
            ccg=-0.69
        )

        assert result.perturbation_id == "test_pert_1"
        assert result.perturbation_type == "planning"
        assert result.tcs == 130.0
        assert result.ccg == -0.69

    def test_ccg_result_to_dict(self):
        """Test converting CCG result to dict."""
        result = CCGResult(
            perturbation_id="test_1",
            experiment_id="exp_001",
            judge_name="claude",
            perturbation_type="planning",
            perturbation_position="early",
            tsd=1,
            ser=3,
            tcs=130.0,
            judge_overall_score=60.0,
            jps=40.0,
            ccg=-0.69
        )

        data = result.to_dict()

        assert data['perturbation_id'] == "test_1"
        assert data['tcs'] == 130.0
        assert data['ccg'] == -0.69


class TestCCGAnalysis:
    """Test CCGAnalysis class."""

    def test_create_ccg_analysis(self):
        """Test creating CCG analysis."""
        results = [
            CCGResult(
                perturbation_id=f"test_{i}",
                experiment_id="exp_001",
                judge_name="claude",
                perturbation_type="planning",
                perturbation_position="early",
                tsd=1,
                ser=i,
                tcs=100.0 + i * 10,
                judge_overall_score=60.0,
                jps=40.0,
                ccg=-0.5
            )
            for i in range(3)
        ]

        analysis = CCGAnalysis(
            experiment_id="exp_001",
            results=results
        )

        assert analysis.experiment_id == "exp_001"
        assert len(analysis.results) == 3

    def test_to_dataframe(self):
        """Test converting analysis to DataFrame."""
        results = [
            CCGResult(
                perturbation_id="test_1",
                experiment_id="exp_001",
                judge_name="claude",
                perturbation_type="planning",
                perturbation_position="early",
                tsd=1,
                ser=3,
                tcs=130.0,
                judge_overall_score=60.0,
                jps=40.0,
                ccg=-0.69
            )
        ]

        analysis = CCGAnalysis(experiment_id="exp_001", results=results)
        df = analysis.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'perturbation_id' in df.columns
        assert 'ccg' in df.columns

    def test_save_csv(self, tmp_path):
        """Test saving analysis to CSV."""
        results = [
            CCGResult(
                perturbation_id="test_1",
                experiment_id="exp_001",
                judge_name="claude",
                perturbation_type="planning",
                perturbation_position="early",
                tsd=1,
                ser=3,
                tcs=130.0,
                judge_overall_score=60.0,
                jps=40.0,
                ccg=-0.69
            )
        ]

        analysis = CCGAnalysis(experiment_id="exp_001", results=results)
        filepath = tmp_path / "ccg_results.csv"
        analysis.save_csv(filepath)

        assert filepath.exists()

        # Read back
        df = pd.read_csv(filepath)
        assert len(df) == 1
        assert df['perturbation_id'][0] == "test_1"


class TestAggregation:
    """Test aggregation functions."""

    def test_aggregate_by_condition(self):
        """Test aggregating CCG results by condition."""
        # Create sample results
        results = []

        # Planning errors
        for pos in ['early', 'middle', 'late']:
            for i in range(3):
                results.append(CCGResult(
                    perturbation_id=f"planning_{pos}_{i}",
                    experiment_id="exp_001",
                    judge_name="claude",
                    perturbation_type="planning",
                    perturbation_position=pos,
                    tsd=1,
                    ser=3,
                    tcs=130.0,
                    judge_overall_score=60.0,
                    jps=40.0,
                    ccg=-0.69
                ))

        # Tool selection errors
        for pos in ['early', 'middle']:
            for i in range(2):
                results.append(CCGResult(
                    perturbation_id=f"tool_{pos}_{i}",
                    experiment_id="exp_001",
                    judge_name="claude",
                    perturbation_type="tool_selection",
                    perturbation_position=pos,
                    tsd=0,
                    ser=2,
                    tcs=20.0,
                    judge_overall_score=80.0,
                    jps=20.0,
                    ccg=0.0
                ))

        aggregates = aggregate_by_condition(results)

        # Should have aggregates for types, positions, and conditions
        assert 'planning' in aggregates
        assert 'tool_selection' in aggregates
        assert 'early' in aggregates
        assert 'planning_early' in aggregates
        assert 'tool_selection_middle' in aggregates

        # Check counts
        assert aggregates['planning']['count'] == 9  # 3 positions × 3 samples
        assert aggregates['tool_selection']['count'] == 4  # 2 positions × 2 samples
        assert aggregates['early']['count'] == 5  # 3 planning + 2 tool


class TestStatisticalAnalysis:
    """Test statistical analysis functions."""

    def test_statistical_analysis(self):
        """Test ANOVA and statistical tests."""
        # Create sample results with variation
        results = []

        # Planning errors (high negative CCG)
        for i in range(10):
            results.append(CCGResult(
                perturbation_id=f"planning_{i}",
                experiment_id="exp_001",
                judge_name="claude",
                perturbation_type="planning",
                perturbation_position="early",
                tsd=1,
                ser=5,
                tcs=150.0,
                judge_overall_score=50.0,
                jps=50.0,
                ccg=-0.67
            ))

        # Parameter errors (high positive CCG)
        for i in range(10):
            results.append(CCGResult(
                perturbation_id=f"parameter_{i}",
                experiment_id="exp_001",
                judge_name="claude",
                perturbation_type="parameter",
                perturbation_position="late",
                tsd=0,
                ser=1,
                tcs=10.0,
                judge_overall_score=60.0,
                jps=40.0,
                ccg=3.0
            ))

        tests = statistical_analysis(results)

        # Should have ANOVA results
        assert 'anova_type' in tests
        assert 'anova_position' in tests
        assert 'overall' in tests

        # Check ANOVA structure
        anova_type = tests['anova_type']
        assert 'f_stat' in anova_type
        assert 'p_value' in anova_type
        assert 'significant' in anova_type

    def test_statistical_analysis_empty(self):
        """Test statistical analysis with no results."""
        tests = statistical_analysis([])
        assert tests == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
