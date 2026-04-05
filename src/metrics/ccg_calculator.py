"""
CCG Calculator - Computes Criticality-Calibration Gap metrics and statistical analysis.

CCG = (JPS - TCS) / TCS

Where:
- JPS (Judge Penalty Score) = 100 - overall_score (from LLM judge)
- TCS (True Criticality Score) = ground-truth error impact

Interpretation:
- CCG < 0: Under-penalization (judge missed critical error)
- CCG > 0: Over-penalization (judge overreacted to minor error)
- CCG = 0: Perfect calibration

This module provides:
- Per-perturbation CCG calculation
- Aggregation by condition, type, position, benchmark
- ANOVA statistical tests
- Effect size calculations (Cohen's d, eta-squared)
- Hypothesis testing support
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


class CCGCalculator:
    """
    Computes Criticality-Calibration Gap (CCG) metrics.

    CCG quantifies the miscalibration between judge-perceived error severity (JPS)
    and ground-truth error criticality (TCS).

    Provides:
    - Per-perturbation CCG calculation
    - Aggregation by multiple dimensions
    - Statistical analysis (ANOVA, Tukey HSD)
    - Effect size computation
    - Hypothesis assessment
    """

    def __init__(self, config: Dict = None):
        """
        Initialize CCG calculator.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}

    def compute_ccg(self, jps: float, tcs: float) -> Optional[float]:
        """
        Compute CCG for a single perturbation.

        Args:
            jps: Judge Penalty Score (100 - overall_score)
            tcs: True Criticality Score

        Returns:
            CCG value, or None if TCS is 0 (excluded from analysis)

        Note:
            - TCS values are capped at 100 to ensure valid range [0, 100]
            - TCS = 0 samples are excluded (return None) to avoid division by zero
              and bias from "perfectly calibrated" appearance (see BUG-002)
        """
        # BUG-001 fix: Cap TCS at 100 to ensure valid range
        tcs = min(tcs, 100)

        # BUG-002 fix: Exclude TCS=0 samples (return None instead of 0)
        # Rationale: Returning 0 makes samples appear "perfectly calibrated"
        # even when JPS varies widely (e.g., JPS=85, TCS=0 should show over-penalization)
        if tcs == 0:
            return None
        return (jps - tcs) / tcs

    def compute_all(
        self,
        evaluations: List[Dict],
        tcs_values: Dict[str, float]
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Compute CCG for all evaluations.

        Args:
            evaluations: List of judge evaluation dicts with:
                - perturbation_id
                - jps or overall_score
                - perturbation_type
                - perturbation_position
                - benchmark (optional)
            tcs_values: Dict mapping perturbation_id -> TCS

        Returns:
            Tuple of:
            - DataFrame with CCG values and metadata (excludes TCS=0 samples)
            - Dict with exclusion stats:
                - total_evaluated: Total samples processed
                - tcs_zero_excluded: Samples excluded due to TCS=0
                - tcs_capped: Samples where TCS was capped from >100 to 100
                - final_sample_size: Samples with valid CCG
        """
        records = []
        exclusion_stats = {
            "total_evaluated": 0,
            "tcs_zero_excluded": 0,
            "tcs_capped": 0,
            "no_score_skipped": 0,
        }

        for eval_dict in evaluations:
            pert_id = eval_dict.get("perturbation_id")

            # Get JPS (computed if only overall_score available)
            if "jps" in eval_dict:
                jps = eval_dict["jps"]
            elif "overall_score" in eval_dict:
                jps = 100 - eval_dict["overall_score"]
            else:
                exclusion_stats["no_score_skipped"] += 1
                continue  # Skip if no score available

            exclusion_stats["total_evaluated"] += 1

            # Get TCS
            raw_tcs = tcs_values.get(pert_id, 50)  # Default to 50 if unknown

            # Track TCS capping (BUG-001)
            if raw_tcs > 100:
                exclusion_stats["tcs_capped"] += 1

            # Cap TCS at 100
            tcs = min(raw_tcs, 100)

            # Compute CCG (returns None for TCS=0)
            ccg = self.compute_ccg(jps, tcs)

            # BUG-002: Track and exclude TCS=0 samples
            if ccg is None:
                exclusion_stats["tcs_zero_excluded"] += 1
                continue  # Exclude from analysis

            records.append({
                "perturbation_id": pert_id,
                "jps": jps,
                "tcs": tcs,
                "tcs_raw": raw_tcs,  # Store original for transparency
                "ccg": ccg,
                "perturbation_type": eval_dict.get("perturbation_type"),
                "position": eval_dict.get("perturbation_position") or eval_dict.get("position"),
                "benchmark": eval_dict.get("benchmark", "unknown"),
                "condition": f"{eval_dict.get('perturbation_type')}_{eval_dict.get('perturbation_position') or eval_dict.get('position')}"
            })

        exclusion_stats["final_sample_size"] = len(records)

        return pd.DataFrame(records), exclusion_stats

    def aggregate(self, df: pd.DataFrame, by: str) -> pd.DataFrame:
        """
        Aggregate CCG statistics by a dimension.

        Args:
            df: DataFrame with CCG values
            by: Column to group by (e.g., "condition", "position", "perturbation_type")

        Returns:
            DataFrame with aggregated statistics
        """
        return df.groupby(by).agg({
            "ccg": ["mean", "std", "count", "min", "max"],
            "jps": "mean",
            "tcs": "mean"
        }).round(3)

    def run_anova(self, df: pd.DataFrame, factors: List[str]) -> Dict:
        """
        Run ANOVA tests for specified factors.

        Args:
            df: DataFrame with CCG values
            factors: List of factor columns to test (e.g., ["perturbation_type", "position"])

        Returns:
            Dict with F-statistics, p-values, and effect sizes for each factor
        """
        try:
            from scipy import stats
        except ImportError:
            return {"error": "scipy not installed"}

        results = {}

        for factor in factors:
            if factor not in df.columns:
                continue

            # Get groups
            groups = [group["ccg"].dropna().values for name, group in df.groupby(factor)]

            # Filter out empty groups
            groups = [g for g in groups if len(g) > 0]

            if len(groups) < 2:
                results[factor] = {"error": "Need at least 2 groups"}
                continue

            # Run one-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups)

            # Compute eta-squared (effect size)
            ss_between = sum(len(g) * (np.mean(g) - df["ccg"].mean())**2 for g in groups)
            ss_total = sum((df["ccg"] - df["ccg"].mean())**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            results[factor] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "eta_squared": float(eta_squared),
                "significant": p_value < 0.05,
                "n_groups": len(groups)
            }

        return results

    def run_tukey_hsd(self, df: pd.DataFrame, factor: str) -> Optional[pd.DataFrame]:
        """
        Run Tukey HSD post-hoc test.

        Args:
            df: DataFrame with CCG values
            factor: Factor column for pairwise comparisons

        Returns:
            DataFrame with Tukey HSD results, or None if test fails
        """
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
        except ImportError:
            return None

        if factor not in df.columns:
            return None

        # Filter out NaN values
        clean_df = df[["ccg", factor]].dropna()

        if len(clean_df) < 3:
            return None

        try:
            tukey = pairwise_tukeyhsd(clean_df["ccg"], clean_df[factor], alpha=0.05)
            return pd.DataFrame(
                data=tukey._results_table.data[1:],
                columns=tukey._results_table.data[0]
            )
        except Exception:
            return None

    def compute_effect_sizes(self, df: pd.DataFrame) -> Dict:
        """
        Compute Cohen's d effect sizes for key comparisons.

        Comparisons:
        - Early vs Late position
        - Planning vs Parameter type
        - Structural (planning+tool_selection) vs Surface (parameter+data_reference)

        Args:
            df: DataFrame with CCG values

        Returns:
            Dict with effect size results
        """
        effect_sizes = {}

        # Early vs Late
        early = df[df["position"] == "early"]["ccg"].dropna()
        late = df[df["position"] == "late"]["ccg"].dropna()
        if len(early) > 1 and len(late) > 1:
            effect_sizes["early_vs_late"] = {
                "cohens_d": self._cohens_d(early, late),
                "early_mean": float(early.mean()),
                "late_mean": float(late.mean()),
                "n_early": len(early),
                "n_late": len(late)
            }

        # Planning vs Parameter
        planning = df[df["perturbation_type"] == "planning"]["ccg"].dropna()
        parameter = df[df["perturbation_type"] == "parameter"]["ccg"].dropna()
        if len(planning) > 1 and len(parameter) > 1:
            effect_sizes["planning_vs_parameter"] = {
                "cohens_d": self._cohens_d(planning, parameter),
                "planning_mean": float(planning.mean()),
                "parameter_mean": float(parameter.mean()),
                "n_planning": len(planning),
                "n_parameter": len(parameter)
            }

        # Structural vs Surface
        structural_types = ["planning", "tool_selection"]
        surface_types = ["parameter", "data_reference"]

        structural = df[df["perturbation_type"].isin(structural_types)]["ccg"].dropna()
        surface = df[df["perturbation_type"].isin(surface_types)]["ccg"].dropna()
        if len(structural) > 1 and len(surface) > 1:
            effect_sizes["structural_vs_surface"] = {
                "cohens_d": self._cohens_d(structural, surface),
                "structural_mean": float(structural.mean()),
                "surface_mean": float(surface.mean()),
                "n_structural": len(structural),
                "n_surface": len(surface)
            }

        # Early structural vs Late surface (key hypothesis comparison)
        early_structural = df[
            (df["position"] == "early") &
            (df["perturbation_type"].isin(structural_types))
        ]["ccg"].dropna()

        late_surface = df[
            (df["position"] == "late") &
            (df["perturbation_type"].isin(surface_types))
        ]["ccg"].dropna()

        if len(early_structural) > 1 and len(late_surface) > 1:
            effect_sizes["early_structural_vs_late_surface"] = {
                "cohens_d": self._cohens_d(early_structural, late_surface),
                "early_structural_mean": float(early_structural.mean()),
                "late_surface_mean": float(late_surface.mean()),
                "n_early_structural": len(early_structural),
                "n_late_surface": len(late_surface)
            }

        return effect_sizes

    def _cohens_d(self, group1, group2) -> float:
        """
        Compute Cohen's d effect size.

        Args:
            group1: First group values
            group2: Second group values

        Returns:
            Cohen's d value
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

        if pooled_std == 0:
            return 0.0

        return float((group1.mean() - group2.mean()) / pooled_std)

    def assess_hypothesis(self, df: pd.DataFrame) -> Dict:
        """
        Assess whether results support the research hypothesis.

        Hypothesis: LLM judges under-penalize early structural failures
        while over-penalizing late local mistakes.

        Evidence required:
        1. Early structural CCG < 0 (under-penalization)
        2. Late surface CCG > 0 (over-penalization)
        3. Significant difference (p < 0.05)
        4. Meaningful effect size (|d| > 0.5)

        Args:
            df: DataFrame with CCG values

        Returns:
            Dict with hypothesis assessment
        """
        structural_types = ["planning", "tool_selection"]
        surface_types = ["parameter", "data_reference"]

        # Early structural errors
        early_structural = df[
            (df["position"] == "early") &
            (df["perturbation_type"].isin(structural_types))
        ]["ccg"].dropna()

        # Late surface errors
        late_surface = df[
            (df["position"] == "late") &
            (df["perturbation_type"].isin(surface_types))
        ]["ccg"].dropna()

        # Compute means
        early_structural_mean = float(early_structural.mean()) if len(early_structural) > 0 else None
        late_surface_mean = float(late_surface.mean()) if len(late_surface) > 0 else None

        # Compute effect size
        effect_size = None
        if len(early_structural) > 1 and len(late_surface) > 1:
            effect_size = self._cohens_d(early_structural, late_surface)

        # Run t-test
        t_stat = None
        p_value = None
        if len(early_structural) > 1 and len(late_surface) > 1:
            try:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(early_structural, late_surface)
                t_stat = float(t_stat)
                p_value = float(p_value)
            except ImportError:
                pass

        # Assess criteria
        criteria = {
            "early_under_penalized": early_structural_mean is not None and early_structural_mean < 0,
            "late_over_penalized": late_surface_mean is not None and late_surface_mean > 0,
            "direction_correct": (
                early_structural_mean is not None and
                late_surface_mean is not None and
                early_structural_mean < late_surface_mean
            ),
            "statistically_significant": p_value is not None and p_value < 0.05,
            "meaningful_effect_size": effect_size is not None and abs(effect_size) > 0.5
        }

        # Overall hypothesis support
        criteria_met = sum(criteria.values())
        total_criteria = len(criteria)

        if criteria_met >= 4:
            support_level = "strong"
        elif criteria_met >= 3:
            support_level = "moderate"
        elif criteria_met >= 2:
            support_level = "weak"
        else:
            support_level = "not supported"

        return {
            "early_structural_mean_ccg": early_structural_mean,
            "late_surface_mean_ccg": late_surface_mean,
            "difference": (
                late_surface_mean - early_structural_mean
                if early_structural_mean is not None and late_surface_mean is not None
                else None
            ),
            "effect_size_cohens_d": effect_size,
            "t_statistic": t_stat,
            "p_value": p_value,
            "criteria": criteria,
            "criteria_met": criteria_met,
            "total_criteria": total_criteria,
            "support_level": support_level,
            "n_early_structural": len(early_structural),
            "n_late_surface": len(late_surface)
        }

    def generate_report(self, df: pd.DataFrame, exclusion_stats: Dict = None) -> Dict:
        """
        Generate comprehensive CCG analysis report.

        Args:
            df: DataFrame with CCG values
            exclusion_stats: Optional dict with exclusion statistics from compute_all()
                - total_evaluated: Total samples processed
                - tcs_zero_excluded: Samples excluded due to TCS=0
                - tcs_capped: Samples where TCS was capped from >100 to 100
                - final_sample_size: Samples with valid CCG

        Returns:
            Complete analysis report dict
        """
        report = {
            "overall": {
                "n_samples": len(df),
                "mean_ccg": float(df["ccg"].mean()) if len(df) > 0 else 0,
                "std_ccg": float(df["ccg"].std()) if len(df) > 0 else 0,
                "min_ccg": float(df["ccg"].min()) if len(df) > 0 else 0,
                "max_ccg": float(df["ccg"].max()) if len(df) > 0 else 0,
                "median_ccg": float(df["ccg"].median()) if len(df) > 0 else 0,
                "mean_jps": float(df["jps"].mean()) if len(df) > 0 else 0,
                "mean_tcs": float(df["tcs"].mean()) if len(df) > 0 else 0
            },
            "by_condition": self._aggregate_to_dict(df, "condition"),
            "by_position": self._aggregate_to_dict(df, "position"),
            "by_type": self._aggregate_to_dict(df, "perturbation_type"),
            "by_benchmark": self._aggregate_to_dict(df, "benchmark"),
            "anova": self.run_anova(df, ["perturbation_type", "position"]),
            "effect_sizes": self.compute_effect_sizes(df),
            "hypothesis_support": self.assess_hypothesis(df)
        }

        # Include exclusion stats if provided (BUG-001, BUG-002 documentation)
        if exclusion_stats:
            report["data_quality"] = {
                "total_evaluated": exclusion_stats.get("total_evaluated", 0),
                "tcs_zero_excluded": exclusion_stats.get("tcs_zero_excluded", 0),
                "tcs_capped_count": exclusion_stats.get("tcs_capped", 0),
                "final_sample_size": exclusion_stats.get("final_sample_size", len(df)),
                "exclusion_rate": (
                    exclusion_stats.get("tcs_zero_excluded", 0) /
                    exclusion_stats.get("total_evaluated", 1)
                    if exclusion_stats.get("total_evaluated", 0) > 0 else 0
                )
            }

        return report

    def _aggregate_to_dict(self, df: pd.DataFrame, by: str) -> Dict:
        """Convert aggregation to nested dict format."""
        if by not in df.columns:
            return {}

        result = {}
        for group_name, group_df in df.groupby(by):
            ccg_values = group_df["ccg"].dropna()
            result[str(group_name)] = {
                "mean_ccg": float(ccg_values.mean()) if len(ccg_values) > 0 else 0,
                "std_ccg": float(ccg_values.std()) if len(ccg_values) > 0 else 0,
                "count": len(ccg_values),
                "mean_jps": float(group_df["jps"].mean()),
                "mean_tcs": float(group_df["tcs"].mean())
            }
        return result

    def save_results(
        self,
        df: pd.DataFrame,
        report: Dict,
        output_dir: str
    ):
        """
        Save CCG results to files.

        Args:
            df: DataFrame with CCG values
            report: Analysis report dict
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save raw results
        df.to_csv(output_path / "ccg_raw_results.csv", index=False)
        print(f"Saved raw results to {output_path / 'ccg_raw_results.csv'}")

        # Save report as JSON
        with open(output_path / "ccg_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved report to {output_path / 'ccg_report.json'}")

    def print_summary(self, report: Dict):
        """Print formatted summary of CCG analysis."""
        print("\n" + "=" * 70)
        print("CCG ANALYSIS SUMMARY")
        print("=" * 70)

        # Data quality section (BUG-001, BUG-002 fixes)
        data_quality = report.get("data_quality", {})
        if data_quality:
            print(f"\nData Quality (after bug fixes):")
            print(f"  Total evaluated:     {data_quality.get('total_evaluated', 'N/A')}")
            print(f"  TCS=0 excluded:      {data_quality.get('tcs_zero_excluded', 0)} (BUG-002 fix)")
            print(f"  TCS capped (>100):   {data_quality.get('tcs_capped_count', 0)} (BUG-001 fix)")
            print(f"  Final sample size:   {data_quality.get('final_sample_size', 'N/A')}")
            exclusion_rate = data_quality.get('exclusion_rate', 0)
            print(f"  Exclusion rate:      {exclusion_rate:.1%}")

        overall = report.get("overall", {})
        print(f"\nOverall Statistics (n={overall.get('n_samples', 0)}):")
        print(f"  Mean CCG:  {overall.get('mean_ccg', 0):.3f} +/- {overall.get('std_ccg', 0):.3f}")
        print(f"  Mean JPS:  {overall.get('mean_jps', 0):.1f}")
        print(f"  Mean TCS:  {overall.get('mean_tcs', 0):.1f}")

        # By position
        print(f"\nBy Position:")
        for pos, stats in report.get("by_position", {}).items():
            print(f"  {pos:10s}: CCG = {stats['mean_ccg']:.3f} (n={stats['count']})")

        # By type
        print(f"\nBy Perturbation Type:")
        for ptype, stats in report.get("by_type", {}).items():
            print(f"  {ptype:20s}: CCG = {stats['mean_ccg']:.3f} (n={stats['count']})")

        # ANOVA
        anova = report.get("anova", {})
        if anova:
            print(f"\nANOVA Tests:")
            for factor, result in anova.items():
                if "error" not in result:
                    sig = "*" if result.get("significant") else ""
                    print(f"  {factor:20s}: F={result['f_statistic']:.2f}, p={result['p_value']:.4f}{sig}, eta2={result['eta_squared']:.3f}")

        # Effect sizes
        effects = report.get("effect_sizes", {})
        if effects:
            print(f"\nEffect Sizes (Cohen's d):")
            for comparison, result in effects.items():
                d = result.get("cohens_d", 0)
                magnitude = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
                print(f"  {comparison:35s}: d = {d:.3f} ({magnitude})")

        # Hypothesis assessment
        hypothesis = report.get("hypothesis_support", {})
        if hypothesis:
            print(f"\nHypothesis Assessment:")
            print(f"  Early structural mean CCG: {hypothesis.get('early_structural_mean_ccg', 'N/A')}")
            print(f"  Late surface mean CCG:     {hypothesis.get('late_surface_mean_ccg', 'N/A')}")
            print(f"  Support level:             {hypothesis.get('support_level', 'N/A')}")
            print(f"  Criteria met:              {hypothesis.get('criteria_met', 0)}/{hypothesis.get('total_criteria', 0)}")

        print("=" * 70)
