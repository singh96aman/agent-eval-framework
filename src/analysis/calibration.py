"""
Calibration Analysis for RQ1: Consequentiality Calibration.

Computes correlation between Judge Penalty Score (JPS) and
Outcome Degradation (OD) to assess judge calibration.

Metrics:
- Spearman r: Rank correlation (primary)
- Pearson r: Linear correlation
- ECE: Expected Calibration Error
- MCE: Maximum Calibration Error
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


@dataclass
class CalibrationReport:
    """Report from calibration analysis."""
    n_samples: int
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    ece: float
    mce: float
    n_bins: int
    bin_stats: List[Dict[str, Any]]
    bootstrap_ci: Dict[str, Tuple[float, float]]
    stratified: Dict[str, Dict[str, Any]]
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "correlation": {
                "spearman_r": self.spearman_r,
                "spearman_p": self.spearman_p,
                "pearson_r": self.pearson_r,
                "pearson_p": self.pearson_p
            },
            "calibration_error": {
                "ece": self.ece,
                "mce": self.mce,
                "n_bins": self.n_bins
            },
            "bin_stats": self.bin_stats,
            "bootstrap_ci": self.bootstrap_ci,
            "stratified": self.stratified,
            "interpretation": self.interpretation
        }


class CalibrationAnalyzer:
    """
    Analyzes calibration between JPS and OD.

    JPS = 100 - overall_score (judge's penalty)
    OD = (baseline_outcome - perturbed_outcome) / 100 (actual impact)

    Good calibration: High JPS when high OD (judge penalizes what matters).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize calibration analyzer.

        Args:
            config: Configuration with:
                - n_bins: Number of bins for ECE (default: 10)
                - bootstrap_iterations: For CI computation (default: 1000)
                - stratify_by: List of stratification dimensions
                - output_dir: Where to save results
        """
        self.config = config
        self.n_bins = config.get("n_bins", 10)
        self.bootstrap_iterations = config.get("bootstrap_iterations", 1000)
        self.stratify_by = config.get("stratify_by", ["benchmark", "perturbation_type", "position"])
        self.output_dir = Path(config.get("output_dir", "results/rq1"))

    def compute_correlation(
        self,
        jps_values: np.ndarray,
        od_values: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute Spearman and Pearson correlations.

        Args:
            jps_values: Judge Penalty Scores
            od_values: Outcome Degradation values

        Returns:
            Dict with correlation coefficients and p-values
        """
        from scipy import stats

        # Spearman (rank correlation - primary metric)
        spearman_r, spearman_p = stats.spearmanr(jps_values, od_values)

        # Pearson (linear correlation)
        pearson_r, pearson_p = stats.pearsonr(jps_values, od_values)

        return {
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p)
        }

    def compute_ece_mce(
        self,
        jps_values: np.ndarray,
        od_values: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        """
        Compute Expected and Maximum Calibration Error.

        ECE measures how well the JPS values predict actual OD.
        We bin by JPS and compare mean JPS to mean OD in each bin.

        Args:
            jps_values: Judge Penalty Scores (0-100 scale)
            od_values: Outcome Degradation values (0-1 scale)
            n_bins: Number of bins

        Returns:
            Tuple of (ECE, MCE, bin_stats)
        """
        # Normalize JPS to 0-1 scale for comparison with OD
        jps_normalized = jps_values / 100.0

        # Create bins based on JPS
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(jps_normalized, bin_edges[1:-1])

        bin_stats = []
        weighted_errors = []
        max_error = 0.0

        for i in range(n_bins):
            mask = bin_indices == i
            n_in_bin = np.sum(mask)

            if n_in_bin > 0:
                mean_jps = float(np.mean(jps_normalized[mask]))
                mean_od = float(np.mean(od_values[mask]))
                error = abs(mean_jps - mean_od)

                bin_stats.append({
                    "bin": i,
                    "bin_range": [float(bin_edges[i]), float(bin_edges[i + 1])],
                    "n_samples": int(n_in_bin),
                    "mean_jps_normalized": mean_jps,
                    "mean_od": mean_od,
                    "calibration_error": error
                })

                weighted_errors.append(n_in_bin * error)
                max_error = max(max_error, error)
            else:
                bin_stats.append({
                    "bin": i,
                    "bin_range": [float(bin_edges[i]), float(bin_edges[i + 1])],
                    "n_samples": 0,
                    "mean_jps_normalized": None,
                    "mean_od": None,
                    "calibration_error": None
                })

        # ECE = weighted average of calibration errors
        total_samples = len(jps_values)
        ece = sum(weighted_errors) / total_samples if total_samples > 0 else 0.0

        return float(ece), float(max_error), bin_stats

    def bootstrap_ci(
        self,
        jps_values: np.ndarray,
        od_values: np.ndarray,
        n_iterations: int = 1000,
        ci_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for correlations.

        Args:
            jps_values: Judge Penalty Scores
            od_values: Outcome Degradation values
            n_iterations: Number of bootstrap samples
            ci_level: Confidence level (default 0.95)

        Returns:
            Dict with CIs for each metric
        """
        from scipy import stats

        n = len(jps_values)
        spearman_samples = []
        pearson_samples = []

        np.random.seed(42)  # Reproducibility

        for _ in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            jps_sample = jps_values[indices]
            od_sample = od_values[indices]

            # Compute correlations
            try:
                sp_r, _ = stats.spearmanr(jps_sample, od_sample)
                pe_r, _ = stats.pearsonr(jps_sample, od_sample)
                spearman_samples.append(sp_r)
                pearson_samples.append(pe_r)
            except Exception:
                continue

        # Compute CIs
        alpha = (1 - ci_level) / 2
        lower_idx = int(alpha * len(spearman_samples))
        upper_idx = int((1 - alpha) * len(spearman_samples))

        spearman_sorted = sorted(spearman_samples)
        pearson_sorted = sorted(pearson_samples)

        return {
            "spearman_r": (
                spearman_sorted[lower_idx] if spearman_sorted else 0,
                spearman_sorted[upper_idx - 1] if spearman_sorted else 0
            ),
            "pearson_r": (
                pearson_sorted[lower_idx] if pearson_sorted else 0,
                pearson_sorted[upper_idx - 1] if pearson_sorted else 0
            )
        }

    def stratified_analysis(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute calibration metrics stratified by different dimensions.

        Args:
            df: DataFrame with jps, od, and stratification columns

        Returns:
            Dict mapping dimension -> group -> metrics
        """
        from scipy import stats

        results = {}

        for dim in self.stratify_by:
            if dim not in df.columns:
                continue

            dim_results = {}
            for group, group_df in df.groupby(dim):
                if len(group_df) < 3:
                    continue

                jps = group_df["jps"].values
                od = group_df["od"].values

                try:
                    sp_r, sp_p = stats.spearmanr(jps, od)
                    pe_r, pe_p = stats.pearsonr(jps, od)
                except Exception:
                    sp_r, sp_p, pe_r, pe_p = 0, 1, 0, 1

                dim_results[str(group)] = {
                    "n": len(group_df),
                    "spearman_r": float(sp_r),
                    "spearman_p": float(sp_p),
                    "pearson_r": float(pe_r),
                    "mean_jps": float(jps.mean()),
                    "mean_od": float(od.mean())
                }

            results[dim] = dim_results

        return results

    def interpret_results(
        self,
        spearman_r: float,
        ece: float
    ) -> str:
        """
        Generate interpretation of calibration results.

        Based on success criteria from Requirements.MD:
        - Judges calibrated: r > 0.5, ECE < 0.1
        - Judges detect but don't size: r 0.2-0.5, ECE 0.1-0.3
        - Judges random: r < 0.15, ECE > 0.3
        """
        if spearman_r > 0.5 and ece < 0.1:
            return (
                "STRONG CALIBRATION: Judges are well-calibrated to actual "
                "outcome damage. JPS strongly correlates with OD (r > 0.5) "
                "and calibration error is low (ECE < 0.1)."
            )
        elif 0.2 <= spearman_r <= 0.5 and 0.1 <= ece <= 0.3:
            return (
                "MODERATE CALIBRATION: Judges detect errors but don't accurately "
                "size their impact. There is a meaningful correlation (r = 0.2-0.5) "
                "but moderate calibration error (ECE = 0.1-0.3). "
                "This suggests judges can identify problematic behaviors but "
                "may over- or under-penalize based on factors other than outcome."
            )
        elif spearman_r < 0.15 or ece > 0.3:
            return (
                "POOR CALIBRATION: Judges show weak or no correlation with actual "
                "outcome damage. Penalty scores appear largely independent of "
                "whether errors actually affect task completion."
            )
        else:
            return (
                f"MIXED RESULTS: Spearman r = {spearman_r:.3f}, ECE = {ece:.3f}. "
                "Results don't cleanly fit expected categories. Further analysis needed."
            )

    def analyze(
        self,
        evaluations: List[Dict[str, Any]],
        perturbations: List[Dict[str, Any]]
    ) -> CalibrationReport:
        """
        Run full calibration analysis.

        Args:
            evaluations: Judge evaluations with overall_score
            perturbations: Perturbations with od field

        Returns:
            CalibrationReport with all metrics
        """
        # Build merged dataset
        od_lookup = {}
        for p in perturbations:
            if p.get("od"):
                od_lookup[p.get("perturbed_trajectory_id")] = {
                    "od": p["od"].get("value", 0),
                    "perturbation_type": p.get("perturbation_type"),
                    "position": p.get("perturbation_position"),
                    "perturbation_id": p.get("perturbation_id")
                }

        records = []
        for e in evaluations:
            traj_id = e.get("trajectory_id")
            if traj_id in od_lookup:
                od_info = od_lookup[traj_id]
                jps = 100 - e.get("overall_score", 50)

                records.append({
                    "trajectory_id": traj_id,
                    "perturbation_id": od_info.get("perturbation_id"),
                    "jps": jps,
                    "od": od_info["od"],
                    "perturbation_type": od_info.get("perturbation_type"),
                    "position": od_info.get("position"),
                    "benchmark": self._extract_benchmark(traj_id)
                })

        if not records:
            return CalibrationReport(
                n_samples=0,
                spearman_r=0, spearman_p=1,
                pearson_r=0, pearson_p=1,
                ece=1, mce=1,
                n_bins=self.n_bins,
                bin_stats=[],
                bootstrap_ci={},
                stratified={},
                interpretation="No valid samples for calibration analysis."
            )

        df = pd.DataFrame(records)
        jps_values = df["jps"].values
        od_values = df["od"].values

        # Compute correlations
        corr = self.compute_correlation(jps_values, od_values)

        # Compute ECE/MCE
        ece, mce, bin_stats = self.compute_ece_mce(
            jps_values, od_values, self.n_bins
        )

        # Bootstrap CIs
        ci = self.bootstrap_ci(
            jps_values, od_values, self.bootstrap_iterations
        )

        # Stratified analysis
        stratified = self.stratified_analysis(df)

        # Interpretation
        interpretation = self.interpret_results(corr["spearman_r"], ece)

        return CalibrationReport(
            n_samples=len(df),
            spearman_r=corr["spearman_r"],
            spearman_p=corr["spearman_p"],
            pearson_r=corr["pearson_r"],
            pearson_p=corr["pearson_p"],
            ece=ece,
            mce=mce,
            n_bins=self.n_bins,
            bin_stats=bin_stats,
            bootstrap_ci=ci,
            stratified=stratified,
            interpretation=interpretation
        )

    def _extract_benchmark(self, trajectory_id: str) -> str:
        """Extract benchmark from trajectory ID."""
        traj_id = trajectory_id.lower()
        if "toolbench" in traj_id:
            return "toolbench"
        elif "gaia" in traj_id:
            return "gaia"
        elif "swe" in traj_id:
            return "swebench"
        return "unknown"

    def plot_calibration_curve(
        self,
        bin_stats: List[Dict[str, Any]],
        output_path: Path
    ):
        """
        Generate calibration curve plot.

        Args:
            bin_stats: Bin statistics from ECE computation
            output_path: Where to save the plot
        """
        valid_bins = [b for b in bin_stats if b["mean_jps_normalized"] is not None]

        if not valid_bins:
            print("   No valid bins for calibration curve")
            return

        jps_means = [b["mean_jps_normalized"] for b in valid_bins]
        od_means = [b["mean_od"] for b in valid_bins]
        bin_sizes = [b["n_samples"] for b in valid_bins]

        fig, ax = plt.subplots(figsize=(8, 8))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)

        # Scatter with size proportional to bin count
        sizes = [max(50, s * 5) for s in bin_sizes]
        scatter = ax.scatter(
            jps_means, od_means, s=sizes, alpha=0.7,
            c=range(len(valid_bins)), cmap='viridis',
            label='Bin means'
        )

        # Connect points
        ax.plot(jps_means, od_means, 'b-', alpha=0.5)

        ax.set_xlabel('Mean JPS (normalized 0-1)', fontsize=12)
        ax.set_ylabel('Mean OD (Outcome Degradation)', fontsize=12)
        ax.set_title('Calibration Curve: JPS vs OD', fontsize=14)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved calibration curve to {output_path}")

    def plot_scatter(
        self,
        jps_values: np.ndarray,
        od_values: np.ndarray,
        output_path: Path,
        spearman_r: float
    ):
        """
        Generate JPS vs OD scatter plot.

        Args:
            jps_values: Judge Penalty Scores
            od_values: Outcome Degradation values
            output_path: Where to save the plot
            spearman_r: Correlation coefficient for annotation
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(jps_values, od_values, alpha=0.5, s=50)

        # Add regression line
        z = np.polyfit(jps_values, od_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(jps_values), max(jps_values), 100)
        ax.plot(x_line, p(x_line), 'r-', alpha=0.7, label=f'Linear fit')

        ax.set_xlabel('JPS (Judge Penalty Score)', fontsize=12)
        ax.set_ylabel('OD (Outcome Degradation)', fontsize=12)
        ax.set_title(f'JPS vs OD Scatter (Spearman r = {spearman_r:.3f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved scatter plot to {output_path}")

    def save_results(
        self,
        report: CalibrationReport,
        evaluations: List[Dict[str, Any]],
        perturbations: List[Dict[str, Any]]
    ):
        """
        Save all calibration results and generate plots.

        Args:
            report: CalibrationReport from analyze()
            evaluations: Original evaluations
            perturbations: Original perturbations
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save statistics JSON
        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"   Saved statistics to {stats_path}")

        # Save findings markdown
        findings_path = self.output_dir / "findings.md"
        self._write_findings(report, findings_path)

        # Generate plots if we have data
        if report.n_samples > 0:
            # Calibration curve
            self.plot_calibration_curve(
                report.bin_stats,
                self.output_dir / "calibration_curve.png"
            )

            # Scatter plot - rebuild data
            od_lookup = {}
            for p in perturbations:
                if p.get("od"):
                    od_lookup[p.get("perturbed_trajectory_id")] = p["od"].get("value", 0)

            jps_list = []
            od_list = []
            for e in evaluations:
                traj_id = e.get("trajectory_id")
                if traj_id in od_lookup:
                    jps_list.append(100 - e.get("overall_score", 50))
                    od_list.append(od_lookup[traj_id])

            if jps_list:
                self.plot_scatter(
                    np.array(jps_list),
                    np.array(od_list),
                    self.output_dir / "jps_vs_od_scatter.png",
                    report.spearman_r
                )

    def _write_findings(self, report: CalibrationReport, output_path: Path):
        """Write findings markdown file."""
        lines = [
            "# RQ1: Consequentiality Calibration Findings",
            "",
            "## Research Question",
            "**Do judge penalty scores correlate with actual outcome damage?**",
            "",
            "## Summary",
            f"- **Samples analyzed**: {report.n_samples}",
            f"- **Spearman r**: {report.spearman_r:.3f} (p = {report.spearman_p:.4f})",
            f"- **Pearson r**: {report.pearson_r:.3f} (p = {report.pearson_p:.4f})",
            f"- **ECE**: {report.ece:.3f}",
            f"- **MCE**: {report.mce:.3f}",
            "",
            "## Interpretation",
            report.interpretation,
            "",
            "## Bootstrap 95% Confidence Intervals",
        ]

        for metric, (low, high) in report.bootstrap_ci.items():
            lines.append(f"- **{metric}**: [{low:.3f}, {high:.3f}]")

        lines.extend([
            "",
            "## Stratified Analysis",
        ])

        for dim, groups in report.stratified.items():
            lines.append(f"\n### By {dim.replace('_', ' ').title()}")
            lines.append("| Group | N | Spearman r | Mean JPS | Mean OD |")
            lines.append("|-------|---|------------|----------|---------|")
            for group, stats in groups.items():
                lines.append(
                    f"| {group} | {stats['n']} | {stats['spearman_r']:.3f} | "
                    f"{stats['mean_jps']:.1f} | {stats['mean_od']:.3f} |"
                )

        lines.extend([
            "",
            "## Calibration Bins",
            "| Bin | Range | N | Mean JPS | Mean OD | Error |",
            "|-----|-------|---|----------|---------|-------|",
        ])

        for b in report.bin_stats:
            if b["mean_jps_normalized"] is not None:
                lines.append(
                    f"| {b['bin']} | [{b['bin_range'][0]:.2f}, {b['bin_range'][1]:.2f}] | "
                    f"{b['n_samples']} | {b['mean_jps_normalized']:.3f} | "
                    f"{b['mean_od']:.3f} | {b['calibration_error']:.3f} |"
                )

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

        print(f"   Saved findings to {output_path}")

    def print_summary(self, report: CalibrationReport):
        """Print summary of calibration results."""
        print("\n" + "=" * 70)
        print("RQ1 CALIBRATION ANALYSIS SUMMARY")
        print("=" * 70)

        print(f"\nSamples: {report.n_samples}")
        print(f"\nCorrelation:")
        print(f"  Spearman r: {report.spearman_r:.3f} (p = {report.spearman_p:.4f})")
        print(f"  Pearson r:  {report.pearson_r:.3f} (p = {report.pearson_p:.4f})")

        print(f"\nCalibration Error:")
        print(f"  ECE: {report.ece:.3f}")
        print(f"  MCE: {report.mce:.3f}")

        print(f"\nBootstrap 95% CI:")
        for metric, (low, high) in report.bootstrap_ci.items():
            print(f"  {metric}: [{low:.3f}, {high:.3f}]")

        print(f"\nInterpretation:")
        print(f"  {report.interpretation}")

        print("=" * 70)
