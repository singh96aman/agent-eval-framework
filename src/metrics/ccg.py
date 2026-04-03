"""
Criticality-Calibration Gap (CCG) metric computation.

The CCG measures the gap between true error criticality (human-annotated)
and judge-perceived criticality (judge penalties).

Formulas:
  TCS (True Criticality Score) = (TSD × 100) + (SER × 10)
  JPS (Judge Penalty Score) = 100 - judge_overall_score
  CCG (Criticality-Calibration Gap) = (JPS - TCS) / TCS

where:
  TSD = Task Success Degradation (0 or 1, from human annotation)
  SER = Subsequent Error Rate (count, from human annotation)
  judge_overall_score = 0-100 score from LLM judge
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from scipy import stats

from src.annotation.tools import Annotation, load_annotation
from src.storage.mongodb import MongoDBStorage


def compute_tcs(task_success_degradation: int, subsequent_error_rate: int) -> float:
    """
    Compute True Criticality Score (TCS) from human annotations.

    Args:
        task_success_degradation: Binary (0 or 1) - did error cause task failure?
        subsequent_error_rate: Count of errors after the perturbation

    Returns:
        TCS score

    Example:
        >>> compute_tcs(1, 3)  # Task failed, 3 downstream errors
        130.0
        >>> compute_tcs(0, 2)  # Task succeeded, 2 downstream errors
        20.0
    """
    return (task_success_degradation * 100) + (subsequent_error_rate * 10)


def compute_jps(judge_overall_score: float) -> float:
    """
    Compute Judge Penalty Score (JPS) from judge evaluation.

    JPS represents how much the judge penalized the trajectory.
    Higher JPS = judge thought trajectory was worse.

    Args:
        judge_overall_score: Judge's 0-100 score (higher = better trajectory)

    Returns:
        JPS score (0-100)

    Example:
        >>> compute_jps(85.0)  # Judge gave high score
        15.0
        >>> compute_jps(40.0)  # Judge heavily penalized
        60.0
    """
    return 100.0 - judge_overall_score


def compute_ccg(jps: float, tcs: float) -> Optional[float]:
    """
    Compute Criticality-Calibration Gap (CCG).

    CCG measures the relative gap between judge perception (JPS)
    and true criticality (TCS).

    Positive CCG: Judge over-penalized (penalty exceeds true criticality)
    Negative CCG: Judge under-penalized (penalty below true criticality)
    Zero CCG: Perfect calibration

    Args:
        jps: Judge Penalty Score (0-100)
        tcs: True Criticality Score (0+)

    Returns:
        CCG value, or None if TCS is 0 (division by zero)

    Example:
        >>> compute_ccg(60.0, 130.0)  # Judge under-penalized
        -0.538...
        >>> compute_ccg(30.0, 20.0)  # Judge over-penalized
        0.5
    """
    if tcs == 0:
        return None  # Cannot compute CCG when TCS is 0

    return (jps - tcs) / tcs


@dataclass
class CCGResult:
    """
    CCG computation result for a single perturbation.

    Attributes:
        perturbation_id: ID of perturbed trajectory
        experiment_id: Experiment ID
        judge_name: Name of judge
        perturbation_type: Type of error (planning, tool_selection, parameter)
        perturbation_position: Position (early, middle, late)
        tsd: Task Success Degradation (0 or 1)
        ser: Subsequent Error Rate (count)
        tcs: True Criticality Score
        judge_overall_score: Judge's 0-100 score
        jps: Judge Penalty Score
        ccg: Criticality-Calibration Gap
        annotator_id: Who annotated
        annotation_notes: Notes from annotation
    """
    perturbation_id: str
    experiment_id: str
    judge_name: str
    perturbation_type: str
    perturbation_position: str
    tsd: int
    ser: int
    tcs: float
    judge_overall_score: float
    jps: float
    ccg: Optional[float]
    annotator_id: str = ""
    annotation_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'perturbation_id': self.perturbation_id,
            'experiment_id': self.experiment_id,
            'judge_name': self.judge_name,
            'perturbation_type': self.perturbation_type,
            'perturbation_position': self.perturbation_position,
            'tsd': self.tsd,
            'ser': self.ser,
            'tcs': self.tcs,
            'judge_overall_score': self.judge_overall_score,
            'jps': self.jps,
            'ccg': self.ccg,
            'annotator_id': self.annotator_id,
            'annotation_notes': self.annotation_notes
        }


@dataclass
class CCGAnalysis:
    """
    Complete CCG analysis results.

    Attributes:
        experiment_id: Experiment ID
        results: List of individual CCG results
        summary: Summary statistics
        by_type: Aggregated by perturbation type
        by_position: Aggregated by position
        by_condition: Aggregated by type × position
        statistical_tests: ANOVA and post-hoc test results
    """
    experiment_id: str
    results: List[CCGResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_position: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_condition: Dict[str, Dict[str, float]] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save_csv(self, filepath: Path) -> None:
        """Save results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"✅ Saved CCG results to {filepath}")

    def print_summary(self) -> None:
        """Print summary statistics to console."""
        print("\n" + "=" * 80)
        print("CCG ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Experiment: {self.experiment_id}")
        print(f"Total results: {len(self.results)}")

        if self.summary:
            print(f"\nOverall CCG:")
            print(f"  Mean: {self.summary.get('ccg_mean', 'N/A'):.3f}")
            print(f"  Std:  {self.summary.get('ccg_std', 'N/A'):.3f}")
            print(f"  Min:  {self.summary.get('ccg_min', 'N/A'):.3f}")
            print(f"  Max:  {self.summary.get('ccg_max', 'N/A'):.3f}")

            print(f"\nTrue Criticality (TCS):")
            print(f"  Mean: {self.summary.get('tcs_mean', 'N/A'):.2f}")

            print(f"\nJudge Penalty (JPS):")
            print(f"  Mean: {self.summary.get('jps_mean', 'N/A'):.2f}")

        if self.by_type:
            print(f"\nBy Perturbation Type:")
            for ptype, stats in self.by_type.items():
                print(f"  {ptype:15s}: CCG = {stats.get('ccg_mean', 'N/A'):.3f} (n={stats.get('count', 0)})")

        if self.by_position:
            print(f"\nBy Position:")
            for pos, stats in self.by_position.items():
                print(f"  {pos:10s}: CCG = {stats.get('ccg_mean', 'N/A'):.3f} (n={stats.get('count', 0)})")

        if self.by_condition:
            print(f"\nBy Condition (Type × Position):")
            for cond, stats in sorted(self.by_condition.items()):
                print(f"  {cond:30s}: CCG = {stats.get('ccg_mean', 'N/A'):.3f} (n={stats.get('count', 0)})")

        if self.statistical_tests:
            print(f"\nStatistical Tests:")
            if 'anova_type' in self.statistical_tests:
                anova_result = self.statistical_tests['anova_type']
                print(f"  Type ANOVA: F={anova_result.get('f_stat', 'N/A'):.2f}, p={anova_result.get('p_value', 'N/A'):.4f}")

            if 'anova_position' in self.statistical_tests:
                anova_result = self.statistical_tests['anova_position']
                print(f"  Position ANOVA: F={anova_result.get('f_stat', 'N/A'):.2f}, p={anova_result.get('p_value', 'N/A'):.4f}")

        print("=" * 80)


def aggregate_by_condition(results: List[CCGResult]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate CCG results by condition (type × position).

    Args:
        results: List of CCG results

    Returns:
        Dictionary mapping condition to aggregated stats
    """
    if not results:
        return {}

    df = pd.DataFrame([r.to_dict() for r in results])

    # Filter out None CCG values
    df = df[df['ccg'].notna()]

    if len(df) == 0:
        return {}

    aggregates = {}

    # By type
    by_type = df.groupby('perturbation_type').agg({
        'ccg': ['mean', 'std', 'count'],
        'tcs': 'mean',
        'jps': 'mean'
    })

    for ptype, row in by_type.iterrows():
        aggregates[ptype] = {
            'ccg_mean': row[('ccg', 'mean')],
            'ccg_std': row[('ccg', 'std')],
            'count': int(row[('ccg', 'count')]),
            'tcs_mean': row[('tcs', 'mean')],
            'jps_mean': row[('jps', 'mean')]
        }

    # By position
    by_position = df.groupby('perturbation_position').agg({
        'ccg': ['mean', 'std', 'count'],
        'tcs': 'mean',
        'jps': 'mean'
    })

    for pos, row in by_position.iterrows():
        aggregates[pos] = {
            'ccg_mean': row[('ccg', 'mean')],
            'ccg_std': row[('ccg', 'std')],
            'count': int(row[('ccg', 'count')]),
            'tcs_mean': row[('tcs', 'mean')],
            'jps_mean': row[('jps', 'mean')]
        }

    # By condition (type × position)
    by_condition = df.groupby(['perturbation_type', 'perturbation_position']).agg({
        'ccg': ['mean', 'std', 'count'],
        'tcs': 'mean',
        'jps': 'mean'
    })

    for (ptype, pos), row in by_condition.iterrows():
        condition_key = f"{ptype}_{pos}"
        aggregates[condition_key] = {
            'ccg_mean': row[('ccg', 'mean')],
            'ccg_std': row[('ccg', 'std')],
            'count': int(row[('ccg', 'count')]),
            'tcs_mean': row[('tcs', 'mean')],
            'jps_mean': row[('jps', 'mean')]
        }

    return aggregates


def statistical_analysis(results: List[CCGResult]) -> Dict[str, Any]:
    """
    Perform statistical analyses on CCG results.

    Includes:
    - ANOVA for perturbation type
    - ANOVA for position
    - Effect sizes (Cohen's d)

    Args:
        results: List of CCG results

    Returns:
        Dictionary with test results
    """
    if not results:
        return {}

    df = pd.DataFrame([r.to_dict() for r in results])
    df = df[df['ccg'].notna()]  # Filter out None CCG

    if len(df) == 0:
        return {}

    tests = {}

    # ANOVA by perturbation type
    type_groups = [group['ccg'].values for name, group in df.groupby('perturbation_type')]
    if len(type_groups) > 1:
        f_stat, p_value = stats.f_oneway(*type_groups)
        tests['anova_type'] = {
            'f_stat': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    # ANOVA by position
    position_groups = [group['ccg'].values for name, group in df.groupby('perturbation_position')]
    if len(position_groups) > 1:
        f_stat, p_value = stats.f_oneway(*position_groups)
        tests['anova_position'] = {
            'f_stat': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    # Overall stats
    tests['overall'] = {
        'ccg_mean': float(df['ccg'].mean()),
        'ccg_std': float(df['ccg'].std()),
        'ccg_median': float(df['ccg'].median()),
        'tcs_mean': float(df['tcs'].mean()),
        'jps_mean': float(df['jps'].mean())
    }

    return tests


def compute_ccg_analysis(
    experiment_id: str,
    judge_name: str,
    storage: Optional[MongoDBStorage] = None,
    annotations_dir: Path = Path("data/annotations")
) -> CCGAnalysis:
    """
    Compute complete CCG analysis for an experiment.

    Args:
        experiment_id: Experiment ID
        judge_name: Name of judge to analyze
        storage: MongoDB storage (if None, creates default)
        annotations_dir: Directory containing annotations

    Returns:
        Complete CCG analysis
    """
    storage = storage or MongoDBStorage()

    # Load all perturbations for this experiment
    perturbations = list(storage.db['perturbations'].find({'experiment_id': experiment_id}))
    print(f"📊 Found {len(perturbations)} perturbations for experiment {experiment_id}")

    # Load judge evaluations
    evaluations = list(storage.db['judge_evaluations'].find({
        'experiment_id': experiment_id,
        'judge_name': judge_name
    }))
    print(f"⚖️  Found {len(evaluations)} judge evaluations from {judge_name}")

    # Build evaluation lookup (perturbation_id -> evaluation)
    eval_lookup = {}
    for eval_doc in evaluations:
        traj_id = eval_doc['trajectory_id']
        if traj_id not in eval_lookup:
            eval_lookup[traj_id] = []
        eval_lookup[traj_id].append(eval_doc)

    # Compute CCG for each annotated perturbation
    results = []
    missing_annotations = []
    missing_evaluations = []

    for pert_doc in perturbations:
        pert_id = pert_doc['perturbation_id']
        perturbed_traj_id = pert_doc['perturbed_trajectory_id']

        # Load annotation
        annotation = load_annotation(pert_id, annotations_dir)
        if not annotation:
            missing_annotations.append(pert_id)
            continue

        # Get judge evaluation (use first sample if multiple)
        if perturbed_traj_id not in eval_lookup or not eval_lookup[perturbed_traj_id]:
            missing_evaluations.append(pert_id)
            continue

        judge_eval = eval_lookup[perturbed_traj_id][0]  # Use first sample

        # Compute metrics
        tcs = compute_tcs(annotation.task_success_degradation, annotation.subsequent_error_rate)
        jps = compute_jps(judge_eval['overall_score'])
        ccg = compute_ccg(jps, tcs)

        result = CCGResult(
            perturbation_id=pert_id,
            experiment_id=experiment_id,
            judge_name=judge_name,
            perturbation_type=pert_doc['perturbation_type'],
            perturbation_position=pert_doc['perturbation_position'],
            tsd=annotation.task_success_degradation,
            ser=annotation.subsequent_error_rate,
            tcs=tcs,
            judge_overall_score=judge_eval['overall_score'],
            jps=jps,
            ccg=ccg,
            annotator_id=annotation.annotator_id,
            annotation_notes=annotation.notes
        )
        results.append(result)

    print(f"✅ Computed CCG for {len(results)} perturbations")

    if missing_annotations:
        print(f"⚠️  Missing annotations: {len(missing_annotations)} perturbations")
        if len(results) == 0:
            print(f"   → Run the 'annotate' phase to create annotations")
    if missing_evaluations:
        print(f"⚠️  Missing evaluations: {len(missing_evaluations)} perturbations")

    if len(results) == 0:
        print(f"\n⚠️  No CCG results computed. Need both annotations and judge evaluations.")
        return CCGAnalysis(experiment_id=experiment_id)

    # Compute aggregates
    aggregates = aggregate_by_condition(results)

    # Extract by type, by position, by condition
    by_type = {k: v for k, v in aggregates.items() if '_' not in k and k in ['planning', 'tool_selection', 'parameter']}
    by_position = {k: v for k, v in aggregates.items() if k in ['early', 'middle', 'late']}
    by_condition = {k: v for k, v in aggregates.items() if '_' in k}

    # Compute statistical tests
    tests = statistical_analysis(results)

    # Overall summary
    ccg_values = [r.ccg for r in results if r.ccg is not None]
    summary = {
        'ccg_mean': np.mean(ccg_values) if ccg_values else 0.0,
        'ccg_std': np.std(ccg_values) if ccg_values else 0.0,
        'ccg_min': np.min(ccg_values) if ccg_values else 0.0,
        'ccg_max': np.max(ccg_values) if ccg_values else 0.0,
        'tcs_mean': np.mean([r.tcs for r in results]),
        'jps_mean': np.mean([r.jps for r in results])
    }

    analysis = CCGAnalysis(
        experiment_id=experiment_id,
        results=results,
        summary=summary,
        by_type=by_type,
        by_position=by_position,
        by_condition=by_condition,
        statistical_tests=tests
    )

    return analysis


def main():
    """
    CLI entry point for CCG computation.

    Usage:
        python -m src.metrics.ccg <experiment_id> <judge_name>
    """
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.metrics.ccg <experiment_id> <judge_name>")
        sys.exit(1)

    experiment_id = sys.argv[1]
    judge_name = sys.argv[2]

    # Compute analysis
    analysis = compute_ccg_analysis(experiment_id, judge_name)

    # Print summary
    analysis.print_summary()

    # Save to CSV
    output_dir = Path(f"results/{experiment_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis.save_csv(output_dir / f"ccg_results_{judge_name}.csv")


if __name__ == "__main__":
    main()
