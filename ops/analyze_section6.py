#!/usr/bin/env python3
"""
Section 6 Aggregation and Report Generation.

This ops script loads per-unit analysis results from MongoDB and computes:
- 6A: Detection metrics (PDR, PNDR, SLA, TIA, CER, AUC, P/R/F1)
- 6B: Calibration metrics (CCorr, CCE, ORR, URR, Failure-ECE)
- 6C: Main claim test (PDR - CCorr gap with bootstrap CI)

Usage:
    python ops/analyze_section6.py --experiment-id exp_trajectory_sampling_v7 --output-dir data/analysis/
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from scipy import stats
from sklearn.metrics import roc_auc_score

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.mongodb import MongoDBStorage

# =============================================================================
# 6A: Detection Metrics
# =============================================================================


def compute_detection_metrics(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute all 6A detection metrics from analysis results.

    Args:
        results: List of analysis result dicts

    Returns:
        Dict with overall and stratified metrics
    """
    if not results:
        return {"error": "No results to analyze"}

    # Separate by perturbation class
    non_placebo = [
        r for r in results if r["ground_truth"]["perturbation_class"] != "placebo"
    ]
    placebo = [
        r for r in results if r["ground_truth"]["perturbation_class"] == "placebo"
    ]
    detected = [r for r in results if r["judge_output"]["error_detected"]]

    # Overall metrics
    overall = {}

    # PDR: Perturbation Detection Rate (TP / (TP + FN) for non-placebo)
    if non_placebo:
        tp = sum(1 for r in non_placebo if r["detection_eval"]["is_true_positive"])
        fn = sum(1 for r in non_placebo if r["detection_eval"]["is_false_negative"])
        overall["pdr"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        overall["pdr_n"] = len(non_placebo)
    else:
        overall["pdr"] = None
        overall["pdr_n"] = 0

    # PNDR: Placebo Non-Detection Rate (TN / (TN + FP) for placebo)
    if placebo:
        tn = sum(1 for r in placebo if r["detection_eval"]["is_true_negative"])
        fp = sum(1 for r in placebo if r["detection_eval"]["is_false_positive"])
        overall["pndr"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        overall["pndr_n"] = len(placebo)
    else:
        overall["pndr"] = None
        overall["pndr_n"] = 0

    # SLA: Step Localization Accuracy (exact matches / detected)
    detected_non_placebo = [
        r for r in non_placebo if r["judge_output"]["error_detected"]
    ]
    if detected_non_placebo:
        correct = sum(
            1
            for r in detected_non_placebo
            if r["detection_eval"]["localization_correct"] is True
        )
        overall["sla"] = correct / len(detected_non_placebo)
        overall["sla_n"] = len(detected_non_placebo)

        # SLA±1: Near matches
        near = sum(
            1
            for r in detected_non_placebo
            if r["detection_eval"]["localization_near"] is True
        )
        overall["sla_pm1"] = near / len(detected_non_placebo)
    else:
        overall["sla"] = None
        overall["sla_pm1"] = None
        overall["sla_n"] = 0

    # TIA: Type Identification Accuracy
    if detected_non_placebo:
        type_correct = sum(
            1
            for r in detected_non_placebo
            if r["detection_eval"]["type_correct"] is True
        )
        overall["tia"] = type_correct / len(detected_non_placebo)
    else:
        overall["tia"] = None

    # CER: Critical Error Recall (PDR for expected_impact=3)
    critical = [r for r in non_placebo if r["ground_truth"]["expected_impact"] == 3]
    if critical:
        critical_detected = sum(
            1 for r in critical if r["detection_eval"]["is_critical_detected"] is True
        )
        overall["cer"] = critical_detected / len(critical)
        overall["cer_n"] = len(critical)
    else:
        overall["cer"] = None
        overall["cer_n"] = 0

    # Detection AUC using error_confidence
    if non_placebo and placebo:
        y_true = [1] * len(non_placebo) + [0] * len(placebo)
        y_score = [r["judge_output"]["error_confidence"] for r in non_placebo] + [
            r["judge_output"]["error_confidence"] for r in placebo
        ]
        try:
            overall["detection_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            overall["detection_auc"] = None
    else:
        overall["detection_auc"] = None

    # Precision, Recall, F1
    tp = sum(1 for r in results if r["detection_eval"]["is_true_positive"])
    fp = sum(1 for r in results if r["detection_eval"]["is_false_positive"])
    fn = sum(1 for r in results if r["detection_eval"]["is_false_negative"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    overall["precision"] = precision
    overall["recall"] = recall
    overall["f1"] = f1
    overall["tp"] = tp
    overall["fp"] = fp
    overall["fn"] = fn
    overall["tn"] = sum(1 for r in results if r["detection_eval"]["is_true_negative"])

    # Stratified analysis
    stratified = {
        "by_class": _stratify_detection(results, "perturbation_class"),
        "by_family": _stratify_detection(results, "perturbation_family"),
        "by_detectability": _stratify_detection(results, "expected_detectability"),
        "by_impact": _stratify_detection(results, "expected_impact"),
        "by_benchmark": _stratify_detection(results, "benchmark"),
    }

    return {
        "overall": overall,
        "stratified": stratified,
        "n_total": len(results),
    }


def _stratify_detection(results: List[Dict], key: str) -> Dict[str, Dict]:
    """Stratify detection metrics by a ground truth key."""
    groups = defaultdict(list)
    for r in results:
        value = r["ground_truth"].get(key)
        groups[str(value)].append(r)

    stratified = {}
    for group_name, group_results in groups.items():
        non_placebo = [
            r
            for r in group_results
            if r["ground_truth"]["perturbation_class"] != "placebo"
        ]
        detected_non_placebo = [
            r for r in non_placebo if r["judge_output"]["error_detected"]
        ]

        metrics = {"n": len(group_results)}

        # PDR
        if non_placebo:
            tp = sum(1 for r in non_placebo if r["detection_eval"]["is_true_positive"])
            fn = sum(1 for r in non_placebo if r["detection_eval"]["is_false_negative"])
            metrics["pdr"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # SLA
        if detected_non_placebo:
            correct = sum(
                1
                for r in detected_non_placebo
                if r["detection_eval"]["localization_correct"] is True
            )
            metrics["sla"] = correct / len(detected_non_placebo)

        stratified[group_name] = metrics

    return stratified


# =============================================================================
# 6B: Calibration Metrics
# =============================================================================


def compute_calibration_metrics(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute all 6B calibration metrics from analysis results.

    Args:
        results: List of analysis result dicts

    Returns:
        Dict with overall and stratified metrics
    """
    if not results:
        return {"error": "No results to analyze"}

    # Filter to results with outcome data
    with_outcome = [
        r for r in results if r["ground_truth"].get("outcome_degradation") is not None
    ]

    overall = {}

    # CCorr: Consequence Correlation (Spearman)
    if len(with_outcome) >= 3:
        predicted = [r["judge_output"]["predicted_impact_score"] for r in with_outcome]
        actual_od = [r["ground_truth"]["outcome_degradation"] for r in with_outcome]

        try:
            corr, pvalue = stats.spearmanr(predicted, actual_od)
            overall["ccorr"] = float(corr) if not np.isnan(corr) else None
            overall["ccorr_pvalue"] = float(pvalue) if not np.isnan(pvalue) else None
        except Exception:
            overall["ccorr"] = None
            overall["ccorr_pvalue"] = None
    else:
        overall["ccorr"] = None
        overall["ccorr_pvalue"] = None

    overall["ccorr_n"] = len(with_outcome)

    # CCE: Consequence Calibration Error
    cce_values = [
        r["calibration_eval"]["cce"]
        for r in results
        if r["calibration_eval"]["cce"] is not None
    ]
    abs_cce_values = [
        r["calibration_eval"]["abs_cce"]
        for r in results
        if r["calibration_eval"]["abs_cce"] is not None
    ]

    if cce_values:
        overall["mean_cce"] = float(np.mean(cce_values))
        overall["std_cce"] = float(np.std(cce_values))
    else:
        overall["mean_cce"] = None
        overall["std_cce"] = None

    if abs_cce_values:
        overall["abs_cce"] = float(np.mean(abs_cce_values))
    else:
        overall["abs_cce"] = None

    # ORR: Over-Reaction Rate (predicted > 0.5 AND true_impact <= 1)
    low_impact = [
        r
        for r in results
        if r["ground_truth"].get("true_impact_level") is not None
        and r["ground_truth"]["true_impact_level"] <= 1
    ]
    if low_impact:
        over_reactions = sum(
            1 for r in low_impact if r["calibration_eval"]["over_reaction"] is True
        )
        overall["orr"] = over_reactions / len(low_impact)
        overall["orr_n"] = len(low_impact)
    else:
        overall["orr"] = None
        overall["orr_n"] = 0

    # URR: Under-Reaction Rate (predicted < 0.5 AND true_impact == 3)
    critical_impact = [
        r for r in results if r["ground_truth"].get("true_impact_level") == 3
    ]
    if critical_impact:
        under_reactions = sum(
            1
            for r in critical_impact
            if r["calibration_eval"]["under_reaction"] is True
        )
        overall["urr"] = under_reactions / len(critical_impact)
        overall["urr_n"] = len(critical_impact)
    else:
        overall["urr"] = None
        overall["urr_n"] = 0

    # Failure-ECE
    failure_ece = compute_failure_ece(results)
    overall["failure_ece"] = failure_ece

    # Impact tier accuracy metrics
    tier_errors = [
        r["calibration_eval"]["impact_tier_error"]
        for r in results
        if r["calibration_eval"]["impact_tier_error"] is not None
    ]
    if tier_errors:
        overall["impact_mae"] = float(np.mean(tier_errors))
        overall["impact_tier_exact"] = sum(1 for e in tier_errors if e == 0) / len(
            tier_errors
        )
    else:
        overall["impact_mae"] = None
        overall["impact_tier_exact"] = None

    # Stratified analysis
    stratified = {
        "by_true_impact": _stratify_calibration(results, "true_impact_level"),
        "by_class": _stratify_calibration(results, "perturbation_class"),
        "by_family": _stratify_calibration(results, "perturbation_family"),
        "by_benchmark": _stratify_calibration(results, "benchmark"),
    }

    return {
        "overall": overall,
        "stratified": stratified,
        "n_total": len(results),
    }


def compute_failure_ece(results: List[Dict], n_bins: int = 10) -> Optional[float]:
    """
    Compute Expected Calibration Error for failure prediction.

    Args:
        results: Analysis results
        n_bins: Number of bins

    Returns:
        ECE value or None
    """
    # Get predictions and actuals
    valid = [r for r in results if r["calibration_eval"]["failure_actual"] is not None]

    if len(valid) < n_bins:
        return None

    predictions = np.array([r["judge_output"]["predicted_failure_prob"] for r in valid])
    actuals = np.array(
        [1 if r["calibration_eval"]["failure_actual"] else 0 for r in valid]
    )

    # Bin by predicted probability
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if mask.sum() == 0:
            continue

        bin_acc = actuals[mask].mean()
        bin_conf = predictions[mask].mean()
        bin_size = mask.sum()

        ece += (bin_size / len(predictions)) * abs(bin_acc - bin_conf)

    return float(ece)


def _stratify_calibration(results: List[Dict], key: str) -> Dict[str, Dict]:
    """Stratify calibration metrics by a ground truth key."""
    groups = defaultdict(list)
    for r in results:
        value = r["ground_truth"].get(key)
        groups[str(value)].append(r)

    stratified = {}
    for group_name, group_results in groups.items():
        metrics = {"n": len(group_results)}

        # CCE
        cce_values = [
            r["calibration_eval"]["cce"]
            for r in group_results
            if r["calibration_eval"]["cce"] is not None
        ]
        if cce_values:
            metrics["mean_cce"] = float(np.mean(cce_values))
            metrics["abs_cce"] = float(np.mean([abs(v) for v in cce_values]))

        stratified[group_name] = metrics

    return stratified


# =============================================================================
# 6C: Agreement + Claim Tests
# =============================================================================


def test_main_claim(
    detection_metrics: Dict[str, Any],
    calibration_metrics: Dict[str, Any],
    results: List[Dict],
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Test the main claim: PDR > CCorr.

    Args:
        detection_metrics: 6A metrics
        calibration_metrics: 6B metrics
        results: Raw analysis results for bootstrapping
        n_bootstrap: Number of bootstrap samples

    Returns:
        Claim test results
    """
    pdr = detection_metrics.get("overall", {}).get("pdr")
    ccorr = calibration_metrics.get("overall", {}).get("ccorr")

    if pdr is None or ccorr is None:
        return {
            "claim": "LLM judges detect errors better than they estimate their downstream consequence",
            "pdr": pdr,
            "ccorr": ccorr,
            "gap": None,
            "gap_ci": None,
            "claim_supported": None,
            "reason": "Insufficient data to compute PDR or CCorr",
        }

    gap = pdr - ccorr

    # Bootstrap CI for gap
    gap_ci = bootstrap_gap_ci(results, n_bootstrap)

    # Claim supported if lower CI bound > 0
    claim_supported = gap_ci[0] > 0 if gap_ci else None

    return {
        "claim": "LLM judges detect errors better than they estimate their downstream consequence",
        "pdr": pdr,
        "ccorr": ccorr,
        "gap": gap,
        "gap_ci": list(gap_ci) if gap_ci else None,
        "claim_supported": claim_supported,
        "interpretation": _interpret_claim(pdr, ccorr, gap, claim_supported),
    }


def bootstrap_gap_ci(
    results: List[Dict],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Optional[Tuple[float, float]]:
    """
    Bootstrap confidence interval for PDR - CCorr gap.

    Args:
        results: Analysis results
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level

    Returns:
        (lower, upper) CI tuple or None
    """
    if len(results) < 10:
        return None

    np.random.seed(42)
    gaps = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(len(results), size=len(results), replace=True)
        sample = [results[i] for i in indices]

        # Compute PDR
        non_placebo = [
            r for r in sample if r["ground_truth"]["perturbation_class"] != "placebo"
        ]
        if non_placebo:
            tp = sum(1 for r in non_placebo if r["detection_eval"]["is_true_positive"])
            fn = sum(1 for r in non_placebo if r["detection_eval"]["is_false_negative"])
            pdr = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            continue

        # Compute CCorr
        with_outcome = [
            r
            for r in sample
            if r["ground_truth"].get("outcome_degradation") is not None
        ]
        if len(with_outcome) >= 3:
            predicted = [
                r["judge_output"]["predicted_impact_score"] for r in with_outcome
            ]
            actual = [r["ground_truth"]["outcome_degradation"] for r in with_outcome]
            try:
                ccorr, _ = stats.spearmanr(predicted, actual)
                if np.isnan(ccorr):
                    continue
            except Exception:
                continue
        else:
            continue

        gaps.append(pdr - ccorr)

    if not gaps:
        return None

    alpha = (1 - ci) / 2
    lower = np.percentile(gaps, alpha * 100)
    upper = np.percentile(gaps, (1 - alpha) * 100)

    return (float(lower), float(upper))


def _interpret_claim(
    pdr: float, ccorr: float, gap: float, supported: Optional[bool]
) -> str:
    """Generate interpretation of claim test results."""
    if supported is None:
        return "Insufficient data to draw conclusions."

    if supported:
        return (
            f"CLAIM SUPPORTED: Judges show significantly better detection (PDR={pdr:.3f}) "
            f"than calibration (CCorr={ccorr:.3f}). The gap of {gap:.3f} is statistically "
            f"significant (95% CI excludes zero). This suggests judges can identify errors "
            f"but struggle to estimate their downstream impact."
        )
    else:
        return (
            f"CLAIM NOT SUPPORTED: The gap between detection (PDR={pdr:.3f}) and "
            f"calibration (CCorr={ccorr:.3f}) is {gap:.3f}, but the 95% CI includes zero, "
            f"suggesting the difference may not be statistically significant."
        )


# =============================================================================
# Human-Judge Agreement
# =============================================================================


def compute_human_agreement(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute human-judge agreement metrics from results with human labels.

    Args:
        results: Analysis results (filters to those with human_comparison)

    Returns:
        Agreement metrics dict
    """
    # Filter to results with human comparison
    with_human = [r for r in results if r.get("human_comparison") is not None]

    if not with_human:
        return {"n_labeled": 0, "error": "No human labels available"}

    metrics = {"n_labeled": len(with_human)}

    # Detection agreement
    detection_data = [
        r["human_comparison"]["detection_agrees"]
        for r in with_human
        if r["human_comparison"].get("detection_agrees") is not None
    ]
    if detection_data:
        metrics["detection_agreement"] = sum(detection_data) / len(detection_data)
        metrics["detection_agreement_n"] = len(detection_data)

    # Type agreement
    type_data = [
        r["human_comparison"]["type_agrees"]
        for r in with_human
        if r["human_comparison"].get("type_agrees") is not None
    ]
    if type_data:
        metrics["type_agreement"] = sum(type_data) / len(type_data)
        metrics["type_agreement_n"] = len(type_data)

    # Impact tier difference
    impact_diffs = [
        r["human_comparison"]["impact_tier_diff"]
        for r in with_human
        if r["human_comparison"].get("impact_tier_diff") is not None
    ]
    if impact_diffs:
        metrics["impact_tier_mae"] = float(np.mean(np.abs(impact_diffs)))
        metrics["impact_tier_bias"] = float(
            np.mean(impact_diffs)
        )  # Positive = judge rates higher
        metrics["impact_tier_n"] = len(impact_diffs)

    # Localization agreement
    loc_data = [
        r["human_comparison"]["localization_agrees"]
        for r in with_human
        if r["human_comparison"].get("localization_agrees") is not None
    ]
    if loc_data:
        metrics["localization_agreement"] = sum(loc_data) / len(loc_data)
        metrics["localization_agreement_n"] = len(loc_data)

    return metrics


def compute_proxy_calibration(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute calibration metrics using expected_impact as proxy ground truth.

    This is used when outcome_degradation has no variance (all baselines failed).

    Args:
        results: Analysis results

    Returns:
        Proxy calibration metrics
    """
    # Filter to non-placebo results with expected_impact
    valid = [
        r
        for r in results
        if r["ground_truth"].get("perturbation_class") != "placebo"
        and r["ground_truth"].get("expected_impact") is not None
    ]

    if len(valid) < 3:
        return {"error": "Insufficient data for proxy calibration"}

    predicted = [r["judge_output"]["predicted_impact_score"] for r in valid]
    expected = [r["ground_truth"]["expected_impact"] for r in valid]

    # Normalize expected_impact (0-3) to (0-1) scale
    expected_normalized = [e / 3.0 for e in expected]

    metrics = {"n": len(valid)}

    # Spearman correlation
    try:
        corr, pvalue = stats.spearmanr(predicted, expected)
        metrics["proxy_ccorr"] = float(corr) if not np.isnan(corr) else None
        metrics["proxy_ccorr_pvalue"] = float(pvalue) if not np.isnan(pvalue) else None
    except Exception:
        metrics["proxy_ccorr"] = None

    # Pearson correlation
    try:
        corr, pvalue = stats.pearsonr(predicted, expected_normalized)
        metrics["proxy_pearson"] = float(corr) if not np.isnan(corr) else None
    except Exception:
        metrics["proxy_pearson"] = None

    # Mean absolute error (on 0-1 scale)
    mae = np.mean(np.abs(np.array(predicted) - np.array(expected_normalized)))
    metrics["proxy_mae"] = float(mae)

    # Tier accuracy
    predicted_tiers = [int(p * 3) if p < 1 else 3 for p in predicted]  # Convert to 0-3
    tier_correct = sum(1 for pt, et in zip(predicted_tiers, expected) if pt == et)
    metrics["proxy_tier_accuracy"] = tier_correct / len(valid)

    # Over/under reaction using expected_impact
    over_reactions = sum(1 for p, e in zip(predicted, expected) if p > 0.5 and e <= 1)
    under_reactions = sum(1 for p, e in zip(predicted, expected) if p < 0.5 and e == 3)

    low_impact = sum(1 for e in expected if e <= 1)
    high_impact = sum(1 for e in expected if e == 3)

    if low_impact > 0:
        metrics["proxy_orr"] = over_reactions / low_impact
    if high_impact > 0:
        metrics["proxy_urr"] = under_reactions / high_impact

    return metrics


# =============================================================================
# Report Generation
# =============================================================================


def generate_reports(
    experiment_id: str,
    results_by_judge: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """
    Generate all JSON reports and summary.

    Args:
        experiment_id: Experiment ID
        results_by_judge: Results grouped by judge model
        output_dir: Output directory
    """
    reports_dir = output_dir / experiment_id / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_judges = list(results_by_judge.keys())
    all_results = [r for results in results_by_judge.values() for r in results]

    # Per-judge metrics
    detection_by_judge = {}
    calibration_by_judge = {}
    claim_by_judge = {}

    human_agreement_by_judge = {}
    proxy_calibration_by_judge = {}

    for judge, results in results_by_judge.items():
        detection_by_judge[judge] = compute_detection_metrics(results)
        calibration_by_judge[judge] = compute_calibration_metrics(results)
        claim_by_judge[judge] = test_main_claim(
            detection_by_judge[judge],
            calibration_by_judge[judge],
            results,
        )
        human_agreement_by_judge[judge] = compute_human_agreement(results)
        proxy_calibration_by_judge[judge] = compute_proxy_calibration(results)

    # Save detection metrics
    detection_report = {
        "experiment_id": experiment_id,
        "computed_at": datetime.utcnow().isoformat(),
        "by_judge": detection_by_judge,
    }
    with open(reports_dir / "detection_metrics.json", "w") as f:
        json.dump(detection_report, f, indent=2, default=str)

    # Save calibration metrics
    calibration_report = {
        "experiment_id": experiment_id,
        "computed_at": datetime.utcnow().isoformat(),
        "by_judge": calibration_by_judge,
    }
    with open(reports_dir / "calibration_metrics.json", "w") as f:
        json.dump(calibration_report, f, indent=2, default=str)

    # Save claim test report
    claim_report = {
        "experiment_id": experiment_id,
        "computed_at": datetime.utcnow().isoformat(),
        "by_judge": claim_by_judge,
    }
    with open(reports_dir / "claim_test_report.json", "w") as f:
        json.dump(claim_report, f, indent=2, default=str)

    # Save human agreement report
    human_agreement_report = {
        "experiment_id": experiment_id,
        "computed_at": datetime.utcnow().isoformat(),
        "by_judge": human_agreement_by_judge,
    }
    with open(reports_dir / "human_agreement.json", "w") as f:
        json.dump(human_agreement_report, f, indent=2, default=str)

    # Save proxy calibration report
    proxy_calibration_report = {
        "experiment_id": experiment_id,
        "computed_at": datetime.utcnow().isoformat(),
        "by_judge": proxy_calibration_by_judge,
    }
    with open(reports_dir / "proxy_calibration.json", "w") as f:
        json.dump(proxy_calibration_report, f, indent=2, default=str)

    # Generate synthesis markdown
    synthesis = generate_synthesis_markdown(
        experiment_id,
        detection_by_judge,
        calibration_by_judge,
        claim_by_judge,
        human_agreement_by_judge,
        proxy_calibration_by_judge,
        len(all_results),
    )
    with open(reports_dir / "synthesis_report.md", "w") as f:
        f.write(synthesis)

    print(f"Reports saved to {reports_dir}")


def generate_synthesis_markdown(
    experiment_id: str,
    detection: Dict[str, Dict],
    calibration: Dict[str, Dict],
    claims: Dict[str, Dict],
    human_agreement: Dict[str, Dict],
    proxy_calibration: Dict[str, Dict],
    n_total: int,
) -> str:
    """Generate synthesis report markdown."""
    lines = [
        "# Section 6 Analysis: Synthesis Report",
        "",
        f"**Experiment:** {experiment_id}",
        f"**Generated:** {datetime.utcnow().isoformat()}Z",
        f"**Total Analysis Results:** {n_total}",
        "",
        "---",
        "",
        "## Main Research Claim",
        "",
        '> **"LLM judges detect errors better than they estimate their downstream consequence."**',
        "",
        "---",
        "",
        "## Summary by Judge",
        "",
        "| Judge | PDR | CCorr | Gap | CI Lower | CI Upper | Supported? |",
        "|-------|-----|-------|-----|----------|----------|------------|",
    ]

    for judge in sorted(claims.keys()):
        c = claims[judge]
        pdr = f"{c['pdr']:.3f}" if c["pdr"] is not None else "N/A"
        ccorr = f"{c['ccorr']:.3f}" if c["ccorr"] is not None else "N/A"
        gap = f"{c['gap']:.3f}" if c["gap"] is not None else "N/A"
        ci = c.get("gap_ci") or [None, None]
        ci_lower = f"{ci[0]:.3f}" if ci[0] is not None else "N/A"
        ci_upper = f"{ci[1]:.3f}" if ci[1] is not None else "N/A"
        supported = (
            "YES"
            if c.get("claim_supported")
            else ("NO" if c.get("claim_supported") is False else "?")
        )

        lines.append(
            f"| {judge} | {pdr} | {ccorr} | {gap} | {ci_lower} | {ci_upper} | {supported} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Detection Metrics (6A)",
            "",
            "| Judge | PDR | PNDR | SLA | TIA | CER | AUC | F1 |",
            "|-------|-----|------|-----|-----|-----|-----|-----|",
        ]
    )

    def fmt(val):
        return f"{val:.3f}" if val is not None else "N/A"

    for judge in sorted(detection.keys()):
        d = detection[judge].get("overall", {})
        lines.append(
            f"| {judge} | "
            f"{fmt(d.get('pdr'))} | "
            f"{fmt(d.get('pndr'))} | "
            f"{fmt(d.get('sla'))} | "
            f"{fmt(d.get('tia'))} | "
            f"{fmt(d.get('cer'))} | "
            f"{fmt(d.get('detection_auc'))} | "
            f"{fmt(d.get('f1'))} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Calibration Metrics (6B)",
            "",
            "| Judge | CCorr | |CCE| | ORR | URR | Failure-ECE |",
            "|-------|-------|------|-----|-----|-------------|",
        ]
    )

    for judge in sorted(calibration.keys()):
        c = calibration[judge].get("overall", {})
        lines.append(
            f"| {judge} | "
            f"{fmt(c.get('ccorr'))} | "
            f"{fmt(c.get('abs_cce'))} | "
            f"{fmt(c.get('orr'))} | "
            f"{fmt(c.get('urr'))} | "
            f"{fmt(c.get('failure_ece'))} |"
        )

    # Proxy Calibration (using expected_impact as ground truth)
    lines.extend(
        [
            "",
            "---",
            "",
            "## Proxy Calibration (using expected_impact)",
            "",
            "*Note: outcome_degradation has no variance (all baselines failed), so expected_impact is used as proxy ground truth.*",
            "",
            "| Judge | Proxy CCorr | Proxy MAE | Tier Accuracy | Proxy ORR | Proxy URR |",
            "|-------|-------------|-----------|---------------|-----------|-----------|",
        ]
    )

    for judge in sorted(proxy_calibration.keys()):
        p = proxy_calibration[judge]
        lines.append(
            f"| {judge} | "
            f"{fmt(p.get('proxy_ccorr'))} | "
            f"{fmt(p.get('proxy_mae'))} | "
            f"{fmt(p.get('proxy_tier_accuracy'))} | "
            f"{fmt(p.get('proxy_orr'))} | "
            f"{fmt(p.get('proxy_urr'))} |"
        )

    # Human-Judge Agreement
    lines.extend(
        [
            "",
            "---",
            "",
            "## Human-Judge Agreement",
            "",
            "| Judge | N Labeled | Detection Agree | Type Agree | Impact MAE | Impact Bias |",
            "|-------|-----------|-----------------|------------|------------|-------------|",
        ]
    )

    for judge in sorted(human_agreement.keys()):
        h = human_agreement[judge]
        n_labeled = h.get("n_labeled", 0)
        lines.append(
            f"| {judge} | "
            f"{n_labeled} | "
            f"{fmt(h.get('detection_agreement'))} | "
            f"{fmt(h.get('type_agreement'))} | "
            f"{fmt(h.get('impact_tier_mae'))} | "
            f"{fmt(h.get('impact_tier_bias'))} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Interpretation",
            "",
        ]
    )

    # Add interpretations
    for judge in sorted(claims.keys()):
        c = claims[judge]
        lines.extend(
            [
                f"### {judge}",
                "",
                c.get("interpretation", "No interpretation available."),
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "*Generated by Section 6 Analysis Pipeline*",
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Section 6 Aggregation and Report Generation"
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Experiment ID to analyze",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--database",
        default="agent_judge_experiment",
        help="MongoDB database name",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SECTION 6 AGGREGATION & REPORT GENERATION")
    print("=" * 60)
    print(f"Experiment: {args.experiment_id}")
    print(f"Output: {args.output_dir}")
    print()

    # Connect to MongoDB
    storage = MongoDBStorage(database=args.database)

    # Load analysis results
    print("Loading analysis results from MongoDB...")
    collection = storage.db["analysis_results"]
    results = list(collection.find({"experiment_id": args.experiment_id}))
    print(f"  Found {len(results)} analysis results")

    if not results:
        print("ERROR: No analysis results found. Run the analysis phase first:")
        print(
            f"  python main.py --config v2/pocv2/perturbation_generation --runner analysis"
        )
        storage.close()
        sys.exit(1)

    # Group by judge
    results_by_judge = defaultdict(list)
    for r in results:
        results_by_judge[r["judge_model"]].append(r)

    print(f"  Judges: {list(results_by_judge.keys())}")
    print()

    # Generate reports
    output_dir = Path(args.output_dir)
    generate_reports(args.experiment_id, results_by_judge, output_dir)

    storage.close()

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Reports saved to: {output_dir / args.experiment_id / 'reports'}")


if __name__ == "__main__":
    main()
