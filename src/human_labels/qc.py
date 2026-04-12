"""
Quality control checks for human annotations.

This module provides functions for:
- Real-time QC during annotation (time bounds, consecutive same)
- Post-hoc QC (gold set accuracy, systematic bias)
- Drift monitoring during annotation batches

Per Section 5A.9, QC thresholds:
- Time per unit: 30s minimum, 600s (10min) maximum
- Consecutive same responses: flag at 5+
- Gold set accuracy: maintain >80%, recalibrate at <70%
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.human_labels.schema import AnnotationRecord


def check_time_bounds(
    annotation: AnnotationRecord,
    min_sec: int = 30,
    max_sec: int = 600,
) -> Dict[str, Any]:
    """
    Check if annotation time is within acceptable bounds.

    Per 5A.9:
    - <30s per unit: flag for review (too fast)
    - >10min per unit: check for issues (too slow)

    Args:
        annotation: AnnotationRecord to check
        min_sec: Minimum acceptable time in seconds
        max_sec: Maximum acceptable time in seconds

    Returns:
        Dict with check results:
        {
            "passed": bool,
            "time_spent": int,
            "issue": str | None,  # "too_fast", "too_slow", or None
            "annotation_id": str
        }
    """
    time_spent = annotation.time_spent_seconds

    issue = None
    if time_spent < min_sec:
        issue = "too_fast"
    elif time_spent > max_sec:
        issue = "too_slow"

    return {
        "passed": issue is None,
        "time_spent": time_spent,
        "issue": issue,
        "annotation_id": annotation.annotation_id,
    }


def check_time_bounds_batch(
    annotations: List[AnnotationRecord],
    min_sec: int = 30,
    max_sec: int = 600,
) -> Dict[str, Any]:
    """
    Check time bounds for a batch of annotations.

    Args:
        annotations: List of AnnotationRecords
        min_sec: Minimum acceptable time in seconds
        max_sec: Maximum acceptable time in seconds

    Returns:
        Dict with batch results:
        {
            "total_checked": int,
            "passed_count": int,
            "too_fast_count": int,
            "too_slow_count": int,
            "too_fast_ids": List[str],
            "too_slow_ids": List[str],
            "mean_time": float,
            "median_time": float,
            "by_annotator": Dict[str, Dict]
        }
    """
    results = {
        "total_checked": len(annotations),
        "passed_count": 0,
        "too_fast_count": 0,
        "too_slow_count": 0,
        "too_fast_ids": [],
        "too_slow_ids": [],
        "mean_time": 0.0,
        "median_time": 0.0,
        "by_annotator": {},
    }

    times = []
    by_annotator: Dict[str, List[int]] = defaultdict(list)

    for ann in annotations:
        check = check_time_bounds(ann, min_sec, max_sec)
        times.append(ann.time_spent_seconds)
        by_annotator[ann.annotator_id].append(ann.time_spent_seconds)

        if check["passed"]:
            results["passed_count"] += 1
        elif check["issue"] == "too_fast":
            results["too_fast_count"] += 1
            results["too_fast_ids"].append(ann.annotation_id)
        elif check["issue"] == "too_slow":
            results["too_slow_count"] += 1
            results["too_slow_ids"].append(ann.annotation_id)

    if times:
        results["mean_time"] = np.mean(times)
        results["median_time"] = np.median(times)

    # Per-annotator summary
    for annotator_id, annotator_times in by_annotator.items():
        results["by_annotator"][annotator_id] = {
            "count": len(annotator_times),
            "mean_time": np.mean(annotator_times),
            "too_fast_count": sum(1 for t in annotator_times if t < min_sec),
            "too_slow_count": sum(1 for t in annotator_times if t > max_sec),
        }

    return results


def check_gold_set_accuracy(
    annotator_annotations: List[AnnotationRecord],
    gold_set: List[Dict[str, Any]],
    dimension: str = "error_detected",
) -> Dict[str, Any]:
    """
    Check annotator accuracy against gold set for drift monitoring.

    Per 5A.9:
    - Target: >80% accuracy on gold set
    - Recalibration trigger: <70% accuracy

    Args:
        annotator_annotations: Annotations from one annotator
        gold_set: List of gold standard labels with:
            - evaluation_unit_id: str
            - gold_error_detected: bool
            - gold_error_trajectory: str
            - gold_error_step_id: str (optional)
            - gold_error_type: str (optional)
        dimension: Which dimension to check accuracy for

    Returns:
        Dict with accuracy results:
        {
            "annotator_id": str,
            "gold_checked": int,
            "gold_correct": int,
            "accuracy": float,
            "status": "passing" | "warning" | "recalibration_needed",
            "incorrect_ids": List[str]
        }
    """
    # Build gold lookup
    gold_by_unit: Dict[str, Dict] = {g["evaluation_unit_id"]: g for g in gold_set}

    annotator_id = (
        annotator_annotations[0].annotator_id if annotator_annotations else "unknown"
    )

    gold_checked = 0
    gold_correct = 0
    incorrect_ids = []

    for ann in annotator_annotations:
        if ann.evaluation_unit_id not in gold_by_unit:
            continue

        gold = gold_by_unit[ann.evaluation_unit_id]
        gold_checked += 1

        # Extract predicted value
        predicted = _get_predicted_value(ann, dimension)
        expected = gold.get(f"gold_{dimension}")

        if predicted == expected:
            gold_correct += 1
        else:
            incorrect_ids.append(ann.annotation_id)

    accuracy = gold_correct / gold_checked if gold_checked > 0 else None

    # Determine status
    if accuracy is None:
        status = "no_gold_overlap"
    elif accuracy >= 0.8:
        status = "passing"
    elif accuracy >= 0.7:
        status = "warning"
    else:
        status = "recalibration_needed"

    return {
        "annotator_id": annotator_id,
        "gold_checked": gold_checked,
        "gold_correct": gold_correct,
        "accuracy": accuracy,
        "status": status,
        "incorrect_ids": incorrect_ids,
    }


def _get_predicted_value(ann: AnnotationRecord, dimension: str) -> Any:
    """Extract predicted value for a dimension from annotation."""
    if dimension == "error_detected" and ann.detectability:
        return ann.detectability.error_detected
    elif dimension == "error_trajectory" and ann.detectability:
        return ann.detectability.error_trajectory
    elif dimension == "error_step_id" and ann.detectability:
        return ann.detectability.error_step_id
    elif dimension == "error_type" and ann.consequence:
        return ann.consequence.error_type
    elif dimension == "impact_tier" and ann.consequence:
        return ann.consequence.impact_tier
    elif dimension == "preference" and ann.preference:
        return ann.preference.preference
    return None


def check_consecutive_same(
    annotations: List[AnnotationRecord],
    threshold: int = 5,
    dimension: str = "error_detected",
) -> Dict[str, Any]:
    """
    Detect patterns of consecutive identical responses (inattention indicator).

    Per 5A.9:
    - Alert at 5+ consecutive same responses

    Args:
        annotations: List of annotations (should be in submission order)
        threshold: Number of consecutive same responses to flag
        dimension: Which dimension to check

    Returns:
        Dict with pattern detection results:
        {
            "by_annotator": {
                "annotator_A": {
                    "max_consecutive": int,
                    "flagged": bool,
                    "flagged_sequences": List[Dict]  # start_idx, end_idx, value
                },
                ...
            },
            "overall_flagged": bool
        }
    """
    # Group by annotator and sort by created_at
    by_annotator: Dict[str, List[AnnotationRecord]] = defaultdict(list)
    for ann in annotations:
        by_annotator[ann.annotator_id].append(ann)

    # Sort each annotator's annotations by time
    for annotator_id in by_annotator:
        by_annotator[annotator_id].sort(key=lambda a: a.created_at)

    results = {
        "by_annotator": {},
        "overall_flagged": False,
    }

    for annotator_id, annotator_anns in by_annotator.items():
        # Extract values
        values = [_get_predicted_value(ann, dimension) for ann in annotator_anns]

        # Find consecutive runs
        max_consecutive = 1
        current_run = 1
        flagged_sequences = []

        for i in range(1, len(values)):
            if values[i] == values[i - 1] and values[i] is not None:
                current_run += 1
                if current_run >= threshold:
                    # Add or extend flagged sequence
                    start_idx = i - current_run + 1
                    if flagged_sequences and flagged_sequences[-1]["end_idx"] == i - 1:
                        flagged_sequences[-1]["end_idx"] = i
                    else:
                        flagged_sequences.append(
                            {
                                "start_idx": start_idx,
                                "end_idx": i,
                                "value": values[i],
                                "length": current_run,
                            }
                        )
            else:
                max_consecutive = max(max_consecutive, current_run)
                current_run = 1

        max_consecutive = max(max_consecutive, current_run)
        flagged = max_consecutive >= threshold

        results["by_annotator"][annotator_id] = {
            "max_consecutive": max_consecutive,
            "flagged": flagged,
            "flagged_sequences": flagged_sequences,
            "total_annotations": len(annotator_anns),
        }

        if flagged:
            results["overall_flagged"] = True

    return results


def run_all_qc_checks(
    annotations: List[AnnotationRecord],
    gold_set: Optional[List[Dict]] = None,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run all QC checks and generate comprehensive report.

    Args:
        annotations: All annotations to check
        gold_set: Optional gold standard labels for accuracy check
        config: Optional configuration overrides

    Returns:
        Dict with all QC results:
        {
            "timestamp": str,
            "total_annotations": int,
            "time_bounds": {...},
            "consecutive_same": {...},
            "gold_accuracy": {...} | None,
            "summary": {
                "issues_found": int,
                "needs_attention": List[str]  # annotator IDs
            }
        }
    """
    if config is None:
        config = {}

    min_sec = config.get("min_time_seconds", 30)
    max_sec = config.get("max_time_seconds", 600)
    consecutive_threshold = config.get("consecutive_threshold", 5)

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_annotations": len(annotations),
        "time_bounds": check_time_bounds_batch(annotations, min_sec, max_sec),
        "consecutive_same": check_consecutive_same(
            annotations, consecutive_threshold, "error_detected"
        ),
        "gold_accuracy": None,
        "summary": {"issues_found": 0, "needs_attention": []},
    }

    # Gold accuracy check if gold set provided
    if gold_set:
        gold_results = {}
        by_annotator: Dict[str, List[AnnotationRecord]] = defaultdict(list)
        for ann in annotations:
            by_annotator[ann.annotator_id].append(ann)

        for annotator_id, annotator_anns in by_annotator.items():
            gold_results[annotator_id] = check_gold_set_accuracy(
                annotator_anns, gold_set, "error_detected"
            )

        report["gold_accuracy"] = gold_results

    # Compute summary
    issues = 0
    needs_attention = set()

    # Check time issues
    if report["time_bounds"]["too_fast_count"] > 0:
        issues += report["time_bounds"]["too_fast_count"]
        for ann_id in report["time_bounds"]["too_fast_ids"]:
            # Find annotator for this annotation
            for ann in annotations:
                if ann.annotation_id == ann_id:
                    needs_attention.add(ann.annotator_id)
                    break

    if report["time_bounds"]["too_slow_count"] > 0:
        issues += report["time_bounds"]["too_slow_count"]

    # Check consecutive same
    if report["consecutive_same"]["overall_flagged"]:
        for annotator_id, data in report["consecutive_same"]["by_annotator"].items():
            if data["flagged"]:
                issues += len(data["flagged_sequences"])
                needs_attention.add(annotator_id)

    # Check gold accuracy
    if report["gold_accuracy"]:
        for annotator_id, data in report["gold_accuracy"].items():
            if data["status"] == "recalibration_needed":
                issues += 1
                needs_attention.add(annotator_id)
            elif data["status"] == "warning":
                needs_attention.add(annotator_id)

    report["summary"]["issues_found"] = issues
    report["summary"]["needs_attention"] = sorted(needs_attention)

    return report


def generate_annotator_quality_report(
    annotations: List[AnnotationRecord],
    gold_set: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Generate per-annotator quality report.

    Args:
        annotations: All annotations
        gold_set: Optional gold standard labels

    Returns:
        Dict with per-annotator metrics:
        {
            "annotator_A": {
                "total_annotations": int,
                "mean_time": float,
                "time_issues": {"too_fast": int, "too_slow": int},
                "consecutive_same_flag": bool,
                "gold_accuracy": float | None,
                "quality_score": float  # 0-1 composite score
            },
            ...
        }
    """
    # Group by annotator
    by_annotator: Dict[str, List[AnnotationRecord]] = defaultdict(list)
    for ann in annotations:
        by_annotator[ann.annotator_id].append(ann)

    report = {}

    for annotator_id, annotator_anns in by_annotator.items():
        # Time stats
        times = [a.time_spent_seconds for a in annotator_anns]
        too_fast = sum(1 for t in times if t < 30)
        too_slow = sum(1 for t in times if t > 600)

        # Consecutive same check
        consec_check = check_consecutive_same(annotator_anns, 5, "error_detected")
        consec_flagged = (
            consec_check["by_annotator"].get(annotator_id, {}).get("flagged", False)
        )

        # Gold accuracy
        gold_acc = None
        if gold_set:
            gold_check = check_gold_set_accuracy(annotator_anns, gold_set)
            gold_acc = gold_check["accuracy"]

        # Compute composite quality score
        score = 1.0

        # Penalize time issues
        time_issue_rate = (too_fast + too_slow) / len(annotator_anns)
        score -= time_issue_rate * 0.3

        # Penalize consecutive same
        if consec_flagged:
            score -= 0.2

        # Factor in gold accuracy
        if gold_acc is not None:
            score = score * 0.7 + gold_acc * 0.3

        score = max(0.0, min(1.0, score))

        report[annotator_id] = {
            "total_annotations": len(annotator_anns),
            "mean_time": np.mean(times) if times else 0.0,
            "time_issues": {"too_fast": too_fast, "too_slow": too_slow},
            "consecutive_same_flag": consec_flagged,
            "gold_accuracy": gold_acc,
            "quality_score": score,
        }

    return report
