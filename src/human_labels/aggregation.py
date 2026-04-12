"""
Aggregation utilities for human labels.

This module provides functions to:
- Aggregate multiple annotations into a single ground-truth label
- Compute majority votes and agreement rates
- Derive localization accuracy against perturbation ground truth

Per Section 5A.4, aggregation produces AggregatedLabel objects ready
for downstream analysis (6A, 6B, 6C).
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from src.human_labels.schema import (
    AggregatedConsequence,
    AggregatedDetectability,
    AggregatedLabel,
    AggregatedPreference,
    AnnotationRecord,
    LocalizationAccuracy,
)


def aggregate_annotations(
    annotations: List[AnnotationRecord],
    low_agreement_threshold: float = 0.6,
) -> List[AggregatedLabel]:
    """
    Aggregate annotations into ground-truth labels by majority vote.

    Groups annotations by evaluation_unit_id and computes:
    - Majority vote for categorical dimensions
    - Mean/std for numerical dimensions
    - Agreement rates for each dimension
    - Flags for low agreement or needed adjudication

    Args:
        annotations: List of AnnotationRecord objects
        low_agreement_threshold: Agreement rate below which to flag

    Returns:
        List of AggregatedLabel objects, one per evaluation unit
    """
    # Group by evaluation unit
    by_unit: Dict[str, List[AnnotationRecord]] = defaultdict(list)
    for ann in annotations:
        by_unit[ann.evaluation_unit_id].append(ann)

    aggregated_labels = []

    for unit_id, unit_annotations in by_unit.items():
        # Separate by mode
        detect_anns = [a for a in unit_annotations if a.detectability is not None]
        conseq_anns = [a for a in unit_annotations if a.consequence is not None]
        pref_anns = [a for a in unit_annotations if a.preference is not None]

        annotation_ids = [a.annotation_id for a in unit_annotations]

        # Aggregate detectability
        agg_detect = None
        if detect_anns:
            agg_detect = _aggregate_detectability(detect_anns)

        # Aggregate consequence
        agg_conseq = None
        if conseq_anns:
            agg_conseq = _aggregate_consequence(conseq_anns)

        # Aggregate preference
        agg_pref = None
        if pref_anns:
            agg_pref = _aggregate_preference(pref_anns)

        # Compute flags
        low_agreement = _check_low_agreement(
            agg_detect, agg_conseq, agg_pref, low_agreement_threshold
        )
        needs_adjudication = _check_needs_adjudication(agg_detect, agg_conseq)

        agg_label = AggregatedLabel(
            evaluation_unit_id=unit_id,
            annotation_ids=annotation_ids,
            num_annotators=len(set(a.annotator_id for a in unit_annotations)),
            aggregated_detectability=agg_detect,
            aggregated_consequence=agg_conseq,
            aggregated_preference=agg_pref,
            low_agreement_flag=low_agreement,
            needs_adjudication=needs_adjudication,
        )
        aggregated_labels.append(agg_label)

    return aggregated_labels


def _aggregate_detectability(
    annotations: List[AnnotationRecord],
) -> AggregatedDetectability:
    """Aggregate detectability labels from multiple annotators."""
    # error_detected
    detected_votes = [a.detectability.error_detected for a in annotations]
    error_detected_majority = _majority_vote(detected_votes)
    error_detected_agreement = _compute_agreement(detected_votes)

    # error_trajectory
    trajectory_votes = [a.detectability.error_trajectory for a in annotations]
    error_trajectory_majority = _majority_vote(trajectory_votes)
    error_trajectory_agreement = _compute_agreement(trajectory_votes)

    # error_step_id
    step_votes = [
        a.detectability.error_step_id
        for a in annotations
        if a.detectability.error_step_id is not None
    ]
    error_step_id_majority = _majority_vote(step_votes) if step_votes else None

    # localization_agreement
    localization_agreement = _compute_localization_agreement(step_votes)

    # confidence
    confidence_values = [a.detectability.confidence for a in annotations]
    mean_confidence = np.mean(confidence_values) if confidence_values else 0.0

    return AggregatedDetectability(
        error_detected_majority=error_detected_majority,
        error_detected_agreement=error_detected_agreement,
        error_trajectory_majority=error_trajectory_majority,
        error_trajectory_agreement=error_trajectory_agreement,
        error_step_id_majority=error_step_id_majority,
        localization_agreement=localization_agreement,
        mean_confidence=mean_confidence,
    )


def _aggregate_consequence(
    annotations: List[AnnotationRecord],
) -> AggregatedConsequence:
    """Aggregate consequence labels from multiple annotators."""
    # error_type
    type_votes = [
        a.consequence.error_type
        for a in annotations
        if a.consequence.error_type is not None
    ]
    error_type_majority = _majority_vote(type_votes) if type_votes else None
    error_type_agreement = _compute_agreement(type_votes) if type_votes else 0.0

    # impact_tier
    impact_values = [
        a.consequence.impact_tier
        for a in annotations
        if a.consequence.impact_tier is not None
    ]
    mean_impact = np.mean(impact_values) if impact_values else 0.0
    impact_std = np.std(impact_values) if len(impact_values) > 1 else 0.0

    # propagation_depth
    prop_values = [
        a.consequence.propagation_depth
        for a in annotations
        if a.consequence.propagation_depth is not None
    ]
    mean_propagation = np.mean(prop_values) if prop_values else 0.0

    # correctness_a
    corr_a_votes = [
        a.consequence.correctness_a
        for a in annotations
        if a.consequence.correctness_a is not None
    ]
    correctness_a_majority = _majority_vote(corr_a_votes) if corr_a_votes else "unknown"

    # correctness_b
    corr_b_votes = [
        a.consequence.correctness_b
        for a in annotations
        if a.consequence.correctness_b is not None
    ]
    correctness_b_majority = _majority_vote(corr_b_votes) if corr_b_votes else "unknown"

    return AggregatedConsequence(
        error_type_majority=error_type_majority,
        error_type_agreement=error_type_agreement,
        mean_impact_tier=mean_impact,
        impact_tier_std=impact_std,
        mean_propagation_depth=mean_propagation,
        correctness_a_majority=correctness_a_majority,
        correctness_b_majority=correctness_b_majority,
    )


def _aggregate_preference(
    annotations: List[AnnotationRecord],
) -> AggregatedPreference:
    """Aggregate preference labels from multiple annotators."""
    pref_votes = [
        a.preference.preference
        for a in annotations
        if a.preference.preference is not None
    ]
    preference_majority = _majority_vote(pref_votes) if pref_votes else "tie"
    preference_agreement = _compute_agreement(pref_votes) if pref_votes else 0.0

    return AggregatedPreference(
        preference_majority=preference_majority,
        preference_agreement=preference_agreement,
    )


def _majority_vote(values: List[Any]) -> Any:
    """Compute majority vote from a list of values."""
    if not values:
        return None
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def _compute_agreement(values: List[Any]) -> float:
    """
    Compute agreement rate as proportion choosing majority.

    Returns:
        Float between 0 and 1
    """
    if len(values) < 2:
        return 1.0

    counter = Counter(values)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(values)


def _compute_localization_agreement(step_ids: List[str]) -> str:
    """
    Compute localization agreement label.

    Returns:
        "exact" if all annotators agree exactly
        "near" if disagreements are within 1 step (needs index info)
        "mixed" if significant disagreement
    """
    if not step_ids:
        return "mixed"

    unique_steps = set(step_ids)
    if len(unique_steps) == 1:
        return "exact"

    # If more than 2 unique values, it's mixed
    if len(unique_steps) > 2:
        return "mixed"

    # For 2 unique values, default to mixed (would need index info for "near")
    return "mixed"


def _check_low_agreement(
    detect: Optional[AggregatedDetectability],
    conseq: Optional[AggregatedConsequence],
    pref: Optional[AggregatedPreference],
    threshold: float,
) -> bool:
    """Check if any key dimension has low agreement."""
    if detect and detect.error_detected_agreement < threshold:
        return True
    if detect and detect.error_trajectory_agreement < threshold:
        return True
    if pref and pref.preference_agreement < threshold:
        return True
    return False


def _check_needs_adjudication(
    detect: Optional[AggregatedDetectability],
    conseq: Optional[AggregatedConsequence],
) -> bool:
    """Check if this unit needs expert adjudication."""
    # Adjudication needed for disagreement on error_detected or error_trajectory
    if detect:
        # If annotators split on whether error exists
        if detect.error_detected_agreement < 0.6:
            return True
        # If annotators split on which trajectory
        if detect.error_trajectory_agreement < 0.6:
            return True
    return False


def derive_localization_accuracy(
    human_step: Optional[str],
    ground_truth_step: Optional[str],
    step_index_map: Optional[Dict[str, int]] = None,
) -> str:
    """
    Compare human-annotated step to perturbation ground truth.

    Per 5A.5:
    - exact: step IDs match exactly
    - near: within 1 step by display_step_index
    - wrong: more than 1 step away or null when error exists

    Args:
        human_step: Step ID from human annotation (error_step_id_majority)
        ground_truth_step: Step ID from perturbation_record.target_step_canonical_id
        step_index_map: Optional mapping from step_id to numeric index

    Returns:
        LocalizationAccuracy value: "exact", "near", or "wrong"
    """
    if ground_truth_step is None:
        # No ground truth - can't evaluate
        return LocalizationAccuracy.WRONG.value

    if human_step is None:
        # Human didn't localize but error exists
        return LocalizationAccuracy.WRONG.value

    if human_step == ground_truth_step:
        return LocalizationAccuracy.EXACT.value

    # Check for near match
    if step_index_map is not None:
        human_idx = step_index_map.get(human_step)
        gt_idx = step_index_map.get(ground_truth_step)

        if human_idx is not None and gt_idx is not None:
            if abs(human_idx - gt_idx) <= 1:
                return LocalizationAccuracy.NEAR.value

    return LocalizationAccuracy.WRONG.value


def derive_error_type_match(
    human_type: Optional[str],
    perturbation_family: str,
) -> bool:
    """
    Check if human-labeled error type matches perturbation family.

    Per 5A.5 Error Type Mapping:
    - planning -> structural
    - tool_selection -> tool_selection
    - parameter -> parameter
    - data_reference -> data_reference
    - unclear -> any (always False)

    Args:
        human_type: ErrorType value from human annotation
        perturbation_family: Perturbation family from perturbation record

    Returns:
        True if types match, False otherwise
    """
    if human_type is None or human_type == "unclear":
        return False

    # Direct mappings
    type_to_family = {
        "planning": "structural",
        "tool_selection": "tool_selection",
        "parameter": "parameter",
        "data_reference": "data_reference",
    }

    expected_family = type_to_family.get(human_type)
    return expected_family == perturbation_family


def compute_aggregation_summary(
    aggregated_labels: List[AggregatedLabel],
) -> Dict[str, Any]:
    """
    Generate summary statistics for aggregated labels.

    Args:
        aggregated_labels: List of AggregatedLabel objects

    Returns:
        Dict with summary statistics:
        {
            "total_units": int,
            "units_with_detectability": int,
            "units_with_consequence": int,
            "units_with_preference": int,
            "low_agreement_count": int,
            "needs_adjudication_count": int,
            "mean_agreement": {
                "error_detected": float,
                "error_trajectory": float,
                "preference": float
            },
            "error_detected_distribution": {
                "true": int,
                "false": int
            },
            "error_trajectory_distribution": {...}
        }
    """
    summary: Dict[str, Any] = {
        "total_units": len(aggregated_labels),
        "units_with_detectability": 0,
        "units_with_consequence": 0,
        "units_with_preference": 0,
        "low_agreement_count": 0,
        "needs_adjudication_count": 0,
        "mean_agreement": {},
        "error_detected_distribution": defaultdict(int),
        "error_trajectory_distribution": defaultdict(int),
        "preference_distribution": defaultdict(int),
    }

    detect_agreements = []
    traj_agreements = []
    pref_agreements = []

    for label in aggregated_labels:
        if label.aggregated_detectability:
            summary["units_with_detectability"] += 1
            detect_agreements.append(
                label.aggregated_detectability.error_detected_agreement
            )
            traj_agreements.append(
                label.aggregated_detectability.error_trajectory_agreement
            )
            summary["error_detected_distribution"][
                str(label.aggregated_detectability.error_detected_majority)
            ] += 1
            summary["error_trajectory_distribution"][
                label.aggregated_detectability.error_trajectory_majority
            ] += 1

        if label.aggregated_consequence:
            summary["units_with_consequence"] += 1

        if label.aggregated_preference:
            summary["units_with_preference"] += 1
            pref_agreements.append(label.aggregated_preference.preference_agreement)
            summary["preference_distribution"][
                label.aggregated_preference.preference_majority
            ] += 1

        if label.low_agreement_flag:
            summary["low_agreement_count"] += 1
        if label.needs_adjudication:
            summary["needs_adjudication_count"] += 1

    # Compute means
    if detect_agreements:
        summary["mean_agreement"]["error_detected"] = np.mean(detect_agreements)
    if traj_agreements:
        summary["mean_agreement"]["error_trajectory"] = np.mean(traj_agreements)
    if pref_agreements:
        summary["mean_agreement"]["preference"] = np.mean(pref_agreements)

    # Convert defaultdicts to dicts
    summary["error_detected_distribution"] = dict(
        summary["error_detected_distribution"]
    )
    summary["error_trajectory_distribution"] = dict(
        summary["error_trajectory_distribution"]
    )
    summary["preference_distribution"] = dict(summary["preference_distribution"])

    return summary
