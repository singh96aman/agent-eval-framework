"""
Inter-annotator agreement metrics for human labels.

This module provides functions to compute agreement statistics:
- Krippendorff's alpha for nominal and ordinal data
- Pairwise agreement rates between annotators
- Localization accuracy (exact/near/wrong)

Per Section 5A.9, target thresholds:
- Krippendorff's alpha > 0.6 for binary dimensions (error_detected)
- Krippendorff's alpha > 0.4 for ordinal dimensions (impact_tier)
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.human_labels.schema import (
    AnnotationRecord,
    LocalizationAccuracy,
)


def compute_krippendorff_alpha(
    annotations: List[AnnotationRecord],
    dimension: str,
    level_of_measurement: str = "nominal",
) -> Dict[str, Any]:
    """
    Compute Krippendorff's alpha for a specific annotation dimension.

    Krippendorff's alpha is a reliability coefficient for measuring agreement
    among annotators on categorical or ordinal data. It handles missing data
    and works with any number of annotators.

    Args:
        annotations: List of AnnotationRecord objects
        dimension: Which dimension to compute alpha for. Options:
            - "error_detected" (binary)
            - "error_trajectory" (nominal: A, B, neither, both)
            - "error_type" (nominal)
            - "impact_tier" (ordinal: 0-3)
            - "propagation_depth" (ordinal: 0-3)
            - "confidence" (ordinal: 1-5)
            - "preference" (nominal: A, B, tie)
        level_of_measurement: "nominal", "ordinal", or "interval"

    Returns:
        Dict with:
        {
            "alpha": float,  # Krippendorff's alpha (-1 to 1)
            "n_units": int,  # Number of units with annotations
            "n_annotators": int,  # Number of unique annotators
            "n_observations": int,  # Total observations
            "dimension": str,
            "level": str,
            "interpretation": str  # "acceptable", "moderate", "low"
        }
    """
    # Group annotations by evaluation unit
    by_unit: Dict[str, List[AnnotationRecord]] = defaultdict(list)
    for ann in annotations:
        by_unit[ann.evaluation_unit_id].append(ann)

    # Extract values for the specified dimension
    values_by_unit: Dict[str, Dict[str, Any]] = {}
    all_annotators = set()

    for unit_id, unit_annotations in by_unit.items():
        values_by_unit[unit_id] = {}
        for ann in unit_annotations:
            value = _extract_dimension_value(ann, dimension)
            if value is not None:
                values_by_unit[unit_id][ann.annotator_id] = value
                all_annotators.add(ann.annotator_id)

    # Filter to units with at least 2 annotations
    units_with_overlap = {
        uid: vals for uid, vals in values_by_unit.items() if len(vals) >= 2
    }

    if not units_with_overlap or len(all_annotators) < 2:
        return {
            "alpha": None,
            "n_units": len(units_with_overlap),
            "n_annotators": len(all_annotators),
            "n_observations": sum(len(v) for v in values_by_unit.values()),
            "dimension": dimension,
            "level": level_of_measurement,
            "interpretation": "insufficient_data",
        }

    # Build reliability data matrix
    annotator_list = sorted(all_annotators)
    unit_list = sorted(units_with_overlap.keys())

    # Create value encoding
    all_values = set()
    for vals in units_with_overlap.values():
        all_values.update(vals.values())
    value_to_code = {v: i for i, v in enumerate(sorted(all_values, key=str))}

    # Build matrix: annotators x units
    # Missing values are np.nan
    matrix = np.full((len(annotator_list), len(unit_list)), np.nan)

    for j, unit_id in enumerate(unit_list):
        for i, annotator_id in enumerate(annotator_list):
            if annotator_id in units_with_overlap[unit_id]:
                value = units_with_overlap[unit_id][annotator_id]
                matrix[i, j] = value_to_code[value]

    # Compute alpha using the formula
    alpha = _compute_alpha(matrix, level_of_measurement)

    # Interpret alpha
    if alpha is None:
        interpretation = "insufficient_data"
    elif alpha >= 0.8:
        interpretation = "high"
    elif alpha >= 0.6:
        interpretation = "acceptable"
    elif alpha >= 0.4:
        interpretation = "moderate"
    else:
        interpretation = "low"

    return {
        "alpha": alpha,
        "n_units": len(units_with_overlap),
        "n_annotators": len(all_annotators),
        "n_observations": sum(len(v) for v in values_by_unit.values()),
        "dimension": dimension,
        "level": level_of_measurement,
        "interpretation": interpretation,
    }


def _compute_alpha(matrix: np.ndarray, level: str) -> Optional[float]:
    """
    Compute Krippendorff's alpha from a data matrix.

    Args:
        matrix: Annotators x units matrix with np.nan for missing values
        level: Level of measurement ("nominal", "ordinal", "interval")

    Returns:
        Alpha value or None if cannot be computed
    """
    n_annotators, n_units = matrix.shape

    # Get all observed values
    observed = matrix[~np.isnan(matrix)]
    if len(observed) < 2:
        return None

    # Compute number of pairable values per unit
    n_pairable = np.sum(~np.isnan(matrix), axis=0)

    # Units with at least 2 annotators
    valid_units = n_pairable >= 2
    if not np.any(valid_units):
        return None

    # Total number of pairable values
    n = np.sum(n_pairable[valid_units])
    if n < 2:
        return None

    # Get unique values
    unique_values = np.unique(observed)
    n_values = len(unique_values)
    if n_values < 2:
        return 1.0  # Perfect agreement if only one value used

    # Compute observed disagreement (D_o)
    do = 0.0
    for j in range(n_units):
        if not valid_units[j]:
            continue
        unit_values = matrix[:, j]
        unit_values = unit_values[~np.isnan(unit_values)]
        m = len(unit_values)
        if m < 2:
            continue

        # Sum of squared differences within this unit
        for i in range(m):
            for k in range(i + 1, m):
                diff = _difference_function(
                    unit_values[i], unit_values[k], level, unique_values
                )
                do += diff

    do = do / (n * (n - 1) / 2) if n > 1 else 0

    # Compute expected disagreement (D_e)
    # Distribution of all values
    value_counts = defaultdict(int)
    for v in observed:
        value_counts[v] += 1

    de = 0.0
    values_list = list(value_counts.keys())
    for i, v1 in enumerate(values_list):
        for v2 in values_list[i + 1 :]:
            diff = _difference_function(v1, v2, level, unique_values)
            de += value_counts[v1] * value_counts[v2] * diff

    de = de / (n * (n - 1) / 2) if n > 1 else 0

    # Compute alpha
    if de == 0:
        return 1.0 if do == 0 else None

    alpha = 1.0 - do / de
    return alpha


def _difference_function(
    v1: float, v2: float, level: str, unique_values: np.ndarray
) -> float:
    """Compute difference metric based on level of measurement."""
    if level == "nominal":
        return 0.0 if v1 == v2 else 1.0
    elif level == "ordinal":
        # For ordinal, difference is squared rank difference
        rank1 = np.where(unique_values == v1)[0][0]
        rank2 = np.where(unique_values == v2)[0][0]
        return (rank1 - rank2) ** 2
    elif level == "interval":
        return (v1 - v2) ** 2
    else:
        return 0.0 if v1 == v2 else 1.0


def _extract_dimension_value(ann: AnnotationRecord, dimension: str) -> Any:
    """Extract the value for a dimension from an annotation."""
    if dimension == "error_detected":
        if ann.detectability:
            return ann.detectability.error_detected
    elif dimension == "error_trajectory":
        if ann.detectability:
            return ann.detectability.error_trajectory
    elif dimension == "error_step_id":
        if ann.detectability:
            return ann.detectability.error_step_id
    elif dimension == "confidence":
        if ann.detectability:
            return ann.detectability.confidence
    elif dimension == "error_type":
        if ann.consequence:
            return ann.consequence.error_type
    elif dimension == "impact_tier":
        if ann.consequence:
            return ann.consequence.impact_tier
    elif dimension == "propagation_depth":
        if ann.consequence:
            return ann.consequence.propagation_depth
    elif dimension == "correctness_a":
        if ann.consequence:
            return ann.consequence.correctness_a
    elif dimension == "correctness_b":
        if ann.consequence:
            return ann.consequence.correctness_b
    elif dimension == "preference":
        if ann.preference:
            return ann.preference.preference
    return None


def compute_pairwise_agreement(
    annotations: List[AnnotationRecord],
    dimension: str = "error_detected",
) -> Dict[str, Any]:
    """
    Compute pairwise agreement rates between annotators.

    Args:
        annotations: List of AnnotationRecord objects
        dimension: Dimension to compute agreement for

    Returns:
        Dict with:
        {
            "overall_agreement": float,  # Proportion of agreement
            "pairwise": {
                ("annotator_A", "annotator_B"): {
                    "agreement_rate": float,
                    "n_shared": int,
                    "n_agree": int
                },
                ...
            },
            "annotator_stats": {
                "annotator_A": {
                    "n_annotations": int,
                    "mean_agreement_with_others": float
                },
                ...
            }
        }
    """
    # Group annotations by evaluation unit
    by_unit: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for ann in annotations:
        value = _extract_dimension_value(ann, dimension)
        if value is not None:
            by_unit[ann.evaluation_unit_id][ann.annotator_id] = value

    # Compute pairwise agreement
    all_annotators = set()
    for vals in by_unit.values():
        all_annotators.update(vals.keys())

    annotator_list = sorted(all_annotators)
    pairwise: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for i, a1 in enumerate(annotator_list):
        for a2 in annotator_list[i + 1 :]:
            n_shared = 0
            n_agree = 0
            for unit_id, vals in by_unit.items():
                if a1 in vals and a2 in vals:
                    n_shared += 1
                    if vals[a1] == vals[a2]:
                        n_agree += 1

            agreement_rate = n_agree / n_shared if n_shared > 0 else None
            pairwise[(a1, a2)] = {
                "agreement_rate": agreement_rate,
                "n_shared": n_shared,
                "n_agree": n_agree,
            }

    # Overall agreement
    total_shared = sum(p["n_shared"] for p in pairwise.values())
    total_agree = sum(p["n_agree"] for p in pairwise.values())
    overall_agreement = total_agree / total_shared if total_shared > 0 else None

    # Per-annotator stats
    annotator_stats: Dict[str, Dict[str, Any]] = {}
    for annotator in annotator_list:
        n_annotations = sum(1 for vals in by_unit.values() if annotator in vals)
        agreements = []
        for (a1, a2), stats in pairwise.items():
            if annotator in (a1, a2) and stats["agreement_rate"] is not None:
                agreements.append(stats["agreement_rate"])
        mean_agreement = np.mean(agreements) if agreements else None
        annotator_stats[annotator] = {
            "n_annotations": n_annotations,
            "mean_agreement_with_others": mean_agreement,
        }

    return {
        "overall_agreement": overall_agreement,
        "pairwise": {
            f"{a1}-{a2}": v for (a1, a2), v in pairwise.items()
        },  # Use string key for JSON
        "annotator_stats": annotator_stats,
    }


def compute_localization_accuracy(
    predicted_step: Optional[str],
    ground_truth_step: Optional[str],
    step_index_map: Optional[Dict[str, int]] = None,
) -> str:
    """
    Compute localization accuracy: exact, near, or wrong.

    Per 5A.5:
    - exact: error_step_id matches target_step_canonical_id exactly
    - near: error_step_id is within 1 step of target (by display_step_index)
    - wrong: error_step_id is more than 1 step away or null when error exists

    Args:
        predicted_step: The step ID predicted by annotator (canonical_step_id)
        ground_truth_step: The actual error step ID from perturbation record
        step_index_map: Optional mapping from step_id to numeric index
                        (for near computation). If None, only exact match is checked.

    Returns:
        LocalizationAccuracy value: "exact", "near", or "wrong"
    """
    if ground_truth_step is None:
        # No ground truth - can't compute accuracy
        return LocalizationAccuracy.WRONG.value

    if predicted_step is None:
        # Error exists but annotator didn't localize
        return LocalizationAccuracy.WRONG.value

    if predicted_step == ground_truth_step:
        return LocalizationAccuracy.EXACT.value

    # Check for near match
    if step_index_map is not None:
        pred_idx = step_index_map.get(predicted_step)
        gt_idx = step_index_map.get(ground_truth_step)

        if pred_idx is not None and gt_idx is not None:
            if abs(pred_idx - gt_idx) <= 1:
                return LocalizationAccuracy.NEAR.value

    return LocalizationAccuracy.WRONG.value


def compute_agreement_report(
    annotations: List[AnnotationRecord],
) -> Dict[str, Any]:
    """
    Generate comprehensive agreement report across all dimensions.

    Args:
        annotations: List of AnnotationRecord objects

    Returns:
        Dict with agreement metrics for each dimension:
        {
            "detectability": {
                "error_detected": {"alpha": ..., "pairwise": ...},
                "error_trajectory": {...},
                "confidence": {...}
            },
            "consequence": {
                "error_type": {...},
                "impact_tier": {...},
                "propagation_depth": {...}
            },
            "preference": {
                "preference": {...}
            },
            "summary": {
                "dimensions_meeting_threshold": int,
                "overall_quality": str
            }
        }
    """
    report: Dict[str, Any] = {
        "detectability": {},
        "consequence": {},
        "preference": {},
        "summary": {},
    }

    # Detectability dimensions
    detect_dims = [
        ("error_detected", "nominal"),
        ("error_trajectory", "nominal"),
        ("confidence", "ordinal"),
    ]
    for dim, level in detect_dims:
        detect_anns = [a for a in annotations if a.detectability is not None]
        if detect_anns:
            alpha_result = compute_krippendorff_alpha(detect_anns, dim, level)
            pairwise_result = compute_pairwise_agreement(detect_anns, dim)
            report["detectability"][dim] = {
                "krippendorff_alpha": alpha_result,
                "pairwise_agreement": pairwise_result,
            }

    # Consequence dimensions
    conseq_dims = [
        ("error_type", "nominal"),
        ("impact_tier", "ordinal"),
        ("propagation_depth", "ordinal"),
    ]
    for dim, level in conseq_dims:
        conseq_anns = [a for a in annotations if a.consequence is not None]
        if conseq_anns:
            alpha_result = compute_krippendorff_alpha(conseq_anns, dim, level)
            pairwise_result = compute_pairwise_agreement(conseq_anns, dim)
            report["consequence"][dim] = {
                "krippendorff_alpha": alpha_result,
                "pairwise_agreement": pairwise_result,
            }

    # Preference dimensions
    pref_anns = [a for a in annotations if a.preference is not None]
    if pref_anns:
        alpha_result = compute_krippendorff_alpha(pref_anns, "preference", "nominal")
        pairwise_result = compute_pairwise_agreement(pref_anns, "preference")
        report["preference"]["preference"] = {
            "krippendorff_alpha": alpha_result,
            "pairwise_agreement": pairwise_result,
        }

    # Summary
    dims_meeting_threshold = 0
    all_alphas = []

    for mode_report in [
        report["detectability"],
        report["consequence"],
        report["preference"],
    ]:
        for dim_report in mode_report.values():
            alpha_data = dim_report.get("krippendorff_alpha", {})
            alpha = alpha_data.get("alpha")
            if alpha is not None:
                all_alphas.append(alpha)
                # Binary dims need > 0.6, ordinal need > 0.4
                level = alpha_data.get("level", "nominal")
                threshold = 0.6 if level == "nominal" else 0.4
                if alpha >= threshold:
                    dims_meeting_threshold += 1

    report["summary"] = {
        "dimensions_meeting_threshold": dims_meeting_threshold,
        "total_dimensions": len(all_alphas),
        "mean_alpha": np.mean(all_alphas) if all_alphas else None,
        "overall_quality": (
            "acceptable"
            if dims_meeting_threshold >= len(all_alphas) * 0.8
            else "needs_review"
        ),
    }

    return report
