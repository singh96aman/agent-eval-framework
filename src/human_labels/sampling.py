"""
Sampling and assignment utilities for human annotation.

This module provides functions to:
- Sample evaluation units for annotation with stratification
- Assign units to annotators with overlap for agreement measurement
- Generate annotation batches

Per Section 5A.8, the sampling strategy ensures:
- Target sample sizes per mode (350 detectability, 200 consequence, 150 preference)
- Overlap for inter-rater agreement (50 full overlap, 100 pairwise)
- Stratification by benchmark, perturbation class, family, and detectability band
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse dimension constants from evaluation sampling
BENCHMARKS = ["toolbench", "swebench"]
PERTURBATION_CLASSES = ["placebo", "fine_grained", "coarse_grained"]
PERTURBATION_FAMILIES = [
    "data_reference",
    "parameter",
    "tool_selection",
    "structural",
    "terminal_flag",
]
DETECTABILITY_BANDS = [0, 1, 2]

# Default sample configuration per 5A.8
DEFAULT_ANNOTATION_CONFIG = {
    "mode_a_target": 350,  # Detectability annotation
    "mode_b_target": 200,  # Consequence annotation
    "mode_c_target": 150,  # Preference annotation
    "full_overlap_count": 50,  # All annotators label these
    "pairwise_overlap_count": 100,  # Each pair overlaps on additional 50
    "num_annotators": 3,
    "stratification": {
        "high_impact": 100,  # expected_impact=3
        "placebo": 75,
        "fine_grained_subtle": 100,  # fine-grained with low detectability
        "coarse_grained": 75,
    },
    "coverage_minimums": {
        "benchmark": 50,
        "perturbation_class": 50,
        "perturbation_family": 20,
        "detectability_band": 75,
    },
}


def _extract_dimension_value(unit: Dict, dimension: str) -> Any:
    """
    Extract a dimension value from an evaluation unit.

    Args:
        unit: Evaluation unit dict
        dimension: One of 'benchmark', 'perturbation_class', 'perturbation_family',
                   'expected_detectability', 'expected_impact'

    Returns:
        The value for the specified dimension
    """
    # Try direct access first
    if dimension in unit:
        return unit[dimension]

    # Check derived_cache
    derived_cache = unit.get("derived_cache", {})
    if dimension in derived_cache:
        return derived_cache[dimension]

    # For perturbation record fields, check perturbed.perturbation_record
    perturbed = unit.get("perturbed", {})
    perturbation_record = perturbed.get("perturbation_record", {})
    if dimension in perturbation_record:
        return perturbation_record[dimension]

    return None


def _build_dimension_index(units: List[Dict], dimension: str) -> Dict[Any, List[int]]:
    """
    Build an index mapping dimension values to unit indices.

    Args:
        units: List of evaluation units
        dimension: Dimension to index by

    Returns:
        Dict mapping dimension values to lists of unit indices
    """
    index: Dict[Any, List[int]] = defaultdict(list)
    for i, unit in enumerate(units):
        value = _extract_dimension_value(unit, dimension)
        if value is not None:
            index[value].append(i)
    return index


def sample_for_annotation(
    units: List[Dict],
    config: Optional[Dict] = None,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Stratified sampling of evaluation units for human annotation.

    Samples units ensuring coverage constraints per 5A.8:
    - High-impact units (expected_impact=3): ~100
    - Placebo units: ~75
    - Fine-grained subtle: ~100
    - Coarse-grained: ~75

    Also ensures minimum coverage:
    - 50+ per benchmark
    - 50+ per perturbation class
    - 20+ per perturbation family
    - 75+ per detectability band

    Args:
        units: List of all evaluation unit dicts
        config: Sampling configuration. If None, uses DEFAULT_ANNOTATION_CONFIG.
        seed: Random seed for reproducibility

    Returns:
        Dict with sampled units by mode:
        {
            "mode_a": List[Dict],  # 350 units for detectability
            "mode_b": List[Dict],  # 200 units for consequence (subset of mode_a)
            "mode_c": List[Dict],  # 150 units for preference (subset)
            "sampling_report": Dict  # Coverage statistics
        }

    Raises:
        ValueError: If not enough units to meet constraints
    """
    if config is None:
        config = DEFAULT_ANNOTATION_CONFIG

    random.seed(seed)
    np.random.seed(seed)

    target_a = config.get("mode_a_target", 350)
    target_b = config.get("mode_b_target", 200)
    target_c = config.get("mode_c_target", 150)
    strat = config.get("stratification", {})
    cov_min = config.get("coverage_minimums", {})

    if len(units) < target_a:
        raise ValueError(
            f"Not enough units ({len(units)}) for target sample size ({target_a})"
        )

    # Build dimension indices
    impact_index = _build_dimension_index(units, "expected_impact")
    class_index = _build_dimension_index(units, "perturbation_class")
    family_index = _build_dimension_index(units, "perturbation_family")
    detect_index = _build_dimension_index(units, "expected_detectability")
    benchmark_index = _build_dimension_index(units, "benchmark")

    # Track selected indices
    selected: set = set()

    # Phase 1: Sample high-impact units (expected_impact=3)
    # Only use default if stratification config exists and doesn't specify
    high_impact_target = (
        strat.get("high_impact", 0) if not strat else strat.get("high_impact", 100)
    )
    if high_impact_target > 0:
        high_impact_pool = impact_index.get(3, [])
        if high_impact_pool:
            n = min(high_impact_target, len(high_impact_pool), target_a)
            selected.update(random.sample(high_impact_pool, n))

    # Phase 2: Sample placebos
    placebo_target = strat.get("placebo", 0) if not strat else strat.get("placebo", 75)
    if placebo_target > 0 and len(selected) < target_a:
        placebo_pool = [i for i in class_index.get("placebo", []) if i not in selected]
        if placebo_pool:
            n = min(placebo_target, len(placebo_pool), target_a - len(selected))
            selected.update(random.sample(placebo_pool, n))

    # Phase 3: Sample fine-grained subtle (fine_grained + detectability 0 or 1)
    subtle_target = (
        strat.get("fine_grained_subtle", 0)
        if not strat
        else strat.get("fine_grained_subtle", 100)
    )
    if subtle_target > 0 and len(selected) < target_a:
        subtle_pool = []
        for i in class_index.get("fine_grained", []):
            if i in selected:
                continue
            detect_val = _extract_dimension_value(units[i], "expected_detectability")
            if detect_val is not None and detect_val <= 1:
                subtle_pool.append(i)
        if subtle_pool:
            n = min(subtle_target, len(subtle_pool), target_a - len(selected))
            selected.update(random.sample(subtle_pool, n))

    # Phase 4: Sample coarse-grained
    coarse_target = (
        strat.get("coarse_grained", 0) if not strat else strat.get("coarse_grained", 75)
    )
    if coarse_target > 0 and len(selected) < target_a:
        coarse_pool = [
            i for i in class_index.get("coarse_grained", []) if i not in selected
        ]
        if coarse_pool:
            n = min(coarse_target, len(coarse_pool), target_a - len(selected))
            selected.update(random.sample(coarse_pool, n))

    # Phase 5: Fill coverage minimums (only if minimums are specified)
    if cov_min:
        dimension_configs = [
            ("benchmark", benchmark_index, cov_min.get("benchmark", 0)),
            ("perturbation_class", class_index, cov_min.get("perturbation_class", 0)),
            (
                "perturbation_family",
                family_index,
                cov_min.get("perturbation_family", 0),
            ),
            (
                "expected_detectability",
                detect_index,
                cov_min.get("detectability_band", 0),
            ),
        ]

        for dim_name, dim_index, minimum in dimension_configs:
            if minimum <= 0:
                continue
            for value, indices in dim_index.items():
                if len(selected) >= target_a:
                    break
                already = len(selected.intersection(indices))
                needed = minimum - already
                if needed > 0:
                    available = [i for i in indices if i not in selected]
                    max_to_add = target_a - len(selected)
                    needed = min(needed, max_to_add, len(available))
                    if needed > 0:
                        selected.update(random.sample(available, needed))

    # Phase 6: Fill to target with random sampling
    if len(selected) < target_a:
        remaining = [i for i in range(len(units)) if i not in selected]
        needed = target_a - len(selected)
        if remaining:
            selected.update(random.sample(remaining, min(needed, len(remaining))))

    # Convert to list of units (mode_a)
    selected_list = sorted(selected)
    mode_a_units = [units[i] for i in selected_list]

    # Mode B: Subset of mode_a (prefer units where error was detected or stratified)
    # For now, just take first target_b units
    mode_b_units = (
        mode_a_units[:target_b] if len(mode_a_units) >= target_b else mode_a_units
    )

    # Mode C: Subset for preference (smaller sample)
    mode_c_units = (
        mode_a_units[:target_c] if len(mode_a_units) >= target_c else mode_a_units
    )

    # Generate sampling report
    report = _generate_sampling_report(mode_a_units, units)

    return {
        "mode_a": mode_a_units,
        "mode_b": mode_b_units,
        "mode_c": mode_c_units,
        "sampling_report": report,
    }


def _generate_sampling_report(sample: List[Dict], population: List[Dict]) -> Dict:
    """Generate coverage statistics for the sample."""
    report = {
        "sample_size": len(sample),
        "population_size": len(population),
        "by_benchmark": {},
        "by_perturbation_class": {},
        "by_perturbation_family": {},
        "by_detectability": {},
        "by_impact": {},
    }

    # Count by each dimension
    for unit in sample:
        benchmark = _extract_dimension_value(unit, "benchmark") or "unknown"
        pclass = _extract_dimension_value(unit, "perturbation_class") or "unknown"
        family = _extract_dimension_value(unit, "perturbation_family") or "unknown"
        detect = _extract_dimension_value(unit, "expected_detectability")
        impact = _extract_dimension_value(unit, "expected_impact")

        report["by_benchmark"][benchmark] = report["by_benchmark"].get(benchmark, 0) + 1
        report["by_perturbation_class"][pclass] = (
            report["by_perturbation_class"].get(pclass, 0) + 1
        )
        report["by_perturbation_family"][family] = (
            report["by_perturbation_family"].get(family, 0) + 1
        )
        if detect is not None:
            detect_key = str(detect)
            report["by_detectability"][detect_key] = (
                report["by_detectability"].get(detect_key, 0) + 1
            )
        if impact is not None:
            impact_key = str(impact)
            report["by_impact"][impact_key] = report["by_impact"].get(impact_key, 0) + 1

    return report


def assign_to_annotators(
    sample: List[Dict],
    annotators: List[str],
    overlap_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Distribute sampled units to annotators with overlap for agreement.

    Per 5A.8:
    - Full overlap: 50 units labeled by ALL annotators
    - Pairwise overlap: 100 units where each pair overlaps on additional 50
    - Single annotator: Remaining units get one annotator each

    Args:
        sample: List of evaluation unit dicts to assign
        annotators: List of annotator IDs
        overlap_config: Configuration for overlap. If None, uses defaults.

    Returns:
        Dict with assignments:
        {
            "assignments_by_annotator": {
                "annotator_A": [unit_ids...],
                "annotator_B": [unit_ids...],
                ...
            },
            "assignments_by_unit": {
                "unit_id": ["annotator_A", "annotator_B"],
                ...
            },
            "full_overlap_units": [unit_ids...],  # All annotators
            "pairwise_overlap_units": {
                ("A", "B"): [unit_ids...],
                ...
            },
            "single_annotator_units": [unit_ids...],
            "workload_summary": {
                "annotator_A": 120,  # Total units
                ...
            }
        }
    """
    if overlap_config is None:
        overlap_config = {
            "full_overlap_count": 50,
            "pairwise_overlap_count": 100,  # Total across all pairs
        }

    n_annotators = len(annotators)
    if n_annotators < 2:
        raise ValueError("Need at least 2 annotators for overlap computation")

    full_overlap_count = min(overlap_config.get("full_overlap_count", 50), len(sample))
    total_pairwise = overlap_config.get("pairwise_overlap_count", 100)

    # Get unit IDs
    unit_ids = [u.get("evaluation_unit_id", f"unit_{i}") for i, u in enumerate(sample)]

    # Shuffle for random assignment
    shuffled_ids = unit_ids.copy()
    random.shuffle(shuffled_ids)

    # Assign full overlap units
    full_overlap_units = shuffled_ids[:full_overlap_count]
    remaining_ids = shuffled_ids[full_overlap_count:]

    # Assign pairwise overlap units
    # Generate all pairs of annotators
    pairs = []
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            pairs.append((annotators[i], annotators[j]))

    # Distribute pairwise overlap evenly
    per_pair = total_pairwise // len(pairs) if pairs else 0
    pairwise_overlap_units: Dict[Tuple[str, str], List[str]] = {}
    pairwise_assigned: set = set()

    idx = 0
    for pair in pairs:
        pair_units = []
        for _ in range(per_pair):
            if idx < len(remaining_ids):
                pair_units.append(remaining_ids[idx])
                pairwise_assigned.add(remaining_ids[idx])
                idx += 1
        pairwise_overlap_units[pair] = pair_units

    # Remaining units get single annotator (round-robin)
    single_units = [uid for uid in remaining_ids if uid not in pairwise_assigned]

    # Build assignments by annotator
    assignments_by_annotator: Dict[str, List[str]] = {a: [] for a in annotators}
    assignments_by_unit: Dict[str, List[str]] = {}

    # Full overlap - all annotators
    for uid in full_overlap_units:
        assignments_by_unit[uid] = list(annotators)
        for a in annotators:
            assignments_by_annotator[a].append(uid)

    # Pairwise overlap
    for pair, uids in pairwise_overlap_units.items():
        for uid in uids:
            assignments_by_unit[uid] = list(pair)
            for a in pair:
                assignments_by_annotator[a].append(uid)

    # Single annotator - round robin
    for i, uid in enumerate(single_units):
        annotator = annotators[i % n_annotators]
        assignments_by_unit[uid] = [annotator]
        assignments_by_annotator[annotator].append(uid)

    # Compute workload summary
    workload_summary = {a: len(uids) for a, uids in assignments_by_annotator.items()}

    return {
        "assignments_by_annotator": assignments_by_annotator,
        "assignments_by_unit": assignments_by_unit,
        "full_overlap_units": full_overlap_units,
        "pairwise_overlap_units": pairwise_overlap_units,
        "single_annotator_units": single_units,
        "workload_summary": workload_summary,
    }


def generate_annotation_batches(
    assignments: Dict[str, Any],
    batch_size: int = 25,
) -> Dict[str, List[List[str]]]:
    """
    Organize annotator assignments into batches.

    Args:
        assignments: Output from assign_to_annotators
        batch_size: Number of units per batch

    Returns:
        Dict mapping annotator ID to list of batches (each batch is list of unit IDs)
    """
    batches_by_annotator: Dict[str, List[List[str]]] = {}

    for annotator, unit_ids in assignments["assignments_by_annotator"].items():
        batches = []
        for i in range(0, len(unit_ids), batch_size):
            batches.append(unit_ids[i : i + batch_size])
        batches_by_annotator[annotator] = batches

    return batches_by_annotator
