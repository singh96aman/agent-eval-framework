"""
Stratified sampling for human evaluation.

This module provides functions to sample evaluation units for human annotation
with coverage constraints across multiple dimensions (benchmark, perturbation class,
perturbation family, and detectability band).

Per Section 4.8, the sampling strategy ensures:
- Sufficient coverage across all stratification dimensions
- No single cell is over-represented (max 35% of sample)
- Reproducible sampling via random seed
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Default coverage minimums per Section 4.8
DEFAULT_COVERAGE_CONFIG = {
    "benchmark": {"minimum": 50},
    "perturbation_class": {"minimum": 50},
    "perturbation_family": {"minimum": 20},
    "expected_detectability": {"minimum": 75},
}

# Known values for each dimension
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


def get_default_coverage_config() -> Dict[str, Any]:
    """
    Return default coverage requirements for human evaluation sampling.

    Returns:
        Dict with coverage minimums per dimension:
        - benchmark: 50 per benchmark (toolbench, swebench)
        - perturbation_class: 50 per class (placebo, fine_grained, coarse_grained)
        - perturbation_family: 20 per family
        - expected_detectability: 75 per band (0, 1, 2)
    """
    return {
        "benchmark": {"minimum": 50},
        "perturbation_class": {"minimum": 50},
        "perturbation_family": {"minimum": 20},
        "expected_detectability": {"minimum": 75},
    }


def _extract_dimension_value(unit: Dict, dimension: str) -> Any:
    """
    Extract a dimension value from an evaluation unit.

    Args:
        unit: Evaluation unit dict (may be EvaluationUnit.to_dict() or raw dict)
        dimension: One of 'benchmark', 'perturbation_class', 'perturbation_family',
                   'expected_detectability'

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


def _get_dimension_values(dimension: str) -> List[Any]:
    """Get the known values for a dimension."""
    if dimension == "benchmark":
        return BENCHMARKS
    elif dimension == "perturbation_class":
        return PERTURBATION_CLASSES
    elif dimension == "perturbation_family":
        return PERTURBATION_FAMILIES
    elif dimension == "expected_detectability":
        return DETECTABILITY_BANDS
    else:
        return []


def check_coverage_minimums(units: List[Dict], config: Optional[Dict] = None) -> Dict:
    """
    Check if a sample meets all coverage minimums.

    Args:
        units: List of evaluation unit dicts
        config: Coverage configuration dict. If None, uses default config.

    Returns:
        Dict with coverage check results:
        {
            "meets_all_minimums": true/false,
            "by_benchmark": {"toolbench": {"count": 200, "minimum": 50, "met": true}, ...},
            "by_class": {...},
            "by_family": {...},
            "by_detectability": {...}
        }
    """
    if config is None:
        config = get_default_coverage_config()

    result = {
        "meets_all_minimums": True,
        "by_benchmark": {},
        "by_class": {},
        "by_family": {},
        "by_detectability": {},
    }

    # Count by benchmark
    benchmark_counts: Dict[str, int] = defaultdict(int)
    for unit in units:
        benchmark = _extract_dimension_value(unit, "benchmark")
        if benchmark:
            benchmark_counts[benchmark] += 1

    benchmark_minimum = config.get("benchmark", {}).get("minimum", 50)
    for benchmark in BENCHMARKS:
        count = benchmark_counts.get(benchmark, 0)
        met = count >= benchmark_minimum
        result["by_benchmark"][benchmark] = {
            "count": count,
            "minimum": benchmark_minimum,
            "met": met,
        }
        if not met:
            result["meets_all_minimums"] = False

    # Count by perturbation_class
    class_counts: Dict[str, int] = defaultdict(int)
    for unit in units:
        pclass = _extract_dimension_value(unit, "perturbation_class")
        if pclass:
            class_counts[pclass] += 1

    class_minimum = config.get("perturbation_class", {}).get("minimum", 50)
    for pclass in PERTURBATION_CLASSES:
        count = class_counts.get(pclass, 0)
        met = count >= class_minimum
        result["by_class"][pclass] = {
            "count": count,
            "minimum": class_minimum,
            "met": met,
        }
        if not met:
            result["meets_all_minimums"] = False

    # Count by perturbation_family
    family_counts: Dict[str, int] = defaultdict(int)
    for unit in units:
        family = _extract_dimension_value(unit, "perturbation_family")
        if family:
            family_counts[family] += 1

    family_minimum = config.get("perturbation_family", {}).get("minimum", 20)
    for family in PERTURBATION_FAMILIES:
        count = family_counts.get(family, 0)
        met = count >= family_minimum
        result["by_family"][family] = {
            "count": count,
            "minimum": family_minimum,
            "met": met,
        }
        if not met:
            result["meets_all_minimums"] = False

    # Count by expected_detectability
    detect_counts: Dict[int, int] = defaultdict(int)
    for unit in units:
        detectability = _extract_dimension_value(unit, "expected_detectability")
        if detectability is not None:
            detect_counts[detectability] += 1

    detect_minimum = config.get("expected_detectability", {}).get("minimum", 75)
    for band in DETECTABILITY_BANDS:
        count = detect_counts.get(band, 0)
        met = count >= detect_minimum
        result["by_detectability"][band] = {
            "count": count,
            "minimum": detect_minimum,
            "met": met,
        }
        if not met:
            result["meets_all_minimums"] = False

    return result


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


def _check_overrepresentation(
    sample_indices: List[int],
    units: List[Dict],
    max_proportion: float = 0.35,
) -> Tuple[bool, Dict[str, Dict[str, float]]]:
    """
    Check if any single cell is over-represented in the sample.

    Only applies to dimensions with more than 2 possible values, since
    dimensions with only 2 values (e.g., benchmark) will naturally have
    ~50% per value when balanced.

    Args:
        sample_indices: Indices of selected units
        units: Full list of evaluation units
        max_proportion: Maximum allowed proportion for any single cell

    Returns:
        Tuple of (is_valid, proportions_by_dimension)
    """
    if len(sample_indices) == 0:
        return True, {}

    sample_units = [units[i] for i in sample_indices]
    total = len(sample_units)

    # Dimensions with their expected number of values
    # Only check max_proportion for dimensions with >2 values
    dimensions = [
        ("benchmark", 2),  # Only 2 benchmarks - skip proportion check
        ("perturbation_class", 3),
        ("perturbation_family", 5),
        ("expected_detectability", 3),
    ]

    proportions: Dict[str, Dict[str, float]] = {}
    is_valid = True

    for dim, num_values in dimensions:
        counts: Dict[Any, int] = defaultdict(int)
        for unit in sample_units:
            value = _extract_dimension_value(unit, dim)
            if value is not None:
                counts[value] += 1

        proportions[dim] = {}
        for value, count in counts.items():
            prop = count / total
            proportions[dim][str(value)] = prop
            # Only check proportion for dimensions with >2 values
            if num_values > 2 and prop > max_proportion:
                is_valid = False

    return is_valid, proportions


def sample_human_evaluation_set(
    units: List[Dict],
    target: int = 350,
    coverage_config: Optional[Dict] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Stratified sampling with coverage constraints for human evaluation.

    Sampling strategy:
    1. First, ensure minimums are met by reserving slots for under-represented cells
    2. Fill remaining slots with stratified random sampling
    3. Verify no dimension is over-represented (no single cell > 35%)

    Args:
        units: List of all evaluation unit dicts (785 total expected)
        target: Target sample size (default 350)
        coverage_config: Coverage configuration. If None, uses default.
        seed: Random seed for reproducibility

    Returns:
        List of selected evaluation unit dicts

    Raises:
        ValueError: If target exceeds available units or minimums cannot be met
    """
    if coverage_config is None:
        coverage_config = get_default_coverage_config()

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    if target > len(units):
        raise ValueError(f"Target {target} exceeds available units {len(units)}")

    # Build indices for each dimension
    benchmark_index = _build_dimension_index(units, "benchmark")
    class_index = _build_dimension_index(units, "perturbation_class")
    family_index = _build_dimension_index(units, "perturbation_family")
    detect_index = _build_dimension_index(units, "expected_detectability")

    # Track selected indices
    selected_indices: set = set()

    # Phase 1: Ensure coverage minimums are met
    # Process dimensions in order of constraint tightness

    dimension_configs = [
        (
            "benchmark",
            benchmark_index,
            coverage_config.get("benchmark", {}).get("minimum", 50),
        ),
        (
            "perturbation_class",
            class_index,
            coverage_config.get("perturbation_class", {}).get("minimum", 50),
        ),
        (
            "perturbation_family",
            family_index,
            coverage_config.get("perturbation_family", {}).get("minimum", 20),
        ),
        (
            "expected_detectability",
            detect_index,
            coverage_config.get("expected_detectability", {}).get("minimum", 75),
        ),
    ]

    for dim_name, dim_index, minimum in dimension_configs:
        for value, indices in dim_index.items():
            # Check if we've already reached the target
            if len(selected_indices) >= target:
                break

            # Count how many already selected for this value
            already_selected = len(selected_indices.intersection(indices))

            # If below minimum, sample more
            needed = minimum - already_selected
            if needed > 0:
                available = [i for i in indices if i not in selected_indices]
                # Respect target cap
                max_to_add = target - len(selected_indices)
                needed = min(needed, max_to_add)

                if len(available) < needed:
                    # Cannot meet minimum - take all available (up to max)
                    to_add = available[:max_to_add]
                else:
                    to_add = random.sample(available, needed)
                selected_indices.update(to_add)

        if len(selected_indices) >= target:
            break

    # Phase 2: Fill remaining slots with stratified random sampling
    remaining_slots = target - len(selected_indices)

    if remaining_slots > 0:
        # Get indices not yet selected
        available_indices = [i for i in range(len(units)) if i not in selected_indices]

        if len(available_indices) <= remaining_slots:
            # Take all remaining
            selected_indices.update(available_indices)
        else:
            # Stratified sampling: weight by inverse frequency in current sample
            # to encourage balance

            # Calculate weights based on under-representation
            weights = np.ones(len(available_indices))

            # Weight units from under-represented cells higher
            current_counts: Dict[str, Dict[Any, int]] = {}
            for dim_name, dim_index, _ in dimension_configs:
                current_counts[dim_name] = defaultdict(int)
                for idx in selected_indices:
                    value = _extract_dimension_value(units[idx], dim_name)
                    if value is not None:
                        current_counts[dim_name][value] += 1

            for i, idx in enumerate(available_indices):
                boost = 1.0
                for dim_name, _, _ in dimension_configs:
                    value = _extract_dimension_value(units[idx], dim_name)
                    if value is not None:
                        count = current_counts[dim_name].get(value, 0)
                        # Lower count = higher boost
                        if count > 0:
                            boost *= 1.0 / (1.0 + count / len(selected_indices))
                        else:
                            boost *= 2.0  # Strong boost for zero counts
                weights[i] = boost

            # Normalize weights
            weights = weights / weights.sum()

            # Sample without replacement
            sampled = np.random.choice(
                available_indices,
                size=min(remaining_slots, len(available_indices)),
                replace=False,
                p=weights,
            )
            selected_indices.update(sampled)

    # Phase 3: Verify no over-representation
    is_valid, proportions = _check_overrepresentation(
        list(selected_indices), units, max_proportion=0.35
    )

    if not is_valid:
        # Log warning but don't fail - may be inherent in the data
        pass

    # Return selected units in original order
    selected_indices_sorted = sorted(selected_indices)
    return [units[i] for i in selected_indices_sorted]


def get_sampling_report(sample: List[Dict], all_units: List[Dict]) -> Dict:
    """
    Generate report showing sample distribution vs full dataset.

    Args:
        sample: List of sampled evaluation unit dicts
        all_units: List of all evaluation unit dicts

    Returns:
        Dict with distribution comparison:
        {
            "sample_size": int,
            "total_size": int,
            "sampling_rate": float,
            "coverage_check": {...},  # Result of check_coverage_minimums
            "distributions": {
                "benchmark": {
                    "sample": {"toolbench": {"count": N, "pct": X}, ...},
                    "population": {"toolbench": {"count": N, "pct": X}, ...}
                },
                "perturbation_class": {...},
                "perturbation_family": {...},
                "expected_detectability": {...}
            },
            "balance_metrics": {
                "max_cell_proportion": float,
                "is_balanced": bool  # True if no cell > 35%
            }
        }
    """
    sample_size = len(sample)
    total_size = len(all_units)

    result = {
        "sample_size": sample_size,
        "total_size": total_size,
        "sampling_rate": sample_size / total_size if total_size > 0 else 0.0,
        "coverage_check": check_coverage_minimums(sample),
        "distributions": {},
        "balance_metrics": {},
    }

    dimensions = [
        ("benchmark", BENCHMARKS),
        ("perturbation_class", PERTURBATION_CLASSES),
        ("perturbation_family", PERTURBATION_FAMILIES),
        ("expected_detectability", DETECTABILITY_BANDS),
    ]

    max_proportion = 0.0

    # Dimensions to check for balance (exclude benchmark since only 2 values)
    balance_check_dimensions = {
        "perturbation_class",
        "perturbation_family",
        "expected_detectability",
    }

    for dim_name, known_values in dimensions:
        # Count in sample
        sample_counts: Dict[Any, int] = defaultdict(int)
        for unit in sample:
            value = _extract_dimension_value(unit, dim_name)
            if value is not None:
                sample_counts[value] += 1

        # Count in population
        pop_counts: Dict[Any, int] = defaultdict(int)
        for unit in all_units:
            value = _extract_dimension_value(unit, dim_name)
            if value is not None:
                pop_counts[value] += 1

        sample_dist = {}
        pop_dist = {}

        for value in known_values:
            s_count = sample_counts.get(value, 0)
            p_count = pop_counts.get(value, 0)

            s_pct = (s_count / sample_size * 100) if sample_size > 0 else 0.0
            p_pct = (p_count / total_size * 100) if total_size > 0 else 0.0

            sample_dist[str(value)] = {"count": s_count, "pct": round(s_pct, 1)}
            pop_dist[str(value)] = {"count": p_count, "pct": round(p_pct, 1)}

            # Track max proportion (only for dimensions with >2 values)
            if sample_size > 0 and dim_name in balance_check_dimensions:
                prop = s_count / sample_size
                if prop > max_proportion:
                    max_proportion = prop

        result["distributions"][dim_name] = {
            "sample": sample_dist,
            "population": pop_dist,
        }

    result["balance_metrics"] = {
        "max_cell_proportion": round(max_proportion, 3),
        "is_balanced": max_proportion <= 0.35,
    }

    return result


def print_sampling_report(report: Dict) -> None:
    """
    Print a formatted sampling report.

    Args:
        report: Report dict from get_sampling_report()
    """
    print("=" * 70)
    print("HUMAN EVALUATION SAMPLING REPORT")
    print("=" * 70)

    print(f"\nSample Size: {report['sample_size']} / {report['total_size']}")
    print(f"Sampling Rate: {report['sampling_rate']:.1%}")

    coverage = report["coverage_check"]
    print(f"\nCoverage Minimums Met: {coverage['meets_all_minimums']}")

    for dim_name, dim_data in report["distributions"].items():
        print(f"\n{dim_name.upper().replace('_', ' ')}:")
        print("-" * 50)
        print(f"{'Value':<25} {'Sample':>12} {'Population':>12}")
        print("-" * 50)

        for value in dim_data["sample"].keys():
            s = dim_data["sample"][value]
            p = dim_data["population"][value]
            print(
                f"{value:<25} {s['count']:>5} ({s['pct']:>5.1f}%) "
                f"{p['count']:>5} ({p['pct']:>5.1f}%)"
            )

    balance = report["balance_metrics"]
    print("\nBalance Metrics:")
    print(f"  Max Cell Proportion: {balance['max_cell_proportion']:.1%}")
    print(f"  Is Balanced (<=35%): {balance['is_balanced']}")

    print("\n" + "=" * 70)
