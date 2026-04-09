"""
Index generation for evaluation units.

Generates summary indexes and statistics for evaluation units to support
filtering, analysis, and experiment tracking.
"""

import json
from collections import defaultdict
from typing import Any, Dict, List


def build_evaluation_unit_index(units: List[Dict]) -> Dict:
    """
    Build a summary index for evaluation units.

    Aggregates evaluation units by various dimensions for efficient
    filtering and analysis. Also includes the full list of unit summaries.

    Args:
        units: List of evaluation unit dicts (from EvaluationUnit.to_dict())

    Returns:
        Index dict with structure:
        {
            "total_units": int,
            "by_benchmark": {"toolbench": N, "swebench": N, ...},
            "by_perturbation_class": {"placebo": N, "fine_grained": N, "coarse_grained": N},
            "by_perturbation_family": {"data_reference": N, ...},
            "by_replay_tier": {"1": N, "2": N, "3": N},
            "by_expected_impact": {"0": N, "1": N, "2": N, "3": N},
            "by_expected_detectability": {"0": N, "1": N, "2": N},
            "units": [
                {
                    "evaluation_unit_id": str,
                    "source_trajectory_id": str,
                    "baseline_variant_id": str,
                    "perturbed_variant_id": str,
                    "benchmark": str,
                    "perturbation_class": str,
                    "perturbation_family": str,
                    "expected_impact": int,
                    "expected_detectability": int,
                    "replay_tier": int or None
                },
                ...
            ]
        }
    """
    # Initialize counters
    by_benchmark: Dict[str, int] = defaultdict(int)
    by_perturbation_class: Dict[str, int] = defaultdict(int)
    by_perturbation_family: Dict[str, int] = defaultdict(int)
    by_replay_tier: Dict[str, int] = defaultdict(int)
    by_expected_impact: Dict[str, int] = defaultdict(int)
    by_expected_detectability: Dict[str, int] = defaultdict(int)

    # Build unit summaries
    unit_summaries: List[Dict[str, Any]] = []

    for unit in units:
        # Extract fields from unit dict
        evaluation_unit_id = unit.get("evaluation_unit_id", "")
        source_trajectory_id = unit.get("source_trajectory_id", "")
        benchmark = unit.get("benchmark", "unknown")
        replay_tier = unit.get("replay_tier")

        # Extract baseline/perturbed variant IDs
        baseline = unit.get("baseline", {})
        perturbed = unit.get("perturbed", {})
        baseline_variant_id = baseline.get("trajectory_variant_id", "")
        perturbed_variant_id = perturbed.get("trajectory_variant_id", "")

        # Extract derived cache fields
        derived_cache = unit.get("derived_cache", {})
        perturbation_class = derived_cache.get("perturbation_class", "unknown")
        perturbation_family = derived_cache.get("perturbation_family", "unknown")
        expected_impact = derived_cache.get("expected_impact", 0)
        expected_detectability = derived_cache.get("expected_detectability", 0)

        # Update counters
        by_benchmark[benchmark] += 1
        by_perturbation_class[perturbation_class] += 1
        by_perturbation_family[perturbation_family] += 1
        by_expected_impact[str(expected_impact)] += 1
        by_expected_detectability[str(expected_detectability)] += 1

        # Handle replay_tier (can be None)
        tier_key = str(replay_tier) if replay_tier is not None else "none"
        by_replay_tier[tier_key] += 1

        # Build unit summary
        unit_summary = {
            "evaluation_unit_id": evaluation_unit_id,
            "source_trajectory_id": source_trajectory_id,
            "baseline_variant_id": baseline_variant_id,
            "perturbed_variant_id": perturbed_variant_id,
            "benchmark": benchmark,
            "perturbation_class": perturbation_class,
            "perturbation_family": perturbation_family,
            "expected_impact": expected_impact,
            "expected_detectability": expected_detectability,
            "replay_tier": replay_tier,
        }
        unit_summaries.append(unit_summary)

    # Build index
    index = {
        "total_units": len(units),
        "by_benchmark": dict(by_benchmark),
        "by_perturbation_class": dict(by_perturbation_class),
        "by_perturbation_family": dict(by_perturbation_family),
        "by_replay_tier": dict(by_replay_tier),
        "by_expected_impact": dict(by_expected_impact),
        "by_expected_detectability": dict(by_expected_detectability),
        "units": unit_summaries,
    }

    return index


def save_index(index: Dict, output_path: str) -> None:
    """
    Save index to JSON file with pretty formatting.

    Args:
        index: Index dict from build_evaluation_unit_index()
        output_path: Path to output JSON file
    """
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)
