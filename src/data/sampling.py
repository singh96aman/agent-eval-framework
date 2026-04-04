"""
Stratified sampling and assignment logic for dataset diversity.

This module implements the sampling strategy from DatasetDiversity.MD:
- 600 total trajectories (400 ToolBench + 100 GAIA + 100 SWE-bench)
- Stratified by domain, complexity, and perturbation condition
- One perturbation per trajectory (main sample)
- Control group with all perturbations (100 trajectories)
"""

import random
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

from src.data.schema import Trajectory, PerturbedTrajectory
from src.data.loaders import (
    classify_trajectory_domain,
    classify_trajectory_complexity,
)


@dataclass
class SamplingConfig:
    """Configuration for stratified sampling."""

    # Target counts by benchmark
    toolbench_count: int = 400
    gaia_count: int = 100
    swebench_count: int = 100

    # Control group size (receives all perturbations)
    control_group_size: int = 100

    # Perturbation types and positions
    perturbation_types: List[str] = field(default_factory=lambda: [
        "planning", "tool_selection", "parameter", "data_reference"
    ])
    positions: List[str] = field(default_factory=lambda: [
        "early", "middle", "late"
    ])

    # Random seed for reproducibility
    random_seed: int = 42

    @property
    def total_trajectories(self) -> int:
        return self.toolbench_count + self.gaia_count + self.swebench_count

    @property
    def main_sample_size(self) -> int:
        return self.total_trajectories - self.control_group_size


@dataclass
class StratifiedSample:
    """Result of stratified sampling."""

    # Main sample: one perturbation per trajectory
    main_sample: List[Tuple[Trajectory, str, str]]  # (traj, ptype, position)

    # Control group: all perturbations per trajectory
    control_group: List[Trajectory]

    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)


def stratified_sample_trajectories(
    toolbench_trajectories: List[Trajectory],
    gaia_trajectories: List[Trajectory],
    swebench_trajectories: List[Trajectory],
    config: Optional[SamplingConfig] = None,
) -> StratifiedSample:
    """
    Create stratified sample from multiple benchmarks.

    Strategy:
    1. Classify trajectories by domain and complexity
    2. Sample proportionally from each stratum
    3. Assign perturbation conditions via round-robin within strata
    4. Reserve control group for within-trajectory replication

    Args:
        toolbench_trajectories: ToolBench trajectories
        gaia_trajectories: GAIA trajectories
        swebench_trajectories: SWE-bench trajectories
        config: Sampling configuration

    Returns:
        StratifiedSample with main sample and control group
    """
    if config is None:
        config = SamplingConfig()

    random.seed(config.random_seed)

    # Combine all trajectories with benchmark labels
    all_trajectories = []
    for traj in toolbench_trajectories:
        all_trajectories.append(("toolbench", traj))
    for traj in gaia_trajectories:
        all_trajectories.append(("gaia", traj))
    for traj in swebench_trajectories:
        all_trajectories.append(("swebench", traj))

    # Shuffle
    random.shuffle(all_trajectories)

    # Classify each trajectory
    classified = []
    for benchmark, traj in all_trajectories:
        domain = classify_trajectory_domain(traj)
        complexity = classify_trajectory_complexity(traj)
        classified.append({
            "benchmark": benchmark,
            "trajectory": traj,
            "domain": domain,
            "complexity": complexity,
        })

    # Sample from each benchmark proportionally
    toolbench_sample = _sample_from_benchmark(
        [c for c in classified if c["benchmark"] == "toolbench"],
        config.toolbench_count
    )
    gaia_sample = _sample_from_benchmark(
        [c for c in classified if c["benchmark"] == "gaia"],
        config.gaia_count
    )
    swebench_sample = _sample_from_benchmark(
        [c for c in classified if c["benchmark"] == "swebench"],
        config.swebench_count
    )

    # Combine samples
    all_samples = toolbench_sample + gaia_sample + swebench_sample

    # Split into control group and main sample
    random.shuffle(all_samples)
    control_group = [s["trajectory"] for s in all_samples[:config.control_group_size]]
    main_pool = all_samples[config.control_group_size:]

    # Assign perturbation conditions to main sample
    main_sample = _assign_perturbation_conditions(
        main_pool, config.perturbation_types, config.positions
    )

    # Calculate statistics
    stats = _calculate_sampling_stats(main_sample, control_group, config)

    return StratifiedSample(
        main_sample=main_sample,
        control_group=control_group,
        stats=stats,
    )


def _sample_from_benchmark(
    classified_trajectories: List[Dict],
    target_count: int,
) -> List[Dict]:
    """
    Sample trajectories from a benchmark with complexity stratification.

    Target distribution: 20% simple, 50% medium, 30% complex
    """
    if not classified_trajectories:
        return []

    # Group by complexity
    by_complexity = defaultdict(list)
    for c in classified_trajectories:
        by_complexity[c["complexity"]].append(c)

    # Target counts
    simple_target = int(target_count * 0.2)
    medium_target = int(target_count * 0.5)
    complex_target = target_count - simple_target - medium_target

    # Sample from each complexity level
    result = []
    result.extend(_sample_up_to(by_complexity["simple"], simple_target))
    result.extend(_sample_up_to(by_complexity["medium"], medium_target))
    result.extend(_sample_up_to(by_complexity["complex"], complex_target))

    # If we haven't reached target, fill from any available
    remaining = target_count - len(result)
    if remaining > 0:
        available = [c for c in classified_trajectories if c not in result]
        result.extend(_sample_up_to(available, remaining))

    return result


def _sample_up_to(items: List, n: int) -> List:
    """Sample up to n items from list."""
    if len(items) <= n:
        return items[:]
    return random.sample(items, n)


def _assign_perturbation_conditions(
    trajectories: List[Dict],
    perturbation_types: List[str],
    positions: List[str],
) -> List[Tuple[Trajectory, str, str]]:
    """
    Assign perturbation conditions using stratified round-robin.

    Each trajectory receives exactly ONE perturbation.
    Conditions are assigned to ensure each (type, position) combination
    is roughly equally represented within each stratum.
    """
    # Build condition list (excluding data_reference × early)
    conditions = []
    for ptype in perturbation_types:
        for pos in positions:
            # Data reference not applicable to early position
            if ptype == "data_reference" and pos == "early":
                continue
            conditions.append((ptype, pos))

    # Group trajectories by (benchmark, domain) stratum
    strata = defaultdict(list)
    for item in trajectories:
        stratum_key = (item["benchmark"], item["domain"])
        strata[stratum_key].append(item["trajectory"])

    # Assign conditions via round-robin within each stratum
    result = []
    condition_idx = 0

    for stratum_key, traj_list in strata.items():
        for traj in traj_list:
            ptype, pos = conditions[condition_idx % len(conditions)]
            result.append((traj, ptype, pos))
            condition_idx += 1

    return result


def _calculate_sampling_stats(
    main_sample: List[Tuple[Trajectory, str, str]],
    control_group: List[Trajectory],
    config: SamplingConfig,
) -> Dict[str, Any]:
    """Calculate statistics for the sample."""
    stats = {
        "total_trajectories": len(main_sample) + len(control_group),
        "main_sample_size": len(main_sample),
        "control_group_size": len(control_group),
    }

    # Count by benchmark
    benchmark_counts = defaultdict(int)
    for traj, _, _ in main_sample:
        benchmark_counts[traj.benchmark] += 1
    for traj in control_group:
        benchmark_counts[traj.benchmark] += 1
    stats["by_benchmark"] = dict(benchmark_counts)

    # Count by perturbation condition
    condition_counts = defaultdict(int)
    for _, ptype, pos in main_sample:
        condition_counts[f"{ptype}_{pos}"] += 1
    stats["by_condition"] = dict(condition_counts)

    # Count by domain
    domain_counts = defaultdict(int)
    for traj, _, _ in main_sample:
        domain = classify_trajectory_domain(traj)
        domain_counts[domain] += 1
    stats["by_domain"] = dict(domain_counts)

    # Count by complexity
    complexity_counts = defaultdict(int)
    for traj, _, _ in main_sample:
        complexity = classify_trajectory_complexity(traj)
        complexity_counts[complexity] += 1
    stats["by_complexity"] = dict(complexity_counts)

    return stats


def generate_sample_report(sample: StratifiedSample) -> str:
    """Generate human-readable report of sampling results."""
    lines = [
        "=" * 60,
        "STRATIFIED SAMPLE REPORT",
        "=" * 60,
        "",
        f"Total trajectories: {sample.stats['total_trajectories']}",
        f"Main sample: {sample.stats['main_sample_size']}",
        f"Control group: {sample.stats['control_group_size']}",
        "",
        "BY BENCHMARK:",
    ]

    for bench, count in sorted(sample.stats.get("by_benchmark", {}).items()):
        lines.append(f"  {bench}: {count}")

    lines.extend(["", "BY PERTURBATION CONDITION:"])
    for cond, count in sorted(sample.stats.get("by_condition", {}).items()):
        lines.append(f"  {cond}: {count}")

    lines.extend(["", "BY DOMAIN:"])
    for domain, count in sorted(sample.stats.get("by_domain", {}).items()):
        lines.append(f"  {domain}: {count}")

    lines.extend(["", "BY COMPLEXITY:"])
    for comp, count in sorted(sample.stats.get("by_complexity", {}).items()):
        lines.append(f"  {comp}: {count}")

    lines.append("=" * 60)

    return "\n".join(lines)


def validate_sample_coverage(
    sample: StratifiedSample,
    min_per_condition: int = 30,
) -> Tuple[bool, List[str]]:
    """
    Validate that sample has adequate coverage.

    Args:
        sample: Stratified sample to validate
        min_per_condition: Minimum samples required per condition

    Returns:
        (is_valid, list of issues)
    """
    issues = []

    # Check condition coverage
    for cond, count in sample.stats.get("by_condition", {}).items():
        if count < min_per_condition:
            issues.append(
                f"Condition '{cond}' has only {count} samples "
                f"(minimum: {min_per_condition})"
            )

    # Check benchmark coverage
    if sample.stats.get("by_benchmark", {}).get("toolbench", 0) < 300:
        issues.append("ToolBench underrepresented (need >= 300)")
    if sample.stats.get("by_benchmark", {}).get("gaia", 0) < 50:
        issues.append("GAIA underrepresented (need >= 50)")
    if sample.stats.get("by_benchmark", {}).get("swebench", 0) < 50:
        issues.append("SWE-bench underrepresented (need >= 50)")

    # Check complexity balance
    complexity = sample.stats.get("by_complexity", {})
    total = sum(complexity.values()) or 1
    simple_pct = complexity.get("simple", 0) / total
    medium_pct = complexity.get("medium", 0) / total
    complex_pct = complexity.get("complex", 0) / total

    if simple_pct < 0.1:
        issues.append(f"Simple complexity underrepresented ({simple_pct:.1%})")
    if medium_pct < 0.3:
        issues.append(f"Medium complexity underrepresented ({medium_pct:.1%})")
    if complex_pct < 0.1:
        issues.append(f"Complex complexity underrepresented ({complex_pct:.1%})")

    return len(issues) == 0, issues
