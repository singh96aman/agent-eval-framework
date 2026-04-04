#!/usr/bin/env python3
"""
Validate trajectory coverage across diversity domains.

This script analyzes the available trajectories and checks coverage for:
- Domain diversity (8 categories for ToolBench)
- Complexity diversity (simple/medium/complex)
- Perturbation applicability (can each type be applied?)
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_toolbench_coverage(max_to_analyze=100):
    """Analyze ToolBench trajectory coverage."""
    print("\n" + "=" * 60)
    print("TOOLBENCH COVERAGE ANALYSIS")
    print(f"Analyzing up to {max_to_analyze} trajectories")
    print("=" * 60)

    from src.data.loaders import (
        load_toolbench_trajectories,
        classify_trajectory_domain,
        classify_trajectory_complexity,
        _has_parameters_in_all_positions,
    )
    from src.perturbations.generator import PerturbationGenerator

    # Load trajectories
    print("\n1. Loading trajectories...")
    trajectories = load_toolbench_trajectories(
        max_trajectories=max_to_analyze,
        min_steps=4,
        max_steps=10,
        filter_successful=True,
        random_seed=42,
        split="eval"
    )

    print(f"   Loaded: {len(trajectories)} trajectories")

    if not trajectories:
        print("   ERROR: No trajectories loaded")
        return

    # Analyze domain distribution
    print("\n2. Domain distribution:")
    domain_counts = defaultdict(list)
    for traj in trajectories:
        domain = classify_trajectory_domain(traj)
        domain_counts[domain].append(traj.trajectory_id)

    for domain in sorted(domain_counts.keys()):
        count = len(domain_counts[domain])
        pct = count / len(trajectories) * 100
        print(f"   {domain}: {count} ({pct:.1f}%)")

    # Analyze complexity distribution
    print("\n3. Complexity distribution:")
    complexity_counts = defaultdict(int)
    for traj in trajectories:
        complexity = classify_trajectory_complexity(traj)
        complexity_counts[complexity] += 1

    for comp in ["simple", "medium", "complex"]:
        count = complexity_counts.get(comp, 0)
        pct = count / len(trajectories) * 100
        print(f"   {comp}: {count} ({pct:.1f}%)")

    # Analyze perturbation applicability
    print("\n4. Perturbation applicability:")
    generator = PerturbationGenerator(random_seed=42)

    perturbation_success = defaultdict(lambda: {"success": 0, "fail": 0})

    # Test each perturbation on a sample
    sample_size = min(20, len(trajectories))
    print(f"   Testing on {sample_size} sample trajectories...")

    for traj in trajectories[:sample_size]:
        system_prompt = traj.metadata.get("system_prompt", "")

        for ptype in ["planning", "tool_selection", "parameter", "data_reference"]:
            for pos in ["early", "middle", "late"]:
                if ptype == "data_reference" and pos == "early":
                    continue

                key = f"{ptype}_{pos}"
                try:
                    perturbed = generator.generate_perturbation(
                        trajectory=traj,
                        perturbation_type=ptype,
                        position=pos,
                        system_prompt=system_prompt
                    )
                    if perturbed:
                        perturbation_success[key]["success"] += 1
                    else:
                        perturbation_success[key]["fail"] += 1
                except Exception:
                    perturbation_success[key]["fail"] += 1

    print("\n   Perturbation success rates:")
    for key in sorted(perturbation_success.keys()):
        stats = perturbation_success[key]
        total = stats["success"] + stats["fail"]
        rate = stats["success"] / total * 100 if total > 0 else 0
        print(f"   {key}: {stats['success']}/{total} ({rate:.1f}%)")

    # Check trajectories with parameters in all positions
    print("\n5. Parameter perturbation eligibility:")
    param_eligible = sum(1 for t in trajectories if _has_parameters_in_all_positions(t))
    print(f"   Trajectories with params in all positions: {param_eligible}/{len(trajectories)}")

    # Summary
    print("\n" + "=" * 60)
    print("COVERAGE SUMMARY")
    print("=" * 60)
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Domains covered: {len(domain_counts)}")
    print(f"Parameter-eligible: {param_eligible} ({param_eligible/len(trajectories)*100:.1f}%)")

    # Identify gaps
    print("\n   Potential gaps:")
    target_domains = ["data_information", "media_entertainment", "ecommerce_shopping",
                      "travel_logistics", "finance_business", "social_communication",
                      "utilities_tools", "sports_gaming"]
    for domain in target_domains:
        if domain not in domain_counts:
            print(f"   - Missing domain: {domain}")
        elif len(domain_counts[domain]) < 5:
            print(f"   - Low count for {domain}: {len(domain_counts[domain])}")

    return trajectories


def main():
    """Run coverage analysis."""
    print("\n" + "=" * 60)
    print("TRAJECTORY COVERAGE VALIDATION")
    print("=" * 60)

    trajectories = analyze_toolbench_coverage(max_to_analyze=100)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
