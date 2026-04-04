#!/usr/bin/env python3
"""
Smoke test for dataset loaders - loads small samples safely.

This script tests that the loaders work without loading entire datasets.
Uses streaming and strict limits to avoid memory issues.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set memory limit warning
MAX_TRAJECTORIES = 10  # Very small for smoke test


def test_toolbench_loader():
    """Test ToolBench loader with small sample."""
    print("\n" + "=" * 60)
    print("Testing ToolBench Loader")
    print("=" * 60)

    try:
        from src.data.loaders import load_toolbench_trajectories

        # Load very small sample
        trajectories = load_toolbench_trajectories(
            max_trajectories=MAX_TRAJECTORIES,
            min_steps=4,
            max_steps=10,
            random_seed=42,
            split="eval"
        )

        if not trajectories:
            print("   WARNING: No trajectories loaded (data may not be available)")
            return []

        print(f"   Loaded {len(trajectories)} trajectories")

        # Show sample info
        for i, traj in enumerate(trajectories[:3]):
            print(f"\n   Sample {i+1}:")
            print(f"     ID: {traj.trajectory_id}")
            print(f"     Steps: {len(traj.steps)}")
            print(f"     Task: {traj.ground_truth.task_description[:80]}...")

        return trajectories

    except Exception as e:
        print(f"   ERROR: {e}")
        return []


def test_perturbation_generation(trajectories):
    """Test perturbation generation on sample trajectories."""
    print("\n" + "=" * 60)
    print("Testing Perturbation Generation")
    print("=" * 60)

    if not trajectories:
        print("   SKIP: No trajectories to test")
        return

    try:
        from src.perturbations.generator import PerturbationGenerator

        generator = PerturbationGenerator(random_seed=42)

        # Test on first trajectory
        traj = trajectories[0]
        system_prompt = traj.metadata.get("system_prompt", "")

        print(f"\n   Testing on: {traj.trajectory_id}")

        # Test each perturbation type
        perturbation_types = ["planning", "tool_selection", "parameter", "data_reference"]
        positions = ["early", "middle", "late"]

        success_count = 0
        total_count = 0

        for ptype in perturbation_types:
            for pos in positions:
                # Skip data_reference for early
                if ptype == "data_reference" and pos == "early":
                    continue

                total_count += 1
                try:
                    perturbed = generator.generate_perturbation(
                        trajectory=traj,
                        perturbation_type=ptype,
                        position=pos,
                        system_prompt=system_prompt
                    )
                    if perturbed:
                        success_count += 1
                        print(f"     {ptype}/{pos}: OK")
                    else:
                        print(f"     {ptype}/{pos}: SKIP (not applicable)")
                except Exception as e:
                    print(f"     {ptype}/{pos}: ERROR - {e}")

        print(f"\n   Results: {success_count}/{total_count} perturbations generated")

    except Exception as e:
        print(f"   ERROR: {e}")


def test_domain_classification(trajectories):
    """Test domain and complexity classification."""
    print("\n" + "=" * 60)
    print("Testing Domain/Complexity Classification")
    print("=" * 60)

    if not trajectories:
        print("   SKIP: No trajectories to test")
        return

    try:
        from src.data.loaders import (
            classify_trajectory_domain,
            classify_trajectory_complexity
        )

        domain_counts = {}
        complexity_counts = {}

        for traj in trajectories:
            domain = classify_trajectory_domain(traj)
            complexity = classify_trajectory_complexity(traj)

            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        print("\n   Domain distribution:")
        for domain, count in sorted(domain_counts.items()):
            print(f"     {domain}: {count}")

        print("\n   Complexity distribution:")
        for comp, count in sorted(complexity_counts.items()):
            print(f"     {comp}: {count}")

    except Exception as e:
        print(f"   ERROR: {e}")


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("SMOKE TEST - Dataset Loaders & Perturbations")
    print(f"Max trajectories: {MAX_TRAJECTORIES}")
    print("=" * 60)

    # Test ToolBench
    trajectories = test_toolbench_loader()

    # Test perturbations
    test_perturbation_generation(trajectories)

    # Test classification
    test_domain_classification(trajectories)

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60 + "\n")

    return len(trajectories) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
