#!/usr/bin/env python3
"""
Sample trajectories for the full study (600 total).

Target: 400 ToolBench + 100 GAIA + 100 SWE-bench

This script:
1. Samples from local ToolBench data (safe, no network)
2. Reports on GAIA/SWE-bench requirements
3. Saves sampled trajectories to data/sampled/
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

OUTPUT_DIR = project_root / "data" / "sampled"


def sample_toolbench(target_count=400):
    """Sample ToolBench trajectories from local data."""
    print("\n" + "=" * 60)
    print(f"SAMPLING TOOLBENCH ({target_count} trajectories)")
    print("=" * 60)

    from src.data.loaders import (
        load_toolbench_trajectories,
        classify_trajectory_domain,
        classify_trajectory_complexity,
        save_trajectories,
    )

    # Load all available trajectories with quality filters
    # Using training set for larger pool and more complex trajectories
    print("\n1. Loading with quality filters (training set)...")
    trajectories = load_toolbench_trajectories(
        max_trajectories=2000,  # Sample from larger pool
        min_steps=4,
        max_steps=15,  # Allow longer trajectories for complex category
        filter_successful=True,
        require_parameters_all_positions=False,  # Relaxed for coverage
        require_tool_diversity=False,  # Relaxed for coverage
        random_seed=42,
        split="train"  # Use training set for more trajectories
    )

    print(f"   Available: {len(trajectories)} trajectories")

    if len(trajectories) < target_count:
        print(f"   WARNING: Only {len(trajectories)} available, target is {target_count}")
        target_count = len(trajectories)

    # Sample with stratification by domain
    print("\n2. Stratified sampling by domain...")
    domain_groups = defaultdict(list)
    for traj in trajectories:
        domain = classify_trajectory_domain(traj)
        domain_groups[domain].append(traj)

    # Calculate samples per domain (proportional)
    sampled = []
    remaining = target_count

    for domain in sorted(domain_groups.keys()):
        group = domain_groups[domain]
        # Proportional allocation
        allocation = int(len(group) / len(trajectories) * target_count)
        allocation = min(allocation, len(group), remaining)
        sampled.extend(group[:allocation])
        remaining -= allocation
        print(f"   {domain}: {allocation} sampled (of {len(group)} available)")

    # Fill remaining from largest group
    if remaining > 0:
        largest_group = max(domain_groups.values(), key=len)
        already_sampled = set(t.trajectory_id for t in sampled)
        extras = [t for t in largest_group if t.trajectory_id not in already_sampled]
        sampled.extend(extras[:remaining])
        print(f"   Added {min(remaining, len(extras))} extras")

    print(f"\n   Total sampled: {len(sampled)}")

    # Populate domain and complexity fields on each trajectory
    print("\n3. Populating domain/complexity metadata...")
    for traj in sampled:
        traj.domain = classify_trajectory_domain(traj)
        traj.complexity = classify_trajectory_complexity(traj)

    # Analyze final sample
    print("\n4. Final sample analysis:")
    domain_counts = defaultdict(int)
    complexity_counts = defaultdict(int)

    for traj in sampled:
        domain_counts[traj.domain] += 1
        complexity_counts[traj.complexity] += 1

    print("   Domains:")
    for domain, count in sorted(domain_counts.items()):
        print(f"     {domain}: {count}")

    print("   Complexity:")
    for comp, count in sorted(complexity_counts.items()):
        print(f"     {comp}: {count}")

    # Save to file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "toolbench_400.json"

    print(f"\n5. Saving to {output_path}...")
    save_trajectories(sampled, str(output_path))
    print(f"   Saved {len(sampled)} trajectories")

    return sampled


def report_gaia_requirements():
    """Report on GAIA trajectory generation requirements."""
    print("\n" + "=" * 60)
    print("GAIA TRAJECTORY GENERATION (100 trajectories)")
    print("=" * 60)

    print("""
GAIA Requirements:
- GAIA contains questions but NOT agent trajectories
- We need to GENERATE trajectories by running an agent
- Design: 50 Claude-generated + 50 GPT-4o-generated

This requires:
1. Access to GAIA dataset (gaia-benchmark/GAIA on HuggingFace)
2. API access to Claude and GPT-4o
3. Agent execution infrastructure

Status: DEFERRED - requires API setup and cost approval

Workaround options:
- Use synthetic trajectories from ToolBench
- Generate minimal GAIA sample (10-20) for validation
- Skip GAIA and increase ToolBench/SWE-bench allocation
""")


def report_swebench_requirements():
    """Report on SWE-bench trajectory requirements."""
    print("\n" + "=" * 60)
    print("SWE-BENCH TRAJECTORIES (100 trajectories)")
    print("=" * 60)

    print("""
SWE-bench Requirements:
- Source: SWE-bench/SWE-smith-trajectories on HuggingFace
- Contains 76k trajectories (LARGE dataset)
- Need to download carefully to avoid memory issues

Download strategy:
1. Use streaming mode to avoid loading entire dataset
2. Sample 100 trajectories with quality filters
3. Save locally for reuse

Status: READY - can download with careful memory management
""")

    # Try a minimal test
    print("\nTesting SWE-bench access (will download ~10 items)...")
    try:
        from datasets import load_dataset

        # Use streaming to avoid memory issues
        dataset = load_dataset(
            "SWE-bench/SWE-smith-trajectories",
            split="train",
            streaming=True
        )

        # Take only 5 items to test
        count = 0
        for item in dataset:
            count += 1
            if count >= 5:
                break

        print(f"   SUCCESS: Accessed {count} items via streaming")
        print("   Full download can proceed with streaming mode")

    except Exception as e:
        print(f"   ERROR: {e}")
        print("   SWE-bench may require authentication or is unavailable")


def main():
    """Run trajectory sampling."""
    print("\n" + "=" * 60)
    print("TRAJECTORY SAMPLING FOR FULL STUDY")
    print("Target: 400 ToolBench + 100 GAIA + 100 SWE-bench = 600")
    print("=" * 60)

    # Sample ToolBench (safe, local data)
    toolbench_sample = sample_toolbench(target_count=400)

    # Report on GAIA (requires API)
    report_gaia_requirements()

    # Report on SWE-bench (requires HuggingFace download)
    report_swebench_requirements()

    # Summary
    print("\n" + "=" * 60)
    print("SAMPLING SUMMARY")
    print("=" * 60)
    print(f"ToolBench: {len(toolbench_sample)} sampled and saved")
    print("GAIA: DEFERRED (requires API setup)")
    print("SWE-bench: READY for streaming download")
    print("\nNext steps:")
    print("1. Review ToolBench sample")
    print("2. Decide on GAIA approach (generate vs skip)")
    print("3. Download SWE-bench sample with streaming")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
