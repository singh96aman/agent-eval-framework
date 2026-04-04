#!/usr/bin/env python3
"""
Regenerate perturbations for all benchmarks with bug fixes.

This script regenerates perturbations using the fixed strategies:
- BUG1 fix: GAIA/SWE-bench use native strategies (no template fallback)
- BUG2 fix: Planning perturbations are semantic (no text corruption)
- BUG4 fix: All benchmarks have all 4 perturbation types

Usage: python scripts/regenerate_perturbations.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Experiment ID for tracking
EXPERIMENT_ID = f"exp_{datetime.now().strftime('%Y%m%d')}_v2"

OUTPUT_DIR = project_root / "data" / "perturbed"
SAMPLED_DIR = project_root / "data" / "sampled"


def load_trajectories(benchmark: str):
    """Load sampled trajectories for a benchmark."""
    from src.data.loaders import load_trajectories_from_json

    path = SAMPLED_DIR / f"{benchmark}_{'400' if benchmark == 'toolbench' else '100'}.json"
    if not path.exists():
        print(f"   WARNING: {path} not found")
        return []

    trajectories = load_trajectories_from_json(str(path))
    print(f"   Loaded {len(trajectories)} {benchmark} trajectories")
    return trajectories


def generate_perturbations_for_benchmark(
    trajectories, benchmark: str, generator
):
    """Generate perturbations for a single benchmark."""
    print(f"\n{'=' * 60}")
    print(f"GENERATING {benchmark.upper()} PERTURBATIONS")
    print(f"{'=' * 60}")

    # Define perturbation conditions (all 4 types x 3 positions)
    # data_reference skips early position (by design)
    conditions = []
    for ptype in ["planning", "tool_selection", "parameter", "data_reference"]:
        for pos in ["early", "middle", "late"]:
            if ptype == "data_reference" and pos == "early":
                continue
            conditions.append((ptype, pos))

    print(f"   Perturbation conditions: {len(conditions)}")

    perturbations = []
    stats = defaultdict(lambda: {"success": 0, "fail": 0})

    for i, traj in enumerate(trajectories):
        # Assign condition via round-robin
        ptype, pos = conditions[i % len(conditions)]

        # Get system prompt (only relevant for ToolBench)
        system_prompt = traj.metadata.get("system_prompt", "")

        try:
            perturbed = generator.generate_perturbation(
                trajectory=traj,
                perturbation_type=ptype,
                position=pos,
                system_prompt=system_prompt
            )

            if perturbed:
                perturbations.append(perturbed)
                stats[f"{ptype}_{pos}"]["success"] += 1
            else:
                stats[f"{ptype}_{pos}"]["fail"] += 1

        except Exception as e:
            stats[f"{ptype}_{pos}"]["fail"] += 1
            if i < 5:
                print(f"   Error on {traj.trajectory_id}: {e}")

        # Progress
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(trajectories)}...")

    print(f"\n   Generated {len(perturbations)} perturbations")

    # Show stats
    print("\n   Statistics:")
    for key in sorted(stats.keys()):
        s = stats[key]
        total = s["success"] + s["fail"]
        rate = s["success"] / total * 100 if total > 0 else 0
        print(f"     {key}: {s['success']}/{total} ({rate:.1f}%)")

    return perturbations


def save_perturbations(perturbations, benchmark: str):
    """Save perturbations to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{benchmark}_perturbations.json"

    print(f"\n   Saving to {output_file}...")
    data = [p.to_dict() for p in perturbations]
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"   Saved {len(perturbations)} perturbations")
    return output_file


def main():
    print("=" * 60)
    print(f"PERTURBATION REGENERATION - {EXPERIMENT_ID}")
    print("=" * 60)
    print("\nThis script regenerates perturbations with bug fixes:")
    print("- BUG1: GAIA/SWE-bench use native strategies")
    print("- BUG2: Planning perturbations are semantic")
    print("- BUG4: All benchmarks have all perturbation types")

    from src.perturbations.generator import PerturbationGenerator

    # Initialize generator
    generator = PerturbationGenerator(random_seed=42)

    all_perturbations = {}
    benchmarks = ["toolbench", "gaia", "swebench"]

    for benchmark in benchmarks:
        trajectories = load_trajectories(benchmark)
        if not trajectories:
            print(f"   Skipping {benchmark} (no trajectories)")
            continue

        perturbations = generate_perturbations_for_benchmark(
            trajectories, benchmark, generator
        )

        if perturbations:
            save_perturbations(perturbations, benchmark)
            all_perturbations[benchmark] = perturbations

    # Final summary
    print("\n" + "=" * 60)
    print("REGENERATION SUMMARY")
    print("=" * 60)

    total = 0
    for benchmark in benchmarks:
        count = len(all_perturbations.get(benchmark, []))
        print(f"{benchmark}: {count} perturbations")
        total += count

    print(f"\nTotal: {total} perturbations")
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print("\nNext step: Run scripts/validate_perturbations.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
