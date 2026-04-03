#!/usr/bin/env python3
"""
Run perturbations on sampled trajectories.

Generates perturbed trajectories for the study.
Uses stratified assignment (one perturbation per trajectory).
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

INPUT_FILE = project_root / "data" / "sampled" / "toolbench_400.json"
OUTPUT_DIR = project_root / "data" / "perturbed"


def run_perturbations():
    """Generate perturbations for sampled trajectories."""
    print("\n" + "=" * 60)
    print("RUNNING PERTURBATIONS")
    print("=" * 60)

    from src.data.loaders import load_trajectories_from_json
    from src.perturbations.generator import PerturbationGenerator
    from src.data.sampling import (
        SamplingConfig,
        stratified_sample_trajectories,
    )

    # Load sampled trajectories
    print(f"\n1. Loading from {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print("   ERROR: Sampled trajectories not found. Run sample_trajectories.py first.")
        return

    trajectories = load_trajectories_from_json(str(INPUT_FILE))
    print(f"   Loaded {len(trajectories)} trajectories")

    # Initialize generator
    generator = PerturbationGenerator(random_seed=42)

    # Define perturbation conditions (excluding data_ref × early)
    conditions = []
    for ptype in ["planning", "tool_selection", "parameter", "data_reference"]:
        for pos in ["early", "middle", "late"]:
            if ptype == "data_reference" and pos == "early":
                continue
            conditions.append((ptype, pos))

    print(f"   Perturbation conditions: {len(conditions)}")

    # Assign conditions using round-robin
    print("\n2. Generating perturbations...")
    perturbations = []
    stats = defaultdict(lambda: {"success": 0, "fail": 0})

    for i, traj in enumerate(trajectories):
        # Assign condition via round-robin
        ptype, pos = conditions[i % len(conditions)]

        # Get system prompt
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
    print("\n3. Perturbation statistics:")
    for key in sorted(stats.keys()):
        s = stats[key]
        total = s["success"] + s["fail"]
        rate = s["success"] / total * 100 if total > 0 else 0
        print(f"   {key}: {s['success']}/{total} ({rate:.1f}%)")

    # Save perturbations
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "toolbench_perturbations.json"

    print(f"\n4. Saving to {output_file}...")
    data = [p.to_dict() for p in perturbations]
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"   Saved {len(perturbations)} perturbations")

    # Summary
    print("\n" + "=" * 60)
    print("PERTURBATION SUMMARY")
    print("=" * 60)
    print(f"Input trajectories: {len(trajectories)}")
    print(f"Successful perturbations: {len(perturbations)}")
    print(f"Success rate: {len(perturbations)/len(trajectories)*100:.1f}%")

    # Count by type
    type_counts = defaultdict(int)
    pos_counts = defaultdict(int)
    for p in perturbations:
        type_counts[p.perturbation_type] += 1
        pos_counts[p.perturbation_position] += 1

    print("\nBy type:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    print("\nBy position:")
    for p, c in sorted(pos_counts.items()):
        print(f"  {p}: {c}")

    return perturbations


if __name__ == "__main__":
    run_perturbations()
