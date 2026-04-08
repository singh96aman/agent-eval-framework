#!/usr/bin/env python3
"""
Dump typed trajectory samples for an experiment to JSON.

Usage:
    python ops/dump_experiment_typed_samples.py exp_trajectory_typing_v1
    python ops/dump_experiment_typed_samples.py exp_trajectory_typing_v1 --limit 10
    python ops/dump_experiment_typed_samples.py exp_trajectory_typing_v1 --benchmark swebench
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.storage.mongodb import MongoDBStorage


def dump_typed_samples(experiment_id: str, limit: int = None, benchmark: str = None):
    """Dump typed trajectories for an experiment to JSON."""

    print(f"Connecting to MongoDB...")
    storage = MongoDBStorage()

    # Query from typed_trajectories collection
    typed_col = storage.db["typed_trajectories"]

    query = {"experiment_id": experiment_id}
    if benchmark:
        query["benchmark"] = benchmark

    cursor = typed_col.find(query)
    if limit:
        cursor = cursor.limit(limit)

    trajectories = list(cursor)
    print(f"Found {len(trajectories)} typed trajectories")

    if not trajectories:
        print(f"No typed trajectories found for experiment '{experiment_id}'")
        storage.close()
        return None

    # Clean MongoDB ObjectId for JSON serialization
    for t in trajectories:
        t.pop('_id', None)
        if 'stored_at' in t:
            t['stored_at'] = str(t['stored_at'])

    # Compute summary statistics
    total_steps = sum(len(t.get('steps', [])) for t in trajectories)
    roles = {}
    total_slots = 0
    for t in trajectories:
        for step in t.get('steps', []):
            role = step.get('step_role', 'unknown')
            roles[role] = roles.get(role, 0) + 1
            total_slots += len(step.get('perturbable_slots', []))

    summary = {
        "total_trajectories": len(trajectories),
        "total_steps": total_steps,
        "total_perturbable_slots": total_slots,
        "step_role_distribution": roles,
        "benchmarks": list(set(t.get('benchmark') for t in trajectories)),
    }

    # Output path with timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "data" / "investigation"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{benchmark}" if benchmark else ""
    output_path = output_dir / f"{experiment_id}_typed{suffix}_{timestamp}.json"

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump({
            "experiment_id": experiment_id,
            "exported_at": datetime.now().isoformat(),
            "summary": summary,
            "trajectories": trajectories
        }, f, indent=2)

    print(f"\nWrote {len(trajectories)} typed trajectories to {output_path}")
    print(f"\nSummary:")
    print(f"  Total steps: {total_steps}")
    print(f"  Total perturbable slots: {total_slots}")
    print(f"  Step roles: {roles}")

    storage.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Dump typed experiment samples to JSON")
    parser.add_argument("experiment_id", help="Experiment ID to dump")
    parser.add_argument("--limit", type=int, help="Max trajectories to dump")
    parser.add_argument("--benchmark", help="Filter by benchmark (toolbench, swebench)")

    args = parser.parse_args()
    dump_typed_samples(args.experiment_id, args.limit, args.benchmark)


if __name__ == "__main__":
    main()
