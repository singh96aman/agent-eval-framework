#!/usr/bin/env python3
"""
Dump all trajectory samples for an experiment to JSON.

Usage:
    python ops/dump_experiment_samples.py exp_trajectory_sampling_v1
    python ops/dump_experiment_samples.py exp_trajectory_sampling_v1 --limit 100
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


def dump_samples(experiment_id: str, limit: int = None):
    """Dump all trajectories for an experiment to JSON."""

    print(f"Connecting to MongoDB...")
    storage = MongoDBStorage()

    # Query directly from trajectories collection
    query = {"experiment_id": experiment_id}
    cursor = storage.trajectories.find(query)
    if limit:
        cursor = cursor.limit(limit)

    trajectories = list(cursor)
    print(f"Found {len(trajectories)} trajectories")

    if not trajectories:
        print(f"No trajectories found for experiment '{experiment_id}'")
        storage.close()
        return None

    # Clean MongoDB ObjectId for JSON serialization
    for t in trajectories:
        t.pop('_id', None)
        if 'stored_at' in t:
            t['stored_at'] = str(t['stored_at'])

    # Output path
    date_str = datetime.now().strftime("%Y_%m_%d")
    output_dir = Path(__file__).parent.parent / "data" / "investigation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_id}_{date_str}.json"

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump({
            "experiment_id": experiment_id,
            "exported_at": datetime.now().isoformat(),
            "count": len(trajectories),
            "trajectories": trajectories
        }, f, indent=2)

    print(f"Wrote {len(trajectories)} trajectories to {output_path}")
    storage.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Dump experiment samples to JSON")
    parser.add_argument("experiment_id", help="Experiment ID to dump")
    parser.add_argument("--limit", type=int, help="Max trajectories to dump")

    args = parser.parse_args()
    dump_samples(args.experiment_id, args.limit)


if __name__ == "__main__":
    main()
