#!/usr/bin/env python3
"""
Delete typed trajectory samples for an experiment from MongoDB.

Usage:
    python ops/delete_experiment_typed_samples.py exp_trajectory_sampling_v7
    python ops/delete_experiment_typed_samples.py exp_trajectory_sampling_v7 --force
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.storage.mongodb import MongoDBStorage


def delete_typed_samples(experiment_id: str, force: bool = False):
    """Delete typed trajectories for an experiment from MongoDB."""

    print(f"Connecting to MongoDB...")
    storage = MongoDBStorage()

    # Query from typed_trajectories collection
    typed_col = storage.db["typed_trajectories"]

    # Count matching documents
    count = typed_col.count_documents({"experiment_id": experiment_id})

    if count == 0:
        print(f"No typed trajectories found for experiment '{experiment_id}'")
        storage.close()
        return

    print(f"\nFound {count} typed trajectories for experiment '{experiment_id}'")

    # Confirmation prompt
    if not force:
        response = input(f"\nAre you sure you want to delete typed_samples for experiment '{experiment_id}'? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("Aborted.")
            storage.close()
            return

    # Delete documents
    result = typed_col.delete_many({"experiment_id": experiment_id})
    print(f"\nDeleted {result.deleted_count} typed trajectories")

    storage.close()


def main():
    parser = argparse.ArgumentParser(description="Delete typed experiment samples from MongoDB")
    parser.add_argument("experiment_id", help="Experiment ID to delete")
    parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()
    delete_typed_samples(args.experiment_id, args.force)


if __name__ == "__main__":
    main()
