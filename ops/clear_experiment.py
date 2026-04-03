#!/usr/bin/env python3
"""
Clear all MongoDB collections for a specific experiment.

This script removes all data associated with a given experiment_id:
- Experiment metadata
- Perturbations
- Annotations
- Judge evaluations
- CCG metrics

NOTE: Trajectories are NOT deleted (they are pure cache, no experiment_id).

Usage:
    python ops/clear_experiment.py <experiment_id>
    python ops/clear_experiment.py <experiment_id> --dry-run
    python ops/clear_experiment.py --all  # Clear ALL experiments (dangerous!)

Examples:
    python ops/clear_experiment.py exp_poc_toolbench_20260402
    python ops/clear_experiment.py exp_poc_toolbench_20260402 --dry-run
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    print("⚠️  Warning: .env file not found, using default MongoDB settings")
    print()

from src.storage.mongodb import MongoDBStorage


def clear_experiment(experiment_id: str, dry_run: bool = False):
    """
    Clear all data for a specific experiment.

    Args:
        experiment_id: Experiment ID to clear
        dry_run: If True, only show what would be deleted
    """
    storage = MongoDBStorage()

    # Check if experiment exists
    experiment = storage.get_experiment(experiment_id)
    if not experiment:
        print(f"❌ Experiment '{experiment_id}' not found in database")
        storage.close()
        return

    print("=" * 70)
    print(f"CLEARING EXPERIMENT: {experiment_id}")
    print("=" * 70)
    print(f"Name: {experiment.get('name', 'N/A')}")
    print(f"Description: {experiment.get('description', 'N/A')}")
    print(f"Created: {experiment.get('created_at', 'N/A')}")
    print("=" * 70)
    print()

    if dry_run:
        print("🔍 DRY RUN MODE - No data will be deleted")
        print()

    # Count documents to be deleted
    collections_to_clear = {
        "experiments": {"experiment_id": experiment_id},
        "trajectories": {"experiment_id": experiment_id},
        "perturbations": {"experiment_id": experiment_id},
        "annotations": {"experiment_id": experiment_id},
        "judge_evaluations": {"experiment_id": experiment_id},
        "ccg_metrics": {"experiment_id": experiment_id},
    }

    total_docs = 0
    collection_counts = {}

    print("📊 Documents to delete:")
    print("-" * 70)

    for collection_name, query in collections_to_clear.items():
        collection = storage.db[collection_name]
        count = collection.count_documents(query)
        collection_counts[collection_name] = count
        total_docs += count
        print(f"  {collection_name:20s}: {count:5d} documents")

    print("-" * 70)
    print(f"  {'TOTAL':20s}: {total_docs:5d} documents")
    print()

    if total_docs == 0:
        print("✓ No data found for this experiment_id")
        storage.close()
        return

    # Confirm deletion (unless dry_run)
    if not dry_run:
        print("⚠️  WARNING: This action cannot be undone!")
        response = input(f"Delete {total_docs} documents? [y/N]: ").strip().lower()
        if response != 'y':
            print("❌ Cancelled")
            storage.close()
            return

        print()
        print("🗑️  Deleting documents...")
        print("-" * 70)

        # Delete from each collection
        for collection_name, query in collections_to_clear.items():
            if collection_counts[collection_name] > 0:
                collection = storage.db[collection_name]
                result = collection.delete_many(query)
                print(f"  {collection_name:20s}: Deleted {result.deleted_count} documents")

        print("-" * 70)
        print(f"  {'TOTAL':20s}: Deleted {total_docs} documents")
        print()
        print("✅ Experiment cleared successfully!")

    else:
        print("✓ Dry run complete - no changes made")

    storage.close()


def clear_all_experiments(dry_run: bool = False):
    """
    Clear ALL experiments from database (dangerous!).

    Args:
        dry_run: If True, only show what would be deleted
    """
    storage = MongoDBStorage()

    # Get all experiments
    experiments = list(storage.db.experiments.find({}))

    if not experiments:
        print("✓ No experiments found in database")
        storage.close()
        return

    print("=" * 70)
    print(f"⚠️  CLEARING ALL EXPERIMENTS ({len(experiments)} experiments)")
    print("=" * 70)
    print()

    if dry_run:
        print("🔍 DRY RUN MODE - No data will be deleted")
        print()

    # Count all documents
    collections_to_clear = [
        "experiments",
        "perturbations",
        "annotations",
        "judge_evaluations",
        "ccg_metrics"
    ]

    total_docs = 0
    collection_counts = {}

    print("📊 Documents to delete:")
    print("-" * 70)

    for collection_name in collections_to_clear:
        collection = storage.db[collection_name]
        count = collection.count_documents({})
        collection_counts[collection_name] = count
        total_docs += count
        print(f"  {collection_name:20s}: {count:5d} documents")

    print("-" * 70)
    print(f"  {'TOTAL':20s}: {total_docs:5d} documents")
    print()

    print("ℹ️  Note: Trajectories are NOT deleted (they are pure cache)")
    print()

    if total_docs == 0:
        print("✓ No data to delete")
        storage.close()
        return

    # Confirm deletion (unless dry_run)
    if not dry_run:
        print("⚠️⚠️⚠️  EXTREME WARNING: This will delete ALL experiment data! ⚠️⚠️⚠️")
        print("⚠️  This action cannot be undone!")
        print()
        response = input("Type 'DELETE ALL' to confirm: ").strip()
        if response != 'DELETE ALL':
            print("❌ Cancelled")
            storage.close()
            return

        print()
        print("🗑️  Deleting ALL experiment data...")
        print("-" * 70)

        # Delete from each collection
        for collection_name in collections_to_clear:
            if collection_counts[collection_name] > 0:
                collection = storage.db[collection_name]
                result = collection.delete_many({})
                print(f"  {collection_name:20s}: Deleted {result.deleted_count} documents")

        print("-" * 70)
        print(f"  {'TOTAL':20s}: Deleted {total_docs} documents")
        print()
        print("✅ All experiments cleared successfully!")

    else:
        print("✓ Dry run complete - no changes made")

    storage.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clear MongoDB collections for a specific experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clear specific experiment (with confirmation)
  python ops/clear_experiment.py exp_poc_toolbench_20260402

  # Dry run to preview what would be deleted
  python ops/clear_experiment.py exp_poc_toolbench_20260402 --dry-run

  # Clear ALL experiments (very dangerous!)
  python ops/clear_experiment.py --all

  # Dry run for all experiments
  python ops/clear_experiment.py --all --dry-run
        """
    )

    parser.add_argument(
        "experiment_id",
        nargs="?",
        help="Experiment ID to clear (e.g., exp_poc_toolbench_20260402)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Clear ALL experiments (DANGEROUS - requires confirmation)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.all:
        if args.experiment_id:
            print("❌ Error: Cannot specify experiment_id with --all flag")
            sys.exit(1)
        clear_all_experiments(dry_run=args.dry_run)

    elif args.experiment_id:
        clear_experiment(args.experiment_id, dry_run=args.dry_run)

    else:
        parser.print_help()
        print()
        print("❌ Error: Must specify either experiment_id or --all flag")
        sys.exit(1)


if __name__ == "__main__":
    main()
