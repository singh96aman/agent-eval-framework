"""
Phase 2: Load real trajectories from HuggingFace and store in MongoDB.

This script loads trajectories from ToolBench and GAIA benchmarks via
HuggingFace and stores them in MongoDB for later perturbation and analysis.

Usage:
    python src/load_trajectories.py --toolbench 25 --gaia 25
    python src/load_trajectories.py --toolbench 10  # ToolBench only
    python src/load_trajectories.py --dry-run       # Test without saving
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import load_toolbench_trajectories, load_gaia_trajectories
from storage.mongodb import MongoDBStorage


def load_and_store_trajectories(
    toolbench_count: int = 25,
    gaia_count: int = 25,
    dry_run: bool = False,
    experiment_name: str = "poc_experiment_v1"
):
    """
    Load trajectories from HuggingFace and store in MongoDB.

    Args:
        toolbench_count: Number of ToolBench trajectories to load
        gaia_count: Number of GAIA trajectories to load
        dry_run: If True, load but don't save to MongoDB
        experiment_name: Name for this experiment run
    """
    print("=" * 70)
    print("PHASE 2: LOAD TRAJECTORIES FROM HUGGINGFACE")
    print("=" * 70)
    print()

    # Load ToolBench trajectories
    print(f"📥 Loading {toolbench_count} ToolBench trajectories...")
    toolbench_trajs = load_toolbench_trajectories(
        max_trajectories=toolbench_count,
        min_steps=2,
        max_steps=20,
        filter_successful=False,  # Include both success and failure
        random_seed=42
    )
    print(f"   ✓ Loaded {len(toolbench_trajs)} ToolBench trajectories")
    print()

    # Load GAIA trajectories
    print(f"📥 Loading {gaia_count} GAIA trajectories...")
    gaia_trajs = load_gaia_trajectories(
        max_trajectories=gaia_count,
        min_steps=1,
        max_steps=20,
        random_seed=42
    )
    print(f"   ✓ Loaded {len(gaia_trajs)} GAIA trajectories")
    print()

    total_trajectories = toolbench_trajs + gaia_trajs
    print(f"📊 Total trajectories loaded: {len(total_trajectories)}")
    print()

    # Show sample
    if total_trajectories:
        sample = total_trajectories[0]
        print("Sample trajectory:")
        print(f"  ID: {sample.trajectory_id}")
        print(f"  Benchmark: {sample.benchmark}")
        print(f"  Steps: {len(sample.steps)}")
        print(f"  Task: {sample.ground_truth.task_description[:80]}...")
        print()

    if dry_run:
        print("🏃 DRY RUN MODE - Not saving to MongoDB")
        print()
        return total_trajectories

    # Store in MongoDB
    print("💾 Storing trajectories in MongoDB...")
    try:
        storage = MongoDBStorage()

        # Create experiment
        experiment_id = storage.create_experiment(
            name=experiment_name,
            description=(
                f"POC experiment with {len(toolbench_trajs)} ToolBench + "
                f"{len(gaia_trajs)} GAIA trajectories"
            ),
            config={
                "toolbench_count": len(toolbench_trajs),
                "gaia_count": len(gaia_trajs),
                "total_count": len(total_trajectories),
                "random_seed": 42
            }
        )
        print(f"   ✓ Created experiment: {experiment_id}")

        # Store trajectories
        stored_count = 0
        for traj in total_trajectories:
            trajectory_id = storage.save_trajectory(
                trajectory=traj,
                experiment_id=experiment_id
            )
            stored_count += 1

            if stored_count % 10 == 0:
                print(f"   ... stored {stored_count}/{len(total_trajectories)}")

        print(f"   ✓ Stored {stored_count} trajectories")
        print()

        storage.close()

        print("=" * 70)
        print("✅ PHASE 2 COMPLETE")
        print("=" * 70)
        print(f"Experiment ID: {experiment_id}")
        print(f"Trajectories: {stored_count}")
        print(f"Database: {storage.database_name}")
        print()
        print("Next: Phase 3 - Perturbation generation")
        print("=" * 70)

        return total_trajectories

    except Exception as e:
        print(f"❌ Error storing trajectories: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Load trajectories from HuggingFace and store in MongoDB"
    )
    parser.add_argument(
        "--toolbench",
        type=int,
        default=25,
        help="Number of ToolBench trajectories to load (default: 25)"
    )
    parser.add_argument(
        "--gaia",
        type=int,
        default=25,
        help="Number of GAIA trajectories to load (default: 25)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load trajectories but don't save to MongoDB"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="poc_experiment_v1",
        help="Name for this experiment run"
    )

    args = parser.parse_args()

    load_and_store_trajectories(
        toolbench_count=args.toolbench,
        gaia_count=args.gaia,
        dry_run=args.dry_run,
        experiment_name=args.experiment_name
    )


if __name__ == "__main__":
    main()
