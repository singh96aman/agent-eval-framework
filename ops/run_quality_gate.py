#!/usr/bin/env python3
"""
Quality Gate Runner - CLI script to validate pipeline outputs.

Loads data from MongoDB and runs quality gates for validation.

Usage:
    python ops/run_quality_gate.py --phase load --all --experiment exp_trajectory_sampling_v8
    python ops/run_quality_gate.py --gate no_synthetic_markers --phase perturb --experiment exp_trajectory_sampling_v8
    python ops/run_quality_gate.py --phase perturb --gates no_synthetic_markers,json_validity --experiment exp_trajectory_sampling_v8
"""

import argparse
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from pymongo import MongoClient

from src.quality_gates import GateRunner, get_gate

# Gate configuration by phase
PHASE_GATES = {
    "load": ["trajectory_count", "grader_pass_rate"],
    "perturb": [
        "no_synthetic_markers",
        "no_structural_corruption",
        "json_validity",
        "position_distribution",
        "perturbation_class_validity",
    ],
    "evaluation_unit": ["blinding_balance", "length_preservation"],
    "compute": ["outcome_variance"],
}

# MongoDB collections for each phase
PHASE_COLLECTIONS = {
    "load": "trajectories",
    "perturb": "perturbed_trajectories",
    "evaluation_unit": "evaluation_units",
    "compute": "outcome_evidence",
}

# Gate configuration thresholds (from Requirements.MD)
GATE_CONFIG = {
    "trajectory_count": {"min": 100},
    "grader_pass_rate": {"min": 0.30},
    "no_synthetic_markers": {"max_rate": 0.0},
    "no_structural_corruption": {"max_rate": 0.0},
    "json_validity": {"min_rate": 0.998},
    "position_distribution": {"max_late_rate": 0.40},
    "perturbation_class_validity": {"min_rate": 0.90},
    "blinding_balance": {"min_rate": 0.45, "max_rate": 0.55},
    "length_preservation": {},
    "outcome_variance": {"min_std": 0.1},
}


def get_mongo_client(db_name: str = "agent_judge_experiment"):
    """Get MongoDB client connection."""
    # Try MongoDB Atlas first (from environment), fallback to localhost
    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri)
    return client[db_name]


def load_data(db, collection_name: str, experiment_id: str):
    """Load data from MongoDB collection."""
    collection = db[collection_name]
    data = list(collection.find({"experiment_id": experiment_id}))
    return data


def run_gates(
    phase: str,
    gate_names: list,
    data: list,
    verbose: bool = True,
) -> bool:
    """Run specified gates and return pass/fail status."""
    runner = GateRunner(phase=phase)

    for gate_name in gate_names:
        gate = get_gate(gate_name)
        runner.add_gate(gate)

    # Get config for gates
    config = {}
    for gate_name in gate_names:
        if gate_name in GATE_CONFIG:
            config.update(GATE_CONFIG[gate_name])

    report = runner.run(data, config)

    # Print report
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Quality Gate Report: {phase}")
        print(f"{'=' * 60}")
        print(f"Records checked: {len(data)}")
        print(f"Timestamp: {report.timestamp}")
        print(f"{'=' * 60}")

        for result in report.results:
            if result.passed:
                status_icon = "\u2705"  # checkmark
                status_text = "PASS"
            else:
                status_icon = "\u274c"  # X
                status_text = "FAIL"

            print(f"{status_icon} [{status_text}] {result.gate_name}")
            print(f"   {result.message}")
            if result.value is not None:
                print(f"   Value: {result.value}, Threshold: {result.threshold}")
            print()

        print(f"{'=' * 60}")
        summary = "ALL PASSED" if report.all_passed else "FAILED"
        icon = "\u2705" if report.all_passed else "\u274c"
        print(f"{icon} Overall: {summary}")
        print(f"   Pass: {report.pass_count}, Fail: {report.fail_count}, Warn: {report.warn_count}")
        print(f"{'=' * 60}")

    return report.all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Run quality gates on pipeline outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all gates for load phase
  python ops/run_quality_gate.py --phase load --all --experiment exp_trajectory_sampling_v8

  # Run a single gate
  python ops/run_quality_gate.py --gate no_synthetic_markers --phase perturb --experiment exp_trajectory_sampling_v8

  # Run multiple specific gates
  python ops/run_quality_gate.py --phase perturb --gates no_synthetic_markers,json_validity --experiment exp_trajectory_sampling_v8

  # List available gates
  python ops/run_quality_gate.py --list-gates
        """,
    )

    parser.add_argument(
        "--phase",
        choices=list(PHASE_GATES.keys()),
        help="Pipeline phase to validate",
    )
    parser.add_argument(
        "--gate",
        help="Single gate name to run",
    )
    parser.add_argument(
        "--gates",
        help="Comma-separated list of gates to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all gates for the specified phase",
    )
    parser.add_argument(
        "--experiment",
        help="Experiment ID to filter data",
    )
    parser.add_argument(
        "--db",
        default="agent_judge_experiment",
        help="MongoDB database name (default: agent_judge_experiment)",
    )
    parser.add_argument(
        "--list-gates",
        action="store_true",
        help="List all available gates by phase",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print pass/fail status",
    )

    args = parser.parse_args()

    # List gates mode
    if args.list_gates:
        print("Available Quality Gates by Phase:")
        print("=" * 40)
        for phase, gates in PHASE_GATES.items():
            print(f"\n{phase}:")
            for gate in gates:
                print(f"  - {gate}")
        return 0

    # Validate arguments
    if not args.phase:
        parser.error("--phase is required (or use --list-gates)")

    if not args.experiment:
        parser.error("--experiment is required")

    if not (args.all or args.gate or args.gates):
        parser.error("Must specify --all, --gate, or --gates")

    # Determine which gates to run
    if args.all:
        gate_names = PHASE_GATES[args.phase]
    elif args.gates:
        gate_names = [g.strip() for g in args.gates.split(",")]
    else:
        gate_names = [args.gate]

    # Validate gate names
    valid_gates = set()
    for gates in PHASE_GATES.values():
        valid_gates.update(gates)

    for gate_name in gate_names:
        if gate_name not in valid_gates:
            print(f"Error: Unknown gate '{gate_name}'")
            print(f"Use --list-gates to see available gates")
            return 1

    # Connect to MongoDB
    try:
        db = get_mongo_client(args.db)
        # Test connection
        db.list_collection_names()
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return 1

    # Load data
    collection_name = PHASE_COLLECTIONS[args.phase]
    print(f"Loading data from {collection_name} for experiment {args.experiment}...")

    data = load_data(db, collection_name, args.experiment)

    if not data:
        print(f"Warning: No data found in {collection_name} for experiment {args.experiment}")
        print("Possible issues:")
        print("  - Experiment ID mismatch")
        print("  - Previous phase not completed")
        print("  - Database connection issue")
        return 1

    print(f"Loaded {len(data)} records")

    # Run gates
    passed = run_gates(
        phase=args.phase,
        gate_names=gate_names,
        data=data,
        verbose=not args.quiet,
    )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
