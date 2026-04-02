#!/usr/bin/env python3
"""
Main entry point for the POC experiment.

This is the single driver script that runs all phases of the experiment
based on configuration files.

Usage:
    python main.py --config poc_phase2_load
    python main.py --config poc_phase2_load --dry-run
    python main.py --config poc_phase2_load --phase load_trajectories
    python main.py --list-configs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.experiment_runner import ExperimentRunner


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load experiment configuration from JSON file.

    Args:
        config_name: Name of config file (without .json extension)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_dir = Path(__file__).parent / "config" / "experiments"

    # Try with and without .json extension
    config_path = config_dir / f"{config_name}.json"
    if not config_path.exists():
        config_path = config_dir / config_name

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_name}\n"
            f"Looked in: {config_dir}\n"
            f"Use --list-configs to see available configurations."
        )

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def list_available_configs():
    """List all available configuration files."""
    config_dir = Path(__file__).parent / "config" / "experiments"

    if not config_dir.exists():
        print("No config directory found.")
        return

    config_files = sorted(config_dir.glob("*.json"))

    if not config_files:
        print("No configuration files found.")
        return

    print("=" * 70)
    print("AVAILABLE CONFIGURATIONS")
    print("=" * 70)
    print()

    for config_file in config_files:
        config_name = config_file.stem
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                exp_info = config.get('experiment', {})
                name = exp_info.get('name', 'N/A')
                phase = config.get('execution', {}).get('phase', 'N/A')
                desc = exp_info.get('description', 'N/A')
        except Exception:
            name = 'Error loading'
            phase = 'N/A'
            desc = 'Could not parse config'

        print(f"📄 {config_name}")
        print(f"   Name: {name}")
        print(f"   Phase: {phase}")
        print(f"   Description: {desc}")
        print()

    print("=" * 70)
    print(f"Usage: python main.py --config <config_name>")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run POC experiment phases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run entire experiment
  python main.py --config poc_experiment --runner all

  # Run specific phases
  python main.py --config poc_experiment --runner load,perturb
  python main.py --config poc_experiment --runner judge,ccg,analyze

  # Run single phase
  python main.py --config poc_experiment --runner load

  # Dry run (test without saving)
  python main.py --config poc_experiment --runner load --dry-run

  # List all available configs
  python main.py --list-configs
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Name of configuration file (in config/experiments/)"
    )
    parser.add_argument(
        "--phase",
        type=str,
        help="(Deprecated: use --runner instead) Override phase specified in config"
    )
    parser.add_argument(
        "--runner",
        type=str,
        help=(
            "Which phases to run (comma-separated or 'all'). "
            "Options: load, perturb, annotate, judge, ccg, analyze, all. "
            "Example: --runner load,perturb or --runner all"
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without saving to database"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available configuration files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # List configs and exit
    if args.list_configs:
        list_available_configs()
        return

    # Require config if not listing
    if not args.config:
        parser.error("--config is required (or use --list-configs)")

    # Load configuration
    try:
        print(f"📋 Loading configuration: {args.config}")
        config = load_config(args.config)
        print(f"✓ Loaded config: {config['experiment']['name']}")
        print()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Override config with command-line args
    if args.phase:
        print("⚠️  Warning: --phase is deprecated, use --runner instead")
        config['execution']['runner'] = args.phase

    if args.runner:
        config['execution']['runner'] = args.runner
        print("⚙️  Runner: {}".format(args.runner))

    if args.dry_run:
        config['execution']['dry_run'] = True
        print("⚙️  Dry run mode enabled")

    if args.verbose:
        config['execution']['verbose'] = True

    if args.resume:
        config['execution']['resume'] = True
        print("⚙️  Resume mode enabled")

    print()

    # Create and run experiment
    try:
        runner = ExperimentRunner(config)
        runner.run()

        print()
        print("=" * 70)
        print("✅ EXPERIMENT PHASE COMPLETE")
        print("=" * 70)

    except KeyboardInterrupt:
        print()
        print("⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ EXPERIMENT FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
