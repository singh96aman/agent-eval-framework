#!/usr/bin/env python3
"""
Main entry point for the experiment runner.

6 phases: load, perturb, sample, annotate, judge, compute

Usage:
    python main.py --config schema_2_template --runner load,perturb
    python main.py --config schema_2_template --runner judge,compute
    python main.py --list-configs
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.experiment_runner import ExperimentRunner


class TeeOutput:
    """Write to both file and console (like Unix tee command)."""

    def __init__(self, file_path: Path, mode='a'):
        self.file = open(file_path, mode)
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_config(config_name: str) -> Dict[str, Any]:
    """Load experiment configuration from JSON file."""
    config_dir = Path(__file__).parent / "config" / "experiments"

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
                desc = exp_info.get('description', 'N/A')
        except Exception:
            name = 'Error loading'
            desc = 'Could not parse config'

        print(f"  {config_name}")
        print(f"   Name: {name}")
        print(f"   Description: {desc}")
        print()

    print("=" * 70)
    print("Usage: python main.py --config <config_name> --runner <phases>")
    print("Phases: load, perturb, sample, annotate, judge, compute")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run experiment phases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config schema_2_template --runner load,perturb
  python main.py --config schema_2_template --runner judge,compute
  python main.py --config schema_2_template --runner load --dry-run
  python main.py --list-configs

Phases: load, perturb, sample, annotate, judge, compute
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Name of configuration file (in config/experiments/)"
    )
    parser.add_argument(
        "--runner",
        type=str,
        help="Phases to run (comma-separated): load,perturb,sample,annotate,judge,compute"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without saving to database"
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
        print(f"Loading configuration: {args.config}")
        config = load_config(args.config)
        print(f"Loaded: {config['experiment']['name']}")
        print()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Print CLI options
    if args.runner:
        print(f"Runner: {args.runner}")
    if args.dry_run:
        print("Dry run mode enabled")

    print()

    # Set up logging
    experiment_info = config.get('experiment', {})
    experiment_name = experiment_info.get('name', 'experiment')
    experiment_id = experiment_info.get('id', args.config)

    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    safe_id = experiment_id.replace("/", "_").replace(":", "_")
    log_path = logs_dir / f"{safe_id}.log"

    print(f"Logging to: {log_path}")
    print()

    # Run experiment
    with TeeOutput(log_path, mode='a') as tee:
        original_stdout = sys.stdout
        sys.stdout = tee

        try:
            print("\n" + "=" * 70)
            print(f"NEW RUN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            print(f"Experiment: {experiment_name}")
            print(f"Config: {args.config}")
            print(f"Runner: {args.runner or 'none'}")
            print("=" * 70)
            print()

            runner = ExperimentRunner(
                config,
                runner_str=args.runner,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
            runner.run()

            print()
            print("=" * 70)
            print("EXPERIMENT PHASE COMPLETE")
            print("=" * 70)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            sys.stdout = original_stdout
            sys.exit(130)
        except Exception as e:
            print()
            print("=" * 70)
            print("EXPERIMENT FAILED")
            print("=" * 70)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout = original_stdout
            sys.exit(1)
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
