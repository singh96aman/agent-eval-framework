#!/usr/bin/env python3
"""
Main entry point for the experiment runner.

Supports two schema versions:
- Schema 1.x: Legacy format (ExperimentRunner)
- Schema 2.x: Simplified format (RunnerV2)

Usage:
    # Schema 2.x (6 phases: load, perturb, sample, annotate, judge, compute)
    python main.py --config schema_2_template --runner load,perturb
    python main.py --config schema_2_template --runner judge,compute

    # Schema 1.x (legacy)
    python main.py --config poc_experiment --runner all

    # List configs
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
from src.runner_v2 import RunnerV2


class TeeOutput:
    """Write to both file and console (like Unix tee command)."""

    def __init__(self, file_path: Path, mode='a'):
        self.file = open(file_path, mode)
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write

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
                schema = config.get('schema', '1.x')
                desc = exp_info.get('description', 'N/A')
        except Exception:
            name = 'Error loading'
            schema = '?'
            desc = 'Could not parse config'

        print(f"📄 {config_name}")
        print(f"   Name: {name}")
        print(f"   Schema: {schema}")
        print(f"   Description: {desc}")
        print()

    print("=" * 70)
    print(f"Usage: python main.py --config <config_name>")
    print("=" * 70)


def get_schema_version(config: Dict[str, Any]) -> str:
    """Extract schema version from config, default to 1.0.0."""
    return config.get("schema", "1.0.0")


def is_schema_v2(config: Dict[str, Any]) -> bool:
    """Check if config uses schema 2.x."""
    version = get_schema_version(config)
    return version.startswith("2.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run experiment phases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Schema 2.x (recommended)
  python main.py --config schema_2_template --runner load,perturb
  python main.py --config schema_2_template --runner judge,compute
  python main.py --config schema_2_template --runner load,perturb,sample,annotate,judge,compute

  # Schema 1.x (legacy)
  python main.py --config poc_experiment --runner all
  python main.py --config poc_experiment --runner load,perturb,judge

  # Dry run (test without saving)
  python main.py --config poc_experiment --runner load --dry-run

  # List all available configs
  python main.py --list-configs

Schema 2.x Phases: load, perturb, sample, annotate, judge, compute
(compute targets are defined in config.compute.targets)
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
    parser.add_argument(
        "--log-bedrock",
        action="store_true",
        help="Log Bedrock API calls with latency"
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

    # Detect schema version early
    schema_v2 = is_schema_v2(config)
    schema_version = get_schema_version(config)

    # Override config with command-line args (schema 1.x only)
    # Schema 2.x passes args directly to RunnerV2
    if not schema_v2:
        if 'execution' not in config:
            config['execution'] = {}

        if args.phase:
            print("⚠️  Warning: --phase is deprecated, use --runner instead")
            config['execution']['runner'] = args.phase

        if args.runner:
            config['execution']['runner'] = args.runner

        if args.dry_run:
            config['execution']['dry_run'] = True

        if args.verbose:
            config['execution']['verbose'] = True

        if args.resume:
            config['execution']['resume'] = True

        if args.log_bedrock:
            config['execution']['log_bedrock'] = True

    # Print CLI options
    if args.runner:
        print("⚙️  Runner: {}".format(args.runner))
    if args.dry_run:
        print("⚙️  Dry run mode enabled")
    if args.verbose:
        print("⚙️  Verbose mode enabled")
    if args.resume:
        print("⚙️  Resume mode enabled")

    print()
    print(f"📋 Schema version: {schema_version}")

    # Set up logging to file
    experiment_info = config.get('experiment', {})
    experiment_name = experiment_info.get('name', 'experiment')
    # Schema 2.x uses 'id', Schema 1.x uses 'experiment_id'
    experiment_id = experiment_info.get('id') or experiment_info.get('experiment_id', args.config)
    verbose = config.get('execution', {}).get('verbose', args.verbose)

    # Create logs directory
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Use experiment_id as log filename (append to same log for same experiment)
    safe_experiment_id = experiment_id.replace("/", "_").replace(":", "_")
    log_filename = f"{safe_experiment_id}.log"
    log_path = logs_dir / log_filename

    print(f"📝 Logging to: {log_path}")
    print()

    # Redirect stdout to both console and file
    with TeeOutput(log_path, mode='a') as tee:
        original_stdout = sys.stdout
        sys.stdout = tee

        try:
            print("\n" + "=" * 70)
            print(f"NEW RUN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            print(f"Experiment: {experiment_name}")
            print(f"Experiment ID: {experiment_id}")
            print(f"Config: {args.config}")
            print(f"Runner: {config.get('execution', {}).get('runner', 'N/A')}")
            print(f"Log file: {log_path}")
            print("=" * 70)
            print()

            # Create and run experiment based on schema version
            if schema_v2:
                print(f"Using RunnerV2 (schema {schema_version})")
                print()
                runner = RunnerV2(
                    config,
                    runner_str=args.runner,
                    dry_run=args.dry_run,
                    verbose=args.verbose or verbose,
                )
            else:
                print(f"Using ExperimentRunner (schema {schema_version})")
                print()
                runner = ExperimentRunner(config)

            runner.run()

            print()
            print("=" * 70)
            print("✅ EXPERIMENT PHASE COMPLETE")
            print("=" * 70)

        except KeyboardInterrupt:
            print()
            print("⚠️  Interrupted by user")
            sys.stdout = original_stdout
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
            sys.stdout = original_stdout
            sys.exit(1)
        finally:
            # Restore stdout
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
