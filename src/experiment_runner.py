"""
Experiment Runner - orchestrates all phases of the POC experiment.

This is the main driver class that handles:
- Phase 1: Load trajectories from HuggingFace
- Phase 2: Generate perturbations
- Phase 3: Human annotation
- Phase 4: Judge evaluation
- Phase 5: CCG computation and analysis
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from data.loaders import load_toolbench_trajectories, load_gaia_trajectories
from storage.mongodb import MongoDBStorage
from data.schema import Trajectory


def generate_experiment_id(config: Dict[str, Any]) -> str:
    """
    Generate a unique experiment ID by hashing the configuration.

    This creates a deterministic hash from the experiment configuration,
    ensuring the same config always produces the same experiment ID.

    Args:
        config: Experiment configuration dictionary

    Returns:
        Unique experiment ID (e.g., "exp_a3f2d9e1")
    """
    # Create a stable JSON representation for hashing
    # Sort keys to ensure deterministic ordering
    config_str = json.dumps(config, sort_keys=True)

    # Generate SHA256 hash and take first 8 characters
    hash_digest = hashlib.sha256(config_str.encode()).hexdigest()[:8]

    return f"exp_{hash_digest}"


class ExperimentRunner:
    """
    Main experiment orchestrator that runs phases based on configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize experiment runner with configuration.

        Args:
            config: Experiment configuration dictionary
        """
        # Load environment variables
        load_dotenv()

        self.config = config
        self.experiment_info = config.get('experiment', {})
        self.execution = config.get('execution', {})

        # Parse runner parameter (which phases to run)
        runner_str = self.execution.get('runner', 'all')
        self.phases_to_run = self._parse_runner(runner_str)

        self.dry_run = self.execution.get('dry_run', False)
        self.verbose = self.execution.get('verbose', True)

        # Initialize storage (unless dry run)
        self.storage: Optional[MongoDBStorage] = None
        if not self.dry_run:
            self.storage = MongoDBStorage()

        # Get or generate experiment ID from config
        # If experiment_id is specified in config, use it
        # Otherwise, generate one from config hash
        self.experiment_id: Optional[str] = self.experiment_info.get('experiment_id')
        if not self.experiment_id:
            self.experiment_id = generate_experiment_id(self.config)

        self.checkpoint_dir = Path(
            self.execution.get('checkpoint_dir', 'results/checkpoints')
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _parse_runner(self, runner_str: str) -> List[str]:
        """
        Parse runner string into list of phases to run.

        Args:
            runner_str: Comma-separated phases or 'all'
                       Examples: 'load', 'load,perturb', 'all'

        Returns:
            List of phase names to execute
        """
        # Mapping of short names to full phase names
        phase_map = {
            'load': 'load_trajectories',
            'perturb': 'generate_perturbations',
            'annotate': 'annotate',
            'judge': 'evaluate_judges',
            'ccg': 'compute_ccg',
            'analyze': 'analyze',
            # Also support full names
            'load_trajectories': 'load_trajectories',
            'generate_perturbations': 'generate_perturbations',
            'evaluate_judges': 'evaluate_judges',
            'compute_ccg': 'compute_ccg',
        }

        # 'all' means run everything in order
        if runner_str == 'all':
            return [
                'load_trajectories',
                'generate_perturbations',
                'annotate',
                'evaluate_judges',
                'compute_ccg',
                'analyze'
            ]

        # Parse comma-separated list
        phases = []
        for phase in runner_str.split(','):
            phase = phase.strip().lower()
            if phase in phase_map:
                phases.append(phase_map[phase])
            else:
                raise ValueError(
                    f"Unknown phase: '{phase}'. "
                    f"Valid options: {', '.join(phase_map.keys())}, all"
                )

        return phases

    def run(self):
        """Run experiment phases specified by runner config."""
        print("=" * 70)
        print(f"EXPERIMENT: {self.experiment_info.get('name', 'Unnamed')}")
        print("=" * 70)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Phases to run: {', '.join(self.phases_to_run)}")
        print(f"Dry Run: {self.dry_run}")
        desc = self.experiment_info.get('description', 'N/A')
        print(f"Description: {desc}")
        print("=" * 70)
        print()

        # Route to appropriate phase handler
        phase_handlers = {
            'load_trajectories': self._phase_load_trajectories,
            'generate_perturbations': self._phase_generate_perturbations,
            'annotate': self._phase_annotate,
            'evaluate_judges': self._phase_evaluate_judges,
            'compute_ccg': self._phase_compute_ccg,
            'analyze': self._phase_analyze,
        }

        # Run each phase in sequence
        for idx, phase in enumerate(self.phases_to_run, 1):
            print()
            print("=" * 70)
            print(
                f"RUNNING PHASE {idx}/{len(self.phases_to_run)}: "
                f"{phase.upper()}"
            )
            print("=" * 70)
            print()

            handler = phase_handlers.get(phase)
            if not handler:
                raise ValueError(
                    f"Unknown phase: {phase}. "
                    f"Valid phases: {list(phase_handlers.keys())}"
                )

            # Run the phase
            handler()

        # Cleanup
        if self.storage:
            self.storage.close()

    def _phase_load_trajectories(self):
        """
        Phase 1: Load trajectories from HuggingFace datasets.

        Loads trajectories from ToolBench and GAIA, then stores them in MongoDB.
        """
        print("📥 PHASE: LOAD TRAJECTORIES")
        print("=" * 70)
        print()

        datasets_config = self.config.get('datasets', {})
        toolbench_config = datasets_config.get('toolbench', {})
        gaia_config = datasets_config.get('gaia', {})

        trajectories = []

        # Load ToolBench
        if toolbench_config.get('enabled', True):
            num_traj = toolbench_config.get('num_trajectories', 25)
            filters = toolbench_config.get('filters', {})

            print(f"📊 Loading {num_traj} ToolBench trajectories...")
            toolbench_trajs = load_toolbench_trajectories(
                max_trajectories=num_traj,
                min_steps=filters.get('min_steps', 2),
                max_steps=filters.get('max_steps', 20),
                filter_successful=filters.get('filter_successful', False),
                random_seed=toolbench_config.get('random_seed', 42)
            )
            print(f"   ✓ Loaded {len(toolbench_trajs)} ToolBench trajectories")
            trajectories.extend(toolbench_trajs)
            print()

        # Load GAIA
        if gaia_config.get('enabled', True):
            num_traj = gaia_config.get('num_trajectories', 25)
            filters = gaia_config.get('filters', {})

            print(f"📊 Loading {num_traj} GAIA trajectories...")
            gaia_trajs = load_gaia_trajectories(
                max_trajectories=num_traj,
                min_steps=filters.get('min_steps', 1),
                max_steps=filters.get('max_steps', 20),
                random_seed=gaia_config.get('random_seed', 42)
            )
            print(f"   ✓ Loaded {len(gaia_trajs)} GAIA trajectories")
            trajectories.extend(gaia_trajs)
            print()

        print(f"📊 Total trajectories loaded: {len(trajectories)}")
        print()

        # Always show first sample for debugging
        if trajectories:
            self._show_sample_trajectory(trajectories[0])

        # Save to MongoDB (unless dry run)
        if self.dry_run:
            print("🏃 DRY RUN MODE - Not saving to MongoDB")
            print()
            return trajectories

        print("💾 Storing trajectories in MongoDB...")
        self._store_trajectories(trajectories)

        return trajectories

    def _store_trajectories(self, trajectories: List[Trajectory]):
        """
        Store trajectories in MongoDB (pure cache).

        NEW ARCHITECTURE: Trajectories are stored WITHOUT experiment_id.
        They are pure cache entries that can be reused across experiments.
        """
        # Use experiment_id from config (already set in __init__)
        # This is either from config or generated from config hash

        # Create experiment record in MongoDB
        experiment_config = {
            "experiment_id": self.experiment_id,
            "name": self.experiment_info.get('name', self.experiment_id),
            "description": self.experiment_info.get('description', ''),
            "config": self.config
        }
        mongo_experiment_id = self.storage.create_experiment(experiment_config)
        print(f"   ✓ Created experiment: {self.experiment_id}")

        # Store trajectories (pure cache, NO experiment_id)
        stored_count = 0
        cache_hits = 0

        for traj in trajectories:
            # Convert Trajectory object to dict (NO experiment_id!)
            traj_dict = traj.to_dict()
            traj_dict["is_perturbed"] = False  # Original trajectory

            # Check if already in cache
            existing = self.storage.get_trajectory(traj_dict["trajectory_id"])
            if existing:
                cache_hits += 1
            else:
                self.storage.save_trajectory(trajectory=traj_dict)
                stored_count += 1

            if (stored_count + cache_hits) % 10 == 0:
                print(f"   ... processed {stored_count + cache_hits}/{len(trajectories)} (new: {stored_count}, cached: {cache_hits})")

        print(f"   ✓ Stored {stored_count} new trajectories")
        print(f"   ✓ Cache hits: {cache_hits}")
        print()

        # Verify data in cache
        print("🔍 Verifying trajectory cache...")
        try:
            total_cached = self.storage.trajectories.count_documents({
                "is_perturbed": False
            })
            print(f"   ✓ Total original trajectories in cache: {total_cached}")
        except Exception as e:
            print(f"   ⚠️  Could not verify: {e}")
        print()

        print("=" * 70)
        print("✅ TRAJECTORY LOADING COMPLETE")
        print("=" * 70)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"New trajectories cached: {stored_count}")
        print(f"Cache hits: {cache_hits}")
        print(f"Database: {self.storage.database_name}")
        print()

    def _show_sample_trajectory(self, traj: Trajectory):
        """Display a sample trajectory for verification."""
        print("=" * 70)
        print("📋 SAMPLE TRAJECTORY")
        print("=" * 70)
        print(f"  ID: {traj.trajectory_id}")
        print(f"  Benchmark: {traj.benchmark}")
        print(f"  Steps: {len(traj.steps)}")
        print(f"  Task: {traj.ground_truth.task_description[:100]}...")
        if traj.steps:
            step1 = traj.steps[0]
            print(f"  First step type: {step1.step_type}")
            print(f"  First step content: {step1.content[:80]}..." if step1.content else "  First step content: [empty]")
            if step1.tool_name:
                print(f"  Tool: {step1.tool_name}")
        print("=" * 70)
        print()

    def _phase_generate_perturbations(self):
        """Phase 2: Generate perturbed trajectories."""
        print("🔧 PHASE: GENERATE PERTURBATIONS")
        print("=" * 70)
        print()
        print("⚠️  This phase is not yet implemented.")
        print("   Coming in Phase 3 of implementation.")
        print()

    def _phase_annotate(self):
        """Phase 3: Human annotation interface."""
        print("✍️  PHASE: ANNOTATION")
        print("=" * 70)
        print()
        print("⚠️  This phase is not yet implemented.")
        print("   Coming in Phase 5 of implementation.")
        print()

    def _phase_evaluate_judges(self):
        """Phase 4: Run judge evaluations."""
        print("⚖️  PHASE: EVALUATE JUDGES")
        print("=" * 70)
        print()
        print("⚠️  This phase is not yet implemented.")
        print("   Coming in Phase 4 of implementation.")
        print()

    def _phase_compute_ccg(self):
        """Phase 5: Compute CCG metrics."""
        print("📊 PHASE: COMPUTE CCG")
        print("=" * 70)
        print()
        print("⚠️  This phase is not yet implemented.")
        print("   Coming in Phase 5 of implementation.")
        print()

    def _phase_analyze(self):
        """Phase 6: Analysis and visualization."""
        print("📈 PHASE: ANALYSIS")
        print("=" * 70)
        print()
        print("⚠️  This phase is not yet implemented.")
        print("   Coming in Phase 6 of implementation.")
        print()
