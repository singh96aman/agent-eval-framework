"""
Experiment Runner - orchestrates all phases of the POC experiment.

This is the main driver class that handles:
- Phase 1: Load trajectories from HuggingFace
- Phase 2: Generate perturbations
- Phase 3: Human annotation
- Phase 4: Judge evaluation
- Phase 5: CCG computation and analysis
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from data.loaders import load_toolbench_trajectories, load_gaia_trajectories
from storage.mongodb import MongoDBStorage
from data.schema import Trajectory


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
        self.phase = self.execution.get('phase', 'load_trajectories')
        self.dry_run = self.execution.get('dry_run', False)
        self.verbose = self.execution.get('verbose', True)

        # Initialize storage (unless dry run)
        self.storage: Optional[MongoDBStorage] = None
        if not self.dry_run:
            self.storage = MongoDBStorage()

        # Experiment state
        self.experiment_id: Optional[str] = None
        self.checkpoint_dir = Path(
            self.execution.get('checkpoint_dir', 'results/checkpoints')
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Run the experiment phase specified in config."""
        print("=" * 70)
        print(f"EXPERIMENT: {self.experiment_info.get('name', 'Unnamed')}")
        print("=" * 70)
        print(f"Phase: {self.phase}")
        print(f"Dry Run: {self.dry_run}")
        print(f"Description: {self.experiment_info.get('description', 'N/A')}")
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

        handler = phase_handlers.get(self.phase)
        if not handler:
            raise ValueError(
                f"Unknown phase: {self.phase}. "
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

        # Show sample
        if trajectories and self.verbose:
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
        """Store trajectories in MongoDB."""
        storage_config = self.config.get('storage', {})
        experiment_name = storage_config.get(
            'experiment_name',
            f"poc_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Create experiment
        self.experiment_id = self.storage.create_experiment(
            name=experiment_name,
            description=self.experiment_info.get('description', ''),
            config=self.config
        )
        print(f"   ✓ Created experiment: {self.experiment_id}")

        # Store trajectories
        stored_count = 0
        for traj in trajectories:
            self.storage.save_trajectory(
                trajectory=traj,
                experiment_id=self.experiment_id
            )
            stored_count += 1

            if stored_count % 10 == 0:
                print(f"   ... stored {stored_count}/{len(trajectories)}")

        print(f"   ✓ Stored {stored_count} trajectories")
        print()
        print("=" * 70)
        print("✅ TRAJECTORY LOADING COMPLETE")
        print("=" * 70)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Trajectories: {stored_count}")
        print(f"Database: {self.storage.database_name}")
        print()

    def _show_sample_trajectory(self, traj: Trajectory):
        """Display a sample trajectory for verification."""
        print("Sample trajectory:")
        print(f"  ID: {traj.trajectory_id}")
        print(f"  Benchmark: {traj.benchmark}")
        print(f"  Steps: {len(traj.steps)}")
        print(f"  Task: {traj.ground_truth.task_description[:80]}...")
        if traj.steps:
            print(f"  First step: {traj.steps[0].action[:60]}...")
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
