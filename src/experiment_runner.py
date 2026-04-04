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
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from src.data.loaders import (
    load_toolbench_trajectories,
    load_gaia_trajectories,
    load_swebench_trajectories,
    load_trajectories_from_json,
    classify_trajectory_domain,
    classify_trajectory_complexity,
)
from src.storage.mongodb import MongoDBStorage
from src.data.schema import Trajectory


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

        # Initialize storage
        # Note: Even in dry-run mode, we need storage to read existing data
        # Dry-run only prevents writing new data (API calls, etc.)
        self.storage: Optional[MongoDBStorage] = MongoDBStorage()

        # Get or generate experiment ID from config
        # If experiment_id is specified in config, use it
        # Otherwise, generate one from config hash
        self.experiment_id: Optional[str] = (
            self.experiment_info.get('experiment_id')
        )
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
            'validate': 'validate_perturbations',
            'annotate': 'annotate',
            'judge': 'evaluate_judges',
            'ccg': 'compute_ccg',
            'analyze': 'analyze',
            'visualize': 'visualize',
            'viz': 'visualize',  # Short alias
            # Also support full names
            'load_trajectories': 'load_trajectories',
            'generate_perturbations': 'generate_perturbations',
            'validate_perturbations': 'validate_perturbations',
            'evaluate_judges': 'evaluate_judges',
            'compute_ccg': 'compute_ccg',
        }

        # 'all' means run everything in order
        if runner_str == 'all':
            return [
                'load_trajectories',
                'generate_perturbations',
                'validate_perturbations',
                'annotate',
                'evaluate_judges',
                'compute_ccg',
                'analyze',
                'visualize'
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
            'validate_perturbations': self._phase_validate_perturbations,
            'annotate': self._phase_annotate,
            'evaluate_judges': self._phase_evaluate_judges,
            'compute_ccg': self._phase_compute_ccg,
            'analyze': self._phase_analyze,
            'visualize': self._phase_visualize,
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

    def _convert_for_json(self, obj: Any) -> Any:
        """
        Recursively convert numpy types to native Python types for JSON serialization.

        Args:
            obj: Object to convert (dict, list, numpy type, etc.)

        Returns:
            Converted object with all numpy types replaced
        """
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _phase_load_trajectories(self):
        """
        Phase 1: Load trajectories from HuggingFace datasets or JSON files.

        Supports two modes per dataset:
        - source: "huggingface" (default) - Load from HuggingFace datasets
        - source: "json" - Load from pre-sampled JSON file (json_path required)

        Loads trajectories from ToolBench, GAIA, and SWE-bench, then stores
        them in MongoDB.
        """
        print("📥 PHASE: LOAD TRAJECTORIES")
        print("=" * 70)
        print()

        datasets_config = self.config.get('datasets', {})
        toolbench_config = datasets_config.get('toolbench', {})
        gaia_config = datasets_config.get('gaia', {})
        swebench_config = datasets_config.get('swebench', {})

        trajectories = []

        # Load ToolBench
        if toolbench_config.get('enabled', True):
            toolbench_trajs = self._load_dataset_trajectories(
                'toolbench', toolbench_config
            )
            trajectories.extend(toolbench_trajs)

        # Load GAIA
        if gaia_config.get('enabled', False):
            gaia_trajs = self._load_dataset_trajectories('gaia', gaia_config)
            trajectories.extend(gaia_trajs)

        # Load SWE-bench
        if swebench_config.get('enabled', False):
            swebench_trajs = self._load_dataset_trajectories(
                'swebench', swebench_config
            )
            trajectories.extend(swebench_trajs)

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

    def _load_dataset_trajectories(
        self,
        dataset_name: str,
        config: Dict[str, Any]
    ) -> List[Trajectory]:
        """
        Load trajectories for a single dataset (ToolBench, GAIA, or SWE-bench).

        Supports two source modes:
        - source: "json" - Load from pre-sampled JSON file
        - source: "huggingface" or "local" (default) - Load from source

        Args:
            dataset_name: Name of the dataset (toolbench, gaia, swebench)
            config: Dataset configuration

        Returns:
            List of Trajectory objects
        """
        source = config.get('source', 'local')
        num_traj = config.get('num_trajectories', 100)
        filters = config.get('filters', {})

        print(f"📊 Loading {dataset_name.upper()} trajectories...")
        print(f"   Source: {source}")

        trajectories = []

        if source == 'json':
            # Load from pre-sampled JSON file
            json_path = config.get('json_path')
            if not json_path:
                print(f"   ❌ ERROR: json_path required for source='json'")
                return []

            # Resolve relative path from project root
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            full_path = project_root / json_path

            if not full_path.exists():
                print(f"   ❌ ERROR: JSON file not found: {full_path}")
                return []

            print(f"   Loading from: {json_path}")
            trajectories = load_trajectories_from_json(str(full_path))

            # Populate domain/complexity metadata if missing
            for traj in trajectories:
                if not hasattr(traj, 'domain') or not traj.domain:
                    traj.domain = classify_trajectory_domain(traj)
                if not hasattr(traj, 'complexity') or not traj.complexity:
                    traj.complexity = classify_trajectory_complexity(traj)

            # Limit to num_trajectories if specified
            if num_traj and len(trajectories) > num_traj:
                trajectories = trajectories[:num_traj]

        else:
            # Load from HuggingFace or local files
            if dataset_name == 'toolbench':
                trajectories = load_toolbench_trajectories(
                    max_trajectories=num_traj,
                    min_steps=filters.get('min_steps', 2),
                    max_steps=filters.get('max_steps', 20),
                    filter_successful=filters.get('filter_successful', False),
                    require_parameters_all_positions=filters.get(
                        'require_parameters_all_positions', False
                    ),
                    require_tool_diversity=filters.get(
                        'require_tool_diversity', False
                    ),
                    random_seed=config.get('random_seed', 42)
                )
            elif dataset_name == 'gaia':
                trajectories = load_gaia_trajectories(
                    max_trajectories=num_traj,
                    min_steps=filters.get('min_steps', 1),
                    max_steps=filters.get('max_steps', 100),
                    random_seed=config.get('random_seed', 42)
                )
            elif dataset_name == 'swebench':
                trajectories = load_swebench_trajectories(
                    max_trajectories=num_traj,
                    min_steps=filters.get('min_steps', 4),
                    max_steps=filters.get('max_steps', 30),
                    filter_successful=filters.get('filter_successful', True),
                    random_seed=config.get('random_seed', 42)
                )

        print(f"   ✓ Loaded {len(trajectories)} {dataset_name} trajectories")
        print()
        return trajectories

    def _store_trajectories(self, trajectories: List[Trajectory]):
        """
        Store trajectories in MongoDB with experiment_id.

        Trajectories are tagged with experiment_id so they can be cleaned up
        per-experiment. This also ensures each experiment gets fresh trajectories
        loaded with the current loader (important after bug fixes).
        """
        # Use experiment_id from config (already set in __init__)
        # This is either from config or generated from config hash

        # Create experiment record in MongoDB (if it doesn't exist)
        experiment_config = {
            "experiment_id": self.experiment_id,
            "name": self.experiment_info.get('name', self.experiment_id),
            "description": self.experiment_info.get('description', ''),
            "config": self.config
        }

        # Check if experiment already exists
        existing_experiment = self.storage.get_experiment(self.experiment_id)
        if existing_experiment:
            print(f"   ℹ️  Experiment already exists: {self.experiment_id}")
            created_at = existing_experiment.get('created_at', 'unknown')
            print(f"   ℹ️  Using existing experiment (created: {created_at})")
        else:
            self.storage.create_experiment(experiment_config)
            print(f"   ✓ Created experiment: {self.experiment_id}")

        # Store trajectories WITH experiment_id
        stored_count = 0
        cache_hits = 0

        for traj in trajectories:
            # Convert Trajectory object to dict WITH experiment_id
            traj_dict = traj.to_dict()
            traj_dict["experiment_id"] = self.experiment_id
            traj_dict["is_perturbed"] = False

            # Check if already exists for THIS experiment
            existing = self.storage.get_trajectory_by_experiment(
                traj_dict["trajectory_id"],
                self.experiment_id
            )
            if existing:
                cache_hits += 1
            else:
                self.storage.save_trajectory(trajectory=traj_dict)
                stored_count += 1

            if (stored_count + cache_hits) % 10 == 0:
                total = stored_count + cache_hits
                print(
                    f"   ... processed {total}/{len(trajectories)} "
                    f"(new: {stored_count}, cached: {cache_hits})"
                )

        print(f"   ✓ Stored {stored_count} new trajectories")
        print(f"   ✓ Cache hits: {cache_hits}")
        print()

        # Verify data in cache for this experiment
        print("🔍 Verifying experiment trajectories...")
        try:
            total_for_exp = self.storage.trajectories.count_documents({
                "experiment_id": self.experiment_id,
                "is_perturbed": False
            })
            print(f"   ✓ Trajectories for this experiment: {total_for_exp}")
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
            if step1.content:
                print(f"  First step content: {step1.content[:80]}...")
            else:
                print("  First step content: [empty]")
            if step1.tool_name:
                print(f"  Tool: {step1.tool_name}")
        print("=" * 70)
        print()

    def _phase_generate_perturbations(self):
        """Phase 2: Generate perturbed trajectories."""
        print("🔧 PHASE: GENERATE PERTURBATIONS")
        print("=" * 70)
        print()

        # Import perturbation generator
        from src.perturbations.generator import PerturbationGenerator

        # Get perturbation config
        pert_config = self.config.get("perturbations", {})
        mode = pert_config.get("mode", "static")
        conditions = pert_config.get("conditions", [])
        batch_size = pert_config.get("batch_size", 10)

        # Get quality scoring config (nested inside perturbations)
        quality_config = pert_config.get("quality_scoring", {})
        quality_scoring_enabled = quality_config.get("enabled", False)
        scorer = None

        if quality_scoring_enabled:
            from src.scoring.quality_scorer import PerturbationQualityScorer
            scorer = PerturbationQualityScorer(
                model_config=quality_config.get("judge_model", {}),
                metrics=quality_config.get("metrics", {}),
                tier_thresholds=quality_config.get("tier_thresholds", {})
            )
            print("Quality scoring: ENABLED")
        else:
            print("Quality scoring: disabled")

        print(f"Perturbation mode: {mode}")
        print(f"Conditions to generate: {len(conditions)}")
        print(f"Batch size: {batch_size} (memory-efficient storage)")
        print()

        # Load original trajectories from MongoDB
        print("📥 Loading original trajectories...")

        # Get original trajectories used in this experiment
        # Strategy: Find unique original_trajectory_ids from existing
        # perturbations, OR if no perturbations exist yet, load from
        # cache based on experiment config
        existing_perturbations = self.storage.get_perturbations_by_experiment(
            self.experiment_id
        )

        if existing_perturbations:
            # Load originals from existing perturbations
            original_ids = list(
                set([p["original_trajectory_id"]
                     for p in existing_perturbations])
            )
            num_existing = len(existing_perturbations)
            print(f"   Found {num_existing} existing perturbations")
            print(f"   Loading {len(original_ids)} original trajectories...")

            trajectories_dict = []
            for traj_id in original_ids:
                traj = self.storage.get_trajectory(traj_id)
                if traj:
                    trajectories_dict.append(traj)
        else:
            # No existing perturbations, need to load fresh trajectories
            # Load trajectories for THIS experiment
            print("   No existing perturbations found")
            print(
                f"   Loading trajectories for experiment: "
                f"{self.experiment_id}..."
            )

            # Load trajectories tagged with this experiment_id
            trajectories_dict = list(self.storage.trajectories.find({
                "experiment_id": self.experiment_id,
                "is_perturbed": False
            }))

            if not trajectories_dict:
                print(
                    f"   ⚠️  No trajectories found for experiment "
                    f"{self.experiment_id}"
                )
                print(
                    "   ℹ️  Make sure to run 'load_trajectories' "
                    "phase first"
                )
                return

        print(f"   ✓ Loaded {len(trajectories_dict)} trajectories")
        print()

        if not trajectories_dict:
            print(
                "❌ No trajectories found. "
                "Run 'load_trajectories' phase first."
            )
            return

        # Convert dicts to Trajectory objects
        from src.data.schema import Trajectory

        # Remove MongoDB-specific fields before converting to Trajectory
        mongodb_fields = [
            "_id", "experiment_id", "is_perturbed",
            "created_at", "updated_at", "stored_at"
        ]
        for t in trajectories_dict:
            for field in mongodb_fields:
                if field in t:
                    del t[field]

        trajectories = [Trajectory.from_dict(t) for t in trajectories_dict]

        # Initialize perturbation generator
        datasets_config = self.config.get("datasets", {})
        toolbench_config = datasets_config.get("toolbench", {})
        random_seed = toolbench_config.get("random_seed", 42)
        generator = PerturbationGenerator(random_seed=random_seed)

        num_trajs = len(trajectories)
        print(f"🔀 Generating perturbations for {num_trajs} trajectories...")
        print()

        # BATCHED GENERATION + STORAGE to avoid OOM
        # Store in batches instead of accumulating all in memory
        batch = []
        batch_scores = []  # Quality scores for batch
        success_count = 0
        failed_count = 0
        stored_count = 0
        cache_hits = 0
        scored_count = 0

        for i, traj in enumerate(trajectories):
            # Extract system prompt from trajectory metadata (if available)
            system_prompt = traj.metadata.get("system_prompt", None)

            # Generate perturbations for this trajectory
            for condition in conditions:
                ptype = condition["type"]
                position = condition["position"]

                try:
                    perturbed = generator.generate_perturbation(
                        trajectory=traj,
                        perturbation_type=ptype,
                        position=position,
                        system_prompt=system_prompt,
                        mode=mode
                    )

                    if perturbed:
                        # Score quality inline if enabled
                        quality_score = None
                        if scorer:
                            try:
                                score_input = {
                                    "original_step_content": perturbed.original_step_content,
                                    "perturbed_step_content": perturbed.perturbed_step_content,
                                    "perturbation_type": perturbed.perturbation_type,
                                    "perturbation_position": perturbed.perturbation_position,
                                    "perturbation_metadata": perturbed.perturbation_metadata
                                }
                                quality_score = scorer.score_perturbation(score_input)
                                scored_count += 1
                            except Exception as e:
                                print(f"   ⚠️  Quality scoring failed: {e}")

                        batch.append(perturbed)
                        batch_scores.append(quality_score)
                        success_count += 1

                        # Store batch when it reaches batch_size
                        if len(batch) >= batch_size:
                            new_stored, new_cached = self._store_perturbation_batch(
                                batch, self.experiment_id, generator,
                                quality_scores=batch_scores
                            )
                            stored_count += new_stored
                            cache_hits += new_cached
                            batch.clear()
                            batch_scores.clear()
                    else:
                        failed_count += 1

                except Exception as e:
                    print(
                        f"   ⚠️  Failed to generate {ptype}/{position} "
                        f"for {traj.trajectory_id}: {e}"
                    )
                    failed_count += 1

            if (i + 1) % 10 == 0:
                total_stored = stored_count + cache_hits
                msg = f"   ... processed {i + 1}/{len(trajectories)} trajectories (stored: {total_stored})"
                if scorer:
                    msg += f" (scored: {scored_count})"
                print(msg)

        # Store remaining batch
        if batch:
            new_stored, new_cached = self._store_perturbation_batch(
                batch, self.experiment_id, generator,
                quality_scores=batch_scores
            )
            stored_count += new_stored
            cache_hits += new_cached
            batch.clear()
            batch_scores.clear()

        print()
        print(f"   ✓ Generated {success_count} perturbations")
        if scorer:
            print(f"   ✓ Scored {scored_count} perturbations")
        if failed_count > 0:
            print(f"   ⚠️  Failed: {failed_count}")
        print()

        # Primary selection (after all perturbations stored)
        # Config is nested inside perturbations
        primary_config = pert_config.get("primary_selection", {})
        if primary_config.get("enabled", False):
            print()
            print("=" * 70)
            print("🎯 PRIMARY SELECTION")
            print("=" * 70)
            print()

            from src.sampling.primary_selector import PrimarySelector

            selector = PrimarySelector(
                total_primary=primary_config.get("total_primary", 600),
                tier_distribution=primary_config.get("tier_distribution"),
                min_per_condition=primary_config.get("min_per_condition", 40),
                random_seed=primary_config.get("random_seed", 42)
            )

            # Load all perturbations for this experiment
            all_perturbations = list(
                self.storage.get_perturbations_by_experiment(self.experiment_id)
            )

            if all_perturbations:
                # Select primaries (use conditions from parent perturbations config)
                primary_ids = selector.select(
                    all_perturbations,
                    conditions=pert_config.get("conditions")
                )

                # Mark primaries in DB
                if primary_ids:
                    self._mark_primaries(primary_ids, all_perturbations, selector)
                    print(f"   ✓ Selected {len(primary_ids)} primary perturbations")

                    # Generate and save selection stats
                    stats = selector.get_selection_stats(all_perturbations, primary_ids)
                    self._save_primary_selection_report(stats)
            else:
                print("   ⚠️  No perturbations found for primary selection")

        print("=" * 70)
        print("✅ PERTURBATION GENERATION COMPLETE")
        print("=" * 70)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Total perturbations: {success_count}")
        print(f"New stored: {stored_count}")
        print(f"Cache hits: {cache_hits}")
        if scorer:
            print(f"Quality scored: {scored_count}")
        print()

    def _store_perturbation_batch(
        self,
        batch: List,
        experiment_id: str,
        generator,
        quality_scores: Optional[List[dict]] = None
    ) -> tuple[int, int]:
        """
        Store a batch of perturbations to MongoDB.

        Args:
            batch: List of perturbation objects
            experiment_id: Experiment identifier
            generator: PerturbationGenerator instance
            quality_scores: Optional list of quality scores (same order as batch)

        Returns:
            Tuple of (new_stored_count, cache_hits_count)
        """
        stored = 0
        cached = 0

        # Ensure quality_scores list matches batch size
        if quality_scores is None:
            quality_scores = [None] * len(batch)

        for perturbed, quality_score in zip(batch, quality_scores):
            # Store perturbed trajectory with experiment_id
            perturbed_traj_dict = perturbed.perturbed_trajectory.to_dict()
            perturbed_traj_dict["experiment_id"] = experiment_id
            perturbed_traj_dict["is_perturbed"] = True

            # Check if already exists for this experiment
            existing = self.storage.get_trajectory_by_experiment(
                perturbed_traj_dict["trajectory_id"],
                experiment_id
            )
            if not existing:
                self.storage.save_trajectory(perturbed_traj_dict)

            # Create perturbation record
            # (links experiment → original → perturbed)
            orig_traj_id = perturbed.original_trajectory.trajectory_id
            pert_traj_id = perturbed.perturbed_trajectory.trajectory_id
            perturbation_id = generator.get_perturbation_id(
                trajectory_id=orig_traj_id,
                perturbation_type=perturbed.perturbation_type,
                position=perturbed.perturbation_position
            )

            perturbation_record = {
                "perturbation_id": perturbation_id,
                "experiment_id": experiment_id,
                "original_trajectory_id": orig_traj_id,
                "perturbed_trajectory_id": pert_traj_id,
                "perturbation_type": perturbed.perturbation_type,
                "perturbation_position": perturbed.perturbation_position,
                "perturbed_step_number": perturbed.perturbed_step_number,
                "perturbation_metadata": perturbed.perturbation_metadata,
                "original_step_content": perturbed.original_step_content,
                "perturbed_step_content": perturbed.perturbed_step_content,
            }

            # Add quality score if available
            if quality_score:
                perturbation_record["quality_score"] = quality_score
                perturbation_record["quality_tier"] = quality_score.get("quality_tier")

            # Initialize primary selection fields
            perturbation_record["is_primary_for_experiment"] = False
            perturbation_record["primary_selection_reason"] = None

            # Check if perturbation already exists
            existing_pert = self.storage.get_perturbation(perturbation_id)
            if existing_pert:
                cached += 1
            else:
                self.storage.save_perturbation(perturbation_record)
                stored += 1

        return stored, cached

    def _mark_primaries(
        self,
        primary_ids: List[str],
        all_perturbations: List[dict],
        selector
    ):
        """
        Mark selected perturbations as primary in MongoDB.

        Args:
            primary_ids: List of perturbation_ids to mark as primary
            all_perturbations: All perturbations (for stats lookup)
            selector: PrimarySelector instance
        """
        primary_set = set(primary_ids)

        for p in all_perturbations:
            if p["perturbation_id"] in primary_set:
                # Update in MongoDB
                self.storage.db["perturbations"].update_one(
                    {"perturbation_id": p["perturbation_id"]},
                    {
                        "$set": {
                            "is_primary_for_experiment": True,
                            "primary_selection_reason": "quota_fill"
                        }
                    }
                )

    def _save_primary_selection_report(self, stats: dict):
        """
        Save primary selection distribution report.

        Args:
            stats: Selection statistics from PrimarySelector
        """
        from pathlib import Path
        import json

        results_dir = Path("results/primary_selection")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON stats
        stats_path = results_dir / f"selection_stats_{self.experiment_id}.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Generate markdown report
        report_path = results_dir / f"distribution_report_{self.experiment_id}.md"
        report = self._generate_distribution_report(stats)
        with open(report_path, "w") as f:
            f.write(report)

        print(f"   ✓ Saved selection report to {results_dir}")

    def _generate_distribution_report(self, stats: dict) -> str:
        """
        Generate markdown distribution report.

        Args:
            stats: Selection statistics

        Returns:
            Markdown report string
        """
        lines = [
            "# Primary Sample Distribution Report",
            "",
            "## Overall",
            f"- Total primary perturbations: {stats['total_selected']}",
            f"- Unique trajectories: {stats['unique_trajectories']}",
            "",
            "## By Perturbation Type",
            "| Type | Count | % |",
            "|------|-------|---|",
        ]

        total = stats["total_selected"] or 1
        for ptype, count in sorted(stats.get("by_type", {}).items()):
            pct = (count / total) * 100
            lines.append(f"| {ptype} | {count} | {pct:.1f}% |")

        lines.extend([
            "",
            "## By Position",
            "| Position | Count | % |",
            "|----------|-------|---|",
        ])

        for pos, count in sorted(stats.get("by_position", {}).items()):
            pct = (count / total) * 100
            lines.append(f"| {pos} | {count} | {pct:.1f}% |")

        lines.extend([
            "",
            "## By Condition (Type x Position)",
            "| Condition | Count |",
            "|-----------|-------|",
        ])

        for cond, count in sorted(stats.get("by_condition", {}).items()):
            lines.append(f"| {cond} | {count} |")

        lines.extend([
            "",
            "## By Quality Tier",
            "| Tier | Count | % |",
            "|------|-------|---|",
        ])

        for tier, count in sorted(stats.get("by_tier", {}).items()):
            pct = (count / total) * 100
            lines.append(f"| {tier} | {count} | {pct:.1f}% |")

        lines.extend([
            "",
            f"*Generated: {stats.get('selection_timestamp', 'N/A')}*"
        ])

        return "\n".join(lines)

    def _phase_validate_perturbations(self):
        """
        Phase 2b: Validate generated perturbations.

        Checks for quality issues:
        - No template fallback usage
        - Planning perturbations are semantic
        - Position and type coverage
        - Uniqueness
        """
        print("✅ PHASE: VALIDATE PERTURBATIONS")
        print("=" * 70)
        print()

        from src.perturbations.validator import (
            PerturbationValidator,
            load_perturbations_from_files,
        )

        # Load perturbations from JSON files
        perturbed_dir = Path("data/perturbed")

        if not perturbed_dir.exists():
            print(f"❌ Perturbation directory not found: {perturbed_dir}")
            print("   Run 'perturb' phase first to generate perturbations")
            return

        print(f"📁 Loading perturbations from: {perturbed_dir}")
        perturbations = load_perturbations_from_files(perturbed_dir)

        if not any(perturbations.values()):
            print("❌ No perturbation files found!")
            print("   Run 'perturb' phase first")
            return

        # Run validation
        validator = PerturbationValidator()
        all_pass = validator.print_report(perturbations, verbose=self.verbose)

        print()
        if all_pass:
            print("=" * 70)
            print("✅ VALIDATION PASSED - Dataset is ready for annotation")
            print("=" * 70)
        else:
            print("=" * 70)
            print("⚠️  VALIDATION ISSUES FOUND - Review before proceeding")
            print("=" * 70)
        print()

    def _phase_annotate(self):
        """Phase 3: Human annotation interface."""
        print("✍️  PHASE: ANNOTATION")
        print("=" * 70)
        print()

        # Import annotation components
        from src.annotation.tools import AnnotationInterface, load_annotation
        import random

        # Get annotation configuration
        annotation_config = self.config.get('annotation', {})
        num_samples = annotation_config.get('num_samples', 25)
        annotator_id = annotation_config.get('annotator_id', 'default')
        sampling_strategy = annotation_config.get('sampling_strategy', 'random')
        skip_annotated = annotation_config.get('skip_annotated', True)
        random_seed = annotation_config.get('random_seed', 42)

        print(f"📊 Configuration:")
        print(f"   Samples to annotate: {num_samples}")
        print(f"   Annotator ID: {annotator_id}")
        print(f"   Sampling strategy: {sampling_strategy}")
        print(f"   Skip already annotated: {skip_annotated}")
        print()

        # Load perturbations that have judge evaluations
        print("📥 Loading perturbations with judge evaluations...")

        # Get all perturbations for this experiment
        all_perturbations = list(self.storage.db['perturbations'].find({
            'experiment_id': self.experiment_id
        }))

        if not all_perturbations:
            print("❌ No perturbations found for this experiment")
            return

        print(f"   Found {len(all_perturbations)} perturbations")

        # Filter to those with judge evaluations (optimized: single query)
        # Get all trajectory IDs that have judge evaluations in one query
        evaluated_traj_ids = set(
            self.storage.db['judge_evaluations'].distinct(
                'trajectory_id',
                {'experiment_id': self.experiment_id}
            )
        )

        # Filter perturbations based on this set
        perturbations_with_evals = [
            pert for pert in all_perturbations
            if pert['perturbed_trajectory_id'] in evaluated_traj_ids
        ]

        print(f"   {len(perturbations_with_evals)} have judge evaluations")

        if not perturbations_with_evals:
            print("❌ No perturbations with judge evaluations found")
            print("   Run the 'judge' phase first to generate evaluations")
            return

        # Filter out already annotated if requested (check MongoDB)
        if skip_annotated:
            # Get set of annotated perturbation IDs from MongoDB
            annotated_ids = set(
                self.storage.db['annotations'].distinct(
                    'perturbation_id',
                    {'annotator_id': annotator_id}
                )
            )

            # Filter to unannotated
            unannotated = [
                pert for pert in perturbations_with_evals
                if pert['perturbation_id'] not in annotated_ids
            ]

            print(f"   {len(unannotated)} not yet annotated")
            perturbations_to_sample = unannotated
        else:
            perturbations_to_sample = perturbations_with_evals

        if not perturbations_to_sample:
            print("✅ All perturbations already annotated!")
            return

        # Sample perturbations based on strategy
        random.seed(random_seed)

        if sampling_strategy == 'random':
            # Random sampling
            sample_size = min(num_samples, len(perturbations_to_sample))
            sampled_perturbations = random.sample(perturbations_to_sample, sample_size)
            print(f"\n📋 Randomly sampled {sample_size} perturbations")

        elif sampling_strategy == 'stratified':
            # Stratified sampling by condition (type × position)
            from collections import defaultdict

            by_condition = defaultdict(list)
            for pert in perturbations_to_sample:
                condition = f"{pert['perturbation_type']}_{pert['perturbation_position']}"
                by_condition[condition].append(pert)

            # Sample evenly from each condition
            samples_per_condition = max(1, num_samples // len(by_condition))
            sampled_perturbations = []

            for condition, perts in by_condition.items():
                sample_size = min(samples_per_condition, len(perts))
                sampled_perturbations.extend(random.sample(perts, sample_size))

            # If we need more, add randomly
            if len(sampled_perturbations) < num_samples:
                remaining = [p for p in perturbations_to_sample if p not in sampled_perturbations]
                additional = min(num_samples - len(sampled_perturbations), len(remaining))
                sampled_perturbations.extend(random.sample(remaining, additional))

            print(f"\n📋 Stratified sample: {len(sampled_perturbations)} perturbations")
            print(f"   Conditions: {len(by_condition)}")

        else:
            print(f"❌ Unknown sampling strategy: {sampling_strategy}")
            return

        # Show distribution
        from collections import Counter
        type_counts = Counter(p['perturbation_type'] for p in sampled_perturbations)
        pos_counts = Counter(p['perturbation_position'] for p in sampled_perturbations)

        print(f"\n📊 Sample distribution:")
        print(f"   By type: {dict(type_counts)}")
        print(f"   By position: {dict(pos_counts)}")
        print()

        # Run annotation interface
        if self.dry_run:
            print("🔍 DRY RUN: Would annotate the following perturbations:")
            for i, pert in enumerate(sampled_perturbations, 1):
                pert_id = pert['perturbation_id']
                ptype = pert['perturbation_type']
                ppos = pert['perturbation_position']
                print(f"   {i}. {pert_id} ({ptype}, {ppos})")
            print()
            return

        # Create annotation interface
        interface = AnnotationInterface(storage=self.storage)

        # Batch annotate
        print("=" * 70)
        print("🎯 STARTING ANNOTATION SESSION")
        print("=" * 70)
        print(f"You will annotate {len(sampled_perturbations)} perturbations")
        print()
        print("For each perturbation, you will answer:")
        print("  1. Did the perturbation cause task failure? (yes/no)")
        print("  2. How many errors occurred after the perturbation? (count)")
        print()
        input("Press Enter to begin...")
        print()

        # Annotate in batches
        perturbation_ids = [p['perturbation_id'] for p in sampled_perturbations]
        completed = interface.batch_annotate(perturbation_ids, annotator_id)

        # Summary
        print("\n" + "=" * 70)
        print("✅ ANNOTATION SESSION COMPLETE")
        print("=" * 70)
        print(f"Completed: {len(completed)}/{len(sampled_perturbations)} annotations")
        print()

        if completed:
            tcs_scores = [ann.compute_tcs() for ann in completed]
            print(f"True Criticality Scores:")
            print(f"   Mean: {sum(tcs_scores) / len(tcs_scores):.2f}")
            print(f"   Min: {min(tcs_scores):.2f}")
            print(f"   Max: {max(tcs_scores):.2f}")

        print()

    def _phase_evaluate_judges(self):
        """Phase 4: Run judge evaluations."""
        print("⚖️  PHASE: EVALUATE JUDGES")
        print("=" * 70)
        print()

        # Import judge components
        from src.judges.claude_judge import create_claude_judge
        from src.judges.gpt_oss_judge import create_gpt_oss_judge
        from src.judges.evaluator import JudgeEvaluator

        # Get judge configuration
        judges_config = self.config.get('judges', {})
        models_config = judges_config.get('models', [])
        samples_per_trajectory = judges_config.get('samples_per_trajectory', 3)

        if not models_config:
            print("❌ No judges configured in experiment config")
            return

        # Create judge instances
        judges = []
        for model_config in models_config:
            provider = model_config.get('provider', 'aws_bedrock')
            name = model_config.get('name', 'unknown')

            try:
                if 'claude' in name.lower():
                    judge = create_claude_judge(model_config)
                    judges.append(judge)
                elif 'gpt' in name.lower() or 'oss' in name.lower():
                    judge = create_gpt_oss_judge(model_config)
                    judges.append(judge)
                else:
                    print(f"⚠️  Unknown judge type: {name} (skipping)")

            except Exception as e:
                print(f"❌ Failed to create judge {name}: {e}")

        if not judges:
            print("❌ No judges could be initialized")
            return

        print(f"✓ Initialized {len(judges)} judges: {[j.name for j in judges]}")
        print()

        # Create evaluator
        evaluator = JudgeEvaluator(
            storage=self.storage,
            judges=judges,
            batch_size=10,
            rate_limit_delay=1.0,
            samples_per_trajectory=samples_per_trajectory
        )

        # Run evaluation
        results = evaluator.evaluate_experiment(
            experiment_id=self.experiment_id,
            resume=True,
            dry_run=self.dry_run
        )

        # Print summary
        print("\n" + "=" * 70)
        print("✅ JUDGE EVALUATION COMPLETE")
        print("=" * 70)

        for judge_name, result in results.items():
            print(f"\n{judge_name}:")
            print(f"   Evaluated: {result.total_evaluated}")
            print(f"   Successful: {result.successful}")
            print(f"   Failed: {result.failed}")
            print(f"   Average score: {result.average_score:.1f}")
            print(f"   Total time: {result.total_time_seconds:.1f}s")
            print(f"   Total tokens: {result.total_tokens:,}")

        print()

    def _phase_compute_ccg(self):
        """Phase 5: Compute CCG metrics."""
        print("📊 PHASE: COMPUTE CCG")
        print("=" * 70)
        print()

        # Import CCG components
        from src.metrics.ccg import compute_ccg_analysis
        from pathlib import Path

        # Get CCG configuration
        ccg_config = self.config.get('ccg', {})
        judges_to_analyze = ccg_config.get('judges', None)

        # If not specified, analyze all judges from the judge config
        if not judges_to_analyze:
            judges_config = self.config.get('judges', {})
            models_config = judges_config.get('models', [])
            judges_to_analyze = [m['name'] for m in models_config]

        if not judges_to_analyze:
            print("❌ No judges specified for CCG analysis")
            return

        print(f"🔍 Analyzing judges: {', '.join(judges_to_analyze)}")
        print()

        # Compute CCG for each judge
        results_dir = Path(f"results/{self.experiment_id}")
        results_dir.mkdir(parents=True, exist_ok=True)

        for judge_name in judges_to_analyze:
            print(f"\n{'=' * 70}")
            print(f"⚖️  ANALYZING: {judge_name}")
            print(f"{'=' * 70}\n")

            try:
                # Compute CCG analysis
                analysis = compute_ccg_analysis(
                    experiment_id=self.experiment_id,
                    judge_name=judge_name,
                    storage=self.storage
                )

                # Print summary
                analysis.print_summary()

                # Save to CSV
                if not self.dry_run:
                    csv_path = results_dir / f"ccg_results_{judge_name}.csv"
                    analysis.save_csv(csv_path)

                    # Also save summary as JSON
                    import json
                    summary_path = results_dir / f"ccg_summary_{judge_name}.json"
                    summary_data = {
                        'experiment_id': analysis.experiment_id,
                        'judge_name': judge_name,
                        'total_results': len(analysis.results),
                        'summary': analysis.summary,
                        'by_type': analysis.by_type,
                        'by_position': analysis.by_position,
                        'by_condition': analysis.by_condition,
                        'statistical_tests': analysis.statistical_tests
                    }

                    # Convert numpy types to native Python types for JSON serialization
                    summary_data = self._convert_for_json(summary_data)

                    with open(summary_path, 'w') as f:
                        json.dump(summary_data, f, indent=2)

                    print(f"\n✅ Saved summary to {summary_path}")

            except Exception as e:
                print(f"❌ Error analyzing {judge_name}: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 70)
        print("✅ CCG COMPUTATION COMPLETE")
        print("=" * 70)
        print(f"\nResults saved to: {results_dir}")
        print()

    def _phase_analyze(self):
        """Phase 6: Analysis (deprecated - use visualize phase)."""
        print("📈 PHASE: ANALYSIS")
        print("=" * 70)
        print()
        print("⚠️  This phase is deprecated.")
        print("   Use 'visualize' phase instead for generating visualizations.")
        print()

    def _phase_visualize(self):
        """Phase 7: Generate visualizations from CCG results."""
        print("📊 PHASE: VISUALIZE")
        print("=" * 70)
        print()

        # Determine results directory
        results_dir = Path('results') / self.experiment_id

        if not results_dir.exists():
            print(f"❌ Results directory not found: {results_dir}")
            print("   Run 'ccg' phase first to generate results")
            return

        # Check for CCG result files
        result_files = list(results_dir.glob('ccg_results_*.csv'))

        if not result_files:
            print(f"❌ No CCG result files found in {results_dir}")
            print("   Run 'ccg' phase first to compute CCG scores")
            return

        print(f"📁 Results directory: {results_dir}")
        print(f"   Found {len(result_files)} result file(s):")
        for rf in result_files:
            judge_name = rf.stem.replace('ccg_results_', '')
            print(f"   - {judge_name}")

        # Get visualization config from experiment config
        viz_config = self.config.get('visualization', {})
        figsize = viz_config.get('figsize', [12, 8])
        dpi = viz_config.get('dpi', 300)
        formats = viz_config.get('formats', ['png', 'pdf'])

        print(f"\n⚙️  Visualization settings:")
        print(f"   Figure size: {figsize[0]}x{figsize[1]}")
        print(f"   DPI: {dpi}")
        print(f"   Formats: {', '.join(formats)}")

        if self.dry_run:
            print("\n🔍 DRY RUN: Would generate visualizations")
            return

        try:
            # Import here to avoid circular dependency
            from visualization.plots import generate_visualizations

            print(f"\n🎨 Generating visualizations...")

            # Generate all visualizations
            generated_files = generate_visualizations(
                results_dir=str(results_dir),
                figsize=tuple(figsize),
                dpi=dpi,
                save_formats=formats
            )

            # Print summary
            total_files = sum(len(files) for files in generated_files.values())
            print(f"\n✅ Successfully generated {total_files} visualization file(s)")

            for viz_type, file_list in generated_files.items():
                if file_list:
                    print(f"   {viz_type.replace('_', ' ').title()}: {len(file_list)} file(s)")

            viz_dir = results_dir / 'visualizations'
            print(f"\n📂 All visualizations saved to: {viz_dir}")

        except Exception as e:
            print(f"\n❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 70)
        print("✅ VISUALIZATION COMPLETE")
        print("=" * 70)
        print()
