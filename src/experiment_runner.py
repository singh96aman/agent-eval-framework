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
            # New phases for Task 05
            'sample_annotation': 'sample_annotation',
            'annotate': 'annotate',  # Interactive human annotation
            'judge_parallel': 'judge_parallel',
            'validate_tcs': 'validate_tcs',
            'ccg_v2': 'compute_ccg_v2',
            'compute_ccg_v2': 'compute_ccg_v2',
            # RQ1: Consequentiality Calibration phases
            'od': 'compute_od',
            'compute_od': 'compute_od',
            'od_validate': 'validate_od',
            'validate_od': 'validate_od',
            'calibration': 'compute_calibration',
            'compute_calibration': 'compute_calibration',
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
            # New phases for Task 05
            'sample_annotation': self._phase_sample_annotation,
            'judge_parallel': self._phase_judge_parallel,
            'validate_tcs': self._phase_validate_tcs,
            'compute_ccg_v2': self._phase_compute_ccg_v2,
            # RQ1: Consequentiality Calibration phases
            'compute_od': self._phase_compute_od,
            'validate_od': self._phase_validate_od,
            'compute_calibration': self._phase_compute_calibration,
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

        # Get quality scoring config (nested inside perturbations)
        quality_config = pert_config.get("quality_scoring", {})
        quality_scoring_enabled = quality_config.get("enabled", False)
        scorer = None

        if quality_scoring_enabled:
            from src.scoring.quality_scorer import create_quality_scorer
            from src.llm import get_bedrock_client
            # Initialize central Bedrock client with logging if enabled
            log_bedrock = self.config.get("execution", {}).get("log_bedrock", False)
            get_bedrock_client(log_calls=log_bedrock)
            scorer = create_quality_scorer(quality_config)
            judge_mode = quality_config.get("judge_mode", "batch")
            print(f"Quality scoring: ENABLED (mode: {judge_mode})")
            if judge_mode == "batch":
                print("   → 1 LLM call per perturbation (all metrics)")
            else:
                print("   → 5 LLM calls per perturbation (1 per metric)")
        else:
            print("Quality scoring: disabled")

        print(f"Perturbation mode: {mode}")
        print(f"Conditions to generate: {len(conditions)}")
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
        generated_count = 0
        failed_count = 0
        stored_count = 0
        skipped_count = 0  # Already scored in DB
        scored_count = 0
        needs_scoring_count = 0  # Exists but no score

        for i, traj in enumerate(trajectories):
            system_prompt = traj.metadata.get("system_prompt", None)

            # Track what needs processing for this trajectory
            to_generate = []  # (condition, perturbation_id) - need to generate & score
            to_score = []     # (perturbation_id, existing_record) - exists, needs score
            already_done = 0  # Already has score

            for condition in conditions:
                ptype = condition["type"]
                position = condition["position"]

                # Build perturbation_id to check DB
                pert_id = generator.get_perturbation_id(
                    trajectory_id=traj.trajectory_id,
                    perturbation_type=ptype,
                    position=position,
                    experiment_id=self.experiment_id
                )

                # Check if exists in DB for THIS experiment
                existing = self.storage.get_perturbation_for_experiment(
                    pert_id, self.experiment_id
                )

                if existing and existing.get("quality_score"):
                    # Already scored - skip entirely
                    already_done += 1
                elif existing:
                    # Exists but no score - need to score
                    to_score.append((pert_id, existing))
                else:
                    # Doesn't exist - need to generate
                    to_generate.append((condition, pert_id))

            skipped_count += already_done

            # Generate new perturbations
            new_perturbations = []  # (perturbation_id, perturbed_obj)
            for condition, pert_id in to_generate:
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
                        new_perturbations.append((pert_id, perturbed))
                        generated_count += 1
                    else:
                        failed_count += 1
                except Exception:
                    print(f"   Warning: Perturbation failed for {ptype}/{position} in {traj.trajectory_id}")
                    failed_count += 1

            # Build list of things to score (new + existing without score)
            to_score_inputs = []

            # Add new perturbations
            for pert_id, perturbed in new_perturbations:
                to_score_inputs.append({
                    "perturbation_id": pert_id,
                    "original_step_content": perturbed.original_step_content,
                    "perturbed_step_content": perturbed.perturbed_step_content,
                    "perturbation_type": perturbed.perturbation_type,
                    "perturbation_position": perturbed.perturbation_position,
                    "perturbation_metadata": perturbed.perturbation_metadata,
                    "_is_new": True,
                    "_perturbed_obj": perturbed
                })

            # Add existing without score
            for pert_id, existing in to_score:
                to_score_inputs.append({
                    "perturbation_id": pert_id,
                    "original_step_content": existing.get("original_step_content", ""),
                    "perturbed_step_content": existing.get("perturbed_step_content", ""),
                    "perturbation_type": existing.get("perturbation_type", ""),
                    "perturbation_position": existing.get("perturbation_position", ""),
                    "perturbation_metadata": existing.get("perturbation_metadata", {}),
                    "_is_new": False,
                    "_perturbed_obj": None
                })
                needs_scoring_count += 1

            # Score all that need scoring
            if to_score_inputs and scorer:
                try:
                    scores = scorer.score_batch(to_score_inputs, batch_size=len(to_score_inputs))
                    scored_count += len(scores)
                except Exception as e:
                    print(f"   ⚠️  Scoring failed for {traj.trajectory_id}: {e}")
                    scores = [None] * len(to_score_inputs)
            else:
                scores = [None] * len(to_score_inputs)

            # Store/update results
            new_stored = 0
            for item, score in zip(to_score_inputs, scores):
                pert_id = item["perturbation_id"]

                if item["_is_new"]:
                    # New perturbation - store it
                    perturbed = item["_perturbed_obj"]
                    self._store_single_perturbation(
                        perturbed, self.experiment_id, generator, score
                    )
                    new_stored += 1
                else:
                    # Existing - update with score
                    if score:
                        self.storage.perturbations.update_one(
                            {"perturbation_id": pert_id},
                            {"$set": {
                                "quality_score": score,
                                "quality_tier": score.get("quality_tier")
                            }}
                        )

            stored_count += new_stored

            # Log trajectory completion
            total_for_traj = already_done + len(to_score_inputs)
            if total_for_traj > 0:
                print(f"   ✓ {traj.trajectory_id}: {new_stored} new, {len(to_score)} updated, {already_done} skipped")

            if (i + 1) % 10 == 0:
                print(f"   ... processed {i + 1}/{len(trajectories)} trajectories")

        print()
        print(f"   ✓ Generated {generated_count} new perturbations")
        print(f"   ✓ Skipped {skipped_count} (already scored)")
        if needs_scoring_count > 0:
            print(f"   ✓ Updated {needs_scoring_count} existing (added scores)")
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
        print(f"New perturbations: {generated_count}")
        print(f"Stored: {stored_count}")
        print(f"Skipped (already scored): {skipped_count}")
        if scorer:
            print(f"Quality scored: {scored_count}")
        print()

    def _store_single_perturbation(
        self,
        perturbed,
        experiment_id: str,
        generator,
        quality_score: Optional[dict] = None
    ):
        """Store a single perturbation to MongoDB."""
        # Store perturbed trajectory
        perturbed_traj_dict = perturbed.perturbed_trajectory.to_dict()
        perturbed_traj_dict["experiment_id"] = experiment_id
        perturbed_traj_dict["is_perturbed"] = True

        existing = self.storage.get_trajectory_by_experiment(
            perturbed_traj_dict["trajectory_id"], experiment_id
        )
        if not existing:
            self.storage.save_trajectory(perturbed_traj_dict)

        # Create perturbation record
        orig_traj_id = perturbed.original_trajectory.trajectory_id
        pert_traj_id = perturbed.perturbed_trajectory.trajectory_id
        perturbation_id = generator.get_perturbation_id(
            trajectory_id=orig_traj_id,
            perturbation_type=perturbed.perturbation_type,
            position=perturbed.perturbation_position,
            experiment_id=experiment_id
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
            "is_primary_for_experiment": False,
            "primary_selection_reason": None,
        }

        if quality_score:
            perturbation_record["quality_score"] = quality_score
            perturbation_record["quality_tier"] = quality_score.get("quality_tier")

        self.storage.save_perturbation(perturbation_record)

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
                position=perturbed.perturbation_position,
                experiment_id=experiment_id
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
                # Update with quality score if we have one and it doesn't
                if quality_score and not existing_pert.get("quality_score"):
                    self.storage.perturbations.update_one(
                        {"perturbation_id": perturbation_id},
                        {"$set": {
                            "quality_score": quality_score,
                            "quality_tier": quality_score.get("quality_tier")
                        }}
                    )
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
        human_annotation_config = self.config.get('human_annotation', {})
        num_samples = annotation_config.get('num_samples', 25)
        annotator_id = annotation_config.get('annotator_id', 'default')
        sampling_strategy = annotation_config.get('sampling_strategy', 'random')
        skip_annotated = annotation_config.get('skip_annotated', True)
        random_seed = annotation_config.get('random_seed', 42)

        # First check for pre-selected samples (from sample_annotation phase)
        print("📥 Checking for pre-selected annotation samples...")
        preselected = list(self.storage.db['perturbations'].find({
            'selected_for_annotation': True
        }))

        if preselected:
            print(f"   Found {len(preselected)} pre-selected samples")
            return self._run_annotation_on_preselected(preselected, annotator_id, skip_annotated)

        # Fall back to legacy sampling behavior
        print("   No pre-selected samples found, using legacy sampling")
        print()

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

    def _run_annotation_on_preselected(
        self,
        preselected: list,
        annotator_id: str,
        skip_annotated: bool
    ):
        """
        Run annotation on pre-selected samples from MongoDB.

        This method handles samples that were flagged by the sample_annotation phase
        with `selected_for_annotation=True`.

        Args:
            preselected: List of perturbation documents from MongoDB
            annotator_id: ID of the annotator
            skip_annotated: Whether to skip already-annotated samples
        """
        from src.annotation.tools import AnnotationInterface
        from collections import Counter

        # Filter out already annotated if requested
        if skip_annotated:
            annotated_ids = set(
                self.storage.db['annotations'].distinct(
                    'perturbation_id',
                    {'annotator_id': annotator_id}
                )
            )
            unannotated = [
                p for p in preselected
                if p['perturbation_id'] not in annotated_ids
            ]
            print(f"   Already annotated: {len(preselected) - len(unannotated)}")
            print(f"   Remaining to annotate: {len(unannotated)}")
            samples_to_annotate = unannotated
        else:
            samples_to_annotate = preselected

        if not samples_to_annotate:
            print("\n✅ All pre-selected samples already annotated!")
            return

        # Show distribution
        type_counts = Counter(p['perturbation_type'] for p in samples_to_annotate)
        pos_counts = Counter(p['perturbation_position'] for p in samples_to_annotate)

        print(f"\n📊 Sample distribution:")
        print(f"   By type: {dict(type_counts)}")
        print(f"   By position: {dict(pos_counts)}")
        print()

        # Dry run check
        if self.dry_run:
            print("🔍 DRY RUN: Would annotate the following perturbations:")
            for i, pert in enumerate(samples_to_annotate[:10], 1):
                pert_id = pert['perturbation_id']
                ptype = pert['perturbation_type']
                ppos = pert['perturbation_position']
                print(f"   {i}. {pert_id} ({ptype}, {ppos})")
            if len(samples_to_annotate) > 10:
                print(f"   ... and {len(samples_to_annotate) - 10} more")
            return

        # Create annotation interface
        interface = AnnotationInterface(storage=self.storage)

        # Start annotation session
        print("=" * 70)
        print("🎯 STARTING ANNOTATION SESSION")
        print("=" * 70)
        print(f"You will annotate {len(samples_to_annotate)} perturbations")
        print()
        print("For each perturbation, you will see:")
        print("  - The original trajectory steps")
        print("  - Side-by-side comparison of original vs perturbed step")
        print("  - Steps that occurred after the perturbation")
        print()
        print("Then answer:")
        print("  1. Did the perturbation cause task failure? (yes/no)")
        print("  2. How many errors occurred after the perturbation? (count)")
        print()
        input("Press Enter to begin...")
        print()

        # Run batch annotation
        perturbation_ids = [p['perturbation_id'] for p in samples_to_annotate]
        completed = interface.batch_annotate(perturbation_ids, annotator_id)

        # Summary
        print("\n" + "=" * 70)
        print("✅ ANNOTATION SESSION COMPLETE")
        print("=" * 70)
        print(f"Completed: {len(completed)}/{len(samples_to_annotate)} annotations")
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

    # ======================================================================
    # TASK 05: NEW PHASES FOR JUDGE EVALUATION & CCG ANALYSIS
    # ======================================================================

    def _phase_sample_annotation(self):
        """Phase: Sample perturbations for human annotation."""
        print("📋 PHASE: SAMPLE FOR ANNOTATION")
        print("=" * 70)
        print()

        from src.annotation.stratified_sampler import StratifiedAnnotationSampler
        from pathlib import Path

        # Get config
        annotation_config = self.config.get("human_annotation", {})
        source_config = self.config.get("source_experiment", {})

        total_samples = annotation_config.get("total_samples", 100)
        output_dir = annotation_config.get("output_dir", "data/annotations")
        random_seed = annotation_config.get("random_seed", 42)

        # Determine source experiment
        source_experiment_id = source_config.get("experiment_id", self.experiment_id)

        print(f"Source experiment: {source_experiment_id}")
        print(f"Target samples: {total_samples}")
        print(f"Output directory: {output_dir}")
        print()

        # Load perturbations from source experiment
        print("📥 Loading perturbations...")
        perturbations = list(self.storage.get_perturbations_by_experiment(
            source_experiment_id
        ))

        # Filter to primary only if configured
        use_primary_only = source_config.get("use_primary_only", True)
        if use_primary_only:
            perturbations = [
                p for p in perturbations
                if p.get("is_primary_for_experiment", False)
            ]
            print(f"   Filtered to primary: {len(perturbations)} perturbations")
        else:
            print(f"   Total perturbations: {len(perturbations)}")

        if not perturbations:
            print("❌ No perturbations found for sampling")
            return

        # Run stratified sampling
        sampler = StratifiedAnnotationSampler(perturbations, random_seed=random_seed)
        selected = sampler.sample(total=total_samples)

        # Print distribution report
        sampler.print_distribution_report(selected)

        # Flag in MongoDB and export backup to JSON
        if not self.dry_run:
            # Flag in MongoDB (primary storage)
            sampler.flag_in_mongodb(selected, self.storage, self.experiment_id)

            # Export to JSON (backup)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = f"{output_dir}/annotation_samples.json"
            sampler.export_for_annotation(selected, output_path)

            print(f"\n📁 Local backup exported to: {output_path}")
        else:
            print("\n🔍 DRY RUN: Would flag samples in MongoDB and export backup")

        print("\n" + "=" * 70)
        print("✅ ANNOTATION SAMPLING COMPLETE")
        print("=" * 70)
        print("\nNext step: Run 'annotate' phase to begin human annotation")
        print("  python main.py --config judge_evaluation_ccg --runner annotate")
        print()

    def _phase_judge_parallel(self):
        """Phase: Run parallel judge evaluation."""
        print("⚖️  PHASE: PARALLEL JUDGE EVALUATION")
        print("=" * 70)
        print()

        from src.judges.parallel_evaluator import ParallelJudgeEvaluator
        from src.judges.claude_judge import create_claude_judge

        # Get configurations
        judges_config = self.config.get("judges", {})
        source_config = self.config.get("source_experiment", {})
        models_config = judges_config.get("models", [])

        if not models_config:
            print("❌ No judges configured")
            return

        # Determine source experiment for perturbations
        source_experiment_id = source_config.get("experiment_id", self.experiment_id)

        # Load perturbations
        print("📥 Loading perturbations...")

        # Check if we should only evaluate annotated samples
        use_annotated_only = source_config.get("use_annotated_only", True)

        if use_annotated_only:
            # Only evaluate samples that were selected for annotation (with ground truth TCS)
            perturbations = list(self.storage.db['perturbations'].find({
                'selected_for_annotation': True
            }))
            print(f"   Loaded {len(perturbations)} annotated perturbations (ground truth TCS)")
        else:
            # Load all from source experiment
            perturbations = list(self.storage.get_perturbations_by_experiment(
                source_experiment_id
            ))

            # Filter to primary only if configured
            use_primary_only = source_config.get("use_primary_only", True)
            if use_primary_only:
                perturbations = [
                    p for p in perturbations
                    if p.get("is_primary_for_experiment", False)
                ]
            print(f"   Loaded {len(perturbations)} perturbations")

        if not perturbations:
            print("❌ No perturbations to evaluate")
            return

        # Evaluate with each judge
        for model_config in models_config:
            name = model_config.get("name", "unknown")
            print(f"\n{'=' * 70}")
            print(f"EVALUATING WITH: {name}")
            print(f"{'=' * 70}")

            try:
                # Create judge
                if 'claude' in name.lower():
                    judge = create_claude_judge(model_config)
                else:
                    print(f"   ⚠️  Unsupported judge type: {name}")
                    continue

                # Create parallel evaluator
                evaluator = ParallelJudgeEvaluator(
                    judge=judge,
                    storage=self.storage,
                    config=judges_config
                )

                if self.dry_run:
                    print("   🔍 DRY RUN: Would evaluate perturbations")
                    continue

                # Run evaluation
                results = evaluator.evaluate_all(
                    perturbations=perturbations,
                    experiment_id=self.experiment_id,
                    resume=True
                )

                # Print summary
                summary = evaluator.get_evaluation_summary(self.experiment_id)
                print(f"\nSummary for {name}:")
                print(f"   Evaluated: {summary.get('count', 0)}")
                print(f"   Average score: {summary.get('avg_score', 0):.1f}")
                print(f"   Average JPS: {summary.get('avg_jps', 0):.1f}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 70)
        print("✅ PARALLEL JUDGE EVALUATION COMPLETE")
        print("=" * 70)
        print()

    def _phase_validate_tcs(self):
        """Phase: Validate heuristic TCS against human annotations."""
        print("📊 PHASE: VALIDATE HEURISTIC TCS")
        print("=" * 70)
        print()

        from src.metrics.criticality_scorer import CriticalityScorer
        from pathlib import Path
        import json

        # Get config
        crit_config = self.config.get("criticality_scoring", {})
        annotation_path = crit_config.get("human_annotation_path", "data/annotations/human_tcs.json")

        # First, try to export annotations from MongoDB to the expected file
        # Only fetch annotations for perturbations selected for this experiment
        print("📥 Fetching annotations from MongoDB...")

        # Get perturbation IDs that were selected for annotation
        selected_pert_ids = self.storage.db['perturbations'].distinct(
            'perturbation_id',
            {'selected_for_annotation': True}
        )

        # Fetch only those annotations
        annotations_from_db = list(self.storage.db['annotations'].find({
            'perturbation_id': {'$in': selected_pert_ids}
        }))

        if annotations_from_db:
            print(f"   Found {len(annotations_from_db)} annotations in MongoDB")

            # Get perturbation metadata for each annotation
            export_data = []
            for ann in annotations_from_db:
                pert_id = ann.get('perturbation_id')
                pert_record = self.storage.db['perturbations'].find_one(
                    {'perturbation_id': pert_id}
                )

                if pert_record:
                    export_data.append({
                        "perturbation_id": pert_id,
                        "perturbation_type": pert_record.get('perturbation_type'),
                        "perturbation_position": pert_record.get('perturbation_position'),
                        "annotation": {
                            "task_success_degradation": ann.get('task_success_degradation'),
                            "subsequent_error_rate": ann.get('subsequent_error_rate'),
                            "tcs_score": ann.get('tcs_score'),  # Pre-computed TCS from MongoDB
                            "notes": ann.get('notes', '')
                        }
                    })

            # Export to JSON file
            Path(annotation_path).parent.mkdir(parents=True, exist_ok=True)
            with open(annotation_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"   Exported to: {annotation_path}")

        elif not Path(annotation_path).exists():
            print("❌ No annotations found in MongoDB or local file")
            print(f"   Expected file: {annotation_path}")
            print("   Run 'annotate' phase first to complete annotations")
            return

        print(f"\nLoading annotations from: {annotation_path}")

        # Create scorer in hybrid mode
        scorer = CriticalityScorer({
            "mode": "hybrid",
            "human_annotation_path": annotation_path,
            "heuristic_tcs": crit_config.get("heuristic_tcs", {})
        })

        # Print validation report
        scorer.print_validation_report()

        print("\n" + "=" * 70)
        print("✅ TCS VALIDATION COMPLETE")
        print("=" * 70)
        print()

    def _phase_compute_ccg_v2(self):
        """Phase: Compute CCG using new calculator with statistical analysis."""
        print("📊 PHASE: COMPUTE CCG (V2)")
        print("=" * 70)
        print()

        from src.metrics.criticality_scorer import CriticalityScorer
        from src.metrics.ccg_calculator import CCGCalculator
        from pathlib import Path
        import json

        # Get configurations
        source_config = self.config.get("source_experiment", {})
        crit_config = self.config.get("criticality_scoring", {})
        ccg_config = self.config.get("ccg_analysis", {})
        judges_config = self.config.get("judges", {})

        # Determine source experiment
        source_experiment_id = source_config.get("experiment_id", self.experiment_id)

        # Get judge names
        models_config = judges_config.get("models", [])
        judge_names = [m.get("name") for m in models_config]

        if not judge_names:
            print("❌ No judges configured")
            return

        print(f"Source experiment: {source_experiment_id}")
        print(f"Judges to analyze: {judge_names}")
        print()

        # Load perturbations
        print("📥 Loading perturbations...")

        # Check if we should only use annotated samples (with ground truth TCS)
        use_annotated_only = source_config.get("use_annotated_only", True)

        if use_annotated_only:
            perturbations = list(self.storage.db['perturbations'].find({
                'selected_for_annotation': True
            }))
            print(f"   Loaded {len(perturbations)} annotated perturbations")
        else:
            perturbations = list(self.storage.get_perturbations_by_experiment(
                source_experiment_id
            ))
            use_primary_only = source_config.get("use_primary_only", True)
            if use_primary_only:
                perturbations = [
                    p for p in perturbations
                    if p.get("is_primary_for_experiment", False)
                ]
            print(f"   Loaded {len(perturbations)} perturbations")

        if not perturbations:
            print("❌ No perturbations found")
            return

        # Initialize criticality scorer
        print("\n📊 Computing TCS values...")
        scorer = CriticalityScorer(crit_config)
        tcs_values = scorer.compute_batch_with_ids(perturbations)
        print(f"   Computed TCS for {len(tcs_values)} perturbations")

        # If human annotations exist, validate heuristic
        if scorer.human_annotations:
            print(f"   Human annotations loaded: {len(scorer.human_annotations)}")
            validation = scorer.validate_heuristic()
            print(f"   Heuristic correlation: {validation.get('pearson_r', 'N/A'):.3f}")

        # Initialize CCG calculator
        calculator = CCGCalculator(ccg_config)

        # Compute CCG for each judge
        output_dir = Path(ccg_config.get("output_dir", f"results/{self.experiment_id}/ccg_analysis"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for judge_name in judge_names:
            print(f"\n{'=' * 70}")
            print(f"ANALYZING: {judge_name}")
            print(f"{'=' * 70}")

            # Load judge evaluations
            evaluations = self.storage.get_judge_outputs(
                experiment_id=self.experiment_id,
                judge_name=judge_name
            )

            if not evaluations:
                print(f"   ⚠️  No evaluations found for {judge_name}")
                continue

            print(f"   Loaded {len(evaluations)} evaluations")

            # Map evaluations to perturbations
            eval_lookup = {}
            for e in evaluations:
                traj_id = e.get("trajectory_id")
                if traj_id not in eval_lookup:
                    eval_lookup[traj_id] = e

            # Build evaluation list with perturbation metadata
            eval_data = []
            for p in perturbations:
                perturbed_traj_id = p.get("perturbed_trajectory_id")
                if perturbed_traj_id in eval_lookup:
                    e = eval_lookup[perturbed_traj_id]
                    eval_data.append({
                        "perturbation_id": p.get("perturbation_id"),
                        "overall_score": e.get("overall_score", 50),
                        "perturbation_type": p.get("perturbation_type"),
                        "perturbation_position": p.get("perturbation_position"),
                        "benchmark": self._get_benchmark_from_trajectory_id(
                            p.get("original_trajectory_id", "")
                        )
                    })

            print(f"   Matched {len(eval_data)} perturbations with evaluations")

            if not eval_data:
                continue

            # Compute CCG (returns tuple with DataFrame and exclusion stats)
            df, exclusion_stats = calculator.compute_all(eval_data, tcs_values)

            # Generate report (include exclusion stats for data quality section)
            report = calculator.generate_report(df, exclusion_stats)

            # Print summary
            calculator.print_summary(report)

            # Save results
            if not self.dry_run:
                # Save raw results
                df.to_csv(output_dir / f"ccg_raw_{judge_name}.csv", index=False)

                # Save report
                report_serializable = self._convert_for_json(report)
                with open(output_dir / f"ccg_report_{judge_name}.json", 'w') as f:
                    json.dump(report_serializable, f, indent=2)

                print(f"\n   Saved results to {output_dir}")

        print("\n" + "=" * 70)
        print("✅ CCG COMPUTATION (V2) COMPLETE")
        print("=" * 70)
        print()

    def _get_benchmark_from_trajectory_id(self, traj_id: str) -> str:
        """Extract benchmark name from trajectory ID."""
        traj_id = traj_id.lower()
        if "toolbench" in traj_id:
            return "toolbench"
        elif "gaia" in traj_id:
            return "gaia"
        elif "swe" in traj_id:
            return "swebench"
        return "unknown"

    # ======================================================================
    # RQ1: CONSEQUENTIALITY CALIBRATION PHASES
    # ======================================================================

    def _phase_compute_od(self):
        """Phase: Compute Outcome Degradation for perturbations."""
        print("📊 PHASE: COMPUTE OUTCOME DEGRADATION (OD)")
        print("=" * 70)
        print()

        from src.replay.od_scorer import ODScorer

        # Get OD configuration
        od_config = self.config.get("od", {})
        source_config = self.config.get("source_experiment", {})

        if not od_config.get("enabled", True):
            print("OD computation disabled in config")
            return

        # Determine source experiment
        source_experiment_id = source_config.get(
            "experiment_id", self.experiment_id
        )

        print(f"Source experiment: {source_experiment_id}")
        print(f"Grader model: {od_config.get('grader_model', 'default')}")
        print()

        # Load perturbations
        print("📥 Loading perturbations...")
        use_primary_only = source_config.get("use_primary_only", True)
        use_annotated_only = source_config.get("use_annotated_only", False)

        if use_annotated_only:
            perturbations = list(self.storage.db['perturbations'].find({
                'selected_for_annotation': True
            }))
            print(f"   Loaded {len(perturbations)} annotated perturbations")
        else:
            perturbations = list(self.storage.get_perturbations_by_experiment(
                source_experiment_id
            ))
            if use_primary_only:
                perturbations = [
                    p for p in perturbations
                    if p.get("is_primary_for_experiment", False)
                ]
            print(f"   Loaded {len(perturbations)} perturbations")

        if not perturbations:
            print("❌ No perturbations found")
            return

        # Check how many already have OD
        already_computed = sum(1 for p in perturbations if p.get("od"))
        print(f"   Already computed: {already_computed}")
        print(f"   Need to compute: {len(perturbations) - already_computed}")
        print()

        if self.dry_run:
            print("🔍 DRY RUN: Would compute OD for perturbations")
            return

        # Initialize OD scorer
        od_config["log_calls"] = self.verbose
        scorer = ODScorer(od_config)

        # Compute OD
        print("🔢 Computing OD values...")
        batch_size = od_config.get("batch_size", 20)
        results = scorer.compute_batch(
            perturbations,
            self.storage,
            batch_size=batch_size,
            resume=True
        )

        # Print stats
        stats = scorer.get_stats()
        print(f"\n✓ Computed OD for {len(results)} perturbations")
        print(f"   Graded: {stats['graded_count']}")
        print(f"   Failed: {stats['failed_count']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")

        # Show OD distribution
        if results:
            od_values = [r.od_value for r in results]
            print(f"\nOD Distribution:")
            print(f"   Mean: {np.mean(od_values):.3f}")
            print(f"   Std:  {np.std(od_values):.3f}")
            print(f"   Min:  {min(od_values):.3f}")
            print(f"   Max:  {max(od_values):.3f}")

        print("\n" + "=" * 70)
        print("✅ OD COMPUTATION COMPLETE")
        print("=" * 70)
        print()

    def _phase_validate_od(self):
        """Phase: Validate OD distribution before calibration."""
        print("✅ PHASE: VALIDATE OD")
        print("=" * 70)
        print()

        # Get validation config
        validation_config = self.config.get("od_validation", {})
        source_config = self.config.get("source_experiment", {})
        output_dir = Path(self.config.get("calibration", {}).get(
            "output_dir", "results/rq1"
        ))

        min_coverage = validation_config.get("min_coverage", 0.7)
        min_variance = validation_config.get("min_variance", 0.1)
        spot_check_count = validation_config.get("spot_check_count", 20)

        # Determine source experiment
        source_experiment_id = source_config.get(
            "experiment_id", self.experiment_id
        )

        # Load perturbations
        print("📥 Loading perturbations...")
        use_primary_only = source_config.get("use_primary_only", True)
        use_annotated_only = source_config.get("use_annotated_only", False)

        if use_annotated_only:
            perturbations = list(self.storage.db['perturbations'].find({
                'selected_for_annotation': True
            }))
        else:
            perturbations = list(self.storage.get_perturbations_by_experiment(
                source_experiment_id
            ))
            if use_primary_only:
                perturbations = [
                    p for p in perturbations
                    if p.get("is_primary_for_experiment", False)
                ]

        print(f"   Total perturbations: {len(perturbations)}")

        # Check coverage
        with_od = [p for p in perturbations if p.get("od")]
        coverage = len(with_od) / len(perturbations) if perturbations else 0

        print(f"\n📊 OD Coverage:")
        print(f"   Perturbations with OD: {len(with_od)}")
        print(f"   Coverage: {coverage:.1%}")
        print(f"   Required: >= {min_coverage:.1%}")

        coverage_pass = coverage >= min_coverage
        print(f"   Status: {'✅ PASS' if coverage_pass else '❌ FAIL'}")

        # Check variance
        if with_od:
            od_values = [p["od"].get("value", 0) for p in with_od]
            variance = np.var(od_values)

            print(f"\n📊 OD Variance:")
            print(f"   Variance: {variance:.4f}")
            print(f"   Required: > {min_variance}")

            variance_pass = variance > min_variance
            print(f"   Status: {'✅ PASS' if variance_pass else '❌ FAIL'}")

            # Distribution stats
            print(f"\n📊 OD Distribution:")
            print(f"   Mean: {np.mean(od_values):.3f}")
            print(f"   Std:  {np.std(od_values):.3f}")
            print(f"   Min:  {min(od_values):.3f}")
            print(f"   Max:  {max(od_values):.3f}")

            # Count OD=0 cases
            zero_od = [p for p in with_od if abs(p["od"].get("value", 0)) < 0.01]
            print(f"\n📊 OD=0 Cases:")
            print(f"   Count: {len(zero_od)}")
            print(f"   Percentage: {len(zero_od) / len(with_od) * 100:.1f}%")

            # Spot check OD=0 cases
            if zero_od and spot_check_count > 0:
                print(f"\n🔍 Spot-checking {min(spot_check_count, len(zero_od))} OD=0 samples:")
                import random
                random.seed(42)
                samples = random.sample(zero_od, min(spot_check_count, len(zero_od)))

                for i, p in enumerate(samples[:5], 1):
                    od_info = p.get("od", {})
                    print(f"   {i}. {p.get('perturbation_id', 'unknown')[:50]}...")
                    print(f"      Baseline: {od_info.get('baseline_outcome', 'N/A')}")
                    print(f"      Perturbed: {od_info.get('perturbed_outcome', 'N/A')}")
        else:
            variance_pass = False

        # Save validation results
        output_dir.mkdir(parents=True, exist_ok=True)
        validation_results = {
            "total_perturbations": len(perturbations),
            "with_od": len(with_od),
            "coverage": float(coverage),
            "coverage_pass": bool(coverage_pass),
            "variance": float(variance) if with_od else 0,
            "variance_pass": bool(variance_pass),
            "od_zero_count": len(zero_od) if with_od else 0,
            "checks_passed": bool(coverage_pass and variance_pass)
        }

        import json
        validation_path = output_dir / "od_validation.json"
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"\n📁 Saved validation results to {validation_path}")

        # Final verdict
        all_pass = coverage_pass and variance_pass
        print("\n" + "=" * 70)
        if all_pass:
            print("✅ OD VALIDATION PASSED - Ready for calibration")
        else:
            print("❌ OD VALIDATION FAILED - Review issues before proceeding")
        print("=" * 70)
        print()

        return all_pass

    def _phase_compute_calibration(self):
        """Phase: Compute calibration metrics (JPS vs OD)."""
        print("📊 PHASE: COMPUTE CALIBRATION")
        print("=" * 70)
        print()

        from src.analysis.calibration import CalibrationAnalyzer

        # Get configurations
        calibration_config = self.config.get("calibration", {})
        source_config = self.config.get("source_experiment", {})
        judge_config = self.config.get("judge_source", {})

        # Determine source experiment for perturbations
        source_experiment_id = source_config.get(
            "experiment_id", self.experiment_id
        )

        # Determine judge experiment
        judge_experiment_id = judge_config.get(
            "experiment_id", self.experiment_id
        )
        judge_name = judge_config.get("judge_name", "claude-sonnet-4.5")

        print(f"Source experiment: {source_experiment_id}")
        print(f"Judge experiment: {judge_experiment_id}")
        print(f"Judge: {judge_name}")
        print()

        # Load perturbations with OD
        print("📥 Loading perturbations with OD...")
        use_primary_only = source_config.get("use_primary_only", True)
        use_annotated_only = source_config.get("use_annotated_only", False)

        if use_annotated_only:
            perturbations = list(self.storage.db['perturbations'].find({
                'selected_for_annotation': True
            }))
        else:
            perturbations = list(self.storage.get_perturbations_by_experiment(
                source_experiment_id
            ))
            if use_primary_only:
                perturbations = [
                    p for p in perturbations
                    if p.get("is_primary_for_experiment", False)
                ]

        # Filter to those with OD
        perturbations = [p for p in perturbations if p.get("od")]
        print(f"   Perturbations with OD: {len(perturbations)}")

        if not perturbations:
            print("❌ No perturbations with OD found")
            print("   Run 'od' phase first to compute OD values")
            return

        # Load judge evaluations
        print("\n📥 Loading judge evaluations...")
        evaluations = self.storage.get_judge_outputs(
            experiment_id=judge_experiment_id,
            judge_name=judge_name
        )
        print(f"   Evaluations: {len(evaluations)}")

        if not evaluations:
            print(f"❌ No evaluations found for {judge_name}")
            return

        # Initialize analyzer
        analyzer = CalibrationAnalyzer(calibration_config)

        if self.dry_run:
            print("\n🔍 DRY RUN: Would compute calibration metrics")
            return

        # Run analysis
        print("\n🔢 Computing calibration metrics...")
        report = analyzer.analyze(evaluations, perturbations)

        # Print summary
        analyzer.print_summary(report)

        # Save results
        print("\n💾 Saving results...")
        analyzer.save_results(report, evaluations, perturbations)

        print("\n" + "=" * 70)
        print("✅ CALIBRATION ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nResults saved to: {analyzer.output_dir}")
        print()
