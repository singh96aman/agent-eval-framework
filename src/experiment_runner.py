"""
Experiment Runner - Schema 2.0.0

7 phases: load, typing, perturb, sample, annotate, judge, compute
Compute targets defined in config.compute.targets
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

from src.storage.mongodb import MongoDBStorage

SCHEMA_VERSION = "2.0.0"

# Valid phases
PHASES = ["load", "typing", "perturb", "sample", "annotate", "judge", "compute"]


def generate_experiment_id(config: Dict[str, Any]) -> str:
    """Generate unique experiment ID from config hash."""
    config_str = json.dumps(config, sort_keys=True)
    hash_digest = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    return f"exp_{hash_digest}"


class ExperimentRunner:
    """
    Experiment runner with 6 phases.

    Phases: load, perturb, sample, annotate, judge, compute
    Compute targets: jps, tcs, od, ccg, calibration (from config)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        runner_str: str = None,
        dry_run: bool = False,
        verbose: bool = True,
    ):
        load_dotenv()

        self.config = config
        self.dry_run = dry_run
        self.verbose = verbose

        # Extract experiment info
        exp_config = config.get("experiment", {})
        self.experiment_id = exp_config.get("id") or generate_experiment_id(config)
        self.experiment_name = exp_config.get("name", "Unnamed")

        # Initialize storage
        storage_config = exp_config.get("storage", {})
        self.storage = MongoDBStorage(
            database=storage_config.get("database", "agent_judge_experiment")
        )

        # Parse runner string
        self.phases_to_run = self._parse_runner(runner_str)

        # Output directory
        compute_config = config.get("compute", {})
        output_config = compute_config.get("output", {})
        self.output_dir = Path(output_config.get("dir", "results")) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _parse_runner(self, runner_str: str) -> List[str]:
        """
        Parse runner string into list of phases.

        Examples:
            "load,perturb" -> ["load", "perturb"]
            "judge,compute" -> ["judge", "compute"]
        """
        phases = []

        if not runner_str:
            return phases

        for part in runner_str.split(","):
            part = part.strip().lower()

            if part in PHASES:
                phases.append(part)
            else:
                raise ValueError(f"Unknown phase: {part}. Valid phases: {PHASES}")

        return phases

    def run(self):
        """Run the experiment."""
        print("=" * 70)
        print(f"EXPERIMENT: {self.experiment_name}")
        print("=" * 70)
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Phases: {', '.join(self.phases_to_run) or 'none'}")
        print(f"Dry Run: {self.dry_run}")
        print(f"Output: {self.output_dir}")
        print("=" * 70)
        print()

        # Phase handlers
        phase_handlers = {
            "load": self._phase_load,
            "typing": self._phase_typing,
            "perturb": self._phase_perturb,
            "sample": self._phase_sample,
            "annotate": self._phase_annotate,
            "judge": self._phase_judge,
            "compute": self._phase_compute,
        }

        for idx, phase in enumerate(self.phases_to_run, 1):
            print()
            print("=" * 70)
            print(f"PHASE {idx}/{len(self.phases_to_run)}: {phase.upper()}")
            print("=" * 70)
            print()

            handler = phase_handlers[phase]
            handler()

        # Cleanup
        if self.storage:
            self.storage.close()

        print()
        print("=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)

    # =========================================================================
    # PHASES
    # =========================================================================

    def _phase_load(self):
        """Load trajectories with stratified sampling and provenance tracking."""
        from src.data.loaders import (
            load_stratified_sample,
            save_sampling_manifest,
        )

        phase_config = self.config.get("phases", {}).get("load", {})

        print("Loading trajectories with stratified sampling...")
        print(
            f"   Seed: {phase_config.get('provenance', {}).get('seed', phase_config.get('sampling', {}).get('seed', 42))}"
        )

        # Use the new stratified sampling function
        all_trajectories, manifest = load_stratified_sample(
            config=phase_config,
            experiment_id=self.experiment_id,
        )

        # Print summary
        print("\n   === SAMPLING SUMMARY ===")
        print(f"   Total trajectories: {len(all_trajectories)}")
        if manifest.counts.get("by_benchmark"):
            print("   By benchmark:")
            for bench, count in manifest.counts["by_benchmark"].items():
                print(f"      {bench}: {count}")
        if manifest.counts.get("by_complexity"):
            print("   By complexity:")
            for comp, count in manifest.counts["by_complexity"].items():
                print(f"      {comp}: {count}")

        if self.dry_run:
            print("\nDRY RUN: Skipping database save")
            return

        # Save manifest for provenance
        manifest_path = self.output_dir / "sampling_manifest.json"
        save_sampling_manifest(manifest, str(manifest_path))

        # Save to storage
        saved = 0
        for traj in all_trajectories:
            traj_dict = traj.to_dict() if hasattr(traj, "to_dict") else traj
            traj_dict["experiment_id"] = self.experiment_id
            self.storage.save_trajectory(traj_dict)
            saved += 1

        print(f"\nSaved {saved} trajectories to database")
        print(f"Manifest saved to: {manifest_path}")

    def _phase_typing(self):
        """
        Type trajectories with enriched fields.

        Adds step roles, dependencies, artifacts, perturbable slots,
        and critical path scores to each trajectory.
        """
        from src.typing.typer import TrajectoryTyper

        phase_config = self.config.get("phases", {}).get("typing", {})

        # Check if typing is enabled
        if not phase_config.get("enabled", True):
            print("Typing phase disabled in config")
            return

        # Get input source - either from config or current experiment
        input_config = self.config.get("input", {})
        source_experiment = input_config.get("source_experiment", self.experiment_id)

        print(f"Loading trajectories from experiment: {source_experiment}")

        # Load trajectories from source experiment
        trajectories = list(
            self.storage.trajectories.find({"experiment_id": source_experiment})
        )

        if not trajectories:
            print(f"No trajectories found for experiment: {source_experiment}")
            print("Run 'load' phase first or check source_experiment in config.")
            return

        print(f"Loaded {len(trajectories)} trajectories")

        # Initialize typer
        use_spacy = (
            phase_config.get("pass1", {})
            .get("components", {})
            .get("entity_extractor", {})
            .get("use_spacy", False)
        )
        typer = TrajectoryTyper(use_spacy=use_spacy)

        if self.dry_run:
            print("\nDRY RUN: Would type trajectories")
            # Type one as sample
            if trajectories:
                sample = typer.type_trajectory(trajectories[0])
                print(f"\nSample typed trajectory: {sample.trajectory_id}")
                print(f"   Steps: {sample.num_steps}")
                print(f"   Environment: {sample.environment_type}")
                if sample.steps:
                    print(f"   Step 1 role: {sample.steps[0].step_role}")
                    print(f"   Step 1 slots: {len(sample.steps[0].perturbable_slots)}")
            return

        # Type all trajectories
        print("\nTyping trajectories...")
        typed_trajectories = []
        errors = []

        for i, traj in enumerate(trajectories):
            try:
                typed = typer.type_trajectory(traj)
                typed_trajectories.append(typed)

                if (i + 1) % 50 == 0:
                    print(f"   Typed {i + 1}/{len(trajectories)} trajectories")
            except Exception as e:
                traj_id = traj.get("trajectory_id", f"index_{i}")
                errors.append({"trajectory_id": traj_id, "error": str(e)})
                print(f"   ERROR typing {traj_id}: {e}")

        print(f"\nTyped {len(typed_trajectories)} trajectories")
        if errors:
            print(f"Errors: {len(errors)}")

        # Get output collection name
        output_config = phase_config.get("output", {})
        collection_name = output_config.get("collection", "typed_trajectories")

        # Create typed_trajectories collection if needed
        typed_collection = self.storage.db[collection_name]

        # Create indexes
        typed_collection.create_index([("trajectory_id", 1)], unique=True)
        typed_collection.create_index([("experiment_id", 1)])
        typed_collection.create_index([("benchmark", 1)])

        # Save typed trajectories
        print(f"\nSaving to collection: {collection_name}")
        saved = 0
        for typed in typed_trajectories:
            doc = typed.to_dict()
            doc["experiment_id"] = self.experiment_id
            doc["source_experiment"] = source_experiment

            try:
                typed_collection.replace_one(
                    {"trajectory_id": doc["trajectory_id"]},
                    doc,
                    upsert=True,
                )
                saved += 1
            except Exception as e:
                print(f"   Error saving {doc['trajectory_id']}: {e}")

        print(f"Saved {saved} typed trajectories")

        # Print summary
        print("\n   === TYPING SUMMARY ===")
        if typed_trajectories:
            roles = {}
            total_slots = 0
            for typed in typed_trajectories:
                for step in typed.steps:
                    roles[step.step_role] = roles.get(step.step_role, 0) + 1
                    total_slots += len(step.perturbable_slots)

            print(f"   Total steps typed: {sum(roles.values())}")
            print("   Step roles:")
            for role, count in sorted(roles.items()):
                print(f"      {role}: {count}")
            print(f"   Total perturbable slots: {total_slots}")
            print(f"   Avg slots per trajectory: {total_slots / len(typed_trajectories):.1f}")

    def _phase_perturb(self):
        """
        Generate controlled perturbations using Section 3 V2 generator.

        Features:
        - Three perturbation classes (placebo, fine-grained, coarse-grained)
        - Five families (data_reference, parameter, tool_selection, structural, terminal_flag)
        - Batch-level balancing for target distribution
        - Heuristic-only QC (no LLM)
        - Impact derivation from Section 2 heuristics
        """
        from src.perturbations.generator_v2 import (
            PerturbationGeneratorV2,
            generate_perturbations_for_batch,
        )
        from src.perturbations.balancer import PerturbationBalancer, balance_perturbation_batch
        from src.perturbations.storage import (
            PerturbationStorage,
            PerturbationExporter,
            build_index_from_perturbations,
            group_by_benchmark,
        )
        from src.typing.schema import TypedTrajectory

        phase_config = self.config.get("phases", {}).get("perturb", {})
        targets_config = self.config.get("targets", {})
        output_config = self.config.get("output", {})

        # Get source experiment
        input_config = self.config.get("input", {})
        source_experiment = input_config.get("experiment_id", self.experiment_id)
        source_collection = input_config.get("collection", "typed_trajectories")

        print(f"Loading typed trajectories from: {source_experiment}")
        print(f"Collection: {source_collection}")

        # Load typed trajectories
        typed_collection = self.storage.db[source_collection]
        raw_docs = list(typed_collection.find({"experiment_id": source_experiment}))

        if not raw_docs:
            print(f"No typed trajectories found for experiment: {source_experiment}")
            return

        # Convert to TypedTrajectory objects
        trajectories = []
        for doc in raw_docs:
            doc.pop("_id", None)
            try:
                typed = TypedTrajectory.from_dict(doc)
                trajectories.append(typed)
            except Exception as e:
                print(f"   Error loading trajectory {doc.get('trajectory_id')}: {e}")

        print(f"Loaded {len(trajectories)} typed trajectories")

        # Get target distribution
        total_target = targets_config.get("total_perturbations", 1500)
        per_trajectory = targets_config.get("per_trajectory", {}).get("target", 3)
        class_weights = targets_config.get("by_class", {
            "placebo": 0.20,
            "fine_grained": 0.50,
            "coarse_grained": 0.30,
        })

        print(f"\nTarget: {total_target} perturbations")
        print(f"Distribution: {class_weights}")
        print(f"Per trajectory: {per_trajectory}")

        if self.dry_run:
            print("\nDRY RUN: Would generate perturbations")
            # Generate sample for one trajectory
            if trajectories:
                generator = PerturbationGeneratorV2(
                    random_seed=phase_config.get("random_seed", 42),
                    enable_qc=True,
                )
                sample_results = generator.generate_for_trajectory(
                    trajectories[0],
                    target_count=per_trajectory,
                    class_weights=class_weights,
                )
                print(f"\nSample: Generated {len(sample_results)} for first trajectory")
                for record, _ in sample_results:
                    print(f"   - {record.perturbation_class}/{record.perturbation_type}")
            return

        # Initialize generator
        generation_config = self.config.get("generation", {})
        llm_config = generation_config.get("llm", {})

        # Get LLM client for Claude-based generators (paraphrase, wrong_plan)
        llm_client = None
        if llm_config.get("provider") == "bedrock":
            try:
                from src.llm.bedrock_client import get_bedrock_client
                llm_client = get_bedrock_client(log_calls=self.verbose)
                print(f"Initialized LLM client for: {llm_config.get('use_for', [])}")
            except Exception as e:
                print(f"Warning: Could not initialize LLM client: {e}")
                print("Generators requiring LLM will be skipped")

        # Generate perturbations
        print(f"\nGenerating perturbations...")

        all_results, index = generate_perturbations_for_batch(
            trajectories=trajectories,
            target_per_trajectory=per_trajectory,
            random_seed=generation_config.get("random_seed", 42),
            llm_client=llm_client,
            verbose=self.verbose,
        )

        print(f"\nGenerated {len(all_results)} perturbations")

        # Extract records for balancing
        records = [record for record, _ in all_results]

        # Balance distribution
        print("\nBalancing distribution...")
        balanced_records, balance_report = balance_perturbation_batch(
            records=records,
            total_target=total_target,
            class_weights=class_weights,
            random_seed=generation_config.get("random_seed", 42),
        )

        print(f"After balancing: {len(balanced_records)} perturbations")
        print(f"Removed: {balance_report['removed']}")
        print(f"Is balanced: {balance_report['is_balanced']}")

        # Filter all_results to balanced records
        balanced_ids = {r.perturbation_id for r in balanced_records}
        balanced_results = [
            (record, traj) for record, traj in all_results
            if record.perturbation_id in balanced_ids
        ]

        # Prepare for storage
        perturbed_trajectories = []
        for record, perturbed_traj in balanced_results:
            pt_dict = {
                "perturbation_id": record.perturbation_id,
                "original_trajectory_id": record.original_trajectory_id,
                "original_trajectory": {
                    "trajectory_id": record.original_trajectory_id,
                    "benchmark": perturbed_traj.benchmark,
                },
                "perturbation_record": record.to_dict(),
                "perturbed_trajectory": perturbed_traj.to_dict(),
                "experiment_id": self.experiment_id,
            }
            perturbed_trajectories.append(pt_dict)

        # Build final index
        final_index = build_index_from_perturbations(
            perturbed_trajectories,
            output_dir=output_config.get("json", {}).get("dir", "data/perturbed"),
        )

        # Save to MongoDB
        mongodb_config = output_config.get("mongodb", {})
        if mongodb_config:
            storage = PerturbationStorage(
                mongodb_client=self.storage.client,
                collection_name=mongodb_config.get("collection", "perturbed_trajectories"),
                index_collection_name=mongodb_config.get("index_collection", "perturbation_index"),
            )

            saved = storage.save_perturbations_batch(perturbed_trajectories, self.experiment_id)
            print(f"\nSaved {saved} perturbations to MongoDB")

            storage.save_index(final_index, self.experiment_id)
            print("Saved perturbation index to MongoDB")

        # Export to JSON
        json_config = output_config.get("json", {})
        if json_config:
            exporter = PerturbationExporter(json_config.get("dir", "data/perturbed"))
            grouped = group_by_benchmark(perturbed_trajectories)
            paths = exporter.export_all(grouped, final_index)

            print(f"\nExported to JSON:")
            for key, path in paths.items():
                print(f"   {key}: {path}")

        # Print summary
        print("\n" + "=" * 50)
        print("PERTURBATION GENERATION SUMMARY")
        print("=" * 50)
        print(final_index.get_distribution_report())

    def _phase_sample(self):
        """Sample perturbations for annotation."""
        from src.annotation.stratified_sampler import StratifiedAnnotationSampler

        phase_config = self.config.get("phases", {}).get("sample", {})
        filter_config = phase_config.get("filter", {})

        # Load perturbations
        perturbations = list(
            self.storage.get_perturbations_by_experiment(self.experiment_id)
        )

        # Apply filters
        if filter_config.get("primary_only", False):
            perturbations = [
                p for p in perturbations if p.get("is_primary_for_experiment", False)
            ]

        print(f"Loaded {len(perturbations)} perturbations")

        if not perturbations:
            print("No perturbations found. Run 'perturb' phase first.")
            return

        # Sample
        total = phase_config.get("total", 100)
        seed = phase_config.get("seed", 42)
        stratify_by = phase_config.get(
            "stratify_by", ["perturbation_type", "position", "benchmark"]
        )

        sampler = StratifiedAnnotationSampler(
            total_samples=total,
            stratify_by=stratify_by,
            random_seed=seed,
        )

        samples = sampler.sample(perturbations)
        print(f"Sampled {len(samples)} perturbations")

        if self.dry_run:
            print("DRY RUN: Skipping database update")
            return

        # Mark samples in database
        for sample in samples:
            self.storage.db["perturbations"].update_one(
                {"perturbation_id": sample["perturbation_id"]},
                {"$set": {"selected_for_annotation": True}},
            )

        print(f"Marked {len(samples)} samples for annotation")

    def _phase_annotate(self):
        """Interactive human annotation."""
        from src.annotation.tools import AnnotationTool

        phase_config = self.config.get("phases", {}).get("annotate", {})

        # Load samples marked for annotation
        samples = list(
            self.storage.db["perturbations"].find({"selected_for_annotation": True})
        )

        if phase_config.get("skip_completed", True):
            samples = [s for s in samples if not s.get("human_annotation")]

        print(f"Found {len(samples)} samples to annotate")

        if not samples:
            print("No samples pending annotation")
            return

        if self.dry_run:
            print("DRY RUN: Skipping annotation")
            return

        # Run annotation tool
        annotator_id = phase_config.get("annotator_id", "researcher")
        metrics_config = phase_config.get("metrics", {})

        tool = AnnotationTool(
            storage=self.storage,
            annotator_id=annotator_id,
            metrics=metrics_config,
        )

        tool.annotate_batch(samples)

    def _phase_judge(self):
        """Run judge evaluation on perturbations."""
        from src.judges.parallel_evaluator import ParallelJudgeEvaluator
        from src.judges.claude_judge import create_claude_judge

        phase_config = self.config.get("phases", {}).get("judge", {})
        models_config = phase_config.get("models", [])
        filter_config = phase_config.get("filter", {})

        if not models_config:
            print("No judge models configured")
            return

        # Load perturbations
        if filter_config.get("annotated_only", False):
            perturbations = list(
                self.storage.db["perturbations"].find({"selected_for_annotation": True})
            )
        else:
            perturbations = list(
                self.storage.get_perturbations_by_experiment(self.experiment_id)
            )

        print(f"Loaded {len(perturbations)} perturbations for evaluation")

        if not perturbations:
            print("No perturbations found")
            return

        if self.dry_run:
            print("DRY RUN: Skipping judge evaluation")
            return

        # Create judges
        judges = []
        for model_config in models_config:
            name = model_config.get("name", "unknown")
            try:
                judge = create_claude_judge(model_config)
                judges.append(judge)
                print(f"   Created judge: {name}")
            except Exception as e:
                print(f"   Failed to create judge {name}: {e}")

        if not judges:
            print("No judges could be initialized")
            return

        # Run parallel evaluation
        parallelism = phase_config.get("parallelism", 2)
        evaluator = ParallelJudgeEvaluator(
            judges=judges,
            storage=self.storage,
            experiment_id=self.experiment_id,
            parallelism=parallelism,
        )

        results = evaluator.evaluate_batch(perturbations)
        print(f"Evaluated {len(results)} perturbations")

    def _phase_compute(self):
        """Run all compute targets from config."""
        compute_config = self.config.get("compute", {})
        targets = compute_config.get("targets", [])

        if not targets:
            print("No compute targets configured")
            return

        print(f"Compute targets: {', '.join(targets)}")
        print()

        # Compute target handlers
        compute_handlers = {
            "jps": self._compute_jps,
            "tcs": self._compute_tcs,
            "od": self._compute_od,
            "ccg": self._compute_ccg,
            "calibration": self._compute_calibration,
        }

        for idx, target in enumerate(targets, 1):
            print("-" * 50)
            print(f"COMPUTE [{idx}/{len(targets)}]: {target.upper()}")
            print("-" * 50)

            handler = compute_handlers.get(target)
            if handler:
                handler()
            else:
                print(f"Unknown compute target: {target}")

            print()

    # =========================================================================
    # COMPUTE TARGETS
    # =========================================================================

    def _compute_jps(self):
        """Compute Judge Penalty Score from judge outputs."""
        compute_config = self.config.get("compute", {}).get("jps", {})
        formula = compute_config.get("formula", "100 - overall_score")

        print(f"Formula: JPS = {formula}")

        # Load judge evaluations
        evaluations = list(
            self.storage.db["judge_outputs"].find({"experiment_id": self.experiment_id})
        )
        print(f"Loaded {len(evaluations)} judge evaluations")

        if not evaluations:
            print("No evaluations found. Run 'judge' phase first.")
            return

        if self.dry_run:
            print("DRY RUN: Would compute JPS")
            return

        # Compute JPS for each evaluation
        computed = 0
        for eval_doc in evaluations:
            metrics = eval_doc.get("metrics", {})
            overall_score = metrics.get("overall_score", 0)

            # Apply formula (simple case: 100 - overall_score)
            jps = 100 - overall_score

            # Update document
            self.storage.db["judge_outputs"].update_one(
                {"_id": eval_doc["_id"]}, {"$set": {"jps": jps}}
            )
            computed += 1

        print(f"Computed JPS for {computed} evaluations")

    def _compute_tcs(self):
        """Compute True Criticality Score from annotations."""
        compute_config = self.config.get("compute", {}).get("tcs", {})
        formula = compute_config.get(
            "formula",
            "(task_success_degradation * 50) + (subsequent_error_rate * 10) + (criticality_rating * 8)",
        )
        validate_config = compute_config.get("validate", {})

        print(f"Formula: TCS = {formula}")

        # Load annotated perturbations
        perturbations = list(
            self.storage.db["perturbations"].find(
                {"selected_for_annotation": True, "human_annotation": {"$exists": True}}
            )
        )

        print(f"Loaded {len(perturbations)} annotated perturbations")

        if not perturbations:
            print("No annotated perturbations found. Run 'annotate' phase first.")
            return

        if self.dry_run:
            print("DRY RUN: Would compute TCS")
            return

        # Compute TCS
        computed = 0
        for p in perturbations:
            annotation = p.get("human_annotation", {})

            tsd = annotation.get("task_success_degradation", 0)
            ser = annotation.get("subsequent_error_rate", 0)
            cr = annotation.get("criticality_rating", 1)

            tcs = (tsd * 50) + (ser * 10) + (cr * 8)

            self.storage.db["perturbations"].update_one(
                {"_id": p["_id"]}, {"$set": {"tcs": tcs}}
            )
            computed += 1

        print(f"Computed TCS for {computed} perturbations")

        # Run validation if enabled
        if validate_config.get("enabled", False):
            print("\nRunning TCS validation...")
            self._validate_tcs(validate_config)

    def _validate_tcs(self, config: Dict):
        """Validate TCS correlation with human judgments."""
        min_correlation = config.get("min_correlation_with_human", 0.4)
        print(f"   Min correlation threshold: {min_correlation}")

    def _compute_od(self):
        """Compute Outcome Degradation by grading trajectories."""
        from src.replay.od_scorer import ODScorer

        compute_config = self.config.get("compute", {}).get("od", {})
        validate_config = compute_config.get("validate", {})

        # Load perturbations
        perturbations = list(
            self.storage.db["perturbations"].find({"selected_for_annotation": True})
        )
        print(f"Loaded {len(perturbations)} perturbations")

        # Check existing
        already_computed = sum(1 for p in perturbations if p.get("od"))
        print(f"Already computed: {already_computed}")
        print(f"Need to compute: {len(perturbations) - already_computed}")

        if self.dry_run:
            print("DRY RUN: Would compute OD")
            return

        # Initialize scorer
        scorer = ODScorer(compute_config)

        # Compute OD
        results = scorer.compute_batch(
            perturbations,
            self.storage,
            batch_size=compute_config.get("batch_size", 20),
            resume=True,
        )

        print(f"Computed OD for {len(results)} perturbations")

        # Run validation if enabled
        if validate_config.get("enabled", False):
            print("\nRunning OD validation...")
            self._validate_od(validate_config)

    def _validate_od(self, config: Dict):
        """Validate OD distribution."""
        perturbations = list(
            self.storage.db["perturbations"].find({"selected_for_annotation": True})
        )
        with_od = [p for p in perturbations if p.get("od")]

        coverage = len(with_od) / len(perturbations) if perturbations else 0
        min_coverage = config.get("min_coverage", 0.7)

        print(f"   Coverage: {coverage:.1%} (min: {min_coverage:.1%})")

        if with_od:
            od_values = [p["od"].get("value", 0) for p in with_od]
            variance = np.var(od_values)
            min_variance = config.get("min_variance", 0.01)

            print(f"   Variance: {variance:.4f} (min: {min_variance})")
            print(f"   Mean OD: {np.mean(od_values):.3f}")

    def _compute_ccg(self):
        """Compute Criticality Calibration Gap."""
        from src.metrics.ccg_calculator import CCGCalculator

        compute_config = self.config.get("compute", {}).get("ccg", {})
        formula = compute_config.get("formula", "(JPS - TCS) / TCS")
        aggregations = compute_config.get("aggregations", ["overall"])

        print(f"Formula: CCG = {formula}")
        print(f"Aggregations: {aggregations}")

        # Load data with both JPS and TCS
        perturbations = list(
            self.storage.db["perturbations"].find(
                {"selected_for_annotation": True, "tcs": {"$exists": True}}
            )
        )

        evaluations = list(
            self.storage.db["judge_outputs"].find(
                {"experiment_id": self.experiment_id, "jps": {"$exists": True}}
            )
        )

        print(f"Perturbations with TCS: {len(perturbations)}")
        print(f"Evaluations with JPS: {len(evaluations)}")

        if self.dry_run:
            print("DRY RUN: Would compute CCG")
            return

        # Match and compute CCG
        calculator = CCGCalculator()
        results = calculator.compute(perturbations, evaluations, aggregations)

        # Save results
        output_path = self.output_dir / "ccg_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Saved CCG results to {output_path}")

        # Print summary
        if "overall" in results:
            overall = results["overall"]
            print(f"\nOverall CCG: {overall.get('mean_ccg', 'N/A'):.3f}")

    def _compute_calibration(self):
        """Compute calibration metrics (JPS vs OD)."""
        from src.analysis.calibration import CalibrationAnalyzer

        compute_config = self.config.get("compute", {}).get("calibration", {})

        x_metric = compute_config.get("x", "od")
        y_metric = compute_config.get("y", "jps")
        metrics = compute_config.get("metrics", ["spearman_r", "pearson_r"])

        print(f"Analyzing: {y_metric} vs {x_metric}")
        print(f"Metrics: {metrics}")

        # Load perturbations with OD
        perturbations = list(
            self.storage.db["perturbations"].find(
                {"selected_for_annotation": True, "od": {"$exists": True}}
            )
        )

        # Load evaluations with JPS
        evaluations = list(
            self.storage.db["judge_outputs"].find(
                {"experiment_id": self.experiment_id, "jps": {"$exists": True}}
            )
        )

        print(f"Perturbations with OD: {len(perturbations)}")
        print(f"Evaluations with JPS: {len(evaluations)}")

        if self.dry_run:
            print("DRY RUN: Would compute calibration")
            return

        # Initialize analyzer
        analyzer = CalibrationAnalyzer(compute_config)

        # Run analysis
        report = analyzer.analyze(evaluations, perturbations)

        # Print summary
        analyzer.print_summary(report)

        # Save results
        output_path = self.output_dir / "calibration_results.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Saved calibration results to {output_path}")
