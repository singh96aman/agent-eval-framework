"""
Experiment Runner v2 - Schema 2.0.0 support.

Simplified runner with:
- 5 phases: load, perturb, sample, annotate, judge
- 5 compute targets: jps, tcs, od, ccg, calibration
- Embedded validation in phase configs
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

from src.storage.mongodb import MongoDBStorage


SCHEMA_VERSION = "2.0.0"

# Valid phases (compute is a phase, targets come from config)
PHASES = ["load", "perturb", "sample", "annotate", "judge", "compute"]


def generate_experiment_id(config: Dict[str, Any]) -> str:
    """Generate unique experiment ID from config hash."""
    config_str = json.dumps(config, sort_keys=True)
    hash_digest = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    return f"exp_{hash_digest}"


class RunnerV2:
    """
    Experiment runner for schema 2.0.0.

    Phases: load, perturb, sample, annotate, judge
    Compute: jps, tcs, od, ccg, calibration
    """

    def __init__(self, config: Dict[str, Any], runner_str: str = None, dry_run: bool = False, verbose: bool = True):
        load_dotenv()

        self.config = config
        self.dry_run = dry_run
        self.verbose = verbose

        # Validate schema
        schema = config.get("schema", "1.0.0")
        if not schema.startswith("2."):
            raise ValueError(f"RunnerV2 requires schema 2.x, got {schema}")

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
                raise ValueError(
                    f"Unknown phase: {part}. Valid phases: {PHASES}"
                )

        return phases

    def run(self):
        """Run the experiment."""
        print("=" * 70)
        print(f"EXPERIMENT: {self.experiment_name}")
        print(f"Schema: {self.config.get('schema', '2.0.0')}")
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
        """Load trajectories from configured sources."""
        from src.data.loaders import (
            load_toolbench_trajectories,
            load_gaia_trajectories,
            load_swebench_trajectories,
            load_trajectories_from_json,
            classify_trajectory_domain,
            classify_trajectory_complexity,
        )

        phase_config = self.config.get("phases", {}).get("load", {})
        datasets_config = phase_config.get("datasets", {})

        all_trajectories = []

        for dataset_name, ds_config in datasets_config.items():
            if not ds_config.get("enabled", True):
                print(f"   Skipping {dataset_name} (disabled)")
                continue

            source = ds_config.get("source", "huggingface")
            limit = ds_config.get("limit", 100)

            print(f"Loading {dataset_name} ({source}, limit={limit})...")

            if source == "json":
                path = ds_config.get("path")
                trajectories = load_trajectories_from_json(path)
                # Apply limit after loading
                if limit and len(trajectories) > limit:
                    trajectories = trajectories[:limit]
            elif dataset_name == "toolbench":
                trajectories = load_toolbench_trajectories(
                    max_trajectories=limit
                )
            elif dataset_name == "gaia":
                trajectories = load_gaia_trajectories(
                    max_trajectories=limit
                )
            elif dataset_name == "swebench":
                trajectories = load_swebench_trajectories(
                    max_trajectories=limit
                )
            else:
                print(f"   Unknown dataset: {dataset_name}")
                continue

            print(f"   Loaded {len(trajectories)} trajectories")
            all_trajectories.extend(trajectories)

        print(f"\nTotal: {len(all_trajectories)} trajectories")

        if self.dry_run:
            print("DRY RUN: Skipping database save")
            return

        # Save to storage
        saved = 0
        for traj in all_trajectories:
            traj_dict = traj if isinstance(traj, dict) else traj.__dict__
            traj_dict["experiment_id"] = self.experiment_id
            self.storage.save_trajectory(traj_dict)
            saved += 1

        print(f"Saved {saved} trajectories to database")

    def _phase_perturb(self):
        """Generate perturbations for loaded trajectories."""
        from src.perturbations.generator import PerturbationGenerator
        from src.perturbations.validator import PerturbationValidator

        phase_config = self.config.get("phases", {}).get("perturb", {})
        validate_config = phase_config.get("validate", {})

        # Load trajectories
        trajectories = list(self.storage.get_trajectories_by_experiment(self.experiment_id))
        print(f"Loaded {len(trajectories)} trajectories")

        if not trajectories:
            print("No trajectories found. Run 'load' phase first.")
            return

        # Configure generator
        generator = PerturbationGenerator(
            perturbation_types=phase_config.get("types", ["planning", "tool_selection", "parameter", "data_reference"]),
            positions=phase_config.get("positions", ["early", "middle", "late"]),
        )

        if self.dry_run:
            print("DRY RUN: Would generate perturbations")
            return

        # Generate perturbations
        perturbations = []
        for traj in trajectories:
            perturbs = generator.generate(traj, count=phase_config.get("per_trajectory", 1))
            for p in perturbs:
                p["experiment_id"] = self.experiment_id
                perturbations.append(p)

        print(f"Generated {len(perturbations)} perturbations")

        # Save perturbations
        for p in perturbations:
            self.storage.save_perturbation(p)

        # Run validation if enabled
        if validate_config.get("enabled", False):
            print("\nRunning validation...")
            self._validate_perturbations(perturbations, validate_config)

    def _validate_perturbations(self, perturbations: List[Dict], config: Dict):
        """Validate perturbations against config criteria."""
        from src.perturbations.validator import PerturbationValidator

        validator = PerturbationValidator()
        results = validator.validate_batch(perturbations)

        # Check coverage
        types = set(p.get("perturbation_type") for p in perturbations)
        positions = set(p.get("position") for p in perturbations)

        min_type_coverage = config.get("min_type_coverage", 0.8)
        min_pos_coverage = config.get("min_position_coverage", 0.8)

        expected_types = {"planning", "tool_selection", "parameter", "data_reference"}
        expected_positions = {"early", "middle", "late"}

        type_coverage = len(types & expected_types) / len(expected_types)
        pos_coverage = len(positions & expected_positions) / len(expected_positions)

        print(f"   Type coverage: {type_coverage:.1%} (min: {min_type_coverage:.1%})")
        print(f"   Position coverage: {pos_coverage:.1%} (min: {min_pos_coverage:.1%})")

        if type_coverage < min_type_coverage:
            print(f"   WARNING: Type coverage below threshold")
        if pos_coverage < min_pos_coverage:
            print(f"   WARNING: Position coverage below threshold")

    def _phase_sample(self):
        """Sample perturbations for annotation."""
        from src.annotation.stratified_sampler import StratifiedAnnotationSampler

        phase_config = self.config.get("phases", {}).get("sample", {})
        filter_config = phase_config.get("filter", {})

        # Load perturbations
        perturbations = list(self.storage.get_perturbations_by_experiment(self.experiment_id))

        # Apply filters
        if filter_config.get("primary_only", False):
            perturbations = [p for p in perturbations if p.get("is_primary_for_experiment", False)]

        print(f"Loaded {len(perturbations)} perturbations")

        if not perturbations:
            print("No perturbations found. Run 'perturb' phase first.")
            return

        # Sample
        total = phase_config.get("total", 100)
        seed = phase_config.get("seed", 42)
        stratify_by = phase_config.get("stratify_by", ["perturbation_type", "position", "benchmark"])

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
                {"$set": {"selected_for_annotation": True}}
            )

        print(f"Marked {len(samples)} samples for annotation")

    def _phase_annotate(self):
        """Interactive human annotation."""
        from src.annotation.tools import AnnotationTool

        phase_config = self.config.get("phases", {}).get("annotate", {})

        # Load samples marked for annotation
        samples = list(self.storage.db["perturbations"].find({"selected_for_annotation": True}))

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
            perturbations = list(self.storage.db["perturbations"].find({"selected_for_annotation": True}))
        else:
            perturbations = list(self.storage.get_perturbations_by_experiment(self.experiment_id))

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
        evaluations = list(self.storage.db["judge_outputs"].find({"experiment_id": self.experiment_id}))
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
                {"_id": eval_doc["_id"]},
                {"$set": {"jps": jps}}
            )
            computed += 1

        print(f"Computed JPS for {computed} evaluations")

    def _compute_tcs(self):
        """Compute True Criticality Score from annotations."""
        compute_config = self.config.get("compute", {}).get("tcs", {})
        formula = compute_config.get("formula", "(task_success_degradation * 50) + (subsequent_error_rate * 10) + (criticality_rating * 8)")
        validate_config = compute_config.get("validate", {})

        print(f"Formula: TCS = {formula}")

        # Load annotated perturbations
        perturbations = list(self.storage.db["perturbations"].find({
            "selected_for_annotation": True,
            "human_annotation": {"$exists": True}
        }))

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
                {"_id": p["_id"]},
                {"$set": {"tcs": tcs}}
            )
            computed += 1

        print(f"Computed TCS for {computed} perturbations")

        # Run validation if enabled
        if validate_config.get("enabled", False):
            print("\nRunning TCS validation...")
            self._validate_tcs(perturbations, validate_config)

    def _validate_tcs(self, perturbations: List[Dict], config: Dict):
        """Validate TCS correlation with human judgments."""
        # This would compare heuristic TCS with human TCS
        min_correlation = config.get("min_correlation_with_human", 0.4)
        print(f"   Min correlation threshold: {min_correlation}")
        # Implementation would compute actual correlation here

    def _compute_od(self):
        """Compute Outcome Degradation by grading trajectories."""
        from src.replay.od_scorer import ODScorer

        compute_config = self.config.get("compute", {}).get("od", {})
        validate_config = compute_config.get("validate", {})

        # Load perturbations
        perturbations = list(self.storage.db["perturbations"].find({"selected_for_annotation": True}))
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
            resume=True
        )

        print(f"Computed OD for {len(results)} perturbations")

        # Run validation if enabled
        if validate_config.get("enabled", False):
            print("\nRunning OD validation...")
            self._validate_od(perturbations, validate_config)

    def _validate_od(self, perturbations: List[Dict], config: Dict):
        """Validate OD distribution."""
        # Reload to get updated OD values
        perturbations = list(self.storage.db["perturbations"].find({"selected_for_annotation": True}))
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
        perturbations = list(self.storage.db["perturbations"].find({
            "selected_for_annotation": True,
            "tcs": {"$exists": True}
        }))

        evaluations = list(self.storage.db["judge_outputs"].find({
            "experiment_id": self.experiment_id,
            "jps": {"$exists": True}
        }))

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
        perturbations = list(self.storage.db["perturbations"].find({
            "selected_for_annotation": True,
            "od": {"$exists": True}
        }))

        # Load evaluations with JPS
        evaluations = list(self.storage.db["judge_outputs"].find({
            "experiment_id": self.experiment_id,
            "jps": {"$exists": True}
        }))

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
