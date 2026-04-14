#!/usr/bin/env python3
"""
Main entry point for the experiment pipeline.

Phases: load, typing, perturb, sample, annotate, judge, compute

Usage:
    python main.py --config v2/pocv2/trajectory_sampling_v9 --phase load
    python main.py --config v2/pocv2/trajectory_sampling_v9 --phase perturb
    python main.py --list-configs
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class TeeOutput:
    """Write to both file and console."""

    def __init__(self, file_path: Path, mode="a"):
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


def get_mongodb_client() -> MongoClient:
    """Get MongoDB client from environment."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not set in environment")
    return MongoClient(uri)


def load_config(config_name: str, validate: bool = True) -> Dict[str, Any]:
    """Load experiment configuration from JSON file with optional validation."""
    from src.config import load_and_validate_config

    config_dir = Path(__file__).parent / "config" / "experiments"

    search_paths = [
        config_dir / "v2" / f"{config_name}.json",
        config_dir / "v1" / f"{config_name}.json",
        config_dir / f"{config_name}.json",
        config_dir / "v2" / config_name,
        config_dir / "v1" / config_name,
        config_dir / config_name,
    ]

    config_path = None
    for path in search_paths:
        if path.exists():
            config_path = path
            break

    if not config_path:
        raise FileNotFoundError(f"Config not found: {config_name}")

    if validate:
        # Use Pydantic validation for schema 3.0.0
        validated = load_and_validate_config(config_path)
        return validated.model_dump(by_alias=True)
    else:
        with open(config_path, "r") as f:
            return json.load(f)


# =============================================================================
# PHASE: LOAD
# =============================================================================
def run_load_phase(config: Dict, db, verbose: bool = True):
    """Load trajectories from datasets into MongoDB."""
    from src.data.loaders import (
        load_toolbench_trajectories,
        load_swebench_trajectories,
        verify_trajectories_batch,
    )
    from src.utils import IDGenerator

    experiment_id = config["experiment"]["id"]
    datasets = config["experiment"].get("datasets", {})
    load_config = config.get("phases", {}).get("load", {})
    sampling = load_config.get("sampling", {})

    # Get version for caching
    version = load_config.get("version", "v1")

    # Check if baseline verification is enabled
    baseline_verify = load_config.get("baseline_verify", False)

    # Initialize ID generator
    id_gen = IDGenerator(config)

    print("\n" + "=" * 70)
    print("PHASE: LOAD")
    print("=" * 70)
    print(f"\nLoad phase version: {version}")

    # Setup judge for baseline verification if enabled
    judge = None
    verify_prompts = None
    if baseline_verify:
        from src.judges import create_claude_judge

        print("\nBaseline verification ENABLED")

        # Get model config (use baseline_verify_model or fall back to judge model)
        raw_model_config = load_config.get("baseline_verify_model")
        if not raw_model_config:
            judge_models = config.get("phases", {}).get("judge", {}).get("models", [])
            if judge_models:
                raw_model_config = judge_models[0]
            else:
                raise ValueError(
                    "baseline_verify enabled but no model config found. "
                    "Set phases.load.baseline_verify_model or phases.judge.models"
                )

        # Transform to format expected by create_claude_judge
        # Input format: {"name": ..., "provider": ..., "model": ..., "max_tokens": ..., "temperature": ...}
        # Expected format: {"model_id": ..., "config": {"temperature": ..., "max_tokens": ...}, "region_name": ...}
        model_config = {
            "model_id": raw_model_config.get("model"),
            "config": {
                "temperature": raw_model_config.get("temperature", 0.0),
                "max_tokens": raw_model_config.get("max_tokens", 1500),
            },
            "region_name": raw_model_config.get("region", "us-east-1"),
        }

        print(f"  Model: {model_config['model_id']}")

        judge = create_claude_judge(model_config)

        # Get prompts config
        verify_prompts_config = load_config.get("baseline_verify_prompts", {})
        verify_prompts = {
            "system": verify_prompts_config.get("system", "SINGLE_TRAJECTORY_SYSTEM_V2"),
            "user": verify_prompts_config.get("user", "SINGLE_TRAJECTORY_USER_V2"),
        }
        print(f"  Prompts: {verify_prompts['system']}, {verify_prompts['user']}")

        # Get min score threshold from config
        verify_min_score = load_config.get("baseline_verify_min_score", 95)
        print(f"  Min score threshold: {verify_min_score}")

    all_trajectories = []
    benchmark_counts = {}
    verification_stats = {}
    cached_counts = {}

    # Load ToolBench
    if datasets.get("toolbench", {}).get("enabled"):
        tb_config = datasets["toolbench"]
        tb_target = sampling.get("targets", {}).get("toolbench", 400)

        # Check for cached trajectories
        cached_tb = db["trajectories"].count_documents({
            "experiment_id": experiment_id,
            "benchmark": "toolbench"
        })

        print(f"\nLoading ToolBench trajectories...")
        print(f"  Target: {tb_target}")

        if cached_tb >= tb_target:
            # Use cached trajectories
            print(f"  Found {cached_tb} cached trajectories, skipping load")
            cached_trajs = list(db["trajectories"].find({
                "experiment_id": experiment_id,
                "benchmark": "toolbench"
            }))
            all_trajectories.extend(cached_trajs)
            benchmark_counts["toolbench"] = len(cached_trajs)
            cached_counts["toolbench"] = cached_tb
        else:
            # Need to load more
            needed = tb_target - cached_tb
            print(f"  Found {cached_tb} cached, need {needed} more")

            # If baseline_verify, load extra candidates (expect ~10% pass rate)
            load_multiplier = 15 if baseline_verify else 1
            load_count = needed * load_multiplier

            print(f"  Loading: {load_count} candidates")

            filters = tb_config.get("filters", {})
            trajectories = load_toolbench_trajectories(
                local_path=tb_config.get("path", "data/toolbench/data"),
                max_trajectories=load_count,
                min_steps=filters.get("min_steps", 3),
                max_steps=filters.get("max_steps", 50),
                filter_successful=filters.get("require_task_success", True),
                require_grader_aligned=filters.get("require_grader_aligned", False),
            )

            print(f"  Loaded: {len(trajectories)} candidates")

            # Convert to dicts for verification
            traj_dicts = [t.to_dict() if hasattr(t, "to_dict") else t for t in trajectories]

            # Baseline verify if enabled
            if baseline_verify and judge:
                print(f"  Verifying with Claude (parallelism={load_config.get('baseline_verify_parallelism', 4)})...")
                verified_trajs, stats = verify_trajectories_batch(
                    trajectories=traj_dicts,
                    judge=judge,
                    prompts=verify_prompts,
                    parallelism=load_config.get("baseline_verify_parallelism", 4),
                    target_count=needed,
                    verbose=verbose,
                    min_score=verify_min_score,
                )
                verification_stats["toolbench"] = stats
                print(f"  Verified: {len(verified_trajs)}/{stats['total_processed']} ({stats['pass_rate']:.1%} pass rate)")
                traj_dicts = verified_trajs

            # Reassign IDs (include experiment_id for uniqueness)
            for i, traj in enumerate(traj_dicts):
                base_id = id_gen.trajectory_id("toolbench", cached_tb + i)
                traj["trajectory_id"] = f"{experiment_id}_{base_id}"
                traj["version"] = version  # Add version field
            benchmark_counts["toolbench"] = len(traj_dicts)
            cached_counts["toolbench"] = cached_tb
            all_trajectories.extend(traj_dicts)

    # Load SWE-bench
    if datasets.get("swebench", {}).get("enabled"):
        sw_config = datasets["swebench"]
        sw_target = sampling.get("targets", {}).get("swebench", 100)

        # Check for cached trajectories
        cached_sw = db["trajectories"].count_documents({
            "experiment_id": experiment_id,
            "benchmark": "swebench"
        })

        print(f"\nLoading SWE-bench trajectories...")
        print(f"  Target: {sw_target}")

        if cached_sw >= sw_target:
            # Use cached trajectories
            print(f"  Found {cached_sw} cached trajectories, skipping load")
            cached_trajs = list(db["trajectories"].find({
                "experiment_id": experiment_id,
                "benchmark": "swebench"
            }))
            all_trajectories.extend(cached_trajs)
            benchmark_counts["swebench"] = len(cached_trajs)
            cached_counts["swebench"] = cached_sw
        else:
            # Need to load more
            needed = sw_target - cached_sw
            print(f"  Found {cached_sw} cached, need {needed} more")

            # If baseline_verify, load extra candidates
            load_multiplier = 15 if baseline_verify else 1
            load_count = needed * load_multiplier

            print(f"  Loading: {load_count} candidates")

            sw_filters = sw_config.get("filters", {})
            trajectories = load_swebench_trajectories(
                split=sw_config.get("split", "tool"),
                max_trajectories=load_count,
                min_steps=sw_filters.get("min_steps", 3),
                max_steps=sw_filters.get("max_steps", 50),
                filter_successful=sw_filters.get("require_task_success", True),
            )

            print(f"  Loaded: {len(trajectories)} candidates")

            # Convert to dicts for verification
            traj_dicts = [t.to_dict() if hasattr(t, "to_dict") else t for t in trajectories]

            # Baseline verify if enabled
            if baseline_verify and judge:
                print(f"  Verifying with Claude (parallelism={load_config.get('baseline_verify_parallelism', 4)})...")
                verified_trajs, stats = verify_trajectories_batch(
                    trajectories=traj_dicts,
                    judge=judge,
                    prompts=verify_prompts,
                    parallelism=load_config.get("baseline_verify_parallelism", 4),
                    target_count=needed,
                    verbose=verbose,
                    min_score=verify_min_score,
                )
                verification_stats["swebench"] = stats
                print(f"  Verified: {len(verified_trajs)}/{stats['total_processed']} ({stats['pass_rate']:.1%} pass rate)")
                traj_dicts = verified_trajs

            # Reassign IDs (include experiment_id for uniqueness)
            for i, traj in enumerate(traj_dicts):
                base_id = id_gen.trajectory_id("swebench", cached_sw + i)
                traj["trajectory_id"] = f"{experiment_id}_{base_id}"
                traj["version"] = version  # Add version field
            benchmark_counts["swebench"] = len(traj_dicts)
            cached_counts["swebench"] = cached_sw
            all_trajectories.extend(traj_dicts)

    # Store in MongoDB (only new trajectories, not cached ones)
    new_trajectories = [t for t in all_trajectories if isinstance(t, dict) and "version" in t]
    cached_trajectories = [t for t in all_trajectories if isinstance(t, dict) and "version" not in t]

    if new_trajectories:
        print(f"\nStoring {len(new_trajectories)} new trajectories in MongoDB...")
        collection = db["trajectories"]

        stored = 0
        for traj in new_trajectories:
            doc = traj.to_dict() if hasattr(traj, "to_dict") else traj
            doc["experiment_id"] = experiment_id
            doc["stored_at"] = datetime.utcnow().isoformat() + "Z"
            if "version" not in doc:
                doc["version"] = version

            # Upsert by trajectory_id
            traj_id = doc.get("trajectory_id")
            if traj_id:
                collection.update_one(
                    {"trajectory_id": traj_id, "experiment_id": experiment_id},
                    {"$set": doc},
                    upsert=True,
                )
                stored += 1

        print(f"  Stored: {stored} trajectories")
    else:
        print(f"\nNo new trajectories to store (all cached)")

    # Summary
    total_cached = sum(cached_counts.values()) if cached_counts else 0
    total_new = len(new_trajectories)
    total = len(all_trajectories)

    print(f"\nLoad summary (version={version}):")
    print(f"  Cached: {total_cached}")
    print(f"  Newly loaded: {total_new}")
    print(f"  Total: {total}")
    print(f"  By benchmark: {benchmark_counts}")

    if verification_stats:
        print(f"\nVerification stats:")
        for benchmark, stats in verification_stats.items():
            print(f"  {benchmark}: {stats['verified_count']}/{stats['total_processed']} verified ({stats['pass_rate']:.1%})")

    print("\nLoad phase complete.")


# =============================================================================
# PHASE: TYPING
# =============================================================================
def run_typing_phase(config: Dict, db, verbose: bool = True):
    """Type trajectories with step roles and perturbable slots."""
    from src.typing.typer import type_trajectory_batch
    from src.data.schema import Trajectory

    experiment_id = config["experiment"]["id"]
    # No version field for typing - it's a transformation, not a generation

    print("\n" + "=" * 70)
    print("PHASE: TYPING")
    print("=" * 70)

    # Check for cached typed trajectories
    typed_collection = db["typed_trajectories"]
    cached_count = typed_collection.count_documents({"experiment_id": experiment_id})

    # Load trajectories from MongoDB
    collection = db["trajectories"]
    total_trajectories = collection.count_documents({"experiment_id": experiment_id})

    print(f"\nTrajectories: {total_trajectories}")
    print(f"Typed (cached): {cached_count}")

    if cached_count >= total_trajectories and total_trajectories > 0:
        print(f"All {cached_count} trajectories already typed, skipping")
        print("\nTyping phase complete.")
        return

    print(f"\nLoading trajectories from MongoDB...")
    docs = list(collection.find({"experiment_id": experiment_id}))
    print(f"  Found: {len(docs)} trajectories")

    if not docs:
        print("No trajectories found. Run load phase first.")
        return

    # Get already typed trajectory IDs
    typed_ids = set(doc["trajectory_id"] for doc in typed_collection.find(
        {"experiment_id": experiment_id},
        {"trajectory_id": 1}
    ))
    print(f"  Already typed: {len(typed_ids)}")

    # Filter to untyped trajectories
    untyped_docs = [doc for doc in docs if doc.get("trajectory_id") not in typed_ids]
    print(f"  Need to type: {len(untyped_docs)}")

    if not untyped_docs:
        print("All trajectories already typed.")
        print("\nTyping phase complete.")
        return

    # Clean MongoDB-specific fields
    def clean_doc(doc):
        return {k: v for k, v in doc.items()
                if k not in ("_id", "experiment_id", "stored_at")}
    cleaned_docs = [clean_doc(doc) for doc in untyped_docs]

    # Type trajectories
    print("\nTyping trajectories...")
    typed_trajectories = type_trajectory_batch(cleaned_docs)
    print(f"  Typed: {len(typed_trajectories)} trajectories")

    # Store typed trajectories
    print("\nStoring typed trajectories...")

    stored = 0
    for typed_traj in typed_trajectories:
        # typed_traj is already a dict from type_trajectory_batch
        doc = typed_traj if isinstance(typed_traj, dict) else typed_traj.to_dict()
        doc["experiment_id"] = experiment_id
        doc["typed_at"] = datetime.utcnow().isoformat() + "Z"

        typed_collection.update_one(
            {"trajectory_id": doc["trajectory_id"], "experiment_id": experiment_id},
            {"$set": doc},
            upsert=True,
        )
        stored += 1

    print(f"  Stored: {stored} typed trajectories")
    print(f"\nTyping summary:")
    print(f"  Cached: {len(typed_ids)}")
    print(f"  Newly typed: {stored}")
    print(f"  Total: {len(typed_ids) + stored}")
    print("\nTyping phase complete.")


# =============================================================================
# PHASE: PERTURB
# =============================================================================
def run_perturb_phase(config: Dict, db, verbose: bool = True):
    """Generate perturbations with incremental storage."""
    from src.perturbations.generator_v2 import generate_perturbations_for_batch
    from src.perturbations.balancer import balance_perturbation_batch
    from src.typing.schema import TypedTrajectory
    from src.llm import get_bedrock_client
    from src.scoring.class_validator import create_validator_from_config

    experiment_id = config["experiment"]["id"]
    phase_config = config.get("phases", {}).get("perturb", {})
    targets = config["experiment"].get("targets", {})
    version = phase_config.get("version", "v1")

    print("\n" + "=" * 70)
    print("PHASE: PERTURB")
    print("=" * 70)
    print(f"Version: {version}")

    # Check for cached perturbations
    perturb_collection = db["perturbed_trajectories"]
    cached_count = perturb_collection.count_documents({"experiment_id": experiment_id})
    total_target = targets.get("total_perturbations", 1500)

    print(f"\nCaching check:")
    print(f"  Cached perturbations: {cached_count}")
    print(f"  Target: {total_target}")

    if cached_count >= total_target:
        print(f"\nFound {cached_count} cached perturbations >= target {total_target}")
        print("Skipping perturbation generation, using cached data.")

        # Still need to ensure evaluation units exist
        eu_collection = db["evaluation_units"]
        eu_count = eu_collection.count_documents({"experiment_id": experiment_id})
        if eu_count >= cached_count:
            print(f"  Evaluation units: {eu_count} (already created)")
            print("\nPerturb phase complete (cached).")
            return
        else:
            print(f"  Evaluation units: {eu_count} (need to create more)")
            # Load cached perturbations to create evaluation units
            cached_docs = list(perturb_collection.find({"experiment_id": experiment_id}))
            from src.perturbations.schema import PerturbationRecord
            all_results = []
            for doc in cached_docs:
                record = PerturbationRecord.from_dict(doc["perturbation_record"])
                from src.typing.schema import TypedTrajectory
                perturbed_traj = TypedTrajectory.from_dict(doc["perturbed_trajectory"])
                all_results.append((record, perturbed_traj))

            # Get balanced records (all cached are already balanced)
            balanced_records = [r for r, _ in all_results]
            print(f"\nCreating evaluation units from {len(balanced_records)} cached perturbations...")
            _create_evaluation_units(config, db, balanced_records, all_results, verbose)
            print("\nPerturb phase complete (cached).")
            return

    needed = total_target - cached_count
    print(f"  Need to generate: {needed} more perturbations")

    # Load typed trajectories
    print("\nLoading typed trajectories from MongoDB...")
    collection = db["typed_trajectories"]
    docs = list(collection.find({"experiment_id": experiment_id}))
    print(f"  Found: {len(docs)} typed trajectories")

    if not docs:
        print("No typed trajectories found. Run typing phase first.")
        return

    trajectories = [TypedTrajectory.from_dict(doc) for doc in docs]

    # Config
    parallelism = phase_config.get("parallelism", 4)
    per_trajectory = targets.get("per_trajectory", {}).get("target", 3)
    class_weights = targets.get("by_class", {})
    random_seed = config["experiment"].get("random_seed", 42)

    print(f"\nTarget: {total_target} perturbations")
    print(f"Per trajectory: {per_trajectory}")
    print(f"Distribution: {class_weights}")
    print(f"Parallelism: {parallelism}")

    # Initialize LLM client
    llm_client = None
    llm_config = phase_config.get("llm", {})
    if llm_config.get("provider") == "bedrock":
        try:
            llm_client = get_bedrock_client(log_calls=verbose)
            print(f"Initialized LLM client for: {llm_config.get('use_for', [])}")
        except Exception as e:
            print(f"Warning: Could not initialize LLM: {e}")

    # Initialize class validator if enabled
    class_validator = create_validator_from_config(config)
    if class_validator:
        print(f"Class validation ENABLED (min_match_rate: {phase_config.get('class_validation', {}).get('min_match_rate', 0.90):.0%})")
    else:
        print("Class validation DISABLED")

    validated_count = [0]  # Track validated perturbations

    # Incremental storage callback
    perturb_collection = db["perturbed_trajectories"]
    saved_count = [0]  # Use list for closure

    def save_batch(batch_perturbations):
        """Save a batch of perturbations to MongoDB with optional class validation."""
        from pymongo import UpdateOne

        operations = []
        timestamp = datetime.utcnow().isoformat() + "Z"

        for record, perturbed_traj in batch_perturbations:
            doc = {
                "perturbation_id": record.perturbation_id,
                "original_trajectory_id": record.original_trajectory_id,
                "perturbation_record": record.to_dict(),
                "perturbed_trajectory": perturbed_traj.to_dict(),
                "experiment_id": experiment_id,
                "stored_at": timestamp,
                "version": version,
            }

            # Run class validation if enabled
            if class_validator:
                perturbation_data = {
                    "perturbation_class": record.perturbation_class,
                    "perturbation_type": record.perturbation_type,
                    "target_step_index": record.target_step_index,
                    "original_value": record.original_value,
                    "perturbed_value": record.perturbed_value,
                    "mutation_method": record.mutation_method,
                }
                validation_result = class_validator.validate_perturbation(perturbation_data)
                doc["class_validation"] = {
                    "class_matches": validation_result.class_matches,
                    "reasoning": validation_result.reasoning,
                    "parse_success": validation_result.parse_success,
                }
                validated_count[0] += 1

            operations.append(
                UpdateOne(
                    {"perturbation_id": record.perturbation_id},
                    {"$set": doc},
                    upsert=True,
                )
            )

        if operations:
            result = perturb_collection.bulk_write(operations)
            saved_count[0] += result.upserted_count + result.modified_count

    # Generate perturbations with incremental save
    print("\nGenerating perturbations...")
    all_results, index = generate_perturbations_for_batch(
        trajectories=trajectories,
        target_per_trajectory=per_trajectory,
        random_seed=random_seed,
        llm_client=llm_client,
        verbose=verbose,
        parallelism=parallelism,
        on_batch_save=save_batch,
        config=config,
    )

    print(f"\nGenerated: {len(all_results)} perturbations")
    print(f"Saved incrementally: {saved_count[0]} perturbations")

    # Print class validation stats if enabled
    if class_validator and validated_count[0] > 0:
        print(f"Class validated: {validated_count[0]} perturbations")
        # Query MongoDB for validation stats
        validation_stats = perturb_collection.aggregate([
            {"$match": {"experiment_id": experiment_id, "class_validation": {"$exists": True}}},
            {"$group": {
                "_id": None,
                "total": {"$sum": 1},
                "matches": {"$sum": {"$cond": [{"$eq": ["$class_validation.class_matches", 1]}, 1, 0]}},
            }}
        ])
        for stat in validation_stats:
            match_rate = stat["matches"] / stat["total"] if stat["total"] > 0 else 0
            print(f"  Class match rate: {match_rate:.1%} ({stat['matches']}/{stat['total']})")

    # Balance distribution
    print("\nBalancing distribution...")
    records = [record for record, _ in all_results]
    balanced_records, balance_report = balance_perturbation_batch(
        records=records,
        total_target=total_target,
        class_weights=class_weights,
        random_seed=random_seed,
    )

    print(f"After balancing: {len(balanced_records)} perturbations")
    print(f"Removed: {balance_report['removed']}")

    # Create evaluation units for balanced perturbations
    print("\nCreating evaluation units...")
    _create_evaluation_units(config, db, balanced_records, all_results, verbose)

    print("\nPerturb phase complete.")


def _create_evaluation_units(config, db, balanced_records, all_results, verbose):
    """Create evaluation units from balanced perturbations."""
    import hashlib
    from src.utils import generate_evaluation_unit_id

    experiment_id = config["experiment"]["id"]
    random_seed = config["experiment"].get("random_seed", 42)

    # Build lookup from perturbation_id to (record, trajectory)
    result_lookup = {r.perturbation_id: (r, t) for r, t in all_results}

    # Build lookup for class_validation from perturbed_trajectories collection
    perturb_collection = db["perturbed_trajectories"]
    class_validation_lookup = {}
    perturb_docs = perturb_collection.find(
        {"experiment_id": experiment_id, "class_validation": {"$exists": True}},
        {"perturbation_id": 1, "class_validation": 1}
    )
    for doc in perturb_docs:
        class_validation_lookup[doc["perturbation_id"]] = doc.get("class_validation")

    eu_collection = db["evaluation_units"]
    created = 0

    for record in balanced_records:
        if record.perturbation_id not in result_lookup:
            continue

        _, perturbed_traj = result_lookup[record.perturbation_id]

        # Load original trajectory
        orig_doc = db["typed_trajectories"].find_one({
            "trajectory_id": record.original_trajectory_id,
            "experiment_id": experiment_id,
        })

        if not orig_doc:
            continue

        # Generate evaluation_unit_id using new scheme
        trajectory_id = record.original_trajectory_id
        eu_id = generate_evaluation_unit_id(trajectory_id, record.perturbation_id)

        # Deterministic blinding based on hash
        blend_input = f"{record.perturbation_id}:{random_seed}"
        blend_hash = int(hashlib.md5(blend_input.encode()).hexdigest(), 16)
        is_a_baseline = (blend_hash % 2) == 0

        eu_doc = {
            "evaluation_unit_id": eu_id,
            "trajectory_id": trajectory_id,
            "perturbation_id": record.perturbation_id,
            "experiment_id": experiment_id,
            "perturbation_class": record.perturbation_class,
            "perturbation_type": record.perturbation_type,
            "blinding": {
                "is_a_baseline": is_a_baseline,
                "trajectory_a": orig_doc if is_a_baseline else perturbed_traj.to_dict(),
                "trajectory_b": perturbed_traj.to_dict() if is_a_baseline else orig_doc,
            },
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        # Include class_validation if available
        if record.perturbation_id in class_validation_lookup:
            eu_doc["class_validation"] = class_validation_lookup[record.perturbation_id]

        eu_collection.update_one(
            {"evaluation_unit_id": eu_doc["evaluation_unit_id"]},
            {"$set": eu_doc},
            upsert=True,
        )
        created += 1

    print(f"  Created: {created} evaluation units")


# =============================================================================
# PHASE: SAMPLE
# =============================================================================
def run_sample_phase(config: Dict, db, verbose: bool = True):
    """Stratified sampling of evaluation units for annotation."""
    from src.annotation.stratified_sampler import StratifiedAnnotationSampler

    experiment_id = config["experiment"]["id"]
    annotate_config = config.get("phases", {}).get("annotate", {})

    print("\n" + "=" * 70)
    print("PHASE: SAMPLE")
    print("=" * 70)

    # Load evaluation units
    print("\nLoading evaluation units from MongoDB...")
    collection = db["evaluation_units"]
    docs = list(collection.find({"experiment_id": experiment_id}))
    print(f"  Found: {len(docs)} evaluation units")

    if not docs:
        print("No evaluation units found. Run perturb phase first.")
        return

    # Sample
    sample_size = annotate_config.get("sample_size", 50)
    stratification = annotate_config.get("stratification", {})

    print(f"\nSampling {sample_size} units with stratification...")

    # StratifiedAnnotationSampler expects perturbations with is_primary_for_experiment field
    # Adapt evaluation units to have the fields the sampler expects
    adapted_docs = []
    for doc in docs:
        adapted = dict(doc)
        # Mark all as primary (evaluation units are already primary)
        adapted["is_primary_for_experiment"] = True
        adapted_docs.append(adapted)

    sampler = StratifiedAnnotationSampler(
        perturbations=adapted_docs,
        random_seed=stratification.get("random_seed", 42),
        stratify_by_class=True,
        class_distribution=stratification.get("by_class") or None,
    )

    sampled = sampler.sample(total=sample_size)
    print(f"  Sampled: {len(sampled)} units")

    # Mark sampled units
    sampled_ids = [doc["evaluation_unit_id"] for doc in sampled]
    collection.update_many(
        {"evaluation_unit_id": {"$in": sampled_ids}},
        {"$set": {"sampled_for_annotation": True}},
    )

    print("\nSample phase complete.")


# =============================================================================
# PHASE: JUDGE
# =============================================================================
def run_judge_phase(config: Dict, db, verbose: bool = True):
    """Run LLM judge evaluation on evaluation units."""
    from src.judges import create_claude_judge, UnitJudgeRunner
    from src.storage.mongodb import MongoDBStorage

    experiment_id = config["experiment"]["id"]
    judge_config = config.get("phases", {}).get("judge", {})

    print("\n" + "=" * 70)
    print("PHASE: JUDGE")
    print("=" * 70)

    base_version = judge_config.get("version", "v1")
    print(f"Version: {base_version}")

    # Load evaluation units
    print("\nLoading evaluation units from MongoDB...")
    collection = db["evaluation_units"]

    query = {"experiment_id": experiment_id}
    if judge_config.get("filter", {}).get("annotated_only"):
        query["sampled_for_annotation"] = True

    docs = list(collection.find(query))
    print(f"  Found: {len(docs)} evaluation units")

    if not docs:
        print("No evaluation units found.")
        return

    # Get modes to run (support both "mode" and "modes" config keys)
    modes = judge_config.get("modes", [judge_config.get("mode", "single_trajectory")])
    if isinstance(modes, str):
        modes = [modes]

    # Check for cached judge outputs per mode
    judge_collection = db["judge_eval_outputs"]
    print(f"\nCaching check for {len(modes)} modes:")
    all_cached = True
    for mode in modes:
        version = f"{base_version}_{mode}"
        cached = judge_collection.count_documents({
            "experiment_id": experiment_id,
            "version": version
        })
        print(f"  {mode}: {cached}/{len(docs)} cached")
        if cached < len(docs):
            all_cached = False

    if all_cached:
        print(f"\nAll modes fully cached. Skipping judge evaluation.")
        print("\nJudge phase complete (cached).")
        return

    # Get judge model config
    models = judge_config.get("models", [])
    if not models:
        print("No judge models configured.")
        return

    model_config = models[0]  # Use first model

    parallelism = judge_config.get("parallelism", 4)

    print(f"\nRunning judge: {model_config['name']}")
    print(f"Modes: {modes}")
    print(f"Parallelism: {parallelism}")

    # Create judge instance
    judge = create_claude_judge({
        "model_id": model_config.get("model", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        "config": {
            "temperature": model_config.get("temperature", 0.0),
            "max_tokens": model_config.get("max_tokens", 1500),
        },
        "region_name": "us-east-1",
    })

    # Create storage instance
    storage_config = config["experiment"]["storage"]
    storage = MongoDBStorage(database=storage_config["database"])

    # Configure runner
    runner_config = {
        "temperature": model_config.get("temperature", 0.0),
        "max_tokens": model_config.get("max_tokens", 1500),
        "checkpoint_interval": 20,
        "rate_limit_delay": 0.5,
    }

    # Get prompts from config
    prompts_config = judge_config.get("prompts", {})

    # Adapt evaluation units: extract perturbed trajectory from blinding structure
    # and add it to the expected location for UnitJudgeRunner
    adapted_units = []
    for doc in docs:
        unit = dict(doc)
        blinding = unit.get("blinding", {})

        # Extract the perturbed trajectory (the one that's not the baseline)
        is_a_baseline = blinding.get("is_a_baseline", True)
        if is_a_baseline:
            # If A is baseline, B is perturbed
            perturbed_traj = blinding.get("trajectory_b")
        else:
            # If A is perturbed, A is what we want
            perturbed_traj = blinding.get("trajectory_a")

        # Add trajectory to the location UnitJudgeRunner expects
        unit["perturbed_trajectory"] = perturbed_traj
        adapted_units.append(unit)

    # Run each mode
    total_outputs = 0
    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"Running mode: {mode}")
        print(f"{'=' * 60}")

        # Create version suffix for this mode
        version = f"{base_version}_{mode}"
        print(f"Version: {version}")

        # Get mode-specific prompts
        mode_prompts = prompts_config.get(mode, {})
        if mode_prompts:
            print(f"Using prompts: {mode_prompts}")

        # Create runner for this mode
        runner = UnitJudgeRunner(
            judge=judge,
            config=runner_config,
            version=version,
            prompts=mode_prompts if mode_prompts else None,
        )

        # Run batch for this mode
        outputs = runner.run_batch(
            units=adapted_units,
            mode=mode,
            resume=True,
            samples_per_unit=1,
            storage=storage,
            experiment_id=experiment_id,
            parallelization=parallelism,
        )

        print(f"Completed {len(outputs)} evaluations for {mode}")
        total_outputs += len(outputs)

    print(f"\n{'=' * 70}")
    print(f"Judge phase complete. Total: {total_outputs} evaluations across {len(modes)} modes")
    print(f"{'=' * 70}")


# =============================================================================
# PHASE: ANNOTATE
# =============================================================================
def run_annotate_phase(config: Dict, db, verbose: bool = True):
    """Launch annotation UI for human labeling."""
    print("\n" + "=" * 70)
    print("PHASE: ANNOTATE")
    print("=" * 70)
    print("\nThis phase requires the Streamlit annotation UI.")
    print("Run: streamlit run src/annotation/app.py")
    print("\nAnnotation is a manual process - skipping automatic execution.")


# =============================================================================
# PHASE: COMPUTE
# =============================================================================
def run_compute_phase(config: Dict, db, verbose: bool = True):
    """Compute metrics from judge evaluations."""
    experiment_id = config["experiment"]["id"]
    compute_config = config.get("phases", {}).get("compute", {})
    version = compute_config.get("version", "v1")

    print("\n" + "=" * 70)
    print("PHASE: COMPUTE")
    print("=" * 70)
    print(f"Version: {version}")

    # Load evaluation units
    eu_collection = db["evaluation_units"]
    judge_collection = db["judge_eval_outputs"]

    units = list(eu_collection.find({"experiment_id": experiment_id}))
    print(f"\nLoaded: {len(units)} evaluation units")

    if not units:
        print("No evaluation units found.")
        return

    # Find all versions with judge outputs for this experiment
    all_judgments = list(judge_collection.find({"experiment_id": experiment_id}))
    if not all_judgments:
        print("No judge evaluations found. Run judge phase first.")
        return

    # Get unique versions
    versions = list(set(j.get("version", "unknown") for j in all_judgments))
    versions.sort()
    print(f"Found versions: {versions}")

    # Build class_validation lookup from evaluation_units
    class_validation_lookup = {}
    for unit in units:
        cv = unit.get("class_validation", {})
        if cv:
            class_validation_lookup[unit["evaluation_unit_id"]] = cv.get("class_matches", 1)

    # Build "clean baseline" lookup using blinded_pair results
    # A baseline is "clean" if blinded_pair judge did NOT find error in baseline trajectory
    clean_baseline_lookup = {}
    blinded_pair_judgments = [j for j in all_judgments if "_blinded_pair" in j.get("version", "")]
    for j in blinded_pair_judgments:
        unit_id = j["evaluation_unit_id"]
        unit = next((u for u in units if u["evaluation_unit_id"] == unit_id), None)
        if not unit:
            continue

        pair_comp = j.get("pair_comparison", {})
        error_traj = pair_comp.get("error_trajectory") if pair_comp else None
        is_a_baseline = unit.get("blinding", {}).get("is_a_baseline", True)

        # Baseline has error if:
        # - error_trajectory == "both"
        # - error_trajectory == "A" and A is baseline
        # - error_trajectory == "B" and B is baseline
        baseline_has_error = (
            error_traj == "both" or
            (error_traj == "A" and is_a_baseline) or
            (error_traj == "B" and not is_a_baseline)
        )
        clean_baseline_lookup[unit_id] = not baseline_has_error

    clean_count = sum(1 for v in clean_baseline_lookup.values() if v)
    print(f"Clean baselines: {clean_count}/{len(clean_baseline_lookup)} ({100*clean_count/len(clean_baseline_lookup):.1f}%)")

    # Compute metrics for each version
    all_results = {}
    for version in versions:
        print(f"\n{'=' * 60}")
        print(f"Computing metrics for: {version}")
        print(f"{'=' * 60}")

        # Filter judgments for this version
        version_judgments = [j for j in all_judgments if j.get("version") == version]
        print(f"  Judge evaluations: {len(version_judgments)}")

        # Build lookup
        judgment_by_unit = {j["evaluation_unit_id"]: j for j in version_judgments}

        # Compute metrics by class (all perturbations and validated-only)
        metrics = {
            "placebo": {"total": 0, "detected": 0},
            "fine_grained": {"total": 0, "detected": 0},
            "coarse_grained": {"total": 0, "detected": 0},
        }
        # Metrics for class_matches=1 only (validated perturbations)
        metrics_validated = {
            "placebo": {"total": 0, "detected": 0},
            "fine_grained": {"total": 0, "detected": 0},
            "coarse_grained": {"total": 0, "detected": 0},
        }
        # Metrics for clean baselines only (no pre-existing errors)
        metrics_clean = {
            "placebo": {"total": 0, "detected": 0},
            "fine_grained": {"total": 0, "detected": 0},
            "coarse_grained": {"total": 0, "detected": 0},
        }

        for unit in units:
            unit_id = unit["evaluation_unit_id"]
            pclass = unit.get("perturbation_class", "unknown")

            if unit_id not in judgment_by_unit:
                continue

            judgment = judgment_by_unit[unit_id]

            # Determine if perturbation was correctly detected
            # For blinded_pair: check if error_trajectory points to perturbed
            # For single_trajectory: just check error_detected
            detected = False

            if "_blinded_pair" in version:
                # blinded_pair: check pair_comparison.error_trajectory
                pair_comp = judgment.get("pair_comparison", {})
                error_traj = pair_comp.get("error_trajectory") if pair_comp else None
                is_a_baseline = unit.get("blinding", {}).get("is_a_baseline", True)

                # Correct detection = judge identifies perturbed trajectory
                if error_traj == "A" and not is_a_baseline:
                    detected = True  # A is perturbed, judge found error in A
                elif error_traj == "B" and is_a_baseline:
                    detected = True  # B is perturbed, judge found error in B
                # "neither" or "both" or wrong trajectory = not detected
            else:
                # single_trajectory: use detection.error_detected
                detection = judgment.get("detection", {})
                detected = detection.get("error_detected", False) if detection else False

            # Track all perturbations
            if pclass in metrics:
                metrics[pclass]["total"] += 1
                if detected:
                    metrics[pclass]["detected"] += 1

            # Track validated perturbations (class_matches=1)
            class_matches = class_validation_lookup.get(unit_id, 1)
            if pclass in metrics_validated and class_matches == 1:
                metrics_validated[pclass]["total"] += 1
                if detected:
                    metrics_validated[pclass]["detected"] += 1

            # Track clean baseline only (no pre-existing errors)
            is_clean = clean_baseline_lookup.get(unit_id, True)
            if pclass in metrics_clean and is_clean:
                metrics_clean[pclass]["total"] += 1
                if detected:
                    metrics_clean[pclass]["detected"] += 1

        # Print results for this version - ALL perturbations
        print("  --- All perturbations ---")
        for pclass, data in metrics.items():
            total = data["total"]
            detected = data["detected"]
            rate = (detected / total * 100) if total > 0 else 0
            print(f"  {pclass}: {detected}/{total} detected ({rate:.1f}%)")

        # Overall PDR (non-placebo)
        non_placebo_total = metrics["fine_grained"]["total"] + metrics["coarse_grained"]["total"]
        non_placebo_detected = metrics["fine_grained"]["detected"] + metrics["coarse_grained"]["detected"]
        pdr = (non_placebo_detected / non_placebo_total * 100) if non_placebo_total > 0 else 0

        # FP rate (placebo detected as error)
        fp_total = metrics["placebo"]["total"]
        fp_detected = metrics["placebo"]["detected"]
        fp_rate = (fp_detected / fp_total * 100) if fp_total > 0 else 0

        print(f"  Overall PDR: {pdr:.1f}%")
        print(f"  False Positive Rate: {fp_rate:.1f}%")

        # Print validated-only results
        validated_total = sum(m["total"] for m in metrics_validated.values())
        if validated_total > 0:
            print("\n  --- Validated only (class_matches=1) ---")
            for pclass, data in metrics_validated.items():
                total = data["total"]
                detected = data["detected"]
                rate = (detected / total * 100) if total > 0 else 0
                print(f"  {pclass}: {detected}/{total} detected ({rate:.1f}%)")

            np_total_v = metrics_validated["fine_grained"]["total"] + metrics_validated["coarse_grained"]["total"]
            np_detected_v = metrics_validated["fine_grained"]["detected"] + metrics_validated["coarse_grained"]["detected"]
            pdr_v = (np_detected_v / np_total_v * 100) if np_total_v > 0 else 0
            fp_total_v = metrics_validated["placebo"]["total"]
            fp_detected_v = metrics_validated["placebo"]["detected"]
            fp_rate_v = (fp_detected_v / fp_total_v * 100) if fp_total_v > 0 else 0
            print(f"  Overall PDR: {pdr_v:.1f}%")
            print(f"  False Positive Rate: {fp_rate_v:.1f}%")

        # Print clean baseline only results
        clean_total = sum(m["total"] for m in metrics_clean.values())
        if clean_total > 0:
            print("\n  --- Clean baselines only (no pre-existing errors) ---")
            for pclass, data in metrics_clean.items():
                total = data["total"]
                detected = data["detected"]
                rate = (detected / total * 100) if total > 0 else 0
                print(f"  {pclass}: {detected}/{total} detected ({rate:.1f}%)")

            np_total_c = metrics_clean["fine_grained"]["total"] + metrics_clean["coarse_grained"]["total"]
            np_detected_c = metrics_clean["fine_grained"]["detected"] + metrics_clean["coarse_grained"]["detected"]
            pdr_c = (np_detected_c / np_total_c * 100) if np_total_c > 0 else 0
            fp_total_c = metrics_clean["placebo"]["total"]
            fp_detected_c = metrics_clean["placebo"]["detected"]
            fp_rate_c = (fp_detected_c / fp_total_c * 100) if fp_total_c > 0 else 0
            print(f"  Overall PDR: {pdr_c:.1f}%")
            print(f"  False Positive Rate: {fp_rate_c:.1f}%")

        all_results[version] = {
            "by_class": metrics,
            "overall_pdr": pdr,
            "false_positive_rate": fp_rate,
        }

        # Store metrics for this version
        metrics_doc = {
            "experiment_id": experiment_id,
            "version": version,
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "by_class": metrics,
            "overall_pdr": pdr,
            "false_positive_rate": fp_rate,
        }

        db["metrics"].update_one(
            {"experiment_id": experiment_id, "version": version},
            {"$set": metrics_doc},
            upsert=True,
        )

    # Print summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print("\n{:<30} {:>15} {:>15} {:>15} {:>10}".format(
        "Version", "Placebo FP", "Fine-grained", "Coarse-grained", "PDR"
    ))
    print("-" * 85)

    for version, results in all_results.items():
        fp = results["by_class"]["placebo"]
        fg = results["by_class"]["fine_grained"]
        cg = results["by_class"]["coarse_grained"]

        fp_rate = (fp["detected"] / fp["total"] * 100) if fp["total"] > 0 else 0
        fg_rate = (fg["detected"] / fg["total"] * 100) if fg["total"] > 0 else 0
        cg_rate = (cg["detected"] / cg["total"] * 100) if cg["total"] > 0 else 0

        print("{:<30} {:>14.1f}% {:>14.1f}% {:>14.1f}% {:>9.1f}%".format(
            version, fp_rate, fg_rate, cg_rate, results["overall_pdr"]
        ))

    print("\nCompute phase complete.")


# =============================================================================
# MAIN
# =============================================================================
PHASES = {
    "load": run_load_phase,
    "typing": run_typing_phase,
    "perturb": run_perturb_phase,
    "sample": run_sample_phase,
    "judge": run_judge_phase,
    "annotate": run_annotate_phase,
    "compute": run_compute_phase,
}


def list_available_configs():
    """List all available configuration files."""
    config_dir = Path(__file__).parent / "config" / "experiments"

    print("=" * 70)
    print("AVAILABLE CONFIGURATIONS")
    print("=" * 70)

    for subdir in ["v2", "v1"]:
        sub_path = config_dir / subdir
        if sub_path.exists():
            print(f"\n[{subdir}]")
            for f in sorted(sub_path.rglob("*.json")):
                rel = f.relative_to(config_dir)
                try:
                    with open(f) as fh:
                        name = json.load(fh).get("experiment", {}).get("name", "")
                    print(f"  {rel}: {name}")
                except Exception:
                    print(f"  {rel}")


def main():
    parser = argparse.ArgumentParser(description="Run experiment phases")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--phase", type=str, help="Phase to run: " + ", ".join(PHASES.keys()))
    parser.add_argument("--list-configs", action="store_true", help="List configs")
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")

    args = parser.parse_args()

    if args.list_configs:
        list_available_configs()
        return

    if not args.config:
        parser.error("--config is required")
    if not args.phase:
        parser.error("--phase is required")

    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    experiment_id = config["experiment"]["id"]
    print(f"Experiment: {config['experiment']['name']}")
    print(f"ID: {experiment_id}")

    # Setup logging
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"{experiment_id}.log"

    # Connect to MongoDB
    print("\nConnecting to MongoDB...")
    client = get_mongodb_client()
    db = client[config["experiment"]["storage"]["database"]]
    print(f"Database: {db.name}")

    # Run phase(s)
    phases = [p.strip() for p in args.phase.split(",")]
    verbose = not args.no_verbose

    with TeeOutput(log_path, mode="a") as tee:
        original_stdout = sys.stdout
        sys.stdout = tee

        try:
            print("\n" + "=" * 70)
            print(f"RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Phases: {', '.join(phases)}")
            print("=" * 70)

            for phase in phases:
                if phase not in PHASES:
                    print(f"Unknown phase: {phase}")
                    print(f"Available: {', '.join(PHASES.keys())}")
                    continue

                if args.dry_run:
                    print(f"\n[DRY RUN] Would run phase: {phase}")
                else:
                    PHASES[phase](config, db, verbose)

            print("\n" + "=" * 70)
            print("COMPLETE")
            print("=" * 70)

        except KeyboardInterrupt:
            print("\nInterrupted")
            sys.exit(130)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
