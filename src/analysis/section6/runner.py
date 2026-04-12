"""
Section 6 Analysis Runner.

Loads evaluation units, judge outputs, outcome evidence, and human labels
from MongoDB, computes per-unit analysis results, and stores them back.
"""

import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.analysis.section6.evaluator import Section6Evaluator
from src.analysis.section6.schema import AnalysisResult
from src.analysis.section6.storage import Section6Storage
from src.evaluation.schema import EvaluationUnit
from src.human_labels.schema import AggregatedLabel
from src.judges.schema import Section5JudgeOutput
from src.outcome_evidence.schema import OutcomeRecord
from src.storage.mongodb import MongoDBStorage


def run_section6_analysis(
    experiment_id: str,
    storage: MongoDBStorage,
    config: Optional[Dict[str, Any]] = None,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Section 6 analysis for all (unit, judge) pairs in an experiment.

    Args:
        experiment_id: The experiment ID
        storage: MongoDBStorage instance
        config: Optional configuration dict
        force: If True, re-compute even if results exist
        verbose: Print progress messages

    Returns:
        Summary dict with counts and statistics
    """
    config = config or {}

    if verbose:
        print("=" * 60)
        print("SECTION 6 ANALYSIS: Detection & Calibration Metrics")
        print("=" * 60)
        print(f"Experiment: {experiment_id}")
        print()

    # Initialize storage for results
    section6_storage = Section6Storage(storage)

    # Load all required data
    if verbose:
        print("Loading data from MongoDB...")

    # 1. Load evaluation units
    eval_units = _load_evaluation_units(storage, experiment_id)
    if verbose:
        print(f"  Evaluation units: {len(eval_units)}")

    # 2. Load judge outputs
    judge_outputs = _load_judge_outputs(storage, experiment_id)
    if verbose:
        print(f"  Judge outputs: {len(judge_outputs)}")
        judges = set(jo.judge_model for jo in judge_outputs)
        print(f"  Judge models: {list(judges)}")

    # 3. Load outcome evidence
    outcome_records = _load_outcome_records(storage, experiment_id)
    if verbose:
        print(f"  Outcome records: {len(outcome_records)}")

    # 4. Load human labels (aggregated)
    human_labels = _load_human_labels(storage, experiment_id)
    if verbose:
        print(f"  Human labels: {len(human_labels)}")

    print()

    # Index data by evaluation_unit_id
    eval_unit_by_id = {eu.evaluation_unit_id: eu for eu in eval_units}
    outcome_by_unit_id = {or_.evaluation_unit_id: or_ for or_ in outcome_records}
    human_by_unit_id = {hl.evaluation_unit_id: hl for hl in human_labels}

    # Group judge outputs by (evaluation_unit_id, judge_model)
    judge_outputs_grouped = defaultdict(list)
    for jo in judge_outputs:
        key = (jo.evaluation_unit_id, jo.judge_model)
        judge_outputs_grouped[key].append(jo)

    # Initialize evaluator
    evaluator = Section6Evaluator(experiment_id)

    # Process each (unit, judge) pair
    if verbose:
        print("Computing analysis results...")

    results_saved = 0
    results_skipped = 0
    errors = []

    total_pairs = len(judge_outputs_grouped)
    for idx, ((unit_id, judge_model), outputs) in enumerate(
        judge_outputs_grouped.items(), 1
    ):
        # Skip if result exists and not forcing
        if not force and section6_storage.exists(unit_id, judge_model):
            results_skipped += 1
            continue

        # Get the evaluation unit
        eval_unit = eval_unit_by_id.get(unit_id)
        if not eval_unit:
            errors.append(f"Missing eval_unit: {unit_id}")
            continue

        # Get outcome record (optional)
        outcome_record = outcome_by_unit_id.get(unit_id)

        # Get human label (optional)
        human_label = human_by_unit_id.get(unit_id)

        # Use the first (deterministic) judge output
        # TODO: Could aggregate multiple samples if needed
        judge_output = outputs[0]

        try:
            # Compute analysis result
            result = evaluator.evaluate_unit(
                eval_unit=eval_unit,
                judge_output=judge_output,
                outcome_record=outcome_record,
                human_label=human_label,
            )

            # Save to MongoDB
            section6_storage.save_result(result)
            results_saved += 1

            if verbose and results_saved % 100 == 0:
                print(f"  Progress: {results_saved}/{total_pairs} saved")

        except Exception as e:
            errors.append(f"Error processing {unit_id}/{judge_model}: {str(e)}")

    if verbose:
        print()
        print("-" * 60)
        print("SUMMARY")
        print("-" * 60)
        print(f"  Total pairs: {total_pairs}")
        print(f"  Saved: {results_saved}")
        print(f"  Skipped (existing): {results_skipped}")
        print(f"  Errors: {len(errors)}")

        if errors:
            print()
            print("Errors:")
            for err in errors[:10]:
                print(f"  - {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

    return {
        "experiment_id": experiment_id,
        "total_pairs": total_pairs,
        "saved": results_saved,
        "skipped": results_skipped,
        "errors": errors,
    }


def _load_evaluation_units(
    storage: MongoDBStorage, experiment_id: str
) -> List[EvaluationUnit]:
    """Load evaluation units from MongoDB."""
    collection = storage.db["evaluation_units"]
    docs = list(collection.find({"experiment_id": experiment_id}))
    return [EvaluationUnit.from_dict(doc) for doc in docs]


def _load_judge_outputs(
    storage: MongoDBStorage, experiment_id: str
) -> List[Section5JudgeOutput]:
    """Load judge outputs from MongoDB."""
    collection = storage.db["judge_eval_outputs"]
    docs = list(collection.find({"experiment_id": experiment_id}))
    return [Section5JudgeOutput.from_dict(doc) for doc in docs]


def _load_outcome_records(
    storage: MongoDBStorage, experiment_id: str
) -> List[OutcomeRecord]:
    """Load outcome evidence records from MongoDB."""
    collection = storage.db["outcome_evidence"]
    docs = list(collection.find({"experiment_id": experiment_id}))
    return [OutcomeRecord.from_dict(doc) for doc in docs]


def _load_human_labels(
    storage: MongoDBStorage, experiment_id: str
) -> List[AggregatedLabel]:
    """
    Load aggregated human labels from MongoDB.

    Auto-aggregates from raw human_labels if aggregated_human_labels is empty.
    """
    collection = storage.db["aggregated_human_labels"]
    docs = list(collection.find({"experiment_id": experiment_id}))

    if docs:
        return [AggregatedLabel.from_dict(doc) for doc in docs]

    # Check for raw human labels and auto-aggregate
    raw_collection = storage.db["human_labels"]
    raw_docs = list(raw_collection.find({"experiment_id": experiment_id}))

    if not raw_docs:
        return []

    # Auto-aggregate
    from src.human_labels.schema import AnnotationRecord
    from src.human_labels.aggregation import aggregate_annotations

    print(f"  Auto-aggregating {len(raw_docs)} raw human labels...")

    annotations = []
    for doc in raw_docs:
        try:
            ann = AnnotationRecord.from_dict(doc)
            annotations.append(ann)
        except Exception:
            pass

    if not annotations:
        return []

    aggregated = aggregate_annotations(annotations)

    # Save to aggregated_human_labels
    for agg in aggregated:
        doc = agg.to_dict()
        doc["experiment_id"] = experiment_id
        collection.update_one(
            {"evaluation_unit_id": agg.evaluation_unit_id},
            {"$set": doc},
            upsert=True,
        )

    print(f"  Saved {len(aggregated)} aggregated labels")

    return aggregated


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Section 6 analysis (detection + calibration metrics)"
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Experiment ID to analyze",
    )
    parser.add_argument(
        "--database",
        default="agent_judge_experiment",
        help="MongoDB database name",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-computation even if results exist",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Connect to MongoDB
    storage = MongoDBStorage(database=args.database)

    try:
        result = run_section6_analysis(
            experiment_id=args.experiment_id,
            storage=storage,
            force=args.force,
            verbose=not args.quiet,
        )

        if result["errors"]:
            exit(1)

    finally:
        storage.close()


if __name__ == "__main__":
    main()
