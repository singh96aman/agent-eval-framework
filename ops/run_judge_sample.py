#!/usr/bin/env python3
"""
Quick script to run judge evaluation on a random sample of perturbations.

Usage: python run_judge_sample.py --sample-size 50
"""

import sys
import json
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage.mongodb import MongoDBStorage
from src.judges.claude_judge import create_claude_judge
from src.judges.gpt_oss_judge import create_gpt_oss_judge
from src.judges.evaluator import JudgeEvaluator

def main():
    # Configuration
    experiment_id = "exp_poc_toolbench_20260402"
    sample_size = 50  # Change this to adjust sample size

    print(f"\n{'='*70}")
    print(f"🎯 JUDGE EVALUATION - RANDOM SAMPLE")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_id}")
    print(f"Sample size: {sample_size} perturbations")
    print(f"{'='*70}\n")

    # Load config
    config_path = Path("config/experiments/poc_experiment_toolbench.json")
    with open(config_path) as f:
        config = json.load(f)

    # Initialize storage
    storage = MongoDBStorage()

    # Load all perturbations
    all_perturbations = storage.get_perturbations_by_experiment(experiment_id)
    print(f"📥 Total perturbations available: {len(all_perturbations)}")

    # Random sample
    if len(all_perturbations) > sample_size:
        random.seed(42)  # For reproducibility
        sampled = random.sample(all_perturbations, sample_size)
        print(f"🎲 Randomly sampled: {sample_size} perturbations")
    else:
        sampled = all_perturbations
        print(f"⚠️  Using all available perturbations ({len(sampled)})")

    print()

    # Create judges from config
    judges_config = config.get('judges', {})
    models_config = judges_config.get('models', [])
    samples_per_trajectory = judges_config.get('samples_per_trajectory', 3)

    judges = []
    for model_config in models_config:
        name = model_config.get('name', 'unknown')
        try:
            if 'claude' in name.lower():
                judge = create_claude_judge(model_config)
                judges.append(judge)
            elif 'gpt' in name.lower() or 'oss' in name.lower():
                judge = create_gpt_oss_judge(model_config)
                judges.append(judge)
        except Exception as e:
            print(f"❌ Failed to create judge {name}: {e}")

    print(f"✓ Initialized {len(judges)} judges: {[j.name for j in judges]}")
    print()

    # Create evaluator
    evaluator = JudgeEvaluator(
        storage=storage,
        judges=judges,
        batch_size=10,
        rate_limit_delay=1.0,
        samples_per_trajectory=samples_per_trajectory
    )

    # Run evaluation on sample
    print(f"\n{'='*70}")
    print(f"🔮 STARTING EVALUATION")
    print(f"{'='*70}")
    print(f"Sample size: {len(sampled)} perturbations")
    print(f"Total evaluations: {len(sampled) * len(judges) * samples_per_trajectory}")
    print(f"{'='*70}\n")

    # Temporarily modify the evaluator to use our sample
    # We'll do this by directly calling _evaluate_with_judge for each judge
    results_by_judge = {}

    for judge in judges:
        print(f"\n{'='*70}")
        print(f"🔮 EVALUATING WITH: {judge.name}")
        print(f"{'='*70}")

        results = evaluator._evaluate_with_judge(
            experiment_id=experiment_id,
            judge=judge,
            perturbations=sampled,  # Use our sample!
            resume=True,
            dry_run=False
        )

        results_by_judge[judge.name] = results

        # Print stats
        stats = judge.get_stats()
        print(f"\n📊 {judge.name} Statistics:")
        print(f"   Total calls: {stats['total_calls']}")
        print(f"   Failed calls: {stats['failed_calls']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Total tokens: {stats['total_tokens']:,}")
        print(f"   Avg time per call: {stats['avg_time_per_call_ms']:.0f}ms")

    print(f"\n{'='*70}")
    print("✅ SAMPLE EVALUATION COMPLETE")
    print(f"{'='*70}\n")

    for judge_name, result in results_by_judge.items():
        print(f"\n{judge_name}:")
        print(f"   Evaluated: {result.total_evaluated}")
        print(f"   Successful: {result.successful}")
        print(f"   Failed: {result.failed}")
        print(f"   Average score: {result.average_score:.1f}")
        print(f"   Total time: {result.total_time_seconds:.1f}s")
        print(f"   Total tokens: {result.total_tokens:,}")

    print()
    storage.close()

if __name__ == "__main__":
    main()
