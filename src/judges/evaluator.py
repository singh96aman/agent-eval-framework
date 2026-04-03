"""
Judge evaluation runner with batching, rate limiting, and progress tracking.
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from src.judges import Judge
from src.judges.schema import JudgeOutput, EvaluationResults
from src.data.schema import Trajectory
from src.storage.mongodb import MongoDBStorage


class JudgeEvaluator:
    """
    Orchestrates batch evaluation of trajectories by LLM judges.

    Features:
    - Batch processing with progress tracking
    - Rate limiting to avoid API throttling
    - Checkpointing for resumability
    - MongoDB storage of results
    - Multiple samples per trajectory
    """

    def __init__(
        self,
        storage: MongoDBStorage,
        judges: List[Judge],
        batch_size: int = 10,
        rate_limit_delay: float = 1.0,
        samples_per_trajectory: int = 3
    ):
        """
        Initialize evaluator.

        Args:
            storage: MongoDB storage backend
            judges: List of Judge instances to use
            batch_size: Number of trajectories to process before checkpoint
            rate_limit_delay: Delay in seconds between API calls
            samples_per_trajectory: How many times to evaluate each trajectory
        """
        self.storage = storage
        self.judges = judges
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.samples_per_trajectory = samples_per_trajectory

    def evaluate_experiment(
        self,
        experiment_id: str,
        resume: bool = True,
        dry_run: bool = False
    ) -> Dict[str, EvaluationResults]:
        """
        Evaluate all perturbations for an experiment with all judges.

        Args:
            experiment_id: Experiment identifier
            resume: If True, skip already-evaluated trajectories
            dry_run: If True, don't actually call APIs or store results

        Returns:
            Dict mapping judge_name -> EvaluationResults
        """
        print("\n" + "=" * 70)
        print("🎯 JUDGE EVALUATION")
        print("=" * 70)
        print(f"Experiment: {experiment_id}")
        print(f"Judges: {[j.name for j in self.judges]}")
        print(f"Samples per trajectory: {self.samples_per_trajectory}")
        print(f"Batch size: {self.batch_size}")
        print(f"Rate limit delay: {self.rate_limit_delay}s")
        print(f"Dry run: {dry_run}")
        print("=" * 70)

        # Load perturbations to evaluate
        perturbations = self.storage.get_perturbations_by_experiment(experiment_id)
        print(f"\n📥 Loaded {len(perturbations)} perturbations to evaluate")

        if len(perturbations) == 0:
            print("⚠️  No perturbations found for this experiment")
            return {}

        # Evaluate with each judge
        results_by_judge = {}

        for judge in self.judges:
            print(f"\n{'=' * 70}")
            print(f"🔮 EVALUATING WITH: {judge.name}")
            print(f"{'=' * 70}")

            results = self._evaluate_with_judge(
                experiment_id=experiment_id,
                judge=judge,
                perturbations=perturbations,
                resume=resume,
                dry_run=dry_run
            )

            results_by_judge[judge.name] = results

            # Print judge stats
            stats = judge.get_stats()
            print(f"\n📊 {judge.name} Statistics:")
            print(f"   Total calls: {stats['total_calls']}")
            print(f"   Failed calls: {stats['failed_calls']}")
            print(f"   Success rate: {stats['success_rate']:.1%}")
            print(f"   Total tokens: {stats['total_tokens']:,}")
            print(f"   Avg time per call: {stats['avg_time_per_call_ms']:.0f}ms")

        print(f"\n{'=' * 70}")
        print("✅ JUDGE EVALUATION COMPLETE")
        print(f"{'=' * 70}\n")

        return results_by_judge

    def _evaluate_with_judge(
        self,
        experiment_id: str,
        judge: Judge,
        perturbations: List[Dict[str, Any]],
        resume: bool,
        dry_run: bool
    ) -> EvaluationResults:
        """
        Evaluate all perturbations with a single judge.

        Args:
            experiment_id: Experiment ID
            judge: Judge instance
            perturbations: List of perturbation dicts
            resume: Skip already-evaluated trajectories
            dry_run: Don't call APIs or store results

        Returns:
            EvaluationResults summary
        """
        start_time = time.time()

        # Filter out already-evaluated if resuming
        if resume:
            perturbations = self._filter_evaluated(
                perturbations,
                judge.name,
                experiment_id
            )
            print(f"   ℹ️  Resume mode: {len(perturbations)} remaining to evaluate")

        # Calculate total work
        total_evaluations = len(perturbations) * self.samples_per_trajectory
        print(f"   📋 Total evaluations: {total_evaluations} "
              f"({len(perturbations)} perturbations × {self.samples_per_trajectory} samples)")

        # Track progress
        successful = 0
        failed = 0
        total_tokens = 0
        evaluation_errors = []

        # Process in batches
        for i, pert in enumerate(perturbations, 1):
            # Convert perturbation dict to Trajectory
            trajectory = Trajectory.from_dict(pert)

            # Evaluate multiple times (for statistical reliability)
            for sample in range(self.samples_per_trajectory):
                if dry_run:
                    print(f"   [DRY RUN] Would evaluate {trajectory.trajectory_id} "
                          f"(sample {sample + 1}/{self.samples_per_trajectory})")
                    continue

                try:
                    # Rate limiting
                    if successful + failed > 0:
                        time.sleep(self.rate_limit_delay)

                    # Evaluate
                    output = judge.evaluate(trajectory)

                    if output:
                        # Store result
                        self.storage.store_judge_output(output, experiment_id)
                        successful += 1
                        total_tokens += output.tokens_used or 0
                    else:
                        failed += 1
                        evaluation_errors.append(
                            f"Evaluation failed for {trajectory.trajectory_id} sample {sample + 1}"
                        )

                except Exception as e:
                    failed += 1
                    error_msg = f"Error evaluating {trajectory.trajectory_id}: {e}"
                    evaluation_errors.append(error_msg)
                    print(f"   ❌ {error_msg}")

            # Progress update
            if i % self.batch_size == 0 or i == len(perturbations):
                completed = successful + failed
                percent = (completed / total_evaluations) * 100
                print(f"   ... progress: {completed}/{total_evaluations} "
                      f"({percent:.1f}%) - success: {successful}, failed: {failed}")

        # Calculate summary
        elapsed = time.time() - start_time
        avg_score = self._calculate_average_score(experiment_id, judge.name)

        results = EvaluationResults(
            experiment_id=experiment_id,
            judge_name=judge.name,
            total_evaluated=len(perturbations),
            successful=successful,
            failed=failed,
            total_time_seconds=elapsed,
            total_tokens=total_tokens,
            average_score=avg_score,
            evaluation_errors=evaluation_errors
        )

        return results

    def _filter_evaluated(
        self,
        perturbations: List[Dict[str, Any]],
        judge_name: str,
        experiment_id: str
    ) -> List[Dict[str, Any]]:
        """
        Filter out perturbations that have already been evaluated.

        Args:
            perturbations: List of perturbation dicts
            judge_name: Judge name to check
            experiment_id: Experiment ID

        Returns:
            Filtered list of perturbations
        """
        remaining = []

        for pert in perturbations:
            trajectory_id = pert['trajectory_id']

            # Check if we already have enough samples
            existing_count = self.storage.count_judge_outputs(
                experiment_id=experiment_id,
                trajectory_id=trajectory_id,
                judge_name=judge_name
            )

            if existing_count < self.samples_per_trajectory:
                remaining.append(pert)

        return remaining

    def _calculate_average_score(
        self,
        experiment_id: str,
        judge_name: str
    ) -> float:
        """
        Calculate average overall_score for a judge's evaluations.

        Args:
            experiment_id: Experiment ID
            judge_name: Judge name

        Returns:
            Average score, or 0.0 if no evaluations
        """
        outputs = self.storage.get_judge_outputs(
            experiment_id=experiment_id,
            judge_name=judge_name
        )

        if not outputs:
            return 0.0

        scores = [o.get('overall_score', 0) for o in outputs]
        return sum(scores) / len(scores) if scores else 0.0

    def get_evaluation_summary(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Get summary statistics for all judge evaluations in an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dict with summary stats
        """
        summary = {
            "experiment_id": experiment_id,
            "judges": {},
            "total_evaluations": 0
        }

        for judge in self.judges:
            outputs = self.storage.get_judge_outputs(
                experiment_id=experiment_id,
                judge_name=judge.name
            )

            if outputs:
                scores = [o.get('overall_score', 0) for o in outputs]
                summary["judges"][judge.name] = {
                    "count": len(outputs),
                    "avg_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores)
                }
                summary["total_evaluations"] += len(outputs)

        return summary
