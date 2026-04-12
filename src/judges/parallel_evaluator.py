"""
Parallel judge evaluation runner with configurable parallelization and checkpointing.

This module provides efficient parallel evaluation of perturbations using ThreadPoolExecutor,
with support for batching, checkpointing, rate limiting, and resume functionality.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from src.judges import Judge
from src.judges.schema import JudgeOutput
from src.data.schema import Trajectory
from src.storage.mongodb import MongoDBStorage


class ParallelJudgeEvaluator:
    """
    Parallel evaluation of perturbations with checkpointing.

    Features:
    - Configurable parallelization (N concurrent LLM calls)
    - Checkpoint to MongoDB every checkpoint_batch_size evaluations
    - Resume from checkpoint (skip already-evaluated perturbations)
    - Rate limiting between batches
    - Graceful error handling (failed evaluations don't block batch)

    Config:
        judge_parallelization: int (number of concurrent calls)
        checkpoint_batch_size: int (save to DB after N perturbations)
        rate_limit_delay_seconds: float (delay between batches)
    """

    def __init__(self, judge: Judge, storage: MongoDBStorage, config: Dict[str, Any]):
        """
        Initialize parallel evaluator.

        Args:
            judge: Judge instance (from src/judges/)
            storage: MongoDBStorage instance
            config: Config dict with judge settings:
                - judge_parallelization: Number of concurrent evaluations
                - checkpoint_batch_size: Save to DB every N evaluations
                - rate_limit_delay_seconds: Delay between batches
                - retry_on_failure: Whether to retry failed evaluations
                - max_retries: Maximum retry attempts
        """
        self.judge = judge
        self.storage = storage
        self.parallelization = config.get("judge_parallelization", 2)
        self.checkpoint_size = config.get("checkpoint_batch_size", 20)
        self.rate_limit_delay = config.get("rate_limit_delay_seconds", 0.5)
        self.retry_on_failure = config.get("retry_on_failure", True)
        self.max_retries = config.get("max_retries", 3)

    def evaluate_all(
        self, perturbations: List[Dict], experiment_id: str, resume: bool = True
    ) -> List[Dict]:
        """
        Evaluate all perturbations with parallelization.

        Args:
            perturbations: List of perturbation dicts from MongoDB
            experiment_id: Experiment ID for storing results
            resume: If True, skip already-evaluated perturbations

        Returns:
            List of evaluation results
        """
        print(f"\n{'=' * 70}")
        print("PARALLEL JUDGE EVALUATION")
        print(f"{'=' * 70}")
        print(f"Judge: {self.judge.name}")
        print(f"Parallelization: {self.parallelization}")
        print(f"Checkpoint size: {self.checkpoint_size}")
        print(f"Total perturbations: {len(perturbations)}")

        # Filter already evaluated if resuming
        if resume:
            perturbations = self._filter_evaluated(perturbations, experiment_id)
            print(f"Remaining after resume filter: {len(perturbations)}")

        if not perturbations:
            print("No perturbations to evaluate.")
            return []

        print(f"{'=' * 70}\n")

        results = []
        total = len(perturbations)
        checkpointed = 0

        # Process in parallel batches
        for i in range(0, total, self.parallelization):
            batch = perturbations[i : i + self.parallelization]
            batch_num = i // self.parallelization + 1
            total_batches = (total + self.parallelization - 1) // self.parallelization

            print(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} perturbations)..."
            )

            batch_results = self._evaluate_batch_parallel(batch, experiment_id)
            results.extend(batch_results)

            # Checkpoint periodically
            successful_in_batch = [
                r for r in batch_results if r.get("status") == "success"
            ]
            if successful_in_batch:
                self._checkpoint(successful_in_batch, experiment_id)
                checkpointed += len(successful_in_batch)

                # Calculate success rate for this batch
                success_count = len(successful_in_batch)
                fail_count = len(batch_results) - success_count
                print(
                    f"  Checkpoint: {checkpointed}/{total} evaluated "
                    f"(batch: {success_count} success, {fail_count} failed)"
                )

            # Rate limiting between batches
            if i + self.parallelization < total:
                time.sleep(self.rate_limit_delay)

        # Summary
        successful = len([r for r in results if r.get("status") == "success"])
        failed = len([r for r in results if r.get("status") == "failed"])

        print(f"\n{'=' * 70}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total evaluated: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"{'=' * 70}\n")

        return results

    def _evaluate_batch_parallel(
        self, batch: List[Dict], experiment_id: str
    ) -> List[Dict]:
        """
        Evaluate batch using ThreadPoolExecutor.

        Args:
            batch: List of perturbation dicts to evaluate
            experiment_id: Experiment ID

        Returns:
            List of evaluation results
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.parallelization) as executor:
            # Submit all tasks
            future_to_pert = {
                executor.submit(self._evaluate_single, pert, experiment_id): pert
                for pert in batch
            }

            # Collect results as they complete
            for future in as_completed(future_to_pert):
                pert = future_to_pert[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Log error but continue
                    pert_id = pert.get("perturbation_id", "unknown")
                    print(f"    Error evaluating {pert_id}: {e}")
                    results.append(
                        {
                            "perturbation_id": pert_id,
                            "trajectory_id": pert.get("perturbed_trajectory_id"),
                            "status": "failed",
                            "error": str(e),
                        }
                    )

        return results

    def _evaluate_single(self, perturbation: Dict, experiment_id: str) -> Dict:
        """
        Evaluate a single perturbation.

        Args:
            perturbation: Perturbation dict from MongoDB
            experiment_id: Experiment ID

        Returns:
            Evaluation result dict
        """
        pert_id = perturbation.get("perturbation_id", "unknown")
        perturbed_traj_id = perturbation.get("perturbed_trajectory_id")

        try:
            # Load trajectory from perturbation
            trajectory = self._load_trajectory(perturbation, experiment_id)

            if not trajectory:
                return {
                    "perturbation_id": pert_id,
                    "trajectory_id": perturbed_traj_id,
                    "status": "failed",
                    "error": f"Trajectory {perturbed_traj_id} not found",
                }

            # Call judge
            output = self.judge.evaluate(
                trajectory,
                retry_on_failure=self.retry_on_failure,
                max_retries=self.max_retries,
            )

            if output:
                # Compute JPS (Judge Penalty Score)
                jps = 100 - output.overall_score

                return {
                    "perturbation_id": pert_id,
                    "trajectory_id": perturbed_traj_id,
                    "evaluation": output.to_dict(),
                    "jps": jps,
                    "overall_score": output.overall_score,
                    "status": "success",
                    # Include perturbation metadata for CCG calculation
                    "perturbation_type": perturbation.get("perturbation_type"),
                    "perturbation_position": perturbation.get("perturbation_position"),
                    "benchmark": self._get_benchmark(perturbation),
                }
            else:
                return {
                    "perturbation_id": pert_id,
                    "trajectory_id": perturbed_traj_id,
                    "status": "failed",
                    "error": "Judge returned None",
                }

        except Exception as e:
            return {
                "perturbation_id": pert_id,
                "trajectory_id": perturbed_traj_id,
                "status": "failed",
                "error": str(e),
            }

    def _filter_evaluated(
        self, perturbations: List[Dict], experiment_id: str
    ) -> List[Dict]:
        """
        Filter out already-evaluated perturbations.

        Args:
            perturbations: List of perturbation dicts
            experiment_id: Experiment ID

        Returns:
            Filtered list of perturbations that need evaluation
        """
        remaining = []

        for pert in perturbations:
            perturbed_traj_id = pert.get("perturbed_trajectory_id")

            # Check if evaluation exists for this trajectory
            existing_count = self.storage.count_judge_outputs(
                experiment_id=experiment_id,
                trajectory_id=perturbed_traj_id,
                judge_name=self.judge.name,
            )

            if existing_count == 0:
                remaining.append(pert)

        return remaining

    def _checkpoint(self, results: List[Dict], experiment_id: str):
        """
        Save results to MongoDB.

        Args:
            results: List of successful evaluation results
            experiment_id: Experiment ID
        """
        for result in results:
            if result.get("status") == "success" and result.get("evaluation"):
                # Create JudgeOutput from the evaluation dict
                evaluation = result["evaluation"]

                # Store using existing storage method
                self.storage.store_judge_output(
                    JudgeOutput.from_dict(evaluation), experiment_id, sample_number=1
                )

    def _load_trajectory(
        self, perturbation: Dict, experiment_id: str
    ) -> Optional[Trajectory]:
        """
        Load trajectory object from perturbation record.

        Args:
            perturbation: Perturbation dict with perturbed_trajectory_id
            experiment_id: Experiment ID (fallback if not in perturbation)

        Returns:
            Trajectory object or None if not found
        """
        perturbed_traj_id = perturbation.get("perturbed_trajectory_id")

        # Use experiment_id from the perturbation (where trajectory is stored)
        # Fall back to passed experiment_id if not present
        source_exp_id = perturbation.get("experiment_id", experiment_id)

        # Get trajectory from MongoDB
        traj_dict = self.storage.get_trajectory_by_experiment(
            perturbed_traj_id, source_exp_id
        )

        if not traj_dict:
            return None

        # Clean dict for Trajectory.from_dict()
        clean_dict = {
            "trajectory_id": traj_dict["trajectory_id"],
            "benchmark": traj_dict["benchmark"],
            "steps": traj_dict["steps"],
            "ground_truth": traj_dict["ground_truth"],
            "metadata": traj_dict.get("metadata", {}),
            "domain": traj_dict.get("domain"),
            "complexity": traj_dict.get("complexity"),
        }

        return Trajectory.from_dict(clean_dict)

    def _get_benchmark(self, perturbation: Dict) -> str:
        """
        Extract benchmark from perturbation or trajectory ID.

        Args:
            perturbation: Perturbation dict

        Returns:
            Benchmark name (toolbench, gaia, swebench)
        """
        traj_id = perturbation.get("original_trajectory_id", "").lower()

        if "toolbench" in traj_id:
            return "toolbench"
        elif "gaia" in traj_id:
            return "gaia"
        elif "swe" in traj_id:
            return "swebench"

        return "unknown"

    def get_evaluation_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for evaluations.

        Args:
            experiment_id: Experiment ID

        Returns:
            Summary dict with counts and averages
        """
        outputs = self.storage.get_judge_outputs(
            experiment_id=experiment_id, judge_name=self.judge.name
        )

        if not outputs:
            return {"count": 0, "avg_score": 0.0, "avg_jps": 0.0}

        scores = [o.get("overall_score", 0) for o in outputs]
        jps_values = [100 - s for s in scores]

        return {
            "count": len(outputs),
            "avg_score": sum(scores) / len(scores),
            "avg_jps": sum(jps_values) / len(jps_values),
            "min_score": min(scores),
            "max_score": max(scores),
        }
