"""
Outcome Degradation (OD) Scorer.

Computes OD by grading final answers from baseline and perturbed trajectories.
OD = (baseline_outcome - perturbed_outcome) / 100

Tier 3 Static Grading: Grade trajectory final answers using LLM.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from src.llm import get_bedrock_client


@dataclass
class ODResult:
    """Result of OD computation for a single perturbation."""
    perturbation_id: str
    baseline_outcome: float  # 0-100
    perturbed_outcome: float  # 0-100
    od_value: float  # (baseline - perturbed) / 100
    tier: int
    method: str
    grader_model: str
    baseline_reasoning: str
    perturbed_reasoning: str
    computed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation_id": self.perturbation_id,
            "value": self.od_value,
            "baseline_outcome": self.baseline_outcome,
            "perturbed_outcome": self.perturbed_outcome,
            "tier": self.tier,
            "method": self.method,
            "grader_model": self.grader_model,
            "baseline_reasoning": self.baseline_reasoning,
            "perturbed_reasoning": self.perturbed_reasoning,
            "computed_at": self.computed_at
        }


OUTCOME_GRADER_PROMPT = """You are an expert evaluator of AI agent task completion.

TASK DESCRIPTION:
{task_description}

AGENT'S FINAL ANSWER/OUTPUT:
{final_answer}

GROUND TRUTH (if available):
{ground_truth}

Evaluate how well the agent completed the task. Consider:
1. Does the final answer address the user's request?
2. Is the answer correct and complete?
3. Are there any significant errors or omissions?

Score from 0-100:
- 100: Perfect completion - all requirements met, correct answer
- 80-99: Good completion - minor issues but task essentially done
- 60-79: Partial completion - task partially done, some errors
- 40-59: Poor completion - significant issues, incomplete
- 20-39: Mostly failed - major errors, barely addressed task
- 0-19: Complete failure - task not completed at all

Respond in JSON format:
{{
    "score": <0-100>,
    "reasoning": "<brief explanation of score>"
}}"""


class ODScorer:
    """
    Computes Outcome Degradation for perturbations.

    OD measures actual impact on task outcome by comparing
    baseline vs perturbed trajectory final answers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OD scorer.

        Args:
            config: OD configuration with:
                - grader_model: Bedrock model ID
                - temperature: Sampling temperature
                - max_tokens: Max tokens for grader response
        """
        self.config = config
        self.grader_model = config.get(
            "grader_model",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1000)
        self.tier = config.get("tier", 3)

        # Initialize Bedrock client
        log_calls = config.get("log_calls", False)
        self.client = get_bedrock_client(log_calls=log_calls)

        # Track stats
        self.graded_count = 0
        self.failed_count = 0

    def extract_final_answer(self, trajectory: Dict[str, Any]) -> str:
        """
        Extract final answer from a trajectory.

        Looks for:
        1. Last assistant message content
        2. Last tool output
        3. Final step content

        Args:
            trajectory: Trajectory dict with steps

        Returns:
            Final answer string
        """
        steps = trajectory.get("steps", [])
        if not steps:
            return "[No steps in trajectory]"

        # Work backwards through steps
        for step in reversed(steps):
            step_type = step.get("step_type", "")
            content = step.get("content", "")

            # Check for final assistant response
            if step_type == "assistant" or step_type == "response":
                if content and len(content.strip()) > 10:
                    return content[:2000]  # Truncate for prompt size

            # Check for tool output that might be the answer
            tool_output = step.get("tool_output", "")
            if tool_output and len(tool_output.strip()) > 10:
                # Only use if it looks like a final answer, not intermediate
                if len(steps) <= 3:
                    return tool_output[:2000]

        # Fallback: use last step content
        last_step = steps[-1]
        content = last_step.get("content", "") or last_step.get("tool_output", "")
        return content[:2000] if content else "[No final answer found]"

    def get_task_description(self, trajectory: Dict[str, Any]) -> str:
        """Extract task description from trajectory."""
        ground_truth = trajectory.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            return ground_truth.get("task_description", "[No task description]")
        return "[No task description]"

    def get_ground_truth_answer(self, trajectory: Dict[str, Any]) -> str:
        """Extract ground truth answer if available."""
        ground_truth = trajectory.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            answer = ground_truth.get("expected_answer", "")
            if not answer:
                answer = ground_truth.get("answer", "")
            return answer or "Not available"
        return "Not available"

    def grade_outcome(
        self,
        trajectory: Dict[str, Any],
        task_description: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Grade a trajectory's outcome.

        Args:
            trajectory: Trajectory dict
            task_description: Override task description

        Returns:
            Tuple of (score 0-100, reasoning)
        """
        # Extract components
        final_answer = self.extract_final_answer(trajectory)
        task = task_description or self.get_task_description(trajectory)
        ground_truth = self.get_ground_truth_answer(trajectory)

        # Build prompt
        prompt = OUTCOME_GRADER_PROMPT.format(
            task_description=task,
            final_answer=final_answer,
            ground_truth=ground_truth
        )

        try:
            # Call LLM
            result = self.client.invoke(
                model_id=self.grader_model,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            response_text = result.get("response", "")

            # Parse JSON response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                score = float(parsed.get("score", 50))
                reasoning = parsed.get("reasoning", "")

                # Clamp score to valid range
                score = max(0, min(100, score))

                self.graded_count += 1
                return score, reasoning
            else:
                # Try to extract score from text
                score_match = re.search(r'"score"\s*:\s*(\d+)', response_text)
                if score_match:
                    score = float(score_match.group(1))
                    score = max(0, min(100, score))
                    self.graded_count += 1
                    return score, "Parsed from text"

                self.failed_count += 1
                return 50.0, "Failed to parse response"

        except Exception as e:
            self.failed_count += 1
            return 50.0, f"Grading error: {str(e)}"

    def compute_od(
        self,
        baseline_trajectory: Dict[str, Any],
        perturbed_trajectory: Dict[str, Any],
        perturbation_id: str
    ) -> ODResult:
        """
        Compute OD for a single perturbation.

        Args:
            baseline_trajectory: Original trajectory dict
            perturbed_trajectory: Perturbed trajectory dict
            perturbation_id: Perturbation identifier

        Returns:
            ODResult with scores and OD value
        """
        # Get task description (same for both)
        task_description = self.get_task_description(baseline_trajectory)

        # Grade both trajectories
        baseline_score, baseline_reasoning = self.grade_outcome(
            baseline_trajectory, task_description
        )
        perturbed_score, perturbed_reasoning = self.grade_outcome(
            perturbed_trajectory, task_description
        )

        # Compute OD = (baseline - perturbed) / 100
        od_value = (baseline_score - perturbed_score) / 100.0

        return ODResult(
            perturbation_id=perturbation_id,
            baseline_outcome=baseline_score,
            perturbed_outcome=perturbed_score,
            od_value=od_value,
            tier=self.tier,
            method="static_grading",
            grader_model=self.grader_model,
            baseline_reasoning=baseline_reasoning,
            perturbed_reasoning=perturbed_reasoning,
            computed_at=datetime.utcnow().isoformat()
        )

    def compute_batch(
        self,
        perturbations: List[Dict[str, Any]],
        storage,
        batch_size: int = 20,
        resume: bool = True
    ) -> List[ODResult]:
        """
        Compute OD for a batch of perturbations.

        Args:
            perturbations: List of perturbation dicts with:
                - perturbation_id
                - original_trajectory_id
                - perturbed_trajectory_id
            storage: MongoDBStorage instance
            batch_size: Number to process before checkpointing
            resume: Skip perturbations that already have OD

        Returns:
            List of ODResult objects
        """
        results = []
        skipped = 0

        for i, pert in enumerate(perturbations):
            pert_id = pert.get("perturbation_id")

            # Check if already computed (resume mode)
            if resume and pert.get("od"):
                skipped += 1
                continue

            # Load trajectories
            baseline_traj = storage.get_trajectory(
                pert.get("original_trajectory_id")
            )
            perturbed_traj = storage.get_trajectory(
                pert.get("perturbed_trajectory_id")
            )

            if not baseline_traj or not perturbed_traj:
                print(f"   Warning: Missing trajectory for {pert_id}")
                continue

            # Compute OD
            try:
                od_result = self.compute_od(
                    baseline_traj, perturbed_traj, pert_id
                )
                results.append(od_result)

                # Update perturbation in MongoDB
                storage.perturbations.update_one(
                    {"perturbation_id": pert_id},
                    {"$set": {"od": od_result.to_dict()}}
                )

            except Exception as e:
                print(f"   Error computing OD for {pert_id}: {e}")
                continue

            # Progress logging
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(perturbations)} "
                      f"(skipped: {skipped})")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get grading statistics."""
        return {
            "graded_count": self.graded_count,
            "failed_count": self.failed_count,
            "success_rate": (
                self.graded_count / (self.graded_count + self.failed_count)
                if (self.graded_count + self.failed_count) > 0 else 0
            )
        }
