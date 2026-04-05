"""
Perturbation Quality Scorer - scores perturbation validity using LLM.

Modes:
- single: N LLM calls per perturbation (1 per custom metric)
- batch: 1 LLM call per perturbation (all metrics at once)
"""

import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from src.llm import get_bedrock_client
from .prompts import format_metric_prompt, format_batch_prompt, get_metric_names


class PerturbationQualityScorer:
    """
    Scores perturbation validity using LLM-as-Judge.

    Modes:
    - single: 1 LLM call per metric (N calls for N metrics)
    - batch: 1 LLM call for all metrics
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        tier_thresholds: Optional[Dict[str, int]] = None,
        judge_mode: str = "batch"
    ):
        """
        Initialize quality scorer with LLM model config.

        Args:
            model_config: LLM model configuration dict
            metrics: Quality metric definitions (optional)
            tier_thresholds: Score thresholds for quality tiers
            judge_mode: "single" (1 call per metric) or "batch" (1 call for all metrics)
        """
        self.model_config = model_config
        self.model_id = model_config.get("model_id")
        self.model_name = model_config.get("name", "quality-scorer")
        self.judge_mode = judge_mode

        config = model_config.get("config", {})
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 500)

        self.metrics = metrics or {}
        self.metric_names = get_metric_names()
        self.tier_thresholds = tier_thresholds or {
            "high": 6,
            "medium": 4,
            "low": 1,
            "invalid": 0
        }

        # Use central Bedrock client (logging controlled globally via --log-bedrock)
        self.bedrock = get_bedrock_client()

        print(f"Initialized PerturbationQualityScorer: {self.model_name} (mode: {judge_mode})")

    def score_perturbation(self, perturbation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single perturbation.

        Args:
            perturbation: Perturbation document

        Returns:
            Quality score dict with all metrics
        """
        if self.judge_mode == "single":
            return self._score_single_mode(perturbation)
        else:
            return self._score_batch_mode(perturbation)

    def _score_single_mode(self, perturbation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score perturbation with 1 LLM call per metric (N calls total).
        """
        original = perturbation.get("original_step_content", "")
        perturbed = perturbation.get("perturbed_step_content", "")
        p_type = perturbation.get("perturbation_type", "unknown")
        p_position = perturbation.get("perturbation_position", "unknown")
        metadata = perturbation.get("perturbation_metadata", {})
        description = metadata.get("description", "")

        score = {}
        total_tokens = 0
        total_time_ms = 0
        reasonings = []

        # Make 1 LLM call per metric
        for metric_name in self.metric_names:
            prompt = format_metric_prompt(
                metric_name=metric_name,
                original_step_content=original,
                perturbed_step_content=perturbed,
                perturbation_type=p_type,
                perturbation_position=p_position,
                perturbation_description=description
            )

            try:
                result = self._call_llm(prompt)
                response_text = result.get("response", "")
                total_tokens += result.get("tokens_used", 0)
                total_time_ms += result.get("time_ms", 0)

                metric_score = self._parse_single_metric_response(
                    response_text, metric_name
                )
                score[metric_name] = metric_score["value"]
                if metric_score.get("reasoning"):
                    reasonings.append(f"{metric_name}: {metric_score['reasoning']}")

            except Exception as e:
                score[metric_name] = 0
                reasonings.append(f"{metric_name}: ERROR - {str(e)}")

        # Calculate total and add metadata
        score["total_score"] = (
            score.get("content_changed", 0) +
            score.get("syntactically_valid", 0) +
            score.get("semantically_meaningful", 0) +
            score.get("type_matches_intent", 0) +
            score.get("realistic_error", 0)
        )
        score["reasoning"] = " | ".join(reasonings)
        score["quality_tier"] = self._assign_tier(score["total_score"])
        score["scored_by"] = self.model_name
        score["scored_at"] = datetime.now(timezone.utc).isoformat()
        score["tokens_used"] = total_tokens
        score["time_ms"] = total_time_ms
        score["llm_calls"] = len(self.metric_names)

        return score

    def _score_batch_mode(self, perturbation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score perturbation with 1 LLM call for all metrics.
        """
        original = perturbation.get("original_step_content", "")
        perturbed = perturbation.get("perturbed_step_content", "")
        p_type = perturbation.get("perturbation_type", "unknown")
        p_position = perturbation.get("perturbation_position", "unknown")
        metadata = perturbation.get("perturbation_metadata", {})
        description = metadata.get("description", "")

        prompt = format_batch_prompt(
            original_step_content=original,
            perturbed_step_content=perturbed,
            perturbation_type=p_type,
            perturbation_position=p_position,
            perturbation_description=description
        )

        try:
            result = self._call_llm(prompt)
            response_text = result.get("response", "")

            score = self._parse_batch_response(response_text)
            score["quality_tier"] = self._assign_tier(score["total_score"])
            score["scored_by"] = self.model_name
            score["scored_at"] = datetime.now(timezone.utc).isoformat()
            score["tokens_used"] = result.get("tokens_used", 0)
            score["time_ms"] = result.get("time_ms", 0)
            score["llm_calls"] = 1

            return score

        except Exception as e:
            return self._default_error_score(str(e))

    def score_batch(
        self,
        perturbations: List[Dict[str, Any]],
        batch_size: int = 10,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Score multiple perturbations.

        Args:
            perturbations: List of perturbation documents
            batch_size: Not used (kept for API compatibility)
            progress_callback: Optional progress callback

        Returns:
            List of quality scores
        """
        scores = []
        for perturbation in perturbations:
            score = self.score_perturbation(perturbation)
            scores.append(score)
        return scores

    def _parse_single_metric_response(
        self, response_text: str, metric_name: str
    ) -> Dict[str, Any]:
        """Parse LLM response for a single metric."""
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response_text[:200]}")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        if metric_name not in data:
            raise ValueError(f"Missing {metric_name} in response")

        # Get max value for this metric
        max_val = 3 if metric_name == "realistic_error" else 1

        return {
            "value": self._clamp(int(data[metric_name]), 0, max_val),
            "reasoning": data.get("reasoning", "")
        }

    def _parse_batch_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response with all metrics."""
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response_text[:200]}")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        score = {
            "content_changed": self._clamp(int(data.get("content_changed", 0)), 0, 1),
            "syntactically_valid": self._clamp(int(data.get("syntactically_valid", 0)), 0, 1),
            "semantically_meaningful": self._clamp(int(data.get("semantically_meaningful", 0)), 0, 1),
            "type_matches_intent": self._clamp(int(data.get("type_matches_intent", 0)), 0, 1),
            "realistic_error": self._clamp(int(data.get("realistic_error", 0)), 0, 3),
            "reasoning": data.get("reasoning", "")
        }

        score["total_score"] = (
            score["content_changed"] +
            score["syntactically_valid"] +
            score["semantically_meaningful"] +
            score["type_matches_intent"] +
            score["realistic_error"]
        )

        return score

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the LLM via central Bedrock client."""
        return self.bedrock.invoke(
            model_id=self.model_id,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

    def _clamp(self, value: int, min_val: int, max_val: int) -> int:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))

    def _assign_tier(self, total_score: int) -> str:
        """Assign quality tier based on total score."""
        if total_score >= self.tier_thresholds["high"]:
            return "high"
        elif total_score >= self.tier_thresholds["medium"]:
            return "medium"
        elif total_score >= self.tier_thresholds["low"]:
            return "low"
        else:
            return "invalid"

    def _default_error_score(self, error_message: str) -> Dict[str, Any]:
        """Return default score when scoring fails."""
        return {
            "content_changed": 0,
            "syntactically_valid": 0,
            "semantically_meaningful": 0,
            "type_matches_intent": 0,
            "realistic_error": 0,
            "total_score": 0,
            "reasoning": f"Scoring failed: {error_message}",
            "quality_tier": "invalid",
            "scored_by": self.model_name,
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "error": error_message,
            "llm_calls": 0
        }


def create_quality_scorer(config: Dict[str, Any]) -> PerturbationQualityScorer:
    """Factory function to create quality scorer from config."""
    model_config = config.get("judge_model", {})
    metrics = config.get("metrics", {})
    tier_thresholds = config.get("tier_thresholds", {})
    judge_mode = config.get("judge_mode", "batch")

    return PerturbationQualityScorer(
        model_config=model_config,
        metrics=metrics,
        tier_thresholds=tier_thresholds,
        judge_mode=judge_mode
    )
