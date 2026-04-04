"""
Perturbation Quality Scorer - scores perturbation validity using LLM.

This module scores perturbations based on validity criteria (not difficulty),
including whether the content changed, is syntactically valid, semantically
meaningful, matches the intended type, and represents a realistic error.
"""

import json
import re
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import boto3
from botocore.exceptions import ClientError

from .prompts import format_quality_scoring_prompt


class PerturbationQualityScorer:
    """
    Scores perturbation validity using LLM-as-Judge.

    Usage:
        scorer = PerturbationQualityScorer(model_config={...})
        score = scorer.score_perturbation(perturbation_doc)
        # Returns: {"total_score": 6, "content_changed": 1, ...}
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        tier_thresholds: Optional[Dict[str, int]] = None,
        region_name: str = "us-east-1"
    ):
        """
        Initialize quality scorer with LLM model config.

        Args:
            model_config: LLM model configuration dict with keys:
                - model_id: Bedrock model ID
                - config: Dict with temperature, max_tokens
            metrics: Quality metric definitions (optional)
            tier_thresholds: Score thresholds for quality tiers
            region_name: AWS region for Bedrock
        """
        self.model_config = model_config
        self.model_id = model_config.get("model_id")
        self.model_name = model_config.get("name", "quality-scorer")

        config = model_config.get("config", {})
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 500)

        self.metrics = metrics or {}
        self.tier_thresholds = tier_thresholds or {
            "high": 6,
            "medium": 4,
            "low": 1,
            "invalid": 0
        }

        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )

        print(f"Initialized PerturbationQualityScorer: {self.model_name}")

    def score_perturbation(
        self,
        perturbation: Dict[str, Any],
        retry_count: int = 0,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Score a single perturbation's validity.

        Args:
            perturbation: Perturbation document with keys:
                - original_step_content: Original step content
                - perturbed_step_content: Perturbed step content
                - perturbation_type: Type of perturbation
                - perturbation_position: Position (early/middle/late)
                - perturbation_metadata: Additional metadata
            retry_count: Current retry attempt
            max_retries: Maximum retry attempts

        Returns:
            Quality score dict with keys:
                - content_changed (0-1)
                - syntactically_valid (0-1)
                - semantically_meaningful (0-1)
                - type_matches_intent (0-1)
                - realistic_error (0-3)
                - total_score (0-7)
                - reasoning (str)
                - quality_tier (high/medium/low/invalid)
                - scored_by (str)
                - scored_at (ISO timestamp)
        """
        original_content = perturbation.get("original_step_content", "")
        perturbed_content = perturbation.get("perturbed_step_content", "")
        perturbation_type = perturbation.get("perturbation_type", "unknown")
        perturbation_position = perturbation.get("perturbation_position", "unknown")
        metadata = perturbation.get("perturbation_metadata", {})

        description = metadata.get("description", None)

        prompt = format_quality_scoring_prompt(
            original_step_content=original_content,
            perturbed_step_content=perturbed_content,
            perturbation_type=perturbation_type,
            perturbation_position=perturbation_position,
            perturbation_description=description
        )

        try:
            llm_result = self._call_llm(prompt)
            response_text = llm_result.get("response", "")

            score = self._parse_score_response(response_text)

            score["quality_tier"] = self._assign_tier(score["total_score"])
            score["scored_by"] = self.model_name
            score["scored_at"] = datetime.now(timezone.utc).isoformat()
            score["tokens_used"] = llm_result.get("tokens_used", 0)
            score["time_ms"] = llm_result.get("time_ms", 0)

            return score

        except Exception as e:
            if retry_count < max_retries:
                time.sleep(1)
                return self.score_perturbation(
                    perturbation,
                    retry_count=retry_count + 1,
                    max_retries=max_retries
                )
            else:
                return self._default_error_score(str(e))

    def score_batch(
        self,
        perturbations: List[Dict[str, Any]],
        batch_size: int = 10,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Score multiple perturbations with progress tracking.

        Args:
            perturbations: List of perturbation documents
            batch_size: Batch size for progress reporting
            progress_callback: Optional callback(scored, total) for progress

        Returns:
            List of quality scores in same order as input
        """
        scores = []
        total = len(perturbations)

        for i, perturbation in enumerate(perturbations):
            score = self.score_perturbation(perturbation)
            scores.append(score)

            if progress_callback and (i + 1) % batch_size == 0:
                progress_callback(i + 1, total)

        return scores

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call the LLM via AWS Bedrock.

        Args:
            prompt: The formatted prompt

        Returns:
            Dict with response, tokens_used, time_ms
        """
        start_time = time.time()

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read())

            content = response_body.get('content', [])
            if not content:
                raise ValueError("Empty response from LLM")

            response_text = content[0].get('text', '')

            usage = response_body.get('usage', {})
            tokens_used = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)

            elapsed_ms = int((time.time() - start_time) * 1000)

            return {
                "response": response_text,
                "tokens_used": tokens_used,
                "time_ms": elapsed_ms
            }

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            raise Exception(f"Bedrock API error ({error_code}): {error_message}")

    def _parse_score_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response into score dict.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed score dict

        Raises:
            ValueError: If response cannot be parsed
        """
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response_text[:200]}")

        try:
            score_data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

        required_fields = [
            "content_changed",
            "syntactically_valid",
            "semantically_meaningful",
            "type_matches_intent",
            "realistic_error"
        ]

        for field in required_fields:
            if field not in score_data:
                raise ValueError(f"Missing required field: {field}")

        score = {
            "content_changed": self._clamp(int(score_data["content_changed"]), 0, 1),
            "syntactically_valid": self._clamp(int(score_data["syntactically_valid"]), 0, 1),
            "semantically_meaningful": self._clamp(int(score_data["semantically_meaningful"]), 0, 1),
            "type_matches_intent": self._clamp(int(score_data["type_matches_intent"]), 0, 1),
            "realistic_error": self._clamp(int(score_data["realistic_error"]), 0, 3),
            "reasoning": score_data.get("reasoning", "")
        }

        score["total_score"] = (
            score["content_changed"] +
            score["syntactically_valid"] +
            score["semantically_meaningful"] +
            score["type_matches_intent"] +
            score["realistic_error"]
        )

        return score

    def _clamp(self, value: int, min_val: int, max_val: int) -> int:
        """Clamp value to range [min_val, max_val]."""
        return max(min_val, min(max_val, value))

    def _assign_tier(self, total_score: int) -> str:
        """
        Assign quality tier based on total score.

        Args:
            total_score: Total quality score (0-7)

        Returns:
            Quality tier: "high", "medium", "low", or "invalid"
        """
        if total_score >= self.tier_thresholds["high"]:
            return "high"
        elif total_score >= self.tier_thresholds["medium"]:
            return "medium"
        elif total_score >= self.tier_thresholds["low"]:
            return "low"
        else:
            return "invalid"

    def _default_error_score(self, error_message: str) -> Dict[str, Any]:
        """
        Return a default score when scoring fails.

        Args:
            error_message: Error description

        Returns:
            Error score dict
        """
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
            "error": error_message
        }


def create_quality_scorer(config: Dict[str, Any]) -> PerturbationQualityScorer:
    """
    Factory function to create quality scorer from config.

    Args:
        config: Quality scoring configuration dict

    Returns:
        Configured PerturbationQualityScorer instance
    """
    model_config = config.get("judge_model", {})
    metrics = config.get("metrics", {})
    tier_thresholds = config.get("tier_thresholds", {})

    return PerturbationQualityScorer(
        model_config=model_config,
        metrics=metrics,
        tier_thresholds=tier_thresholds
    )
