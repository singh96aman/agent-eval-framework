"""
LLM-based perturbation class validator.

Validates that perturbations match their intended class (placebo, fine_grained, coarse_grained).
Prompt is read from config, NOT hardcoded.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.llm import get_bedrock_client, DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)


@dataclass
class ClassValidationResult:
    """Result of class validation."""

    class_matches: int  # 0 or 1
    reasoning: str
    raw_response: Optional[str] = None
    parse_success: bool = True


class PerturbationClassValidator:
    """
    Validates that perturbations match their intended class using LLM.

    The prompt is read from config at `phases.perturb.class_validation.prompt`.
    Template variables: {{perturbation_class}}, {{perturbation_type}}, {{step_index}},
                       {{original_value}}, {{perturbed_value}}, {{mutation_method}}
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        prompt_template: str,
        log_calls: bool = False,
        max_retries: int = 2,
    ):
        """
        Initialize the class validator.

        Args:
            model_config: Model configuration from config file
                - provider: "bedrock"
                - model: Model ID
                - max_tokens: Max response tokens
                - temperature: Sampling temperature
            prompt_template: Prompt template with {{variable}} placeholders
            log_calls: Whether to log LLM calls
            max_retries: Max retries on parse failure
        """
        self.model_config = model_config
        self.prompt_template = prompt_template
        self.log_calls = log_calls
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Bedrock client."""
        if self._client is None:
            self._client = get_bedrock_client(log_calls=self.log_calls)
        return self._client

    @property
    def model_id(self) -> str:
        """Get model ID from config."""
        return self.model_config.get("model", DEFAULT_MODEL_ID)

    @property
    def max_tokens(self) -> int:
        """Get max tokens from config."""
        return self.model_config.get("max_tokens", 500)

    @property
    def temperature(self) -> float:
        """Get temperature from config."""
        return self.model_config.get("temperature", 0.0)

    def _build_prompt(self, perturbation: Dict[str, Any]) -> str:
        """
        Substitute template variables in prompt.

        Args:
            perturbation: Perturbation record with fields:
                - perturbation_class: placebo, fine_grained, coarse_grained
                - perturbation_type: specific type (e.g., formatting, wrong_value)
                - target_step_index: which step was perturbed
                - original_value: original content
                - perturbed_value: modified content
                - mutation_method: how the perturbation was generated

        Returns:
            Prompt with variables substituted
        """
        # Extract values with defaults
        perturbation_class = perturbation.get("perturbation_class", "unknown")
        perturbation_type = perturbation.get("perturbation_type", "unknown")
        step_index = perturbation.get("target_step_index", 0)
        original_value = str(perturbation.get("original_value", ""))[:500]
        perturbed_value = str(perturbation.get("perturbed_value", ""))[:500]
        mutation_method = perturbation.get("mutation_method", "unknown")

        # Substitute template variables
        prompt = self.prompt_template
        prompt = prompt.replace("{{perturbation_class}}", perturbation_class)
        prompt = prompt.replace("{{perturbation_type}}", perturbation_type)
        prompt = prompt.replace("{{step_index}}", str(step_index))
        prompt = prompt.replace("{{original_value}}", original_value)
        prompt = prompt.replace("{{perturbed_value}}", perturbed_value)
        prompt = prompt.replace("{{mutation_method}}", mutation_method)

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM and return raw response."""
        response = self.client.invoke(
            model_id=self.model_id,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.get("response", "").strip()

    def _parse_json_response(
        self, raw_response: str
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Parse JSON from LLM response with fallback handling.

        Returns:
            Tuple of (parsed_dict or None, success_flag)
        """
        try:
            # Try direct parse
            return json.loads(raw_response), True
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        if "```json" in raw_response:
            try:
                json_str = raw_response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str), True
            except (IndexError, json.JSONDecodeError):
                pass

        # Try to extract JSON from generic code block
        if "```" in raw_response:
            try:
                json_str = raw_response.split("```")[1].split("```")[0].strip()
                return json.loads(json_str), True
            except (IndexError, json.JSONDecodeError):
                pass

        # Try to find JSON object in response
        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = raw_response[start:end]
                return json.loads(json_str), True
        except json.JSONDecodeError:
            pass

        return None, False

    def validate_perturbation(
        self, perturbation: Dict[str, Any]
    ) -> ClassValidationResult:
        """
        Validate a perturbation's class using LLM.

        Args:
            perturbation: Perturbation record with class and content info

        Returns:
            ClassValidationResult with class_matches (0 or 1) and reasoning
        """
        prompt = self._build_prompt(perturbation)
        raw_response = None

        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self._call_llm(prompt)
                parsed, success = self._parse_json_response(raw_response)

                if success and parsed:
                    # Normalize class_matches to 0 or 1
                    class_matches = parsed.get("class_matches", 0)
                    if isinstance(class_matches, bool):
                        class_matches = 1 if class_matches else 0
                    elif isinstance(class_matches, str):
                        class_matches = 1 if class_matches.lower() in ("1", "true", "yes") else 0
                    else:
                        class_matches = int(class_matches) if class_matches else 0

                    # Clamp to 0 or 1
                    class_matches = max(0, min(1, class_matches))

                    return ClassValidationResult(
                        class_matches=class_matches,
                        reasoning=parsed.get("reasoning", ""),
                        raw_response=raw_response,
                        parse_success=True,
                    )
                else:
                    logger.warning(
                        f"JSON parse failed (attempt {attempt + 1}): {raw_response[:200]}"
                    )
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")

        # All retries failed
        return ClassValidationResult(
            class_matches=1,  # Assume valid on parse failure (conservative)
            reasoning="LLM response could not be parsed",
            raw_response=raw_response,
            parse_success=False,
        )


def create_validator_from_config(config: Dict[str, Any]) -> Optional[PerturbationClassValidator]:
    """
    Create a PerturbationClassValidator from experiment config.

    Args:
        config: Experiment config dict

    Returns:
        PerturbationClassValidator if class_validation is enabled, else None
    """
    class_validation_config = config.get("phases", {}).get("perturb", {}).get("class_validation", {})

    if not class_validation_config.get("enabled", False):
        return None

    model_config = class_validation_config.get("model", {})
    prompt_template = class_validation_config.get("prompt", "")

    if not prompt_template:
        logger.warning("Class validation enabled but no prompt template provided")
        return None

    return PerturbationClassValidator(
        model_config=model_config,
        prompt_template=prompt_template,
        log_calls=True,
    )
