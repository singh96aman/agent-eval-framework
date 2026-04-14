"""
LLM-based Perturbation Generator.

Replaces heuristic fallback mutations (that produce detectable artifacts like
"_old", "_mutated", "_wrong") with LLM-generated plausible mutations.

This module provides contextual, semantically meaningful perturbations that
are indistinguishable from natural errors.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.llm import get_bedrock_client, DEFAULT_MODEL_ID
from src.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class LLMMutationResult:
    """Result of an LLM-based mutation."""

    mutated_value: Any
    mutation_type: str
    reasoning: Optional[str] = None
    raw_llm_response: Optional[str] = None
    parse_success: bool = True
    prompt_name: Optional[str] = None


class LLMPerturbationGenerator:
    """
    Generate semantically meaningful perturbations using LLM.

    Replaces heuristic fallbacks that produce detectable artifacts.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        log_calls: bool = False,
        max_retries: int = 2,
    ):
        """
        Initialize the LLM perturbation generator.

        Args:
            model_id: Claude model ID for generation
            log_calls: Whether to log LLM calls
            max_retries: Max retries on parse failure
        """
        self.model_id = model_id
        self.log_calls = log_calls
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Bedrock client."""
        if self._client is None:
            self._client = get_bedrock_client(log_calls=self.log_calls)
        return self._client

    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call LLM and return raw response."""
        response = self.client.invoke(
            model_id=self.model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,  # Allow some creativity for realistic errors
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

    def generate_wrong_parameter(
        self,
        tool_name: str,
        param_name: str,
        current_value: Any,
        value_type: str,
        task_context: str = "",
    ) -> Optional[LLMMutationResult]:
        """
        Generate a plausible but incorrect parameter value.

        Args:
            tool_name: Name of the tool being called
            param_name: Name of the parameter
            current_value: Current parameter value
            value_type: Type of the value (string, int, etc.)
            task_context: Optional task description for context

        Returns:
            LLMMutationResult or None if generation failed
        """
        prompt_template = get_prompt("WRONG_PARAMETER_PROMPT_V1")
        prompt = prompt_template.format(
            tool_name=tool_name,
            param_name=param_name,
            current_value=current_value,
            value_type=value_type,
            task_context=task_context or "General tool usage",
        )

        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self._call_llm(prompt)
                parsed, success = self._parse_json_response(raw_response)

                if success and parsed:
                    return LLMMutationResult(
                        mutated_value=parsed.get("mutated_value", current_value),
                        mutation_type=parsed.get("mutation_type", "llm_generated"),
                        reasoning=parsed.get("reasoning"),
                        raw_llm_response=raw_response,
                        parse_success=True,
                        prompt_name="WRONG_PARAMETER_PROMPT_V1",
                    )
                else:
                    logger.warning(
                        f"JSON parse failed (attempt {attempt + 1}): {raw_response[:200]}"
                    )
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")

        # All retries failed - return with parse_success=False
        return LLMMutationResult(
            mutated_value=current_value,  # Return original as fallback
            mutation_type="llm_parse_failed",
            reasoning="LLM response could not be parsed",
            raw_llm_response=raw_response if "raw_response" in dir() else None,
            parse_success=False,
            prompt_name="WRONG_PARAMETER_PROMPT_V1",
        )

    def generate_value_mutation(
        self,
        original_value: Any,
        value_type: str,
        context: str = "",
    ) -> Optional[LLMMutationResult]:
        """
        Generate a generic value mutation (replaces "_mutated" fallback).

        Args:
            original_value: The value to mutate
            value_type: Type of the value
            context: Optional context for the mutation

        Returns:
            LLMMutationResult or None if generation failed
        """
        prompt_template = get_prompt("VALUE_MUTATION_PROMPT_V1")
        prompt = prompt_template.format(
            original_value=original_value,
            value_type=value_type,
            context=context or "Parameter value in tool call",
        )

        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self._call_llm(prompt)
                parsed, success = self._parse_json_response(raw_response)

                if success and parsed:
                    return LLMMutationResult(
                        mutated_value=parsed.get("mutated_value", original_value),
                        mutation_type=parsed.get("mutation_type", "llm_generated"),
                        reasoning=parsed.get("reasoning"),
                        raw_llm_response=raw_response,
                        parse_success=True,
                        prompt_name="VALUE_MUTATION_PROMPT_V1",
                    )
                else:
                    logger.warning(
                        f"JSON parse failed (attempt {attempt + 1}): {raw_response[:200]}"
                    )
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")

        return LLMMutationResult(
            mutated_value=original_value,
            mutation_type="llm_parse_failed",
            reasoning="LLM response could not be parsed",
            raw_llm_response=raw_response if "raw_response" in dir() else None,
            parse_success=False,
            prompt_name="VALUE_MUTATION_PROMPT_V1",
        )

    def generate_wrong_date(
        self,
        original_date: str,
        date_format: str = "YYYY-MM-DD",
        context: str = "",
    ) -> Optional[LLMMutationResult]:
        """
        Generate a plausible but incorrect date (replaces "_wrong" fallback).

        Args:
            original_date: The original date string
            date_format: Expected format of the date
            context: Optional context

        Returns:
            LLMMutationResult or None if generation failed
        """
        prompt_template = get_prompt("WRONG_DATE_PROMPT_V1")
        prompt = prompt_template.format(
            original_date=original_date,
            date_format=date_format,
            context=context or "Date parameter in tool call",
        )

        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self._call_llm(prompt)
                parsed, success = self._parse_json_response(raw_response)

                if success and parsed:
                    return LLMMutationResult(
                        mutated_value=parsed.get("mutated_date", original_date),
                        mutation_type=parsed.get("mutation_type", "llm_generated"),
                        reasoning=parsed.get("reasoning"),
                        raw_llm_response=raw_response,
                        parse_success=True,
                        prompt_name="WRONG_DATE_PROMPT_V1",
                    )
                else:
                    logger.warning(
                        f"JSON parse failed (attempt {attempt + 1}): {raw_response[:200]}"
                    )
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")

        return LLMMutationResult(
            mutated_value=original_date,
            mutation_type="llm_parse_failed",
            reasoning="LLM response could not be parsed",
            raw_llm_response=raw_response if "raw_response" in dir() else None,
            parse_success=False,
            prompt_name="WRONG_DATE_PROMPT_V1",
        )

    def generate_wrong_identifier(
        self,
        original_id: str,
        id_type: str = "identifier",
        context: str = "",
    ) -> Optional[LLMMutationResult]:
        """
        Generate a plausible but incorrect identifier (replaces suffix array).

        Args:
            original_id: The original identifier
            id_type: Type of identifier (variable, file, api_key, etc.)
            context: Optional context

        Returns:
            LLMMutationResult or None if generation failed
        """
        prompt_template = get_prompt("WRONG_IDENTIFIER_PROMPT_V1")
        prompt = prompt_template.format(
            original_id=original_id,
            id_type=id_type,
            context=context or "Identifier in tool call",
        )

        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self._call_llm(prompt)
                parsed, success = self._parse_json_response(raw_response)

                if success and parsed:
                    return LLMMutationResult(
                        mutated_value=parsed.get("mutated_id", original_id),
                        mutation_type=parsed.get("mutation_type", "llm_generated"),
                        reasoning=parsed.get("reasoning"),
                        raw_llm_response=raw_response,
                        parse_success=True,
                        prompt_name="WRONG_IDENTIFIER_PROMPT_V1",
                    )
                else:
                    logger.warning(
                        f"JSON parse failed (attempt {attempt + 1}): {raw_response[:200]}"
                    )
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")

        return LLMMutationResult(
            mutated_value=original_id,
            mutation_type="llm_parse_failed",
            reasoning="LLM response could not be parsed",
            raw_llm_response=raw_response if "raw_response" in dir() else None,
            parse_success=False,
            prompt_name="WRONG_IDENTIFIER_PROMPT_V1",
        )


# Singleton instance for easy import
_default_generator: Optional[LLMPerturbationGenerator] = None


def get_llm_generator(
    model_id: str = DEFAULT_MODEL_ID,
    log_calls: bool = False,
) -> LLMPerturbationGenerator:
    """Get or create the default LLM perturbation generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = LLMPerturbationGenerator(
            model_id=model_id, log_calls=log_calls
        )
    return _default_generator
