"""
Judge evaluation system for LLM-based trajectory assessment.

Provides:
- Base Judge interface
- Claude Sonnet 4.5 judge (AWS Bedrock)
- GPT-OSS 120B judge (AWS Bedrock)
- Evaluation runner with batching and rate limiting
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import json
from datetime import datetime

from src.data.schema import Trajectory
from src.judges.schema import JudgeOutput, StepError, ErrorSeverity
from src.judges.prompts import build_evaluation_prompt, JUDGE_SYSTEM_PROMPT


class Judge(ABC):
    """
    Abstract base class for LLM judges.

    Subclasses must implement:
    - _call_llm(): Make the actual API call
    - _parse_response(): Parse LLM response into JudgeOutput
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60
    ):
        """
        Initialize judge.

        Args:
            name: Human-readable judge name (e.g., "claude-sonnet-4.5")
            model_id: Model identifier for API calls
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: API call timeout in seconds
        """
        self.name = name
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Stats tracking
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time_ms = 0
        self.failed_calls = 0

    @abstractmethod
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Make the actual LLM API call.

        Args:
            system_prompt: System message
            user_prompt: User message with trajectory

        Returns:
            Dict with keys:
                - "response": The LLM's text response
                - "tokens_used": Token count (if available)
                - "time_ms": Time taken in milliseconds

        Raises:
            Exception: If API call fails
        """
        pass

    @abstractmethod
    def _parse_response(
        self,
        response_text: str,
        trajectory_id: str
    ) -> JudgeOutput:
        """
        Parse LLM response into structured JudgeOutput.

        Args:
            response_text: Raw LLM response
            trajectory_id: ID of evaluated trajectory

        Returns:
            Structured JudgeOutput object

        Raises:
            ValueError: If response cannot be parsed
        """
        pass

    def evaluate(
        self,
        trajectory: Trajectory,
        retry_on_failure: bool = True,
        max_retries: int = 3
    ) -> Optional[JudgeOutput]:
        """
        Evaluate a trajectory and return structured output.

        This is the main public method. It handles:
        - Prompt construction
        - LLM API call with retries
        - Response parsing
        - Error handling

        Args:
            trajectory: Trajectory to evaluate
            retry_on_failure: Whether to retry on API failures
            max_retries: Maximum number of retry attempts

        Returns:
            JudgeOutput object, or None if evaluation failed after retries
        """
        start_time = time.time()

        # Build prompts
        system_prompt = JUDGE_SYSTEM_PROMPT
        user_prompt = build_evaluation_prompt(trajectory)

        # Try evaluation with retries
        for attempt in range(max_retries):
            try:
                # Call LLM
                llm_result = self._call_llm(system_prompt, user_prompt)

                # Parse response
                output = self._parse_response(
                    llm_result["response"],
                    trajectory.trajectory_id
                )

                # Add metadata
                output.evaluation_time_ms = llm_result.get("time_ms")
                output.tokens_used = llm_result.get("tokens_used")

                # Update stats
                self.total_calls += 1
                self.total_tokens += output.tokens_used or 0
                self.total_time_ms += output.evaluation_time_ms or 0

                return output

            except Exception as e:
                self.failed_calls += 1

                if attempt < max_retries - 1 and retry_on_failure:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    print(f"   ⚠️  Attempt {attempt + 1} failed: {e}")
                    print(f"   ⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Final failure
                    print(f"   ❌ Evaluation failed after {max_retries} attempts: {e}")
                    return None

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this judge."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "success_rate": (
                (self.total_calls - self.failed_calls) / self.total_calls
                if self.total_calls > 0 else 0
            ),
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "avg_time_per_call_ms": (
                self.total_time_ms / self.total_calls
                if self.total_calls > 0 else 0
            )
        }


def parse_json_response(
    response_text: str,
    trajectory_id: str,
    judge_name: str,
    model_id: str
) -> JudgeOutput:
    """
    Parse a JSON-formatted judge response into JudgeOutput.

    This is a helper function used by concrete judge implementations.

    Args:
        response_text: Raw LLM response (should contain JSON)
        trajectory_id: ID of evaluated trajectory
        judge_name: Name of the judge
        model_id: Model identifier

    Returns:
        JudgeOutput object

    Raises:
        ValueError: If response cannot be parsed
    """
    # Try to extract JSON from response (may be wrapped in markdown)
    try:
        # Look for JSON between ```json and ``` or just parse directly
        json_str = response_text.strip()

        if "```json" in json_str:
            start = json_str.index("```json") + 7
            end = json_str.index("```", start)
            json_str = json_str[start:end].strip()
        elif "```" in json_str:
            start = json_str.index("```") + 3
            end = json_str.index("```", start)
            json_str = json_str[start:end].strip()

        data = json.loads(json_str)

    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse JSON from response: {e}\nResponse: {response_text[:200]}")

    # Parse step errors
    step_errors = []
    for err in data.get("step_errors", []):
        try:
            step_errors.append(StepError(
                step_index=err["step_index"],
                error_type=err["error_type"],
                severity=ErrorSeverity(err["severity"]),
                description=err["description"],
                impacts_task_success=err.get("impacts_task_success", False)
            ))
        except (KeyError, ValueError) as e:
            # Skip malformed step errors
            print(f"   ⚠️  Skipping malformed step error: {e}")
            continue

    # Create JudgeOutput
    return JudgeOutput(
        trajectory_id=trajectory_id,
        judge_name=judge_name,
        model_id=model_id,
        task_success=int(data["task_success"]),
        completeness=float(data["completeness"]),
        efficiency_errors=int(data.get("efficiency_errors", 0)),
        hallucination=int(data.get("hallucination", 0)),
        sycophancy=int(data.get("sycophancy", 0)),
        overall_score=float(data["overall_score"]),
        step_errors=step_errors,
        reasoning=data.get("reasoning", ""),
        timestamp=datetime.utcnow()
    )
