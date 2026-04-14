"""
Tier 3 Grading: Final-Answer Grading for Outcome Evidence.

Provides multiple grading strategies:
- ExactMatchGrader: Normalize + string compare (GAIA)
- FuzzyMatchGrader: Levenshtein distance + threshold
- HeuristicGrader: Task-specific rules (ToolBench)
- LLMGrader: Wrapper around ODScorer for LLM-based grading

This module reuses and extends src/replay/od_scorer.py.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.replay.od_scorer import ODScorer


@dataclass
class GradingResult:
    """Result of grading a trajectory's final answer."""

    score: float  # 0-1 normalized
    passed: bool
    reasoning: str
    grader_type: str
    raw_prediction: Optional[str] = None
    raw_expected: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "score": self.score,
            "passed": self.passed,
            "reasoning": self.reasoning,
            "grader_type": self.grader_type,
            "raw_prediction": self.raw_prediction,
            "raw_expected": self.raw_expected,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradingResult":
        """Create GradingResult from dictionary."""
        return cls(
            score=data["score"],
            passed=data["passed"],
            reasoning=data["reasoning"],
            grader_type=data["grader_type"],
            raw_prediction=data.get("raw_prediction"),
            raw_expected=data.get("raw_expected"),
        )


class BaseGrader(ABC):
    """Abstract base class for graders."""

    @abstractmethod
    def grade(
        self,
        trajectory: Dict[str, Any],
        expected: Optional[str] = None,
    ) -> GradingResult:
        """
        Grade a trajectory's final answer.

        Args:
            trajectory: Trajectory dict with steps and ground_truth
            expected: Optional override for expected answer

        Returns:
            GradingResult with score, pass/fail, and reasoning
        """
        pass

    def extract_final_answer(self, trajectory: Dict[str, Any]) -> str:
        """Extract final answer from trajectory.

        For ToolBench format, extracts the 'final_answer' from the
        Finish action's Action Input JSON.
        """
        import json as json_module
        steps = trajectory.get("steps", [])
        if not steps:
            return ""

        # Work backwards through steps
        for step in reversed(steps):
            content = step.get("content") or step.get("raw_text", "")
            tool_name = step.get("tool_name", "")

            # ToolBench format: Look for Finish action with final_answer
            if tool_name and "finish" in tool_name.lower():
                # Try to extract final_answer from tool_input
                tool_input = step.get("tool_input")
                if isinstance(tool_input, dict):
                    final_answer = tool_input.get("final_answer")
                    if final_answer:
                        return str(final_answer)

                # Try parsing from metadata
                metadata = step.get("metadata", {})
                action_input = metadata.get("action_input", "")
                if action_input:
                    try:
                        parsed = json_module.loads(action_input)
                        final_answer = parsed.get("final_answer")
                        if final_answer:
                            return str(final_answer)
                    except (json_module.JSONDecodeError, TypeError):
                        pass

            # Check for reasoning content (not tool output)
            step_type = step.get("step_type") or step.get("step_role", "")
            if step_type in ("reasoning", "assistant", "response", "final_response"):
                # Prefer thought/reasoning over raw API output
                if content and not content.strip().startswith("{"):
                    return content

        # Fallback: last step content (excluding raw JSON)
        last_step = steps[-1]
        content = last_step.get("content") or last_step.get("raw_text", "")
        if content and not content.strip().startswith("{"):
            return content

        # Very last fallback
        return content if content else ""

    def get_expected_answer(self, trajectory: Dict[str, Any]) -> Optional[str]:
        """Extract expected answer from trajectory ground truth."""
        ground_truth = trajectory.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            answer = ground_truth.get("expected_answer")
            if not answer:
                answer = ground_truth.get("answer")
            if not answer:
                answer = ground_truth.get("Final answer")
            return answer
        return None


class ExactMatchGrader(BaseGrader):
    """
    Exact match grader for GAIA-style tasks.

    Normalizes both answers and compares for exact match.
    """

    def __init__(self, case_sensitive: bool = False, strip_whitespace: bool = True):
        """
        Initialize ExactMatchGrader.

        Args:
            case_sensitive: Whether comparison is case-sensitive
            strip_whitespace: Whether to strip whitespace
        """
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""

        result = text
        if self.strip_whitespace:
            result = result.strip()
            # Collapse multiple spaces
            result = re.sub(r"\s+", " ", result)

        if not self.case_sensitive:
            result = result.lower()

        # Remove common artifacts
        result = re.sub(r"^(the answer is|answer:)\s*", "", result, flags=re.IGNORECASE)

        return result

    def grade(
        self,
        trajectory: Dict[str, Any],
        expected: Optional[str] = None,
    ) -> GradingResult:
        """Grade using exact match comparison."""
        predicted = self.extract_final_answer(trajectory)
        expected_answer = expected or self.get_expected_answer(trajectory)

        if expected_answer is None:
            return GradingResult(
                score=0.0,
                passed=False,
                reasoning="No expected answer available",
                grader_type="exact_match",
                raw_prediction=predicted,
                raw_expected=None,
            )

        normalized_predicted = self.normalize(predicted)
        normalized_expected = self.normalize(expected_answer)

        # Check for exact match
        is_match = normalized_predicted == normalized_expected

        return GradingResult(
            score=1.0 if is_match else 0.0,
            passed=is_match,
            reasoning=(
                "Exact match"
                if is_match
                else f"Mismatch: '{normalized_predicted}' vs '{normalized_expected}'"
            ),
            grader_type="exact_match",
            raw_prediction=predicted,
            raw_expected=expected_answer,
        )


class FuzzyMatchGrader(BaseGrader):
    """
    Fuzzy match grader using Levenshtein distance.

    Useful for text answers where minor variations are acceptable.
    """

    def __init__(self, threshold: float = 0.8):
        """
        Initialize FuzzyMatchGrader.

        Args:
            threshold: Similarity threshold for passing (0-1)
        """
        self.threshold = threshold

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are
                # one character longer than s2
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def similarity(self, s1: str, s2: str) -> float:
        """Compute similarity score (0-1) between two strings."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        max_len = max(len(s1), len(s2))
        distance = self.levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)

    def normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        result = text.strip().lower()
        result = re.sub(r"\s+", " ", result)
        return result

    def grade(
        self,
        trajectory: Dict[str, Any],
        expected: Optional[str] = None,
    ) -> GradingResult:
        """Grade using fuzzy match comparison."""
        predicted = self.extract_final_answer(trajectory)
        expected_answer = expected or self.get_expected_answer(trajectory)

        if expected_answer is None:
            return GradingResult(
                score=0.0,
                passed=False,
                reasoning="No expected answer available",
                grader_type="fuzzy_match",
                raw_prediction=predicted,
                raw_expected=None,
            )

        normalized_predicted = self.normalize(predicted)
        normalized_expected = self.normalize(expected_answer)

        sim_score = self.similarity(normalized_predicted, normalized_expected)
        passed = sim_score >= self.threshold

        return GradingResult(
            score=sim_score,
            passed=passed,
            reasoning=(f"Similarity: {sim_score:.2f} (threshold: {self.threshold})"),
            grader_type="fuzzy_match",
            raw_prediction=predicted,
            raw_expected=expected_answer,
        )


class HeuristicGrader(BaseGrader):
    """
    Heuristic grader for ToolBench-style tasks.

    Uses task-specific rules to determine success.
    """

    def __init__(
        self,
        success_patterns: Optional[list] = None,
        failure_patterns: Optional[list] = None,
    ):
        """
        Initialize HeuristicGrader.

        Args:
            success_patterns: Regex patterns indicating success
            failure_patterns: Regex patterns indicating failure
        """
        self.success_patterns = success_patterns or [
            r"successfully",
            r"completed",
            r"here is",
            r"here are",
            r"the result is",
            r"found",
        ]
        self.failure_patterns = failure_patterns or [
            r"unable to",
            r"could not",
            r"failed to",
            r"error",
            r"sorry",
            r"apologize",
            r"cannot",
            r"can't",
        ]

    def _count_pattern_matches(self, text: str, patterns: list) -> int:
        """Count how many patterns match in text."""
        count = 0
        text_lower = text.lower()
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                count += 1
        return count

    def _count_api_errors(self, trajectory: Dict[str, Any]) -> int:
        """Count API errors in trajectory.

        Handles common API response patterns where {"error": ""} means success.
        """
        count = 0
        # Error patterns that indicate actual failures (not just field names)
        error_patterns = ["404", "500", "timeout", "rate limit", "connection refused"]

        for step in trajectory.get("steps", []):
            tool_output = str(
                step.get("tool_output") or step.get("observation", "")
            )

            # Try to parse as JSON to check error field properly
            if tool_output.strip().startswith("{"):
                try:
                    import json
                    parsed = json.loads(tool_output)
                    # Check if "error" field has actual error content
                    error_val = parsed.get("error", None)
                    if error_val and str(error_val).strip():
                        # Non-empty error field = real error
                        count += 1
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass

            # Fallback: check for error patterns in raw text
            text_lower = tool_output.lower()
            if any(err in text_lower for err in error_patterns):
                count += 1

        return count

    def grade(
        self,
        trajectory: Dict[str, Any],
        expected: Optional[str] = None,
    ) -> GradingResult:
        """Grade using heuristic rules."""
        predicted = self.extract_final_answer(trajectory)

        if not predicted:
            return GradingResult(
                score=0.0,
                passed=False,
                reasoning="No final answer found",
                grader_type="heuristic",
                raw_prediction=predicted,
                raw_expected=expected,
            )

        # Count pattern matches
        success_matches = self._count_pattern_matches(predicted, self.success_patterns)
        failure_matches = self._count_pattern_matches(predicted, self.failure_patterns)

        # Count API errors
        api_errors = self._count_api_errors(trajectory)

        # Compute heuristic score
        # Base score starts at 0.5
        score = 0.5

        # Adjust based on success patterns (+0.1 each, max +0.4)
        score += min(success_matches * 0.1, 0.4)

        # Adjust based on failure patterns (-0.15 each)
        score -= failure_matches * 0.15

        # Penalty for API errors (-0.05 each)
        score -= api_errors * 0.05

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        # Pass threshold at 0.5
        passed = score >= 0.5

        return GradingResult(
            score=score,
            passed=passed,
            reasoning=(
                f"Heuristic: {success_matches} success signals, "
                f"{failure_matches} failure signals, {api_errors} API errors"
            ),
            grader_type="heuristic",
            raw_prediction=predicted,
            raw_expected=expected,
        )


class LLMGrader(BaseGrader):
    """
    LLM-based grader using ODScorer.

    Wraps the existing ODScorer for LLM-based outcome grading.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLMGrader.

        Args:
            config: Configuration dict for ODScorer
        """
        self.config = config or {}
        self._scorer: Optional[ODScorer] = None

    @property
    def scorer(self) -> ODScorer:
        """Lazy initialization of ODScorer."""
        if self._scorer is None:
            self._scorer = ODScorer(self.config)
        return self._scorer

    def grade(
        self,
        trajectory: Dict[str, Any],
        expected: Optional[str] = None,
    ) -> GradingResult:
        """Grade using LLM-based evaluation."""
        # Get task description for context
        task_description = None
        ground_truth = trajectory.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            task_description = ground_truth.get("task_description")

        # Use ODScorer to grade
        score, reasoning = self.scorer.grade_outcome(trajectory, task_description)

        # Normalize score from 0-100 to 0-1
        normalized_score = score / 100.0

        # Pass threshold at 0.6 (60/100)
        passed = normalized_score >= 0.6

        return GradingResult(
            score=normalized_score,
            passed=passed,
            reasoning=reasoning,
            grader_type="llm",
            raw_prediction=self.extract_final_answer(trajectory),
            raw_expected=expected or self.get_expected_answer(trajectory),
        )


def get_grader(benchmark: str, config: Optional[Dict[str, Any]] = None) -> BaseGrader:
    """
    Factory function to get appropriate grader for a benchmark.

    Args:
        benchmark: Benchmark name (gaia, toolbench, swebench)
        config: Optional configuration for graders

    Returns:
        Appropriate grader instance
    """
    benchmark_lower = benchmark.lower()

    if benchmark_lower == "gaia":
        return ExactMatchGrader()
    elif benchmark_lower == "toolbench":
        return HeuristicGrader()
    elif benchmark_lower == "swebench":
        # SWE-bench uses test suite (Tier 1), but fallback to LLM grading
        return LLMGrader(config or {})
    else:
        # Default to LLM grader
        return LLMGrader(config or {})
