"""
Prompt Quality Gates (6 gates).

Validates prompts for:
- Priming detection (no error expectation bias)
- Blinding integrity (no baseline/perturbed leakage)
- Vocabulary alignment (types match schema)
- Schema alignment (responses parse correctly)
- Neutrality (A and B equal framing)
- Output parsability (real responses parse)
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.quality_gates.base import BaseGate, GateResult


# =============================================================================
# Priming Detection Gate
# =============================================================================


class PrimingDetectionGate(BaseGate):
    """Gate: Detect priming language that biases toward error detection."""

    name = "priming_detection"
    description = "Verify prompt does not prime for error expectation"

    # Patterns that suggest priming
    PRIMING_PATTERNS = [
        r"one.*contain.*error",
        r"one.*has.*error",
        r"find.*the.*error",
        r"identify.*the.*error",
        r"which.*has.*error",
        r"which.*contains.*error",
        r"the.*error.*is",
        r"there.*is.*an.*error",
        r"may.*contain.*error",
    ]

    def check(
        self, prompt: str, config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check for priming patterns in prompt."""
        if not prompt:
            return self._skip("No prompt to check")

        prompt_lower = prompt.lower()
        found_patterns = []

        for pattern in self.PRIMING_PATTERNS:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                found_patterns.append(pattern)

        if len(found_patterns) == 0:
            return self._pass(
                "No priming patterns detected",
                value=0,
                threshold=0,
            )
        else:
            return self._fail(
                f"Found {len(found_patterns)} priming patterns",
                value=len(found_patterns),
                threshold=0,
                details={"patterns": found_patterns},
            )


# =============================================================================
# Blinding Integrity Gate
# =============================================================================


class BlindingIntegrityGate(BaseGate):
    """Gate: Verify prompt maintains blinding (no baseline/perturbed terms)."""

    name = "blinding_integrity"
    description = "Verify prompt does not leak baseline/perturbed labels"

    # Terms that break blinding
    BLINDING_VIOLATIONS = [
        r"\bbaseline\b",
        r"\bperturbed\b",
        r"\boriginal\b",
        r"\bmodified\b",
        r"\bcorrect.*trajectory\b",
        r"\bincorrect.*trajectory\b",
        r"\bground.*truth\b",
    ]

    def check(
        self, prompt: str, config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check for blinding violations in prompt."""
        if not prompt:
            return self._skip("No prompt to check")

        prompt_lower = prompt.lower()
        violations = []

        for pattern in self.BLINDING_VIOLATIONS:
            matches = re.findall(pattern, prompt_lower, re.IGNORECASE)
            if matches:
                violations.extend(matches)

        if len(violations) == 0:
            return self._pass(
                "Blinding integrity maintained",
                value=0,
                threshold=0,
            )
        else:
            return self._fail(
                f"Found {len(violations)} blinding violations",
                value=len(violations),
                threshold=0,
                details={"terms": list(set(violations))},
            )


# =============================================================================
# Vocabulary Alignment Gate
# =============================================================================


class VocabularyAlignmentGate(BaseGate):
    """Gate: Verify prompt vocabulary matches schema types."""

    name = "vocabulary_alignment"
    description = "Verify error types in prompt match schema"

    # Valid error types from schema
    VALID_TYPES = {
        "planning",
        "tool_selection",
        "parameter",
        "data_reference",
        "structural",
        "terminal_flag",
        "other",
    }

    def check(
        self, prompt: str, config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check vocabulary alignment."""
        config = config or {}
        valid_types = config.get("valid_types", self.VALID_TYPES)

        if not prompt:
            return self._skip("No prompt to check")

        # Extract error types from prompt (look for quoted strings)
        type_pattern = r'"(planning|tool_selection|parameter|data_reference|structural|reasoning|other)"'
        found_types = set(re.findall(type_pattern, prompt, re.IGNORECASE))

        # Check if all found types are valid
        invalid_types = found_types - valid_types

        if len(invalid_types) == 0:
            return self._pass(
                f"All {len(found_types)} types are valid",
                value=list(found_types),
                threshold=list(valid_types),
            )
        else:
            return self._warn(
                f"Found {len(invalid_types)} non-standard types",
                value=list(invalid_types),
                threshold=list(valid_types),
            )


# =============================================================================
# Schema Alignment Gate
# =============================================================================


class SchemaAlignmentGate(BaseGate):
    """Gate: Verify mock/example response in prompt parses correctly."""

    name = "schema_alignment"
    description = "Verify example JSON in prompt is valid"

    def check(
        self, prompt: str, config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check schema alignment."""
        if not prompt:
            return self._skip("No prompt to check")

        # Find JSON blocks in prompt
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, prompt)

        if len(json_matches) == 0:
            return self._skip("No JSON examples found in prompt")

        valid_count = 0
        invalid_examples = []

        for json_str in json_matches:
            # Skip template placeholders
            if "<" in json_str and ">" in json_str:
                valid_count += 1
                continue

            try:
                json.loads(json_str)
                valid_count += 1
            except json.JSONDecodeError:
                invalid_examples.append(json_str[:100])

        if len(invalid_examples) == 0:
            return self._pass(
                f"All {valid_count} JSON examples valid",
                value=valid_count,
            )
        else:
            return self._warn(
                f"Found {len(invalid_examples)} invalid JSON examples",
                value=len(invalid_examples),
                details={"examples": invalid_examples[:3]},
            )


# =============================================================================
# Neutrality Gate
# =============================================================================


class NeutralityGate(BaseGate):
    """Gate: Verify A and B have equal framing in prompt."""

    name = "neutrality"
    description = "Verify A and B trajectories have equal framing"

    def check(
        self, prompt: str, config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check neutrality of A/B framing."""
        if not prompt:
            return self._skip("No prompt to check")

        # Check for biased framing
        a_positive = len(re.findall(
            r"trajectory.?a.{0,20}(correct|good|better|proper)",
            prompt, re.IGNORECASE
        ))
        b_positive = len(re.findall(
            r"trajectory.?b.{0,20}(correct|good|better|proper)",
            prompt, re.IGNORECASE
        ))
        a_negative = len(re.findall(
            r"trajectory.?a.{0,20}(error|wrong|bad|incorrect)",
            prompt, re.IGNORECASE
        ))
        b_negative = len(re.findall(
            r"trajectory.?b.{0,20}(error|wrong|bad|incorrect)",
            prompt, re.IGNORECASE
        ))

        # Both should have equal framing
        bias_score = abs(a_positive - b_positive) + abs(a_negative - b_negative)

        if bias_score == 0:
            return self._pass(
                "A and B have neutral framing",
                value=bias_score,
                threshold=0,
            )
        else:
            return self._warn(
                f"Potential A/B framing bias (score: {bias_score})",
                value=bias_score,
                threshold=0,
                details={
                    "a_positive": a_positive,
                    "b_positive": b_positive,
                    "a_negative": a_negative,
                    "b_negative": b_negative,
                },
            )



# =============================================================================
# Output Parsability Gate
# =============================================================================


class OutputParsabilityGate(BaseGate):
    """Gate: Verify actual judge outputs parse correctly."""

    name = "output_parsability"
    description = "Verify judge outputs can be parsed"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check output parsability."""
        config = config or {}
        min_rate = config.get("min_rate", 0.95)

        if not data:
            return self._skip("No outputs to check")

        valid_count = 0
        invalid_examples = []

        for output in data:
            raw_response = output.get("raw_response", "")
            if self._try_parse(raw_response):
                valid_count += 1
            else:
                invalid_examples.append({
                    "id": output.get("judge_output_id"),
                    "response": raw_response[:100],
                })

        rate = valid_count / len(data)

        if rate >= min_rate:
            return self._pass(
                f"Parse rate {rate:.2%} >= {min_rate:.0%}",
                value=rate,
                threshold=min_rate,
            )
        else:
            return self._fail(
                f"Parse rate {rate:.2%} < {min_rate:.0%}",
                value=rate,
                threshold=min_rate,
                details={"invalid_examples": invalid_examples[:5]},
            )

    def _try_parse(self, response: str) -> bool:
        """Try to parse a response as JSON."""
        if not response:
            return False

        # Try direct parse
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown
        if "```json" in response:
            try:
                json_str = response.split("```json")[1].split("```")[0]
                json.loads(json_str.strip())
                return True
            except (IndexError, json.JSONDecodeError):
                pass

        # Try finding JSON object
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json.loads(response[start:end])
                return True
        except json.JSONDecodeError:
            pass

        return False


# =============================================================================
# Gate Registry
# =============================================================================

PROMPT_GATES = {
    "priming_detection": PrimingDetectionGate,
    "blinding_integrity": BlindingIntegrityGate,
    "vocabulary_alignment": VocabularyAlignmentGate,
    "schema_alignment": SchemaAlignmentGate,
    "neutrality": NeutralityGate,
    "output_parsability": OutputParsabilityGate,
}


def get_prompt_gate(name: str) -> BaseGate:
    """Get a prompt gate by name."""
    if name not in PROMPT_GATES:
        raise KeyError(f"Unknown prompt gate: {name}")
    return PROMPT_GATES[name]()
