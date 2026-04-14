"""
Placebo perturbation generators for control group testing.

Placebo perturbations change text WITHOUT changing meaning.
A well-calibrated judge should NOT penalize these changes.

Types:
- PARAPHRASE: LLM-based thought text paraphrasing
- FORMATTING: Heuristic whitespace/formatting changes in JSON
- SYNONYM: Heuristic word substitution with synonyms
- REORDER_ARGS: Heuristic JSON key reordering
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from src.llm import get_bedrock_client, DEFAULT_MODEL_ID
from src.perturbations.schema import (
    PerturbationClass,
    PerturbationFamily,
    PerturbationRecord,
    PerturbationType,
)
from src.typing.schema import TypedStep

# Use centralized model config
CLAUDE_MODEL_ID = DEFAULT_MODEL_ID


class BasePlaceboGenerator(ABC):
    """Base class for all placebo perturbation generators."""

    @property
    @abstractmethod
    def perturbation_type(self) -> PerturbationType:
        """Return the perturbation type this generator produces."""
        pass

    @abstractmethod
    def generate(
        self, typed_step: TypedStep, trajectory_id: str
    ) -> Optional[PerturbationRecord]:
        """
        Generate a placebo perturbation for the given step.

        Args:
            typed_step: The typed step to perturb
            trajectory_id: ID of the trajectory containing this step

        Returns:
            PerturbationRecord if perturbation was generated, None if not applicable
        """
        pass

    @abstractmethod
    def validate_equivalence(self, original: str, perturbed: str) -> bool:
        """
        Verify that semantic markers are preserved between original and perturbed text.

        Args:
            original: Original text
            perturbed: Perturbed text

        Returns:
            True if semantically equivalent, False otherwise
        """
        pass

    def _extract_semantic_markers(self, text: str) -> Dict[str, Set[str]]:
        """
        Extract semantic markers that must be preserved.

        Returns dict with:
        - numbers: All numeric values
        - file_paths: All file paths
        - identifiers: Variable names, function names, etc.
        """
        markers: Dict[str, Set[str]] = {
            "numbers": set(),
            "file_paths": set(),
            "identifiers": set(),
        }

        # Extract numbers (integers and floats)
        number_pattern = r"\b\d+(?:\.\d+)?\b"
        markers["numbers"] = set(re.findall(number_pattern, text))

        # Extract file paths (Unix and Windows style)
        filepath_pattern = r"(?:/[\w./\-_]+|[A-Za-z]:\\[\w\\./\-_]+|\./[\w./\-_]+)"
        markers["file_paths"] = set(re.findall(filepath_pattern, text))

        # Extract identifiers (camelCase, snake_case, etc.)
        identifier_pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*\b"
        potential_ids = re.findall(identifier_pattern, text)
        # Filter to likely identifiers (contain underscore or mixed case)
        markers["identifiers"] = set(
            id
            for id in potential_ids
            if "_" in id
            or (any(c.isupper() for c in id) and any(c.islower() for c in id))
        )

        return markers

    def _markers_preserved(
        self,
        original_markers: Dict[str, Set[str]],
        perturbed_markers: Dict[str, Set[str]],
    ) -> bool:
        """Check if all original markers are present in perturbed text."""
        for key in original_markers:
            if not original_markers[key].issubset(perturbed_markers[key]):
                return False
        return True


class PlaceboParaphraseGenerator(BasePlaceboGenerator):
    """
    Uses Claude LLM to paraphrase thought text while preserving semantics.

    Must preserve: tool_name, tool_arguments, numbers, file paths.
    """

    @property
    def perturbation_type(self) -> PerturbationType:
        return PerturbationType.PARAPHRASE

    def __init__(self, log_llm_calls: bool = False):
        self.log_llm_calls = log_llm_calls

    def generate(
        self, typed_step: TypedStep, trajectory_id: str
    ) -> Optional[PerturbationRecord]:
        """Generate a paraphrased version of the step's raw_text."""
        original_text = typed_step.raw_text

        # Skip very short text
        if len(original_text.strip()) < 20:
            return None

        # Extract what must be preserved
        preserved_items = self._get_preserved_items(typed_step)

        # Call LLM to paraphrase
        paraphrased = self._paraphrase_with_llm(original_text, preserved_items)

        if paraphrased is None:
            return None

        # Validate semantic equivalence
        if not self.validate_equivalence(original_text, paraphrased):
            return None

        # Create perturbation record
        return PerturbationRecord.create(
            original_trajectory_id=trajectory_id,
            perturbation_class=PerturbationClass.PLACEBO,
            perturbation_family=PerturbationFamily.DATA_REFERENCE,
            perturbation_type=PerturbationType.PARAPHRASE,
            target_step_index=typed_step.step_index,
            target_slot="raw_text",
            original_value=original_text,
            perturbed_value=paraphrased,
            mutation_method="llm_paraphrase",
            expected_impact=0,  # Placebo should have no impact
            notes="LLM-based paraphrase preserving semantic markers",
        )

    def _get_preserved_items(self, typed_step: TypedStep) -> Dict[str, Any]:
        """Extract items that must be preserved in paraphrase."""
        preserved = {
            "tool_name": typed_step.tool_name,
            "tool_arguments": typed_step.tool_arguments,
            "numbers": [],
            "file_paths": [],
        }

        # Extract from raw_text
        markers = self._extract_semantic_markers(typed_step.raw_text)
        preserved["numbers"] = list(markers["numbers"])
        preserved["file_paths"] = list(markers["file_paths"])

        return preserved

    def _paraphrase_with_llm(
        self, text: str, preserved_items: Dict[str, Any]
    ) -> Optional[str]:
        """Call Claude to paraphrase text while preserving semantic markers."""
        # Build the preservation list
        preserve_list = []
        if preserved_items.get("tool_name"):
            preserve_list.append(f"- Tool name: {preserved_items['tool_name']}")
        if preserved_items.get("tool_arguments"):
            preserve_list.append(
                f"- Tool arguments: {json.dumps(preserved_items['tool_arguments'])}"
            )
        if preserved_items.get("numbers"):
            preserve_list.append(f"- Numbers: {', '.join(preserved_items['numbers'])}")
        if preserved_items.get("file_paths"):
            preserve_list.append(
                f"- File paths: {', '.join(preserved_items['file_paths'])}"
            )

        preserve_str = "\n".join(preserve_list) if preserve_list else "None"

        prompt = f"""Paraphrase the following text while preserving its exact meaning.
You MUST preserve these items exactly as written:
{preserve_str}

Rules:
1. Keep the same meaning and intent
2. Use different words and sentence structure where possible
3. DO NOT change any technical terms, tool names, file paths, or numbers
4. DO NOT add new information or remove existing information
5. Keep the same length (within 20% of original)

Original text:
{text}

Respond with ONLY the paraphrased text, nothing else."""

        try:
            client = get_bedrock_client(log_calls=self.log_llm_calls)
            response = client.invoke(
                model_id=CLAUDE_MODEL_ID,
                prompt=prompt,
                max_tokens=len(text) * 2,  # Allow some room
                temperature=0.5,
            )
            return response["response"].strip()
        except Exception as e:
            # Log error but don't crash
            if self.log_llm_calls:
                print(f"   [PlaceboParaphrase] LLM error: {e}")
            return None

    def validate_equivalence(self, original: str, perturbed: str) -> bool:
        """Verify semantic markers are preserved."""
        original_markers = self._extract_semantic_markers(original)
        perturbed_markers = self._extract_semantic_markers(perturbed)

        # Check all markers preserved
        if not self._markers_preserved(original_markers, perturbed_markers):
            return False

        # Check length is similar (within 50% to allow for paraphrasing)
        len_ratio = len(perturbed) / len(original) if len(original) > 0 else 1
        if len_ratio < 0.5 or len_ratio > 1.5:
            return False

        return True


class PlaceboFormattingGenerator(BasePlaceboGenerator):
    """
    Heuristic-based formatting changes without LLM.

    Changes whitespace in JSON: {"a":1} -> { "a": 1 }
    Changes quote styles where safe.
    Normalizes spacing.
    """

    @property
    def perturbation_type(self) -> PerturbationType:
        return PerturbationType.FORMATTING

    def generate(
        self, typed_step: TypedStep, trajectory_id: str
    ) -> Optional[PerturbationRecord]:
        """Generate a formatting-changed version of tool_arguments or raw_text."""
        # Prefer to format tool_arguments if present
        if typed_step.tool_arguments:
            original = typed_step.tool_arguments
            formatted = self._reformat_json(original)

            if formatted is None or formatted == original:
                return None

            return PerturbationRecord.create(
                original_trajectory_id=trajectory_id,
                perturbation_class=PerturbationClass.PLACEBO,
                perturbation_family=PerturbationFamily.PARAMETER,
                perturbation_type=PerturbationType.FORMATTING,
                target_step_index=typed_step.step_index,
                target_slot="tool_arguments",
                original_value=original,
                perturbed_value=formatted,
                mutation_method="heuristic_json_formatting",
                expected_impact=0,
                notes="JSON whitespace reformatting",
            )

        # Otherwise try to reformat JSON in raw_text
        original_text = typed_step.raw_text
        formatted_text = self._reformat_json_in_text(original_text)

        if formatted_text is None or formatted_text == original_text:
            return None

        return PerturbationRecord.create(
            original_trajectory_id=trajectory_id,
            perturbation_class=PerturbationClass.PLACEBO,
            perturbation_family=PerturbationFamily.DATA_REFERENCE,
            perturbation_type=PerturbationType.FORMATTING,
            target_step_index=typed_step.step_index,
            target_slot="raw_text",
            original_value=original_text,
            perturbed_value=formatted_text,
            mutation_method="heuristic_text_formatting",
            expected_impact=0,
            notes="Text whitespace/formatting normalization",
        )

    def _reformat_json(self, obj: Any) -> Optional[Any]:
        """
        Reformat JSON object by changing whitespace style.
        Returns a new object (dict/list) that is semantically equivalent.
        """
        if isinstance(obj, dict):
            # For dict, we just return it - the change is in serialization
            # The semantic equivalence is preserved
            return obj
        elif isinstance(obj, list):
            return obj
        else:
            return None

    def _reformat_json_in_text(self, text: str) -> Optional[str]:
        """Find and reformat JSON snippets in text."""
        # Pattern to find JSON objects
        json_pattern = r"\{[^{}]*\}"
        matches = list(re.finditer(json_pattern, text))

        if not matches:
            return None

        result = text
        offset = 0

        for match in matches:
            try:
                json_str = match.group()
                parsed = json.loads(json_str)

                # Toggle between compact and spaced format
                if " " not in json_str.replace('" ', "").replace(' "', ""):
                    # Compact format -> add spaces
                    reformatted = json.dumps(parsed, separators=(", ", ": "))
                else:
                    # Spaced format -> make compact
                    reformatted = json.dumps(parsed, separators=(",", ":"))

                if reformatted != json_str:
                    start = match.start() + offset
                    end = match.end() + offset
                    result = result[:start] + reformatted + result[end:]
                    offset += len(reformatted) - len(json_str)
            except json.JSONDecodeError:
                continue

        return result if result != text else None

    def validate_equivalence(self, original: str, perturbed: str) -> bool:
        """Verify that formatting changes preserve semantics."""
        # For JSON strings, parse and compare
        try:
            orig_parsed = json.loads(original)
            pert_parsed = json.loads(perturbed)
            return orig_parsed == pert_parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # For text, extract and compare semantic markers
        original_markers = self._extract_semantic_markers(original)
        perturbed_markers = self._extract_semantic_markers(perturbed)
        return self._markers_preserved(original_markers, perturbed_markers)


class PlaceboSynonymGenerator(BasePlaceboGenerator):
    """
    Heuristic word substitution with synonyms.

    Only swaps common synonyms in thought text (not in tool arguments).
    """

    # Protected tokens that must not be modified (ReAct format markers)
    PROTECTED_TOKENS: List[str] = [
        "Action:",
        "Action Input:",
        "Observation:",
        "Thought:",
        "Final Answer:",
        "Finish",
        "Tool:",
        "Tool Input:",
        "Tool Output:",
    ]

    # Common synonym pairs for agent contexts
    SYNONYM_PAIRS: List[Tuple[str, str]] = [
        ("retrieve", "fetch"),
        ("search", "look up"),
        ("find", "locate"),
        ("examine", "inspect"),
        ("check", "verify"),
        ("create", "generate"),
        ("delete", "remove"),
        ("modify", "alter"),
        ("update", "change"),
        ("execute", "run"),
        ("start", "begin"),
        ("finish", "complete"),
        ("get", "obtain"),
        ("put", "place"),
        ("send", "transmit"),
        ("receive", "accept"),
        ("read", "access"),
        ("write", "save"),
        ("open", "access"),
        ("close", "terminate"),
        ("analyze", "examine"),
        ("compute", "calculate"),
        ("display", "show"),
        ("output", "return"),
        ("input", "provide"),
        ("validate", "verify"),
        ("error", "issue"),
        ("problem", "issue"),
        ("contains", "includes"),
        ("using", "with"),
        ("via", "through"),
    ]

    @property
    def perturbation_type(self) -> PerturbationType:
        return PerturbationType.SYNONYM

    def __init__(self):
        # Build bidirectional lookup
        self.synonym_map: Dict[str, str] = {}
        for word1, word2 in self.SYNONYM_PAIRS:
            self.synonym_map[word1.lower()] = word2
            self.synonym_map[word2.lower()] = word1

    def generate(
        self, typed_step: TypedStep, trajectory_id: str
    ) -> Optional[PerturbationRecord]:
        """Generate a synonym-substituted version of raw_text."""
        original_text = typed_step.raw_text

        # Find and replace synonyms
        substituted, changes = self._substitute_synonyms(original_text)

        if not changes or substituted == original_text:
            return None

        return PerturbationRecord.create(
            original_trajectory_id=trajectory_id,
            perturbation_class=PerturbationClass.PLACEBO,
            perturbation_family=PerturbationFamily.DATA_REFERENCE,
            perturbation_type=PerturbationType.SYNONYM,
            target_step_index=typed_step.step_index,
            target_slot="raw_text",
            original_value=original_text,
            perturbed_value=substituted,
            mutation_method="heuristic_synonym_substitution",
            expected_impact=0,
            notes=f"Synonym substitutions: {', '.join(changes)}",
        )

    def _substitute_synonyms(self, text: str) -> Tuple[str, List[str]]:
        """
        Substitute synonyms in text.

        Returns:
            Tuple of (substituted_text, list of changes made)
        """
        changes: List[str] = []
        result = text

        # Find protected token positions (ranges where substitution is forbidden)
        protected_ranges: List[Tuple[int, int]] = []
        for token in self.PROTECTED_TOKENS:
            pos = 0
            while True:
                idx = text.find(token, pos)
                if idx == -1:
                    break
                protected_ranges.append((idx, idx + len(token)))
                pos = idx + 1

        def is_protected(start: int, end: int) -> bool:
            """Check if position overlaps with any protected range."""
            for pstart, pend in protected_ranges:
                if start < pend and end > pstart:
                    return True
            return False

        # Find word boundaries and positions
        word_positions: List[Tuple[int, int, str]] = []

        for match in re.finditer(r"\b\w+\b", text):
            word_positions.append((match.start(), match.end(), match.group()))

        # Process in reverse order to maintain positions
        for start, end, word in reversed(word_positions):
            # Skip words inside protected tokens
            if is_protected(start, end):
                continue

            word_lower = word.lower()
            if word_lower in self.synonym_map:
                replacement = self.synonym_map[word_lower]
                # Preserve case
                if word.isupper():
                    replacement = replacement.upper()
                elif word[0].isupper():
                    replacement = replacement.capitalize()

                result = result[:start] + replacement + result[end:]
                changes.insert(0, f"{word}->{replacement}")

        return result, changes

    def validate_equivalence(self, original: str, perturbed: str) -> bool:
        """Verify that synonym substitutions preserve semantics."""
        # Semantic markers should be preserved
        original_markers = self._extract_semantic_markers(original)
        perturbed_markers = self._extract_semantic_markers(perturbed)

        if not self._markers_preserved(original_markers, perturbed_markers):
            return False

        # Word count should be similar (synonyms may be multi-word)
        orig_words = len(original.split())
        pert_words = len(perturbed.split())
        if abs(orig_words - pert_words) > max(3, orig_words * 0.2):
            return False

        return True


class PlaceboReorderArgsGenerator(BasePlaceboGenerator):
    """
    Reorders JSON keys in tool_arguments.

    Only for arguments that are order-independent (standard JSON objects).
    """

    @property
    def perturbation_type(self) -> PerturbationType:
        return PerturbationType.REORDER_ARGS

    def generate(
        self, typed_step: TypedStep, trajectory_id: str
    ) -> Optional[PerturbationRecord]:
        """Generate a key-reordered version of tool_arguments."""
        if not typed_step.tool_arguments:
            return None

        if not isinstance(typed_step.tool_arguments, dict):
            return None

        # Need at least 2 keys to reorder
        if len(typed_step.tool_arguments) < 2:
            return None

        original = typed_step.tool_arguments
        reordered = self._reorder_keys(original)

        # Compare key orders, not just values (dicts compare equal regardless of order)
        if list(reordered.keys()) == list(original.keys()):
            return None

        return PerturbationRecord.create(
            original_trajectory_id=trajectory_id,
            perturbation_class=PerturbationClass.PLACEBO,
            perturbation_family=PerturbationFamily.PARAMETER,
            perturbation_type=PerturbationType.REORDER_ARGS,
            target_step_index=typed_step.step_index,
            target_slot="tool_arguments",
            original_value=original,
            perturbed_value=reordered,
            mutation_method="heuristic_key_reorder",
            expected_impact=0,
            notes="JSON key reordering (order-independent semantics)",
        )

    def _reorder_keys(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reorder dictionary keys.

        Uses reverse alphabetical order to ensure a different order from typical
        insertion order (which is often alphabetical).
        """
        if not isinstance(obj, dict):
            return obj

        # Get keys in reverse alphabetical order
        sorted_keys = sorted(obj.keys(), reverse=True)

        # Check if already in reverse order
        if list(obj.keys()) == sorted_keys:
            # Try regular alphabetical
            sorted_keys = sorted(obj.keys())

        # Still same? No reordering possible
        if list(obj.keys()) == sorted_keys:
            return obj

        # Build reordered dict
        reordered: Dict[str, Any] = {}
        for key in sorted_keys:
            value = obj[key]
            # Recursively reorder nested dicts
            if isinstance(value, dict):
                reordered[key] = self._reorder_keys(value)
            else:
                reordered[key] = value

        return reordered

    def validate_equivalence(self, original: str, perturbed: str) -> bool:
        """Verify that key reordering preserves semantics."""
        # For JSON strings
        try:
            orig_parsed = json.loads(original)
            pert_parsed = json.loads(perturbed)
            return orig_parsed == pert_parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # For dict objects (when called with dicts converted to strings)
        return True  # Key reordering always preserves semantics in JSON


def get_placebo_generator(perturbation_type: PerturbationType) -> BasePlaceboGenerator:
    """
    Factory function to get the appropriate placebo generator.

    Args:
        perturbation_type: The type of placebo perturbation to generate

    Returns:
        BasePlaceboGenerator instance

    Raises:
        ValueError: If perturbation_type is not a valid placebo type
    """
    generators = {
        PerturbationType.PARAPHRASE: PlaceboParaphraseGenerator,
        PerturbationType.FORMATTING: PlaceboFormattingGenerator,
        PerturbationType.SYNONYM: PlaceboSynonymGenerator,
        PerturbationType.REORDER_ARGS: PlaceboReorderArgsGenerator,
    }

    if perturbation_type not in generators:
        valid_types = ", ".join(t.value for t in generators.keys())
        raise ValueError(
            f"Invalid placebo perturbation type: {perturbation_type}. "
            f"Valid types: {valid_types}"
        )

    return generators[perturbation_type]()


def get_all_placebo_generators() -> List[BasePlaceboGenerator]:
    """
    Get instances of all placebo generators.

    Returns:
        List of all BasePlaceboGenerator instances
    """
    return [
        PlaceboParaphraseGenerator(),
        PlaceboFormattingGenerator(),
        PlaceboSynonymGenerator(),
        PlaceboReorderArgsGenerator(),
    ]
