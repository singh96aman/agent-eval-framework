"""
Fine-grained perturbation generators for Section 3 Phase 2B.

Fine-grained perturbations inject SINGLE LOCAL-SLOT errors to test
whether judges can detect subtle mistakes in specific values.

This module provides:
- DataReferenceGenerator: Mutates extracted values (WRONG_VALUE, OFF_BY_ONE, TYPO_IN_ID, COPIED_VALUE_ERROR)
- ParameterGenerator: Mutates tool arguments (THRESHOLD_SHIFT, QUERY_DRIFT, WRONG_PARAMETER)
- ToolSelectionNearNeighborGenerator: Swaps with similar tool in same family

Each generator works with TypedStep's perturbable_slots and applies
value_type-specific mutations.
"""

import random
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Set, Tuple

from src.perturbations.schema import (
    PerturbationClass,
    PerturbationFamily,
    PerturbationType,
    PerturbationRecord,
)
from src.perturbations.tool_similarity import ToolSimilarityMatcher
from src.typing.schema import (
    TypedStep,
    TypedTrajectory,
    PerturbableSlot,
    ValueType,
)
from src.perturbations.llm_generator import (
    LLMPerturbationGenerator,
    LLMMutationResult,
    get_llm_generator,
)


class BaseFineGrainedGenerator(ABC):
    """
    Base class for fine-grained perturbation generators.

    Fine-grained perturbations target a single slot and apply
    value-type-specific mutations that create subtle but meaningful errors.
    """

    # Perturbation family this generator handles
    perturbation_family: PerturbationFamily
    # Perturbation types this generator can produce
    supported_types: List[PerturbationType]

    def __init__(
        self,
        random_seed: Optional[int] = None,
        use_llm: bool = True,
        llm_generator: Optional[LLMPerturbationGenerator] = None,
    ):
        """
        Initialize generator with optional random seed and LLM generator.

        Args:
            random_seed: Seed for reproducibility
            use_llm: Whether to use LLM for mutations (replaces artifact-producing heuristics)
            llm_generator: Optional pre-configured LLM generator instance
        """
        self.random = random.Random(random_seed)
        self.use_llm = use_llm
        self._llm_generator = llm_generator
        # Track LLM metadata for last mutation
        self._last_llm_result: Optional[LLMMutationResult] = None

    @property
    def llm_generator(self) -> Optional[LLMPerturbationGenerator]:
        """Lazy initialization of LLM generator."""
        if self.use_llm and self._llm_generator is None:
            self._llm_generator = get_llm_generator()
        return self._llm_generator

    def _char_transposition(self, value: str) -> Optional[Tuple[str, str]]:
        """
        Apply character transposition mutation.

        Swaps two adjacent characters at a random position.

        Args:
            value: String value to mutate

        Returns:
            Tuple of (mutated_value, "char_transposition") or None if too short
        """
        if len(value) < 2:
            return None
        chars = list(value)
        idx = self.random.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return ("".join(chars), "char_transposition")

    @abstractmethod
    def generate(
        self,
        typed_step: TypedStep,
        trajectory_id: str,
        perturbation_type: Optional[PerturbationType] = None,
        trajectory: Optional[TypedTrajectory] = None,
    ) -> Optional[PerturbationRecord]:
        """
        Generate a fine-grained perturbation for the given step.

        Args:
            typed_step: The typed step to perturb
            trajectory_id: ID of the trajectory containing this step
            perturbation_type: Specific type to generate (or None for random)
            trajectory: Full trajectory (needed for COPIED_VALUE_ERROR)

        Returns:
            PerturbationRecord if successful, None if no eligible slots
        """
        pass

    def mutate_value(self, value: Any, value_type: str) -> Tuple[Any, str]:
        """
        Apply mutation based on value type.

        Args:
            value: Original value to mutate
            value_type: Type of the value (from ValueType enum)

        Returns:
            Tuple of (mutated_value, mutation_method_description)
        """
        # Dispatch to type-specific mutation
        if value_type == ValueType.INTEGER.value or value_type == "integer":
            return self._mutate_integer(value)
        elif value_type == ValueType.FILEPATH.value or value_type == "filepath":
            return self._mutate_filepath(value)
        elif value_type == ValueType.SEARCH_QUERY.value or value_type == "search_query":
            return self._mutate_search_query(value)
        elif value_type == ValueType.LINE_NUMBER.value or value_type == "line_number":
            return self._mutate_line_number(value)
        elif value_type == ValueType.IDENTIFIER.value or value_type == "identifier":
            return self._mutate_identifier(value)
        elif value_type == ValueType.FLOAT.value or value_type == "float":
            return self._mutate_float(value)
        elif value_type == ValueType.STRING.value or value_type == "string":
            return self._mutate_string(value)
        elif value_type == ValueType.ENTITY_NAME.value or value_type == "entity_name":
            return self._mutate_entity_name(value)
        elif value_type == ValueType.URL.value or value_type == "url":
            return self._mutate_url(value)
        elif value_type == ValueType.IPV4.value or value_type == "ipv4":
            return self._mutate_ipv4(value)
        elif value_type == ValueType.DATE.value or value_type == "date":
            return self._mutate_date(value)
        elif value_type == ValueType.BOOLEAN.value or value_type == "boolean":
            return self._mutate_boolean(value)
        else:
            # Generic fallback
            return self._mutate_generic(value)

    def _mutate_integer(self, value: Any) -> Tuple[Any, str]:
        """Mutate integer value: +/-1, +/-2, +/-10, digit swap."""
        try:
            int_val = int(value)
        except (ValueError, TypeError):
            # Use LLM instead of artifact-producing fallback
            if self.use_llm and self.llm_generator:
                result = self.llm_generator.generate_value_mutation(
                    original_value=value,
                    value_type="integer",
                    context="Integer parameter that could not be parsed",
                )
                if result and result.parse_success:
                    self._last_llm_result = result
                    return (result.mutated_value, f"llm_{result.mutation_type}")
            # Non-LLM fallback: return None to skip this mutation
            return (None, "skip_unparseable")

        strategy = self.random.choice(
            ["offset_1", "offset_2", "offset_10", "digit_swap"]
        )

        if strategy == "offset_1":
            offset = self.random.choice([-1, 1])
            return (int_val + offset, f"offset_{'+' if offset > 0 else ''}{offset}")
        elif strategy == "offset_2":
            offset = self.random.choice([-2, 2])
            return (int_val + offset, f"offset_{'+' if offset > 0 else ''}{offset}")
        elif strategy == "offset_10":
            offset = self.random.choice([-10, 10])
            return (int_val + offset, f"offset_{'+' if offset > 0 else ''}{offset}")
        else:  # digit_swap
            str_val = str(abs(int_val))
            if len(str_val) >= 2:
                chars = list(str_val)
                # Swap two adjacent digits
                idx = self.random.randint(0, len(chars) - 2)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                result = int("".join(chars))
                if int_val < 0:
                    result = -result
                return (result, "digit_swap")
            else:
                # Single digit - just offset
                return (int_val + 1, "offset_+1")

    def _mutate_filepath(self, value: Any) -> Tuple[Any, str]:
        """Mutate filepath: char substitution, extension change, dir swap."""
        path = str(value)
        strategy = self.random.choice(["char_sub", "ext_change", "dir_swap", "typo"])

        if strategy == "char_sub" and len(path) > 3:
            # Substitute a character
            chars = list(path)
            # Find a valid position (not a path separator)
            valid_positions = [i for i, c in enumerate(chars) if c.isalnum()]
            if valid_positions:
                idx = self.random.choice(valid_positions)
                if chars[idx].isalpha():
                    # Swap with adjacent letter
                    if chars[idx].lower() < "z":
                        chars[idx] = chr(ord(chars[idx]) + 1)
                    else:
                        chars[idx] = chr(ord(chars[idx]) - 1)
                else:
                    # Numeric - change digit
                    chars[idx] = str((int(chars[idx]) + 1) % 10)
                return ("".join(chars), "char_substitution")

        if strategy == "ext_change":
            # Change extension
            if "." in path:
                base, ext = path.rsplit(".", 1)
                ext_map = {
                    "py": "pyx",
                    "js": "jsx",
                    "ts": "tsx",
                    "c": "cpp",
                    "h": "hpp",
                }
                new_ext = ext_map.get(ext, ext + "x")
                return (f"{base}.{new_ext}", "extension_change")

        if strategy == "dir_swap":
            # Swap directory components
            if "/" in path:
                parts = path.split("/")
                if len(parts) >= 3:
                    # Swap two directory parts
                    idx = self.random.randint(0, len(parts) - 3)
                    parts[idx], parts[idx + 1] = parts[idx + 1], parts[idx]
                    return ("/".join(parts), "dir_swap")

        # Typo fallback
        if "_" in path:
            # wait.py -> wait_py (common typo)
            return (path.replace(".", "_", 1), "underscore_typo")
        elif "." in path:
            # Add underscore before extension
            base, ext = path.rsplit(".", 1)
            return (f"{base}_.{ext}", "trailing_underscore")

        return (path + "_", "append_underscore")

    def _mutate_search_query(self, value: Any) -> Tuple[Any, str]:
        """Mutate search query: add/remove word, swap entity name."""
        query = str(value)
        words = query.split()

        if len(words) == 0:
            return (query + " extra", "append_word")

        strategy = self.random.choice(
            ["add_word", "remove_word", "swap_entity", "change_word"]
        )

        if strategy == "add_word" or len(words) == 1:
            # Add a plausible extra word
            modifiers = ["base", "large", "small", "v2", "latest", "old", "new"]
            modifier = self.random.choice(modifiers)
            insert_pos = self.random.randint(0, len(words))
            words.insert(insert_pos, modifier)
            return (" ".join(words), "add_word")

        elif strategy == "remove_word" and len(words) > 2:
            # Remove a word (not the first or last usually)
            idx = self.random.randint(0, len(words) - 1)
            removed = words.pop(idx)
            return (" ".join(words), f"remove_word_{removed}")

        elif strategy == "swap_entity":
            # Swap similar entities (bert -> gpt, etc.)
            entity_swaps = {
                "bert": "gpt",
                "gpt": "bert",
                "base": "large",
                "large": "base",
                "python": "java",
                "java": "python",
                "linux": "windows",
                "windows": "linux",
                "v1": "v2",
                "v2": "v1",
            }
            for i, word in enumerate(words):
                word_lower = word.lower()
                if word_lower in entity_swaps:
                    # Preserve case
                    replacement = entity_swaps[word_lower]
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    words[i] = replacement
                    return (
                        " ".join(words),
                        f"swap_entity_{word_lower}_to_{replacement}",
                    )
            # No entity found, change a word
            idx = self.random.randint(0, len(words) - 1)
            words[idx] = (
                words[idx] + "s" if not words[idx].endswith("s") else words[idx][:-1]
            )
            return (" ".join(words), "pluralize_word")

        else:  # change_word
            idx = self.random.randint(0, len(words) - 1)
            original = words[idx]
            # Small change to word
            if len(original) > 2:
                words[idx] = original[:-1]  # Remove last char
            else:
                words[idx] = original + "s"
            return (" ".join(words), f"modify_word_{original}")

    def _mutate_line_number(self, value: Any) -> Tuple[Any, str]:
        """Mutate line number: +/-1, +/-5, +/-10."""
        try:
            line = int(value)
        except (ValueError, TypeError):
            return (value, "no_mutation")

        offsets = [
            (-1, "off_by_minus_1"),
            (1, "off_by_plus_1"),
            (-5, "off_by_minus_5"),
            (5, "off_by_plus_5"),
            (-10, "off_by_minus_10"),
            (10, "off_by_plus_10"),
        ]
        offset, method = self.random.choice(offsets)
        result = max(1, line + offset)  # Line numbers can't be < 1
        return (result, method)

    def _mutate_identifier(self, value: Any) -> Tuple[Any, str]:
        """Mutate identifier: append number, swap suffix."""
        ident = str(value)
        strategy = self.random.choice(["append_num", "swap_suffix", "change_num"])

        if strategy == "append_num":
            # user_123 -> user_1234 or user_123_1
            if ident[-1].isdigit():
                return (ident + str(self.random.randint(0, 9)), "append_digit")
            else:
                return (ident + "_" + str(self.random.randint(1, 99)), "append_number")

        elif strategy == "swap_suffix":
            # user_123 -> user_124
            match = re.search(r"(\d+)$", ident)
            if match:
                num = int(match.group(1))
                new_num = num + self.random.choice([-1, 1])
                return (ident[: match.start()] + str(new_num), "increment_suffix")
            else:
                # No number suffix, add one
                return (ident + "_1", "add_suffix_number")

        else:  # change_num
            # Change any number in identifier
            def increment_match(m):
                return str(int(m.group()) + self.random.choice([-1, 1]))

            new_ident = re.sub(r"\d+", increment_match, ident, count=1)
            if new_ident != ident:
                return (new_ident, "change_embedded_number")
            return (ident + "_v2", "append_version")

    def _mutate_float(self, value: Any) -> Tuple[Any, str]:
        """Mutate float value with small perturbations."""
        try:
            float_val = float(value)
        except (ValueError, TypeError):
            return (value, "no_mutation")

        strategy = self.random.choice(["small_offset", "scale", "round"])

        if strategy == "small_offset":
            offset = (
                self.random.uniform(-0.1, 0.1) * abs(float_val)
                if float_val != 0
                else 0.01
            )
            return (round(float_val + offset, 4), "small_offset")
        elif strategy == "scale":
            scale = self.random.choice([0.9, 1.1, 0.95, 1.05])
            return (round(float_val * scale, 4), f"scale_{scale}")
        else:  # round
            # Change precision
            return (round(float_val, 1), "reduce_precision")

    def _mutate_string(self, value: Any) -> Tuple[Any, str]:
        """Mutate generic string."""
        s = str(value)
        if len(s) == 0:
            return ("_", "replace_empty")

        strategy = self.random.choice(["typo", "case", "truncate"])

        if strategy == "typo" and len(s) > 2:
            # Introduce typo - swap adjacent chars
            idx = self.random.randint(0, len(s) - 2)
            chars = list(s)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return ("".join(chars), "adjacent_char_swap")
        elif strategy == "case":
            if s.islower():
                return (s.upper(), "to_upper")
            elif s.isupper():
                return (s.lower(), "to_lower")
            else:
                return (s.swapcase(), "swap_case")
        else:  # truncate
            if len(s) > 3:
                return (s[:-1], "truncate_last")
            return (s + "_", "append_underscore")

    def _mutate_entity_name(self, value: Any) -> Tuple[Any, str]:
        """Mutate entity name (similar to identifier but with name-specific logic)."""
        name = str(value)

        # Common entity mutations
        if "_" in name:
            parts = name.split("_")
            # Change last part
            parts[-1] = parts[-1] + str(self.random.randint(1, 9))
            return ("_".join(parts), "append_digit_to_name")

        # CamelCase
        if any(c.isupper() for c in name[1:]):
            # Change last word
            return (name + "V2", "append_version")

        # Simple name
        return (name + str(self.random.randint(1, 9)), "append_digit")

    def _mutate_url(self, value: Any) -> Tuple[Any, str]:
        """Mutate URL."""
        url = str(value)

        strategy = self.random.choice(["path_typo", "domain_typo", "param_change"])

        if strategy == "path_typo" and "/" in url:
            # Typo in path segment
            parts = url.split("/")
            if len(parts) > 3:
                idx = self.random.randint(3, len(parts) - 1)
                if parts[idx]:
                    parts[idx] = (
                        parts[idx][:-1] if len(parts[idx]) > 1 else parts[idx] + "s"
                    )
            return ("/".join(parts), "path_typo")

        if strategy == "domain_typo":
            # Common domain typos
            typos = [
                ("google", "gogle"),
                ("github", "gitub"),
                ("stackoverflow", "stackoverfow"),
            ]
            for correct, wrong in typos:
                if correct in url.lower():
                    return (url.replace(correct, wrong), f"domain_typo_{correct}")

        # Param change fallback
        if "?" in url:
            return (url + "&extra=1", "add_param")
        return (url + "?v=1", "add_query_param")

    def _mutate_ipv4(self, value: Any) -> Tuple[Any, str]:
        """Mutate IPv4 address."""
        ip = str(value)
        parts = ip.split(".")

        if len(parts) == 4:
            # Change last octet
            try:
                last = int(parts[-1])
                parts[-1] = str((last + self.random.randint(1, 10)) % 256)
                return (".".join(parts), "change_last_octet")
            except ValueError:
                pass

        return (ip + ".1", "append_octet")

    def _mutate_date(self, value: Any) -> Tuple[Any, str]:
        """Mutate date value."""
        date_str = str(value)

        # Try ISO format YYYY-MM-DD
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        if match:
            year, month, day = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
            strategy = self.random.choice(["day", "month", "year"])

            if strategy == "day":
                new_day = ((day + self.random.randint(1, 5) - 1) % 28) + 1
                return (f"{year:04d}-{month:02d}-{new_day:02d}", "change_day")
            elif strategy == "month":
                new_month = ((month + self.random.randint(1, 3) - 1) % 12) + 1
                return (f"{year:04d}-{new_month:02d}-{day:02d}", "change_month")
            else:
                return (
                    f"{year + self.random.choice([-1, 1]):04d}-{month:02d}-{day:02d}",
                    "change_year",
                )

        # Use LLM for non-ISO format dates instead of artifact
        if self.use_llm and self.llm_generator:
            result = self.llm_generator.generate_wrong_date(
                original_date=date_str,
                date_format="unknown",
                context="Date that does not match ISO format",
            )
            if result and result.parse_success:
                self._last_llm_result = result
                return (result.mutated_value, f"llm_{result.mutation_type}")
        # Non-LLM fallback: skip mutation
        return (None, "skip_unparseable_date")

    def _mutate_boolean(self, value: Any) -> Tuple[Any, str]:
        """Mutate boolean value (flip)."""
        if isinstance(value, bool):
            return (not value, "flip_boolean")
        if str(value).lower() in ("true", "false"):
            return (
                "false" if str(value).lower() == "true" else "true",
                "flip_boolean_string",
            )
        return (value, "no_mutation")

    def _mutate_generic(self, value: Any) -> Tuple[Any, str]:
        """Generic mutation for unknown types."""
        str_val = str(value)
        if not str_val:
            return (None, "skip_empty_value")

        # Use LLM for generic mutations instead of artifact
        if self.use_llm and self.llm_generator:
            result = self.llm_generator.generate_value_mutation(
                original_value=str_val,
                value_type="unknown",
                context="Generic value requiring mutation",
            )
            if result and result.parse_success:
                self._last_llm_result = result
                return (result.mutated_value, f"llm_{result.mutation_type}")
        # Non-LLM fallback: apply simple character transposition
        result = self._char_transposition(str_val)
        if result:
            return result
        return (None, "skip_short_value")

    def _filter_eligible_slots(
        self,
        slots: List[PerturbableSlot],
        allowed_families: Set[str],
    ) -> List[PerturbableSlot]:
        """
        Filter slots to those eligible for this generator.

        Args:
            slots: List of perturbable slots
            allowed_families: Set of allowed perturbation family names

        Returns:
            Filtered list of eligible slots
        """
        eligible = []
        for slot in slots:
            # Check if any allowed family is in the slot's allowed types
            slot_families = set(slot.allowed_perturbation_types)
            if slot_families & allowed_families:
                eligible.append(slot)
        return eligible


class DataReferenceGenerator(BaseFineGrainedGenerator):
    """
    Generator for data reference perturbations.

    Mutates extracted values that flow from one step to another:
    - WRONG_VALUE: integer 12->10, string "bert"->"gpt"
    - OFF_BY_ONE: line_number 108->107, index 5->4
    - TYPO_IN_ID: filepath "wait.py"->"wait_py", entity "user_123"->"user_124"
    - COPIED_VALUE_ERROR: use value from wrong step
    """

    perturbation_family = PerturbationFamily.DATA_REFERENCE
    supported_types = [
        PerturbationType.WRONG_VALUE,
        PerturbationType.OFF_BY_ONE,
        PerturbationType.TYPO_IN_ID,
        PerturbationType.COPIED_VALUE_ERROR,
    ]

    def generate(
        self,
        typed_step: TypedStep,
        trajectory_id: str,
        perturbation_type: Optional[PerturbationType] = None,
        trajectory: Optional[TypedTrajectory] = None,
    ) -> Optional[PerturbationRecord]:
        """Generate a data reference perturbation."""
        # Filter to eligible slots
        eligible_slots = self._filter_eligible_slots(
            typed_step.perturbable_slots,
            {"data_reference"},
        )

        if not eligible_slots:
            return None

        # Choose perturbation type if not specified
        if perturbation_type is None:
            # COPIED_VALUE_ERROR requires trajectory
            available_types = list(self.supported_types)
            if trajectory is None:
                available_types = [
                    t
                    for t in available_types
                    if t != PerturbationType.COPIED_VALUE_ERROR
                ]
            perturbation_type = self.random.choice(available_types)

        # Special handling for COPIED_VALUE_ERROR
        if perturbation_type == PerturbationType.COPIED_VALUE_ERROR:
            return self._generate_copied_value_error(
                typed_step, trajectory_id, trajectory, eligible_slots
            )

        # Choose a slot
        slot = self.random.choice(eligible_slots)

        # Apply type-specific mutation
        if perturbation_type == PerturbationType.WRONG_VALUE:
            mutated_value, method = self._apply_wrong_value(slot)
        elif perturbation_type == PerturbationType.OFF_BY_ONE:
            mutated_value, method = self._apply_off_by_one(slot)
        elif perturbation_type == PerturbationType.TYPO_IN_ID:
            mutated_value, method = self._apply_typo_in_id(slot)
        else:
            # Fallback to generic mutation
            mutated_value, method = self.mutate_value(
                slot.current_value, slot.value_type
            )

        # Ensure mutation actually changed the value
        if mutated_value == slot.current_value:
            return None

        # Determine expected impact
        expected_impact = self._estimate_impact(typed_step, slot)

        return PerturbationRecord.create(
            original_trajectory_id=trajectory_id,
            perturbation_class=PerturbationClass.FINE_GRAINED,
            perturbation_family=PerturbationFamily.DATA_REFERENCE,
            perturbation_type=perturbation_type,
            target_step_index=typed_step.step_index,
            target_slot=slot.slot,
            original_value=slot.current_value,
            perturbed_value=mutated_value,
            mutation_method=method,
            expected_impact=expected_impact,
            notes=f"value_type={slot.value_type}",
        )

    def _apply_wrong_value(self, slot: PerturbableSlot) -> Tuple[Any, str]:
        """Apply WRONG_VALUE mutation - significant value change."""
        value = slot.current_value
        value_type = slot.value_type

        # Larger changes than generic mutations
        if value_type in (ValueType.INTEGER.value, "integer"):
            try:
                int_val = int(value)
                # Change by 10-20%
                change = max(2, abs(int_val) // 5)
                offset = self.random.choice([-change, change])
                return (int_val + offset, f"wrong_value_offset_{offset}")
            except (ValueError, TypeError):
                pass

        if value_type in (ValueType.STRING.value, "string"):
            # String swaps
            swaps = {
                "bert": "gpt",
                "gpt": "bert",
                "true": "false",
                "false": "true",
                "yes": "no",
                "no": "yes",
                "enabled": "disabled",
                "disabled": "enabled",
            }
            str_val = str(value).lower()
            for orig, repl in swaps.items():
                if orig in str_val:
                    return (
                        str(value).replace(orig, repl),
                        f"wrong_value_swap_{orig}_to_{repl}",
                    )

        # Fallback to type-specific mutation
        return self.mutate_value(value, value_type)

    def _apply_off_by_one(self, slot: PerturbableSlot) -> Tuple[Any, str]:
        """Apply OFF_BY_ONE mutation - subtle numeric error."""
        value = slot.current_value
        value_type = slot.value_type

        # Only applies to numeric types
        if value_type in (
            ValueType.INTEGER.value,
            "integer",
            ValueType.LINE_NUMBER.value,
            "line_number",
        ):
            try:
                int_val = int(value)
                offset = self.random.choice([-1, 1])
                result = max(0, int_val + offset)  # Prevent negative for indices
                return (result, f"off_by_one_{'+1' if offset > 0 else '-1'}")
            except (ValueError, TypeError):
                pass

        if value_type in (ValueType.FLOAT.value, "float"):
            try:
                float_val = float(value)
                offset = self.random.choice([-0.1, 0.1, -1.0, 1.0])
                return (round(float_val + offset, 4), f"off_by_{offset}")
            except (ValueError, TypeError):
                pass

        # For identifiers with numbers
        if value_type in (ValueType.IDENTIFIER.value, "identifier"):
            return self._mutate_identifier(value)

        return self.mutate_value(value, value_type)

    def _apply_typo_in_id(self, slot: PerturbableSlot) -> Tuple[Any, str]:
        """Apply TYPO_IN_ID mutation - character-level error."""
        value = slot.current_value
        value_type = slot.value_type

        str_val = str(value)

        if value_type in (ValueType.FILEPATH.value, "filepath"):
            # wait.py -> wait_py
            if "." in str_val:
                return (str_val.replace(".", "_", 1), "typo_dot_to_underscore")

        if value_type in (
            ValueType.IDENTIFIER.value,
            "identifier",
            ValueType.ENTITY_NAME.value,
            "entity_name",
        ):
            # user_123 -> user_124
            match = re.search(r"(\d+)$", str_val)
            if match:
                num = int(match.group(1))
                new_num = num + 1
                return (
                    str_val[: match.start()] + str(new_num),
                    "typo_increment_suffix",
                )

            # Character swap
            if len(str_val) >= 2:
                idx = self.random.randint(0, len(str_val) - 2)
                chars = list(str_val)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                return ("".join(chars), "typo_char_swap")

        # Generic typo
        if len(str_val) >= 3:
            idx = self.random.randint(0, len(str_val) - 2)
            chars = list(str_val)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return ("".join(chars), "typo_adjacent_swap")

        return self.mutate_value(value, value_type)

    def _generate_copied_value_error(
        self,
        typed_step: TypedStep,
        trajectory_id: str,
        trajectory: Optional[TypedTrajectory],
        eligible_slots: List[PerturbableSlot],
    ) -> Optional[PerturbationRecord]:
        """Generate COPIED_VALUE_ERROR - use value from wrong step."""
        if trajectory is None or len(trajectory.steps) < 2:
            return None

        # Find a slot to perturb
        slot = self.random.choice(eligible_slots)

        # Find a similar value from a different step
        current_step_idx = typed_step.step_index
        wrong_value = None
        source_step_idx = None

        for step in trajectory.steps:
            if step.step_index == current_step_idx:
                continue

            # Look for a value of the same type in this step
            for other_slot in step.perturbable_slots:
                if other_slot.value_type == slot.value_type:
                    if other_slot.current_value != slot.current_value:
                        wrong_value = other_slot.current_value
                        source_step_idx = step.step_index
                        break

            if wrong_value is not None:
                break

        if wrong_value is None:
            # No suitable value found, fall back to wrong_value
            return self.generate(
                typed_step, trajectory_id, PerturbationType.WRONG_VALUE, trajectory
            )

        expected_impact = self._estimate_impact(typed_step, slot)

        return PerturbationRecord.create(
            original_trajectory_id=trajectory_id,
            perturbation_class=PerturbationClass.FINE_GRAINED,
            perturbation_family=PerturbationFamily.DATA_REFERENCE,
            perturbation_type=PerturbationType.COPIED_VALUE_ERROR,
            target_step_index=typed_step.step_index,
            target_slot=slot.slot,
            original_value=slot.current_value,
            perturbed_value=wrong_value,
            mutation_method=f"copied_from_step_{source_step_idx}",
            expected_impact=expected_impact,
            notes=f"value_type={slot.value_type}, source_step={source_step_idx}",
        )

    def _estimate_impact(self, step: TypedStep, slot: PerturbableSlot) -> int:
        """Estimate expected impact of perturbation (0-3)."""
        # Base impact on step properties
        impact = 1  # Default subtle

        if step.is_terminal_step or step.produces_final_answer:
            impact = 3  # Critical
        elif step.affects_final_answer and step.affects_final_answer.value:
            impact = 2  # Moderate
        elif step.critical_path_score and step.critical_path_score.value > 0.7:
            impact = 2  # Moderate

        return impact


class ParameterGenerator(BaseFineGrainedGenerator):
    """
    Generator for parameter perturbations.

    Mutates tool arguments:
    - THRESHOLD_SHIFT: top_k 5->50, limit 10->100
    - QUERY_DRIFT: "bert base layers"->"bert large layers"
    - WRONG_PARAMETER: change one arg value plausibly
    """

    perturbation_family = PerturbationFamily.PARAMETER
    supported_types = [
        PerturbationType.THRESHOLD_SHIFT,
        PerturbationType.QUERY_DRIFT,
        PerturbationType.WRONG_PARAMETER,
    ]

    # Common threshold/limit parameter names
    THRESHOLD_PARAMS = {
        "top_k",
        "top_n",
        "limit",
        "max_results",
        "count",
        "num_results",
        "k",
        "n",
        "threshold",
        "max",
        "min",
        "batch_size",
        "page_size",
    }

    # Common query parameter names
    QUERY_PARAMS = {
        "query",
        "q",
        "search",
        "search_query",
        "text",
        "prompt",
        "keywords",
        "terms",
        "filter",
    }

    def generate(
        self,
        typed_step: TypedStep,
        trajectory_id: str,
        perturbation_type: Optional[PerturbationType] = None,
        trajectory: Optional[TypedTrajectory] = None,
    ) -> Optional[PerturbationRecord]:
        """Generate a parameter perturbation."""
        # Filter to eligible slots
        eligible_slots = self._filter_eligible_slots(
            typed_step.perturbable_slots,
            {"parameter"},
        )

        if not eligible_slots:
            return None

        # Choose perturbation type if not specified
        if perturbation_type is None:
            perturbation_type = self.random.choice(self.supported_types)

        # Try to find appropriate slot for the perturbation type
        slot = None

        if perturbation_type == PerturbationType.THRESHOLD_SHIFT:
            # Prefer numeric slots with threshold-like names
            threshold_slots = [s for s in eligible_slots if self._is_threshold_slot(s)]
            if threshold_slots:
                slot = self.random.choice(threshold_slots)

        elif perturbation_type == PerturbationType.QUERY_DRIFT:
            # Prefer string slots with query-like names or search_query type
            query_slots = [s for s in eligible_slots if self._is_query_slot(s)]
            if query_slots:
                slot = self.random.choice(query_slots)

        # Fallback to any eligible slot
        if slot is None:
            slot = self.random.choice(eligible_slots)
            # Adjust type based on slot
            if self._is_threshold_slot(slot):
                perturbation_type = PerturbationType.THRESHOLD_SHIFT
            elif self._is_query_slot(slot):
                perturbation_type = PerturbationType.QUERY_DRIFT
            else:
                perturbation_type = PerturbationType.WRONG_PARAMETER

        # Apply mutation
        if perturbation_type == PerturbationType.THRESHOLD_SHIFT:
            mutated_value, method = self._apply_threshold_shift(slot)
        elif perturbation_type == PerturbationType.QUERY_DRIFT:
            mutated_value, method = self._apply_query_drift(slot)
        else:  # WRONG_PARAMETER
            mutated_value, method = self._apply_wrong_parameter(slot)

        # Ensure mutation actually changed the value
        if mutated_value == slot.current_value:
            return None

        expected_impact = self._estimate_impact(typed_step, slot)

        return PerturbationRecord.create(
            original_trajectory_id=trajectory_id,
            perturbation_class=PerturbationClass.FINE_GRAINED,
            perturbation_family=PerturbationFamily.PARAMETER,
            perturbation_type=perturbation_type,
            target_step_index=typed_step.step_index,
            target_slot=slot.slot,
            original_value=slot.current_value,
            perturbed_value=mutated_value,
            mutation_method=method,
            expected_impact=expected_impact,
            notes=f"value_type={slot.value_type}",
        )

    def _is_threshold_slot(self, slot: PerturbableSlot) -> bool:
        """Check if slot is a threshold/limit parameter."""
        # Check value type
        if slot.value_type in (
            ValueType.INTEGER.value,
            "integer",
            ValueType.FLOAT.value,
            "float",
        ):
            # Check slot name
            slot_name = slot.slot.lower().split(".")[-1]
            return any(p in slot_name for p in self.THRESHOLD_PARAMS)
        return False

    def _is_query_slot(self, slot: PerturbableSlot) -> bool:
        """Check if slot is a query parameter."""
        # Check value type
        if slot.value_type in (
            ValueType.SEARCH_QUERY.value,
            "search_query",
            ValueType.STRING.value,
            "string",
        ):
            # Check slot name
            slot_name = slot.slot.lower().split(".")[-1]
            return any(p in slot_name for p in self.QUERY_PARAMS)
        return False

    def _apply_threshold_shift(self, slot: PerturbableSlot) -> Tuple[Any, str]:
        """Apply THRESHOLD_SHIFT - scale threshold by order of magnitude."""
        value = slot.current_value

        try:
            if slot.value_type in (ValueType.INTEGER.value, "integer"):
                int_val = int(value)
                # Scale by 10x or 0.1x
                if int_val == 0:
                    return (10, "threshold_shift_from_zero")
                scale = self.random.choice([10, 0.1])
                result = int(int_val * scale)
                if result == int_val:
                    result = int_val * 10
                return (result, f"threshold_shift_scale_{scale}x")

            elif slot.value_type in (ValueType.FLOAT.value, "float"):
                float_val = float(value)
                if float_val == 0:
                    return (1.0, "threshold_shift_from_zero")
                scale = self.random.choice([10.0, 0.1])
                return (round(float_val * scale, 4), f"threshold_shift_scale_{scale}x")
        except (ValueError, TypeError):
            pass

        return self.mutate_value(value, slot.value_type)

    def _apply_query_drift(self, slot: PerturbableSlot) -> Tuple[Any, str]:
        """Apply QUERY_DRIFT - modify search query subtly."""
        return self._mutate_search_query(slot.current_value)

    def _apply_wrong_parameter(self, slot: PerturbableSlot) -> Tuple[Any, str]:
        """Apply WRONG_PARAMETER - plausible but wrong value."""
        value = slot.current_value
        value_type = slot.value_type

        # Type-specific wrong values
        if value_type in (ValueType.BOOLEAN.value, "boolean"):
            return self._mutate_boolean(value)

        if value_type in (ValueType.INTEGER.value, "integer"):
            # Common wrong values for integers
            try:
                int_val = int(value)
                wrong_values = [0, 1, -1, 100, 1000, int_val * 2, int_val // 2]
                wrong_values = [v for v in wrong_values if v != int_val]
                if wrong_values:
                    return (self.random.choice(wrong_values), "wrong_common_value")
            except (ValueError, TypeError):
                pass

        if value_type in (ValueType.STRING.value, "string"):
            str_val = str(value)
            # Use LLM instead of artifact-producing suffixes
            if self.use_llm and self.llm_generator:
                result = self.llm_generator.generate_wrong_identifier(
                    original_id=str_val,
                    id_type="string_parameter",
                    context="Parameter value that needs a plausible wrong value",
                )
                if result and result.parse_success:
                    self._last_llm_result = result
                    return (result.mutated_value, f"llm_{result.mutation_type}")
            # Non-LLM fallback: use character transposition instead of suffixes
            result = self._char_transposition(str_val)
            if result:
                return result
            # Single char: duplicate it
            return (str_val + str_val, "char_duplicate")

        return self.mutate_value(value, value_type)

    def _estimate_impact(self, step: TypedStep, slot: PerturbableSlot) -> int:
        """Estimate expected impact of parameter perturbation."""
        impact = 1  # Default subtle

        # Threshold shifts can have large impact
        if self._is_threshold_slot(slot):
            impact = 2  # Moderate - affects result quantity

        # Query drift can significantly change results
        if self._is_query_slot(slot):
            impact = 2  # Moderate

        # Terminal steps are critical
        if step.is_terminal_step or step.produces_final_answer:
            impact = 3

        return impact


class ToolSelectionNearNeighborGenerator(BaseFineGrainedGenerator):
    """
    Generator for near-neighbor tool selection perturbations.

    Swaps tool with a similar tool in the same family:
    - grep -> ripgrep
    - web_search -> google_search

    Must be same FAMILY (search tools, file tools, etc.)
    """

    perturbation_family = PerturbationFamily.TOOL_SELECTION
    supported_types = [PerturbationType.NEAR_NEIGHBOR_TOOL]

    # Tool families for near-neighbor matching
    TOOL_FAMILIES = {
        "search": {
            "grep",
            "ripgrep",
            "rg",
            "ack",
            "ag",
            "find",
            "locate",
            "search",
            "web_search",
            "google_search",
            "bing_search",
        },
        "file_read": {
            "cat",
            "head",
            "tail",
            "less",
            "more",
            "read",
            "view",
            "read_file",
            "get_file",
            "file_content",
        },
        "file_write": {
            "write",
            "write_file",
            "save",
            "save_file",
            "create_file",
            "echo",
            "tee",
        },
        "file_edit": {
            "edit",
            "sed",
            "awk",
            "patch",
            "apply_patch",
            "modify",
            "str_replace_editor",
            "edit_file",
        },
        "execute": {
            "run",
            "exec",
            "execute",
            "bash",
            "sh",
            "shell",
            "run_command",
            "execute_command",
        },
        "list": {
            "ls",
            "dir",
            "list",
            "list_files",
            "list_dir",
            "get_directory",
            "tree",
        },
        "api": {
            "api_call",
            "http_get",
            "http_post",
            "fetch",
            "request",
            "curl",
            "wget",
        },
    }

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize with tool similarity matcher."""
        super().__init__(random_seed)
        self.tool_matcher = ToolSimilarityMatcher()

    def generate(
        self,
        typed_step: TypedStep,
        trajectory_id: str,
        perturbation_type: Optional[PerturbationType] = None,
        trajectory: Optional[TypedTrajectory] = None,
    ) -> Optional[PerturbationRecord]:
        """Generate a near-neighbor tool selection perturbation."""
        # This generator works on the tool_name, not perturbable_slots
        if not typed_step.tool_name:
            return None

        original_tool = typed_step.tool_name

        # Find the tool's family
        tool_family = self._get_tool_family(original_tool)

        # Find near-neighbor tools
        neighbors = self._find_near_neighbors(original_tool, tool_family)

        if not neighbors:
            return None

        # Choose a random neighbor
        new_tool = self.random.choice(neighbors)

        expected_impact = self._estimate_impact(typed_step, original_tool, new_tool)

        return PerturbationRecord.create(
            original_trajectory_id=trajectory_id,
            perturbation_class=PerturbationClass.FINE_GRAINED,
            perturbation_family=PerturbationFamily.TOOL_SELECTION,
            perturbation_type=PerturbationType.NEAR_NEIGHBOR_TOOL,
            target_step_index=typed_step.step_index,
            target_slot="tool_name",
            original_value=original_tool,
            perturbed_value=new_tool,
            mutation_method=f"near_neighbor_swap_family_{tool_family or 'unknown'}",
            expected_impact=expected_impact,
            notes=f"tool_family={tool_family}",
        )

    def _get_tool_family(self, tool_name: str) -> Optional[str]:
        """Determine the family of a tool."""
        tool_lower = tool_name.lower()

        for family, tools in self.TOOL_FAMILIES.items():
            # Exact match
            if tool_lower in tools:
                return family
            # Partial match (tool name contains family tool)
            for t in tools:
                if t in tool_lower or tool_lower in t:
                    return family

        return None

    def _find_near_neighbors(
        self, tool_name: str, tool_family: Optional[str]
    ) -> List[str]:
        """Find near-neighbor tools in the same family."""
        neighbors = []

        if tool_family and tool_family in self.TOOL_FAMILIES:
            family_tools = self.TOOL_FAMILIES[tool_family]
            for tool in family_tools:
                if tool.lower() != tool_name.lower():
                    neighbors.append(tool)

        # Also try name-based similarity
        tool_lower = tool_name.lower()

        # Common tool swaps
        tool_swaps = {
            "grep": ["ripgrep", "rg", "ack"],
            "ripgrep": ["grep", "rg", "ack"],
            "rg": ["grep", "ripgrep", "ack"],
            "cat": ["head", "tail", "less"],
            "head": ["cat", "tail"],
            "tail": ["cat", "head"],
            "ls": ["dir", "tree", "find"],
            "find": ["ls", "locate", "grep"],
            "web_search": ["google_search", "bing_search", "search"],
            "google_search": ["web_search", "bing_search"],
            "read_file": ["cat", "get_file", "view_file"],
            "write_file": ["save_file", "create_file"],
            "execute": ["run", "bash", "shell"],
            "bash": ["sh", "shell", "execute"],
        }

        if tool_lower in tool_swaps:
            neighbors.extend(tool_swaps[tool_lower])

        # Remove duplicates and original tool
        neighbors = list(set(n for n in neighbors if n.lower() != tool_lower))

        return neighbors

    def _estimate_impact(
        self, step: TypedStep, original_tool: str, new_tool: str
    ) -> int:
        """Estimate impact of tool swap."""
        # Near-neighbor swaps are subtle - usually impact 1-2
        impact = 1

        # If terminal step, higher impact
        if step.is_terminal_step or step.produces_final_answer:
            impact = 2

        # Different tool families would be coarse-grained, not fine-grained
        # So within-family swaps are subtle

        return impact

    def index_tools_from_trajectory(self, trajectory: TypedTrajectory):
        """
        Index tools used in a trajectory for better neighbor matching.

        This allows finding neighbors among tools actually available
        in the trajectory's context.
        """
        tools_used = set()
        for step in trajectory.steps:
            if step.tool_name:
                tools_used.add(step.tool_name)

        # The tool matcher can use this information
        # For now, we rely on our predefined families
        return tools_used


def get_fine_grained_generator(
    perturbation_family: PerturbationFamily,
    perturbation_type: Optional[PerturbationType] = None,
    random_seed: Optional[int] = None,
    llm_client: Optional[Any] = None,
) -> BaseFineGrainedGenerator:
    """
    Factory function to get the appropriate fine-grained generator.

    Args:
        perturbation_family: The perturbation family
        perturbation_type: Optional specific type (for validation)
        random_seed: Random seed for reproducibility
        llm_client: Optional LLM client for generating semantic perturbations

    Returns:
        Appropriate generator instance

    Raises:
        ValueError: If family/type combination is invalid
    """
    generators = {
        PerturbationFamily.DATA_REFERENCE: DataReferenceGenerator,
        PerturbationFamily.PARAMETER: ParameterGenerator,
        PerturbationFamily.TOOL_SELECTION: ToolSelectionNearNeighborGenerator,
    }

    if perturbation_family not in generators:
        raise ValueError(
            f"No fine-grained generator for family {perturbation_family}. "
            f"Valid families: {list(generators.keys())}"
        )

    generator_class = generators[perturbation_family]

    # Pass LLM generator if client is provided (enables LLM-based mutations)
    llm_generator = None
    if llm_client is not None:
        from src.perturbations.llm_generator import LLMPerturbationGenerator
        llm_generator = LLMPerturbationGenerator()

    generator = generator_class(random_seed, llm_generator=llm_generator)

    # Validate perturbation type if specified
    if perturbation_type is not None:
        if perturbation_type not in generator.supported_types:
            raise ValueError(
                f"Perturbation type {perturbation_type} not supported by "
                f"{generator_class.__name__}. Supported types: {generator.supported_types}"
            )

    return generator


# Convenience functions for direct access
def generate_data_reference_perturbation(
    typed_step: TypedStep,
    trajectory_id: str,
    perturbation_type: Optional[PerturbationType] = None,
    trajectory: Optional[TypedTrajectory] = None,
    random_seed: Optional[int] = None,
) -> Optional[PerturbationRecord]:
    """Generate a data reference perturbation."""
    generator = DataReferenceGenerator(random_seed)
    return generator.generate(typed_step, trajectory_id, perturbation_type, trajectory)


def generate_parameter_perturbation(
    typed_step: TypedStep,
    trajectory_id: str,
    perturbation_type: Optional[PerturbationType] = None,
    trajectory: Optional[TypedTrajectory] = None,
    random_seed: Optional[int] = None,
) -> Optional[PerturbationRecord]:
    """Generate a parameter perturbation."""
    generator = ParameterGenerator(random_seed)
    return generator.generate(typed_step, trajectory_id, perturbation_type, trajectory)


def generate_tool_selection_perturbation(
    typed_step: TypedStep,
    trajectory_id: str,
    trajectory: Optional[TypedTrajectory] = None,
    random_seed: Optional[int] = None,
) -> Optional[PerturbationRecord]:
    """Generate a near-neighbor tool selection perturbation."""
    generator = ToolSelectionNearNeighborGenerator(random_seed)
    return generator.generate(typed_step, trajectory_id, None, trajectory)
