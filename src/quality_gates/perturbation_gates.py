"""
Perturbation Quality Gates (7 gates).

Validates perturbation quality per Requirements.MD Section 4.3.
All gates use ONLY regex/parsing - NO LLM calls.
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.quality_gates.base import BaseGate, GateResult


# =============================================================================
# Synthetic Markers Gate (Detailed)
# =============================================================================


class GateNoSyntheticMarkers(BaseGate):
    """Gate: No synthetic artifact markers in perturbed values."""

    name = "gate_no_synthetic_markers"
    description = "Verify no _old, _mutated, _wrong markers"

    # Comprehensive artifact patterns
    ARTIFACT_PATTERNS = [
        r"_old\b",
        r"_mutated\b",
        r"_wrong\b",
        r"_backup\b",
        r"_test\b",
        r"_v1\b",
        r"_original\b",
        r"_modified\b",
        r"_perturbed\b",
        r"_changed\b",
    ]

    ARTIFACT_REGEX = re.compile(
        "|".join(ARTIFACT_PATTERNS),
        re.IGNORECASE
    )

    def check(
        self, perturbations: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check for synthetic markers in perturbations."""
        if not perturbations:
            return self._skip("No perturbations to check")

        violations = []
        for p in perturbations:
            perturbed_value = str(p.get("perturbed_value", ""))
            matches = self.ARTIFACT_REGEX.findall(perturbed_value)
            if matches:
                violations.append({
                    "perturbation_id": p.get("perturbation_id"),
                    "markers": matches,
                    "value_preview": perturbed_value[:100],
                })

        if len(violations) == 0:
            return self._pass(
                f"No synthetic markers in {len(perturbations)} perturbations",
                value=0,
                threshold=0,
            )
        else:
            rate = len(violations) / len(perturbations)
            return self._fail(
                f"Found {len(violations)} ({rate:.1%}) with synthetic markers",
                value=len(violations),
                threshold=0,
                details={"examples": violations[:5]},
            )


# =============================================================================
# Structural Corruption Gate
# =============================================================================


class GateNoStructuralCorruption(BaseGate):
    """Gate: No structural corruption in perturbations."""

    name = "gate_no_structural_corruption"
    description = "Verify no structural corruption (duplicate markers, etc.)"

    # Patterns indicating structural corruption
    CORRUPTION_PATTERNS = [
        (r"Action:.*Action:", "duplicate_action"),
        (r"Tool:.*Tool:", "duplicate_tool"),
        (r"Thought:.*Thought:", "duplicate_thought"),
        (r"\{.*\{.*\}.*\}.*\{", "nested_json_corruption"),
        (r"Step \d+.*Step \d+.*Step \d+", "step_marker_repetition"),
    ]

    def check(
        self, perturbations: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check for structural corruption."""
        if not perturbations:
            return self._skip("No perturbations to check")

        violations = []
        for p in perturbations:
            content = str(p.get("perturbed_value", ""))

            for pattern, pattern_name in self.CORRUPTION_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                    violations.append({
                        "perturbation_id": p.get("perturbation_id"),
                        "corruption_type": pattern_name,
                    })
                    break  # One violation per perturbation

        if len(violations) == 0:
            return self._pass(
                f"No structural corruption in {len(perturbations)} perturbations",
                value=0,
                threshold=0,
            )
        else:
            return self._fail(
                f"Found {len(violations)} structural corruptions",
                value=len(violations),
                threshold=0,
                details={"examples": violations[:5]},
            )


# =============================================================================
# JSON Validity Gate
# =============================================================================


class GateJSONValidity(BaseGate):
    """Gate: All JSON tool arguments are valid."""

    name = "gate_json_validity"
    description = "Verify tool arguments are valid JSON"

    def check(
        self, perturbations: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check JSON validity."""
        if not perturbations:
            return self._skip("No perturbations to check")

        invalid = []
        checked = 0

        for p in perturbations:
            perturbed = p.get("perturbed_value")
            slot = p.get("target_slot", "")

            # Only check JSON-like values (tool arguments)
            if not isinstance(perturbed, str):
                continue

            perturbed = perturbed.strip()
            if not (perturbed.startswith("{") or perturbed.startswith("[")):
                continue

            checked += 1
            try:
                json.loads(perturbed)
            except json.JSONDecodeError as e:
                invalid.append({
                    "perturbation_id": p.get("perturbation_id"),
                    "slot": slot,
                    "error": str(e)[:50],
                })

        if checked == 0:
            return self._skip("No JSON values to check")

        if len(invalid) == 0:
            return self._pass(
                f"All {checked} JSON values are valid",
                value=checked,
                threshold=checked,
            )
        else:
            rate = len(invalid) / checked
            return self._fail(
                f"{len(invalid)} ({rate:.1%}) invalid JSON values",
                value=len(invalid),
                threshold=0,
                details={"examples": invalid[:5]},
            )


# =============================================================================
# Placebo Semantics Gate
# =============================================================================


class GatePlaceboPreservesSemantics(BaseGate):
    """Gate: Placebo perturbations preserve semantics."""

    name = "gate_placebo_preserves_semantics"
    description = "Verify placebos don't change meaning"

    def check(
        self, perturbations: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check placebo semantic preservation."""
        if not perturbations:
            return self._skip("No perturbations to check")

        # Filter to placebos
        placebos = [
            p for p in perturbations
            if p.get("perturbation_class") == "placebo"
        ]

        if not placebos:
            return self._skip("No placebo perturbations to check")

        violations = []
        for p in placebos:
            original = str(p.get("original_value", ""))
            perturbed = str(p.get("perturbed_value", ""))

            # Check semantic markers preservation
            if not self._semantics_preserved(original, perturbed):
                violations.append({
                    "perturbation_id": p.get("perturbation_id"),
                    "perturbation_type": p.get("perturbation_type"),
                })

        if len(violations) == 0:
            return self._pass(
                f"All {len(placebos)} placebos preserve semantics",
                value=len(placebos),
            )
        else:
            rate = len(violations) / len(placebos)
            return self._warn(
                f"{len(violations)} ({rate:.1%}) placebos may not preserve semantics",
                value=len(violations),
                details={"examples": violations[:5]},
            )

    def _semantics_preserved(self, original: str, perturbed: str) -> bool:
        """Check if semantic markers are preserved."""
        # Extract numbers
        orig_nums = set(re.findall(r'\b\d+\.?\d*\b', original))
        pert_nums = set(re.findall(r'\b\d+\.?\d*\b', perturbed))
        if orig_nums != pert_nums:
            return False

        # Extract file paths
        path_pattern = r'[/\\][\w\-./\\]+'
        orig_paths = set(re.findall(path_pattern, original))
        pert_paths = set(re.findall(path_pattern, perturbed))
        if orig_paths != pert_paths:
            return False

        # Length check (within 50%)
        if len(original) > 0:
            ratio = len(perturbed) / len(original)
            if ratio < 0.5 or ratio > 1.5:
                return False

        return True


# =============================================================================
# Length Preservation Gate
# =============================================================================


class GateLengthPreservation(BaseGate):
    """Gate: Perturbations preserve trajectory length."""

    name = "gate_length_preservation"
    description = "Verify perturbations don't change step count"

    def check(
        self, perturbations: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check length preservation."""
        if not perturbations:
            return self._skip("No perturbations to check")

        violations = []
        for p in perturbations:
            # Check if this is a structural perturbation that removes steps
            pert_type = p.get("perturbation_type", "")

            # These types are expected to change length
            if pert_type in ["skipped_prerequisite", "premature_termination"]:
                continue

            # For non-structural, check length preservation metadata
            orig_len = p.get("original_step_count")
            pert_len = p.get("perturbed_step_count")

            if orig_len is not None and pert_len is not None:
                if orig_len != pert_len:
                    violations.append({
                        "perturbation_id": p.get("perturbation_id"),
                        "original_len": orig_len,
                        "perturbed_len": pert_len,
                    })

        if len(violations) == 0:
            return self._pass(
                "Length preservation verified",
                value=0,
                threshold=0,
            )
        else:
            return self._fail(
                f"Found {len(violations)} length mismatches",
                value=len(violations),
                threshold=0,
                details={"examples": violations[:5]},
            )


# =============================================================================
# Position Distribution Gate
# =============================================================================


class GatePositionDistribution(BaseGate):
    """Gate: Perturbation positions are well-distributed."""

    name = "gate_position_distribution"
    description = "Verify perturbations aren't clustered at end"

    def check(
        self, perturbations: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check position distribution."""
        config = config or {}
        max_last_step_rate = config.get("max_last_step_rate", 0.40)

        if not perturbations:
            return self._skip("No perturbations to check")

        # Count by position
        position_counts = {"early": 0, "middle": 0, "late": 0, "other": 0}
        for p in perturbations:
            pos = p.get("perturbation_position", "other")
            if pos in position_counts:
                position_counts[pos] += 1
            else:
                position_counts["other"] += 1

        total = len(perturbations)
        late_rate = position_counts["late"] / total if total > 0 else 0

        if late_rate <= max_last_step_rate:
            return self._pass(
                f"Late position rate {late_rate:.1%} <= {max_last_step_rate:.0%}",
                value=late_rate,
                threshold=max_last_step_rate,
                details={"distribution": position_counts},
            )
        else:
            return self._fail(
                f"Late position rate {late_rate:.1%} > {max_last_step_rate:.0%}",
                value=late_rate,
                threshold=max_last_step_rate,
                details={"distribution": position_counts},
            )


# =============================================================================
# Non-Placebo Meaningful Gate
# =============================================================================


class GateNonPlaceboMeaningful(BaseGate):
    """Gate: Non-placebo perturbations create meaningful changes."""

    name = "gate_non_placebo_meaningful"
    description = "Verify non-placebo perturbations actually change values"

    def check(
        self, perturbations: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check non-placebo perturbations are meaningful."""
        if not perturbations:
            return self._skip("No perturbations to check")

        # Filter to non-placebos
        non_placebos = [
            p for p in perturbations
            if p.get("perturbation_class") != "placebo"
        ]

        if not non_placebos:
            return self._skip("No non-placebo perturbations to check")

        no_change = []
        for p in non_placebos:
            original = str(p.get("original_value", ""))
            perturbed = str(p.get("perturbed_value", ""))

            # Check for actual change
            if original == perturbed:
                no_change.append({
                    "perturbation_id": p.get("perturbation_id"),
                    "perturbation_type": p.get("perturbation_type"),
                })

        if len(no_change) == 0:
            return self._pass(
                f"All {len(non_placebos)} non-placebos create meaningful changes",
                value=len(non_placebos),
            )
        else:
            rate = len(no_change) / len(non_placebos)
            return self._fail(
                f"{len(no_change)} ({rate:.1%}) non-placebos have no change",
                value=len(no_change),
                threshold=0,
                details={"examples": no_change[:5]},
            )


# =============================================================================
# Gate Registry
# =============================================================================

PERTURBATION_GATES = {
    "gate_no_synthetic_markers": GateNoSyntheticMarkers,
    "gate_no_structural_corruption": GateNoStructuralCorruption,
    "gate_json_validity": GateJSONValidity,
    "gate_placebo_preserves_semantics": GatePlaceboPreservesSemantics,
    "gate_length_preservation": GateLengthPreservation,
    "gate_position_distribution": GatePositionDistribution,
    "gate_non_placebo_meaningful": GateNonPlaceboMeaningful,
}


def get_perturbation_gate(name: str) -> BaseGate:
    """Get a perturbation gate by name."""
    if name not in PERTURBATION_GATES:
        raise KeyError(f"Unknown perturbation gate: {name}")
    return PERTURBATION_GATES[name]()
