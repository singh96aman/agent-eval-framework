"""
Pipeline Quality Gates (21 gates).

Validates data at each pipeline phase using ONLY regex/parsing - NO LLM calls.

Gates are organized by phase:
- Load phase: trajectory_count, grader_pass_rate
- Perturb phase: no_synthetic_markers, json_validity, position_distribution
- Create units phase: blinding_balance, length_preservation
- Compute phase: outcome_variance
"""

import json
import re
import statistics
from typing import Any, Dict, List, Optional

from src.quality_gates.base import BaseGate, GateResult


# =============================================================================
# Load Phase Gates
# =============================================================================


class TrajectoryCountGate(BaseGate):
    """Gate: Minimum trajectory count."""

    name = "trajectory_count"
    description = "Verify minimum number of trajectories loaded"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check trajectory count >= min threshold."""
        config = config or {}
        min_count = config.get("min", 100)

        count = len(data) if data else 0

        if count >= min_count:
            return self._pass(
                f"Trajectory count {count} >= {min_count}",
                value=count,
                threshold=min_count,
            )
        else:
            return self._fail(
                f"Trajectory count {count} < {min_count}",
                value=count,
                threshold=min_count,
            )


class GraderPassRateGate(BaseGate):
    """Gate: Grader pass rate above threshold."""

    name = "grader_pass_rate"
    description = "Verify grader pass rate is above minimum"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check grader pass rate >= min threshold."""
        config = config or {}
        min_rate = config.get("min", 0.30)

        if not data:
            return self._fail("No data to check", value=0.0, threshold=min_rate)

        # Count passed trajectories
        # Check grader_passed field first, fall back to task_success
        passed = 0
        for t in data:
            if t.get("grader_passed") is True:
                passed += 1
            elif t.get("grader_passed") is None:
                # Fall back to task_success if grader_passed not set
                # This handles pre-filtered data where all are assumed passed
                gt = t.get("ground_truth", {})
                if isinstance(gt, dict) and gt.get("task_success") is True:
                    passed += 1
        rate = passed / len(data)

        if rate >= min_rate:
            return self._pass(
                f"Grader pass rate {rate:.2%} >= {min_rate:.0%}",
                value=rate,
                threshold=min_rate,
            )
        else:
            return self._fail(
                f"Grader pass rate {rate:.2%} < {min_rate:.0%}",
                value=rate,
                threshold=min_rate,
            )


# =============================================================================
# Perturb Phase Gates
# =============================================================================

# Artifact markers to detect
ARTIFACT_MARKERS = [
    r"_old\b",
    r"_mutated\b",
    r"_wrong\b",
    r"_backup\b",
    r"_test\b",
    r"_v1\b",
]

ARTIFACT_PATTERN = re.compile("|".join(ARTIFACT_MARKERS), re.IGNORECASE)


class NoSyntheticMarkersGate(BaseGate):
    """Gate: No synthetic artifact markers in perturbations."""

    name = "no_synthetic_markers"
    description = "Verify no artifact markers (_old, _mutated, etc.)"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check for artifact markers in perturbed values."""
        config = config or {}
        max_rate = config.get("max_rate", 0.0)

        if not data:
            return self._skip("No perturbations to check")

        violations = []
        for p in data:
            perturbed_value = str(p.get("perturbed_value", ""))
            if ARTIFACT_PATTERN.search(perturbed_value):
                violations.append({
                    "perturbation_id": p.get("perturbation_id"),
                    "value": perturbed_value[:100],
                })

        rate = len(violations) / len(data)

        if rate <= max_rate:
            return self._pass(
                f"Artifact rate {rate:.2%} <= {max_rate:.0%}",
                value=rate,
                threshold=max_rate,
                details={"violation_count": len(violations)},
            )
        else:
            return self._fail(
                f"Artifact rate {rate:.2%} > {max_rate:.0%}",
                value=rate,
                threshold=max_rate,
                details={
                    "violation_count": len(violations),
                    "examples": violations[:5],
                },
            )


class NoStructuralCorruptionGate(BaseGate):
    """Gate: No structural corruption in perturbations."""

    name = "no_structural_corruption"
    description = "Verify no structural corruption (e.g., Action:.*Action:)"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check for structural corruption patterns."""
        if not data:
            return self._skip("No perturbations to check")

        # Pattern: duplicate Action: markers
        corruption_pattern = re.compile(r"Action:.*Action:", re.IGNORECASE)

        violations = []
        for p in data:
            content = str(p.get("perturbed_value", ""))
            if corruption_pattern.search(content):
                violations.append({
                    "perturbation_id": p.get("perturbation_id"),
                    "pattern": "duplicate_action",
                })

        if len(violations) == 0:
            return self._pass(
                "No structural corruption detected",
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


class JSONValidityGate(BaseGate):
    """Gate: All JSON tool arguments are valid."""

    name = "json_validity"
    description = "Verify all tool arguments are valid JSON"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check JSON validity in perturbed values."""
        config = config or {}
        min_rate = config.get("min_rate", 1.0)

        if not data:
            return self._skip("No perturbations to check")

        valid_count = 0
        invalid_examples = []

        for p in data:
            perturbed = p.get("perturbed_value")
            # Only check if it looks like JSON
            if isinstance(perturbed, str) and (
                perturbed.strip().startswith("{") or
                perturbed.strip().startswith("[")
            ):
                try:
                    json.loads(perturbed)
                    valid_count += 1
                except json.JSONDecodeError:
                    invalid_examples.append({
                        "perturbation_id": p.get("perturbation_id"),
                        "value": perturbed[:100],
                    })
            else:
                # Non-JSON values are valid
                valid_count += 1

        rate = valid_count / len(data)

        if rate >= min_rate:
            return self._pass(
                f"JSON validity rate {rate:.2%} >= {min_rate:.0%}",
                value=rate,
                threshold=min_rate,
            )
        else:
            return self._fail(
                f"JSON validity rate {rate:.2%} < {min_rate:.0%}",
                value=rate,
                threshold=min_rate,
                details={"invalid_examples": invalid_examples[:5]},
            )


class PositionDistributionGate(BaseGate):
    """Gate: Perturbation position distribution is reasonable."""

    name = "position_distribution"
    description = "Verify perturbations are not clustered at last step"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check position distribution."""
        config = config or {}
        max_last_step_rate = config.get("max_last_step_rate", 0.40)

        if not data:
            return self._skip("No perturbations to check")

        # Count perturbations at last step
        last_step_count = 0
        for p in data:
            position = p.get("perturbation_position", "")
            if position == "late" or position == "last":
                last_step_count += 1

        rate = last_step_count / len(data)

        if rate <= max_last_step_rate:
            return self._pass(
                f"Last-step rate {rate:.2%} <= {max_last_step_rate:.0%}",
                value=rate,
                threshold=max_last_step_rate,
            )
        else:
            return self._fail(
                f"Last-step rate {rate:.2%} > {max_last_step_rate:.0%}",
                value=rate,
                threshold=max_last_step_rate,
            )


class ClassDistributionGate(BaseGate):
    """Gate: Perturbation class distribution matches target."""

    name = "class_distribution"
    description = "Verify class distribution (placebo/fine/coarse)"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check class distribution matches target."""
        config = config or {}
        tolerance = config.get("tolerance", 0.05)
        target = config.get("target", {
            "placebo": 0.20,
            "fine_grained": 0.50,
            "coarse_grained": 0.30,
        })

        if not data:
            return self._skip("No perturbations to check")

        # Count by class
        counts = {"placebo": 0, "fine_grained": 0, "coarse_grained": 0}
        for p in data:
            pclass = p.get("perturbation_class", "unknown")
            if pclass in counts:
                counts[pclass] += 1

        total = sum(counts.values())
        if total == 0:
            return self._fail("No perturbations with known class")

        # Check each class
        violations = []
        for cls, tgt in target.items():
            actual = counts.get(cls, 0) / total
            if abs(actual - tgt) > tolerance:
                violations.append({
                    "class": cls,
                    "target": tgt,
                    "actual": actual,
                    "diff": abs(actual - tgt),
                })

        if len(violations) == 0:
            return self._pass(
                "Class distribution within tolerance",
                value=counts,
                threshold=target,
            )
        else:
            return self._fail(
                f"{len(violations)} classes outside tolerance",
                value=counts,
                threshold=target,
                details={"violations": violations},
            )


class PerturbationClassValidityGate(BaseGate):
    """Gate: Perturbation class validation passes minimum rate."""

    name = "perturbation_class_validity"
    description = "Verify perturbations match their claimed class (via LLM validation)"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """
        Check class_matches rate >= min_rate.

        Expects perturbations with class_validation.class_matches field (0 or 1).

        Args:
            data: List of perturbation records with class_validation field
            config: Optional config with min_rate (default 0.90)

        Returns:
            GateResult indicating pass/fail
        """
        config = config or {}
        min_rate = config.get("min_rate", 0.90)

        if not data:
            return self._skip("No perturbations to check")

        # Count perturbations with class_matches = 1
        total_validated = 0
        matches = 0
        mismatches = []

        for p in data:
            class_validation = p.get("class_validation", {})
            if not class_validation:
                continue

            total_validated += 1
            class_matches = class_validation.get("class_matches", 1)

            if class_matches == 1:
                matches += 1
            else:
                mismatches.append({
                    "perturbation_id": p.get("perturbation_id"),
                    "perturbation_class": p.get("perturbation_class"),
                    "perturbation_type": p.get("perturbation_type"),
                    "reasoning": class_validation.get("reasoning", "")[:100],
                })

        if total_validated == 0:
            return self._skip("No perturbations have class_validation data")

        rate = matches / total_validated

        if rate >= min_rate:
            return self._pass(
                f"Class match rate {rate:.2%} >= {min_rate:.0%}",
                value=rate,
                threshold=min_rate,
                details={
                    "total_validated": total_validated,
                    "matches": matches,
                    "mismatches": len(mismatches),
                },
            )
        else:
            return self._fail(
                f"Class match rate {rate:.2%} < {min_rate:.0%}",
                value=rate,
                threshold=min_rate,
                details={
                    "total_validated": total_validated,
                    "matches": matches,
                    "mismatches": len(mismatches),
                    "mismatch_examples": mismatches[:5],
                },
            )


# =============================================================================
# Create Units Phase Gates
# =============================================================================


class BlindingBalanceGate(BaseGate):
    """Gate: A/B blinding balance is within range."""

    name = "blinding_balance"
    description = "Verify A/B randomization is 45-55% balanced"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check blinding balance."""
        config = config or {}
        min_ratio = config.get("min", 0.45)
        max_ratio = config.get("max", 0.55)

        if not data:
            return self._skip("No evaluation units to check")

        # Count A=baseline assignments
        # Handle both flat and nested blinding structures
        a_is_baseline = 0
        for u in data:
            # Check nested blinding dict first
            blinding = u.get("blinding", {})
            if isinstance(blinding, dict) and blinding.get("is_a_baseline"):
                a_is_baseline += 1
            # Fallback to flat structure
            elif u.get("is_a_baseline", False):
                a_is_baseline += 1
        ratio = a_is_baseline / len(data)

        if min_ratio <= ratio <= max_ratio:
            return self._pass(
                f"Balance ratio {ratio:.2%} in [{min_ratio:.0%}, {max_ratio:.0%}]",
                value=ratio,
                threshold=(min_ratio, max_ratio),
            )
        else:
            return self._fail(
                f"Balance ratio {ratio:.2%} outside [{min_ratio:.0%}, {max_ratio:.0%}]",
                value=ratio,
                threshold=(min_ratio, max_ratio),
            )


class LengthPreservationGate(BaseGate):
    """Gate: Baseline and perturbed trajectories have same length."""

    name = "length_preservation"
    description = "Verify trajectory lengths are preserved"

    # Perturbation types that intentionally change trajectory length
    LENGTH_CHANGING_TYPES = {"skipped_prerequisite"}

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check length preservation."""
        if not data:
            return self._skip("No evaluation units to check")

        mismatches = []
        skipped_count = 0
        for u in data:
            baseline = u.get("baseline", {})
            perturbed = u.get("perturbed", {})

            # Get perturbation type
            pert_record = perturbed.get("perturbation_record", {})
            pert_type = pert_record.get("perturbation_type", "")

            # Skip length-changing perturbation types
            if pert_type in self.LENGTH_CHANGING_TYPES:
                skipped_count += 1
                continue

            b_steps = len(baseline.get("trajectory", {}).get("steps", []))
            p_steps = len(perturbed.get("trajectory", {}).get("steps", []))

            if b_steps != p_steps:
                mismatches.append({
                    "unit_id": u.get("evaluation_unit_id"),
                    "baseline_len": b_steps,
                    "perturbed_len": p_steps,
                    "perturbation_type": pert_type,
                })

        if len(mismatches) == 0:
            msg = "All trajectory lengths preserved"
            if skipped_count > 0:
                msg += f" ({skipped_count} length-changing types skipped)"
            return self._pass(msg, value=0, threshold=0)
        else:
            return self._fail(
                f"Found {len(mismatches)} length mismatches",
                value=len(mismatches),
                threshold=0,
                details={"mismatches": mismatches[:5]},
            )


# =============================================================================
# Compute Phase Gates
# =============================================================================


class OutcomeVarianceGate(BaseGate):
    """Gate: Outcome degradation has sufficient variance."""

    name = "outcome_variance"
    description = "Verify outcome degradation has variance for CCorr"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check outcome variance."""
        config = config or {}
        min_std = config.get("min_std", 0.1)

        if not data:
            return self._skip("No outcome records to check")

        # Extract outcome degradation values
        od_values = []
        for d in data:
            od = d.get("outcome_degradation")
            if od is not None:
                od_values.append(od)

        if len(od_values) < 2:
            return self._fail(
                "Insufficient outcome data for variance calculation",
                value=len(od_values),
            )

        std = statistics.stdev(od_values)

        if std >= min_std:
            return self._pass(
                f"Outcome std {std:.3f} >= {min_std}",
                value=std,
                threshold=min_std,
            )
        else:
            return self._fail(
                f"Outcome std {std:.3f} < {min_std} (CCorr may be meaningless)",
                value=std,
                threshold=min_std,
            )


# =============================================================================
# Additional Gates
# =============================================================================


class MinStepsGate(BaseGate):
    """Gate: Trajectories have minimum steps."""

    name = "min_steps"
    description = "Verify trajectories have minimum step count"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check minimum steps."""
        config = config or {}
        min_steps = config.get("min", 3)

        if not data:
            return self._skip("No data to check")

        violations = []
        for t in data:
            steps = t.get("steps", [])
            if len(steps) < min_steps:
                violations.append({
                    "trajectory_id": t.get("trajectory_id"),
                    "step_count": len(steps),
                })

        if len(violations) == 0:
            return self._pass(
                f"All trajectories have >= {min_steps} steps",
                value=len(data),
                threshold=min_steps,
            )
        else:
            return self._fail(
                f"{len(violations)} trajectories have < {min_steps} steps",
                value=len(violations),
                threshold=0,
                details={"examples": violations[:5]},
            )


class MaxStepsGate(BaseGate):
    """Gate: Trajectories have maximum steps."""

    name = "max_steps"
    description = "Verify trajectories have maximum step count"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check maximum steps."""
        config = config or {}
        max_steps = config.get("max", 50)

        if not data:
            return self._skip("No data to check")

        violations = []
        for t in data:
            steps = t.get("steps", [])
            if len(steps) > max_steps:
                violations.append({
                    "trajectory_id": t.get("trajectory_id"),
                    "step_count": len(steps),
                })

        if len(violations) == 0:
            return self._pass(
                f"All trajectories have <= {max_steps} steps",
                value=len(data),
                threshold=max_steps,
            )
        else:
            return self._fail(
                f"{len(violations)} trajectories have > {max_steps} steps",
                value=len(violations),
                threshold=0,
                details={"examples": violations[:5]},
            )


class TaskSuccessGate(BaseGate):
    """Gate: Task success rate above threshold."""

    name = "task_success"
    description = "Verify task success rate"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check task success rate."""
        config = config or {}
        min_rate = config.get("min_rate", 0.0)

        if not data:
            return self._skip("No data to check")

        success_count = sum(
            1 for t in data
            if t.get("ground_truth", {}).get("task_success", False)
        )
        rate = success_count / len(data)

        if rate >= min_rate:
            return self._pass(
                f"Task success rate {rate:.2%} >= {min_rate:.0%}",
                value=rate,
                threshold=min_rate,
            )
        else:
            return self._fail(
                f"Task success rate {rate:.2%} < {min_rate:.0%}",
                value=rate,
                threshold=min_rate,
            )


class UniqueIDsGate(BaseGate):
    """Gate: All IDs are unique."""

    name = "unique_ids"
    description = "Verify all trajectory/perturbation IDs are unique"

    def check(
        self, data: List[Dict], config: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """Check ID uniqueness."""
        config = config or {}
        id_field = config.get("id_field", "trajectory_id")

        if not data:
            return self._skip("No data to check")

        ids = [d.get(id_field) for d in data if d.get(id_field)]
        unique_ids = set(ids)

        if len(ids) == len(unique_ids):
            return self._pass(
                f"All {len(ids)} IDs are unique",
                value=len(ids),
            )
        else:
            duplicates = len(ids) - len(unique_ids)
            return self._fail(
                f"Found {duplicates} duplicate IDs",
                value=duplicates,
                threshold=0,
            )


# =============================================================================
# Gate Registry
# =============================================================================

PIPELINE_GATES = {
    # Load phase
    "trajectory_count": TrajectoryCountGate,
    "grader_pass_rate": GraderPassRateGate,
    "task_success": TaskSuccessGate,
    "min_steps": MinStepsGate,
    "max_steps": MaxStepsGate,
    "unique_ids": UniqueIDsGate,
    # Perturb phase
    "no_synthetic_markers": NoSyntheticMarkersGate,
    "no_structural_corruption": NoStructuralCorruptionGate,
    "json_validity": JSONValidityGate,
    "position_distribution": PositionDistributionGate,
    "class_distribution": ClassDistributionGate,
    "perturbation_class_validity": PerturbationClassValidityGate,
    # Create units phase
    "blinding_balance": BlindingBalanceGate,
    "length_preservation": LengthPreservationGate,
    # Compute phase
    "outcome_variance": OutcomeVarianceGate,
}


def get_gate(name: str) -> BaseGate:
    """Get a pipeline gate by name."""
    if name not in PIPELINE_GATES:
        raise KeyError(f"Unknown gate: {name}")
    return PIPELINE_GATES[name]()
