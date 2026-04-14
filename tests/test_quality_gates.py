"""
Tests for quality gates.

Tests all pipeline, prompt, and perturbation gates with
both PASS and FAIL scenarios.
"""

import pytest
from src.quality_gates import (
    GateRunner,
    GateStatus,
    get_gate,
    get_prompt_gate,
    get_perturbation_gate,
    PIPELINE_GATES,
    PROMPT_GATES,
    PERTURBATION_GATES,
)


class TestGateRunner:
    """Tests for GateRunner."""

    def test_runner_creates_report(self):
        """Test runner creates report with results."""
        runner = GateRunner(phase="test")
        runner.add_gate(get_gate("trajectory_count"))
        report = runner.run([{"id": 1}, {"id": 2}], {"min": 1})

        assert report.phase == "test"
        assert len(report.results) == 1
        assert report.all_passed

    def test_runner_handles_crash(self):
        """Test runner handles gate exceptions gracefully."""
        runner = GateRunner(phase="test")
        runner.add_gate(get_gate("trajectory_count"))

        # Pass None which will cause issues
        report = runner.run(None)

        # Should not crash, but return fail
        assert len(report.results) == 1


# =============================================================================
# Pipeline Gates Tests
# =============================================================================


class TestTrajectoryCountGate:
    """Tests for trajectory count gate."""

    def test_pass_when_above_threshold(self):
        """Test passes when count >= threshold."""
        gate = get_gate("trajectory_count")
        data = [{"id": i} for i in range(100)]
        result = gate.check(data, {"min": 100})

        assert result.status == GateStatus.PASS
        assert result.value == 100

    def test_fail_when_below_threshold(self):
        """Test fails when count < threshold."""
        gate = get_gate("trajectory_count")
        data = [{"id": i} for i in range(50)]
        result = gate.check(data, {"min": 100})

        assert result.status == GateStatus.FAIL
        assert result.value == 50


class TestNoSyntheticMarkersGate:
    """Tests for synthetic markers gate."""

    def test_pass_with_clean_data(self):
        """Test passes with no artifact markers."""
        gate = get_gate("no_synthetic_markers")
        data = [
            {"perturbation_id": "1", "perturbed_value": "clean_value"},
            {"perturbation_id": "2", "perturbed_value": "another_clean"},
        ]
        result = gate.check(data)

        assert result.status == GateStatus.PASS

    def test_fail_with_old_marker(self):
        """Test fails with _old marker."""
        gate = get_gate("no_synthetic_markers")
        data = [
            {"perturbation_id": "1", "perturbed_value": "value_old"},
        ]
        result = gate.check(data)

        assert result.status == GateStatus.FAIL

    def test_fail_with_mutated_marker(self):
        """Test fails with _mutated marker."""
        gate = get_gate("no_synthetic_markers")
        data = [
            {"perturbation_id": "1", "perturbed_value": "value_mutated"},
        ]
        result = gate.check(data)

        assert result.status == GateStatus.FAIL

    def test_fail_with_wrong_marker(self):
        """Test fails with _wrong marker."""
        gate = get_gate("no_synthetic_markers")
        data = [
            {"perturbation_id": "1", "perturbed_value": "date_wrong"},
        ]
        result = gate.check(data)

        assert result.status == GateStatus.FAIL


class TestJSONValidityGate:
    """Tests for JSON validity gate."""

    def test_pass_with_valid_json(self):
        """Test passes with valid JSON."""
        gate = get_gate("json_validity")
        data = [
            {"perturbation_id": "1", "perturbed_value": '{"key": "value"}'},
            {"perturbation_id": "2", "perturbed_value": '["a", "b"]'},
        ]
        result = gate.check(data)

        assert result.status == GateStatus.PASS

    def test_fail_with_invalid_json(self):
        """Test fails with invalid JSON."""
        gate = get_gate("json_validity")
        data = [
            {"perturbation_id": "1", "perturbed_value": '{"key": value}'},
        ]
        result = gate.check(data, {"min_rate": 1.0})

        assert result.status == GateStatus.FAIL

    def test_pass_with_non_json(self):
        """Test passes non-JSON values (they're valid by default)."""
        gate = get_gate("json_validity")
        data = [
            {"perturbation_id": "1", "perturbed_value": "plain string"},
        ]
        result = gate.check(data)

        # Non-JSON values count as valid, so this should pass
        assert result.status == GateStatus.PASS


class TestBlindingBalanceGate:
    """Tests for blinding balance gate."""

    def test_pass_with_balanced_data(self):
        """Test passes with 50% balance."""
        gate = get_gate("blinding_balance")
        data = [
            {"is_a_baseline": True},
            {"is_a_baseline": False},
            {"is_a_baseline": True},
            {"is_a_baseline": False},
        ]
        result = gate.check(data)

        assert result.status == GateStatus.PASS
        assert result.value == 0.5

    def test_fail_with_imbalanced_data(self):
        """Test fails with imbalanced data."""
        gate = get_gate("blinding_balance")
        data = [
            {"is_a_baseline": True},
            {"is_a_baseline": True},
            {"is_a_baseline": True},
            {"is_a_baseline": False},
        ]
        result = gate.check(data)

        assert result.status == GateStatus.FAIL
        assert result.value == 0.75


class TestOutcomeVarianceGate:
    """Tests for outcome variance gate."""

    def test_pass_with_variance(self):
        """Test passes with sufficient variance."""
        gate = get_gate("outcome_variance")
        data = [
            {"outcome_degradation": 0.0},
            {"outcome_degradation": 0.5},
            {"outcome_degradation": 1.0},
        ]
        result = gate.check(data, {"min_std": 0.1})

        assert result.status == GateStatus.PASS

    def test_fail_with_no_variance(self):
        """Test fails with no variance (all same)."""
        gate = get_gate("outcome_variance")
        data = [
            {"outcome_degradation": 0.5},
            {"outcome_degradation": 0.5},
            {"outcome_degradation": 0.5},
        ]
        result = gate.check(data, {"min_std": 0.1})

        assert result.status == GateStatus.FAIL


# =============================================================================
# Prompt Gates Tests
# =============================================================================


class TestPrimingDetectionGate:
    """Tests for priming detection gate."""

    def test_pass_with_neutral_prompt(self):
        """Test passes with neutral prompt."""
        gate = get_prompt_gate("priming_detection")
        prompt = "Compare these two trajectories and analyze their quality."
        result = gate.check(prompt)

        assert result.status == GateStatus.PASS

    def test_fail_with_priming_prompt(self):
        """Test fails with priming language."""
        gate = get_prompt_gate("priming_detection")
        prompt = "One of these trajectories contains an error. Find the error."
        result = gate.check(prompt)

        assert result.status == GateStatus.FAIL


class TestBlindingIntegrityGate:
    """Tests for blinding integrity gate."""

    def test_pass_with_blinded_prompt(self):
        """Test passes with properly blinded prompt."""
        gate = get_prompt_gate("blinding_integrity")
        prompt = "Compare Trajectory A and Trajectory B."
        result = gate.check(prompt)

        assert result.status == GateStatus.PASS

    def test_fail_with_baseline_mention(self):
        """Test fails when baseline is mentioned."""
        gate = get_prompt_gate("blinding_integrity")
        prompt = "Compare the baseline trajectory with the perturbed one."
        result = gate.check(prompt)

        assert result.status == GateStatus.FAIL


class TestNeutralityGate:
    """Tests for neutrality gate."""

    def test_pass_with_neutral_framing(self):
        """Test passes with equal A/B framing."""
        gate = get_prompt_gate("neutrality")
        prompt = """
        Trajectory A:
        [steps]

        Trajectory B:
        [steps]
        """
        result = gate.check(prompt)

        assert result.status == GateStatus.PASS


# =============================================================================
# Perturbation Gates Tests
# =============================================================================


class TestGateNoSyntheticMarkers:
    """Tests for detailed synthetic markers gate."""

    def test_pass_clean(self):
        """Test passes with clean perturbations."""
        gate = get_perturbation_gate("gate_no_synthetic_markers")
        data = [
            {"perturbation_id": "1", "perturbed_value": "clean"},
            {"perturbation_id": "2", "perturbed_value": "also clean"},
        ]
        result = gate.check(data)

        assert result.status == GateStatus.PASS

    def test_fail_with_any_marker(self):
        """Test fails with any artifact marker."""
        gate = get_perturbation_gate("gate_no_synthetic_markers")
        markers = ["_old", "_mutated", "_wrong", "_backup", "_test", "_v1"]

        for marker in markers:
            data = [{"perturbation_id": "1", "perturbed_value": f"value{marker}"}]
            result = gate.check(data)
            assert result.status == GateStatus.FAIL, f"Should fail for {marker}"


class TestGatePlaceboPreservesSemantics:
    """Tests for placebo semantics gate."""

    def test_pass_with_preserved_semantics(self):
        """Test passes when semantics preserved."""
        gate = get_perturbation_gate("gate_placebo_preserves_semantics")
        data = [
            {
                "perturbation_id": "1",
                "perturbation_class": "placebo",
                "original_value": "Search for item 123 in /data/file.txt",
                "perturbed_value": "Look for item 123 in /data/file.txt",
            }
        ]
        result = gate.check(data)

        assert result.status == GateStatus.PASS

    def test_warn_with_changed_numbers(self):
        """Test warns when numbers change."""
        gate = get_perturbation_gate("gate_placebo_preserves_semantics")
        data = [
            {
                "perturbation_id": "1",
                "perturbation_class": "placebo",
                "original_value": "Item 123",
                "perturbed_value": "Item 456",
            }
        ]
        result = gate.check(data)

        assert result.status == GateStatus.WARN


class TestGateNonPlaceboMeaningful:
    """Tests for non-placebo meaningful gate."""

    def test_pass_with_meaningful_change(self):
        """Test passes when non-placebo creates change."""
        gate = get_perturbation_gate("gate_non_placebo_meaningful")
        data = [
            {
                "perturbation_id": "1",
                "perturbation_class": "fine_grained",
                "original_value": "original",
                "perturbed_value": "changed",
            }
        ]
        result = gate.check(data)

        assert result.status == GateStatus.PASS

    def test_fail_with_no_change(self):
        """Test fails when non-placebo has no change."""
        gate = get_perturbation_gate("gate_non_placebo_meaningful")
        data = [
            {
                "perturbation_id": "1",
                "perturbation_class": "fine_grained",
                "original_value": "same",
                "perturbed_value": "same",
            }
        ]
        result = gate.check(data)

        assert result.status == GateStatus.FAIL


# =============================================================================
# Registry Tests
# =============================================================================


class TestGateRegistries:
    """Tests for gate registries."""

    def test_all_pipeline_gates_instantiate(self):
        """Test all pipeline gates can be instantiated."""
        for name in PIPELINE_GATES:
            gate = get_gate(name)
            assert gate is not None
            assert hasattr(gate, "check")

    def test_all_prompt_gates_instantiate(self):
        """Test all prompt gates can be instantiated."""
        for name in PROMPT_GATES:
            gate = get_prompt_gate(name)
            assert gate is not None
            assert hasattr(gate, "check")

    def test_all_perturbation_gates_instantiate(self):
        """Test all perturbation gates can be instantiated."""
        for name in PERTURBATION_GATES:
            gate = get_perturbation_gate(name)
            assert gate is not None
            assert hasattr(gate, "check")

    def test_unknown_gate_raises(self):
        """Test unknown gate name raises KeyError."""
        with pytest.raises(KeyError):
            get_gate("nonexistent_gate")
