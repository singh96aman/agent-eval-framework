"""
Tests for perturbation class validator.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.scoring.class_validator import (
    PerturbationClassValidator,
    ClassValidationResult,
    create_validator_from_config,
)
from src.quality_gates.pipeline_gates import PerturbationClassValidityGate


class TestPerturbationClassValidator:
    """Tests for PerturbationClassValidator."""

    @pytest.fixture
    def model_config(self):
        """Sample model config."""
        return {
            "provider": "bedrock",
            "model": "test-model-id",
            "max_tokens": 500,
            "temperature": 0.0,
        }

    @pytest.fixture
    def prompt_template(self):
        """Sample prompt template."""
        return (
            "Validate perturbation class.\n"
            "Class: {{perturbation_class}}\n"
            "Type: {{perturbation_type}}\n"
            "Step: {{step_index}}\n"
            "Original: {{original_value}}\n"
            "Perturbed: {{perturbed_value}}\n"
            "Method: {{mutation_method}}\n"
            "Respond with JSON: {\"class_matches\": 0 or 1, \"reasoning\": \"...\"}"
        )

    @pytest.fixture
    def validator(self, model_config, prompt_template):
        """Create validator instance."""
        return PerturbationClassValidator(
            model_config=model_config,
            prompt_template=prompt_template,
            log_calls=False,
        )

    def test_build_prompt_substitutes_variables(self, validator):
        """Test that _build_prompt substitutes all template variables."""
        perturbation = {
            "perturbation_class": "fine_grained",
            "perturbation_type": "wrong_value",
            "target_step_index": 3,
            "original_value": "original text",
            "perturbed_value": "perturbed text",
            "mutation_method": "llm_generated",
        }

        prompt = validator._build_prompt(perturbation)

        assert "fine_grained" in prompt
        assert "wrong_value" in prompt
        assert "3" in prompt
        assert "original text" in prompt
        assert "perturbed text" in prompt
        assert "llm_generated" in prompt
        assert "{{" not in prompt  # No unsubstituted variables

    def test_build_prompt_truncates_long_values(self, validator):
        """Test that long values are truncated to 500 chars."""
        long_value = "x" * 1000
        perturbation = {
            "perturbation_class": "placebo",
            "perturbation_type": "formatting",
            "target_step_index": 0,
            "original_value": long_value,
            "perturbed_value": long_value,
            "mutation_method": "template",
        }

        prompt = validator._build_prompt(perturbation)

        # Should have truncated value (500 chars)
        assert "x" * 500 in prompt
        assert "x" * 501 not in prompt

    def test_build_prompt_handles_missing_fields(self, validator):
        """Test that missing fields get default values."""
        perturbation = {}  # Empty

        prompt = validator._build_prompt(perturbation)

        assert "unknown" in prompt  # Default for missing class/type/method
        assert "0" in prompt  # Default for step_index

    def test_parse_json_response_direct(self, validator):
        """Test parsing direct JSON response."""
        response = '{"class_matches": 1, "reasoning": "looks good"}'
        parsed, success = validator._parse_json_response(response)

        assert success
        assert parsed["class_matches"] == 1
        assert parsed["reasoning"] == "looks good"

    def test_parse_json_response_markdown_block(self, validator):
        """Test parsing JSON from markdown code block."""
        response = '''Here's the result:
```json
{"class_matches": 0, "reasoning": "mismatch detected"}
```
'''
        parsed, success = validator._parse_json_response(response)

        assert success
        assert parsed["class_matches"] == 0

    def test_parse_json_response_embedded(self, validator):
        """Test extracting JSON from response with surrounding text."""
        response = 'Analysis complete. {"class_matches": 1, "reasoning": "valid"} End.'
        parsed, success = validator._parse_json_response(response)

        assert success
        assert parsed["class_matches"] == 1

    def test_parse_json_response_invalid(self, validator):
        """Test handling of invalid JSON."""
        response = "This is not JSON at all"
        parsed, success = validator._parse_json_response(response)

        assert not success
        assert parsed is None

    @patch.object(PerturbationClassValidator, "_call_llm")
    def test_validate_perturbation_success(self, mock_call_llm, validator):
        """Test successful validation."""
        mock_call_llm.return_value = '{"class_matches": 1, "reasoning": "correct class"}'

        perturbation = {
            "perturbation_class": "fine_grained",
            "perturbation_type": "wrong_value",
            "target_step_index": 2,
            "original_value": "value1",
            "perturbed_value": "value2",
            "mutation_method": "llm",
        }

        result = validator.validate_perturbation(perturbation)

        assert isinstance(result, ClassValidationResult)
        assert result.class_matches == 1
        assert result.reasoning == "correct class"
        assert result.parse_success

    @patch.object(PerturbationClassValidator, "_call_llm")
    def test_validate_perturbation_mismatch(self, mock_call_llm, validator):
        """Test validation detecting class mismatch."""
        mock_call_llm.return_value = '{"class_matches": 0, "reasoning": "too severe for placebo"}'

        perturbation = {
            "perturbation_class": "placebo",
            "perturbation_type": "formatting",
            "target_step_index": 1,
            "original_value": "hello",
            "perturbed_value": "completely different",
            "mutation_method": "template",
        }

        result = validator.validate_perturbation(perturbation)

        assert result.class_matches == 0
        assert "severe" in result.reasoning

    @patch.object(PerturbationClassValidator, "_call_llm")
    def test_validate_perturbation_normalizes_boolean(self, mock_call_llm, validator):
        """Test that boolean class_matches is normalized to int."""
        mock_call_llm.return_value = '{"class_matches": true, "reasoning": "ok"}'

        result = validator.validate_perturbation({"perturbation_class": "placebo"})

        assert result.class_matches == 1
        assert isinstance(result.class_matches, int)

    @patch.object(PerturbationClassValidator, "_call_llm")
    def test_validate_perturbation_parse_failure_returns_conservative(
        self, mock_call_llm, validator
    ):
        """Test that parse failures return class_matches=1 (conservative)."""
        mock_call_llm.return_value = "Invalid response"

        result = validator.validate_perturbation({"perturbation_class": "fine_grained"})

        assert result.class_matches == 1  # Conservative default
        assert not result.parse_success


class TestCreateValidatorFromConfig:
    """Tests for create_validator_from_config."""

    def test_returns_none_when_disabled(self):
        """Test that None is returned when class_validation is disabled."""
        config = {
            "phases": {
                "perturb": {
                    "class_validation": {
                        "enabled": False,
                    }
                }
            }
        }

        validator = create_validator_from_config(config)
        assert validator is None

    def test_returns_none_when_missing(self):
        """Test that None is returned when class_validation config is missing."""
        config = {"phases": {"perturb": {}}}

        validator = create_validator_from_config(config)
        assert validator is None

    def test_returns_none_when_no_prompt(self):
        """Test that None is returned when no prompt template is provided."""
        config = {
            "phases": {
                "perturb": {
                    "class_validation": {
                        "enabled": True,
                        "model": {"model": "test"},
                        "prompt": "",  # Empty prompt
                    }
                }
            }
        }

        validator = create_validator_from_config(config)
        assert validator is None

    def test_creates_validator_when_enabled(self):
        """Test that validator is created when properly configured."""
        config = {
            "phases": {
                "perturb": {
                    "class_validation": {
                        "enabled": True,
                        "model": {
                            "provider": "bedrock",
                            "model": "test-model",
                            "max_tokens": 500,
                            "temperature": 0.0,
                        },
                        "prompt": "Test prompt {{perturbation_class}}",
                    }
                }
            }
        }

        validator = create_validator_from_config(config)

        assert validator is not None
        assert isinstance(validator, PerturbationClassValidator)
        assert validator.model_id == "test-model"
        assert "{{perturbation_class}}" in validator.prompt_template


class TestPerturbationClassValidityGate:
    """Tests for PerturbationClassValidityGate quality gate."""

    @pytest.fixture
    def gate(self):
        """Create gate instance."""
        return PerturbationClassValidityGate()

    def test_skip_when_no_data(self, gate):
        """Test that gate skips when no data provided."""
        result = gate.check([])
        assert result.status.value == "skip"

    def test_skip_when_no_validation_data(self, gate):
        """Test that gate skips when no class_validation data exists."""
        data = [
            {"perturbation_id": "1", "perturbation_class": "placebo"},
            {"perturbation_id": "2", "perturbation_class": "fine_grained"},
        ]
        result = gate.check(data)
        assert result.status.value == "skip"

    def test_pass_when_all_match(self, gate):
        """Test that gate passes when all perturbations match their class."""
        data = [
            {"perturbation_id": "1", "class_validation": {"class_matches": 1}},
            {"perturbation_id": "2", "class_validation": {"class_matches": 1}},
            {"perturbation_id": "3", "class_validation": {"class_matches": 1}},
        ]
        result = gate.check(data)
        assert result.status.value == "pass"
        assert result.value == 1.0

    def test_pass_when_above_threshold(self, gate):
        """Test that gate passes when match rate >= 90%."""
        data = [
            {"perturbation_id": f"{i}", "class_validation": {"class_matches": 1}}
            for i in range(9)
        ]
        data.append({"perturbation_id": "10", "class_validation": {"class_matches": 0}})

        result = gate.check(data)
        assert result.status.value == "pass"
        assert result.value == 0.9

    def test_fail_when_below_threshold(self, gate):
        """Test that gate fails when match rate < 90%."""
        data = [
            {"perturbation_id": f"{i}", "class_validation": {"class_matches": 1}}
            for i in range(8)
        ]
        data.extend([
            {"perturbation_id": "9", "class_validation": {"class_matches": 0}},
            {"perturbation_id": "10", "class_validation": {"class_matches": 0}},
        ])

        result = gate.check(data)
        assert result.status.value == "fail"
        assert result.value == 0.8

    def test_custom_min_rate(self, gate):
        """Test that custom min_rate config is respected."""
        data = [
            {"perturbation_id": "1", "class_validation": {"class_matches": 1}},
            {"perturbation_id": "2", "class_validation": {"class_matches": 0}},
        ]

        # With default 90% threshold, 50% should fail
        result = gate.check(data, config={})
        assert result.status.value == "fail"

        # With 50% threshold, 50% should pass
        result = gate.check(data, config={"min_rate": 0.50})
        assert result.status.value == "pass"

    def test_provides_mismatch_examples(self, gate):
        """Test that gate provides examples of mismatches on failure."""
        data = [
            {
                "perturbation_id": "1",
                "perturbation_class": "placebo",
                "perturbation_type": "formatting",
                "class_validation": {
                    "class_matches": 0,
                    "reasoning": "semantic change detected"
                }
            },
        ]

        result = gate.check(data, config={"min_rate": 0.90})
        assert result.status.value == "fail"
        assert "mismatch_examples" in result.details
        assert len(result.details["mismatch_examples"]) == 1
        assert result.details["mismatch_examples"][0]["perturbation_class"] == "placebo"
