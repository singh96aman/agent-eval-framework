"""
Tests for prompt registry.

Tests prompt lookup, validation, and backward compatibility.
"""

import pytest
from src.prompts import (
    get_prompt,
    get_prompt_safe,
    list_prompts,
    validate_prompt_config,
    PROMPT_REGISTRY,
    # Judge prompts
    BLINDED_PAIR_SYSTEM_V1,
    BLINDED_PAIR_SYSTEM_V2,
    SINGLE_TRAJECTORY_SYSTEM_V1,
    # Perturbation prompts
    WRONG_PARAMETER_PROMPT_V1,
    VALUE_MUTATION_PROMPT_V1,
    # Backward compat
    BLINDED_PAIR_SYSTEM,
    BLINDED_PAIR_USER,
)


class TestPromptRegistry:
    """Tests for prompt registry functions."""

    def test_get_prompt_returns_string(self):
        """Test get_prompt returns prompt string."""
        prompt = get_prompt("BLINDED_PAIR_SYSTEM_V2")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_prompt_raises_for_unknown(self):
        """Test get_prompt raises KeyError for unknown prompt."""
        with pytest.raises(KeyError):
            get_prompt("NONEXISTENT_PROMPT")

    def test_get_prompt_safe_returns_none(self):
        """Test get_prompt_safe returns None for unknown."""
        result = get_prompt_safe("NONEXISTENT_PROMPT")
        assert result is None

    def test_get_prompt_safe_returns_default(self):
        """Test get_prompt_safe returns custom default."""
        result = get_prompt_safe("NONEXISTENT_PROMPT", default="fallback")
        assert result == "fallback"

    def test_list_prompts_all(self):
        """Test list_prompts returns all prompts."""
        prompts = list_prompts()
        assert len(prompts) > 0
        assert "BLINDED_PAIR_SYSTEM_V1" in prompts
        assert "WRONG_PARAMETER_PROMPT_V1" in prompts

    def test_list_prompts_judge_category(self):
        """Test list_prompts filters by judge category."""
        prompts = list_prompts(category="judge")
        assert "BLINDED_PAIR_SYSTEM_V1" in prompts
        assert "WRONG_PARAMETER_PROMPT_V1" not in prompts

    def test_list_prompts_perturbation_category(self):
        """Test list_prompts filters by perturbation category."""
        prompts = list_prompts(category="perturbation")
        assert "WRONG_PARAMETER_PROMPT_V1" in prompts
        assert "BLINDED_PAIR_SYSTEM_V1" not in prompts

    def test_list_prompts_invalid_category(self):
        """Test list_prompts raises for invalid category."""
        with pytest.raises(ValueError):
            list_prompts(category="invalid")


class TestPromptValidation:
    """Tests for config validation."""

    def test_valid_config_passes(self):
        """Test valid config passes validation."""
        config = {
            "prompts": {
                "judge": {
                    "blinded_pair": {
                        "system": "BLINDED_PAIR_SYSTEM_V2",
                        "user": "BLINDED_PAIR_USER_V1",
                    }
                },
                "perturbation": {
                    "wrong_parameter": "WRONG_PARAMETER_PROMPT_V1",
                }
            }
        }
        result = validate_prompt_config(config)
        assert result is True

    def test_invalid_config_raises(self):
        """Test invalid config raises ValueError."""
        config = {
            "prompts": {
                "judge": {
                    "blinded_pair": {
                        "system": "NONEXISTENT_PROMPT",
                    }
                }
            }
        }
        with pytest.raises(ValueError):
            validate_prompt_config(config)

    def test_empty_config_passes(self):
        """Test empty config passes validation."""
        config = {}
        result = validate_prompt_config(config)
        assert result is True


class TestPromptContent:
    """Tests for prompt content quality."""

    def test_v2_prompt_is_neutral(self):
        """Test V2 blinded pair prompt is neutral (no priming)."""
        prompt = BLINDED_PAIR_SYSTEM_V2
        prompt_lower = prompt.lower()

        # Should not contain priming language
        assert "one.*contain.*error" not in prompt_lower
        assert "find the error" not in prompt_lower

        # Should have neutral framing
        assert "compare" in prompt_lower or "analyze" in prompt_lower

    def test_prompts_have_json_instruction(self):
        """Test all prompts instruct JSON response."""
        for name, prompt in PROMPT_REGISTRY.items():
            # System prompts should mention JSON
            if "SYSTEM" in name:
                assert "json" in prompt.lower(), f"{name} should mention JSON"

    def test_perturbation_prompts_have_no_marker_instruction(self):
        """Test perturbation prompts warn against markers."""
        marker_prompts = [
            "WRONG_PARAMETER_PROMPT_V1",
            "VALUE_MUTATION_PROMPT_V1",
            "WRONG_DATE_PROMPT_V1",
            "WRONG_IDENTIFIER_PROMPT_V1",
        ]

        for name in marker_prompts:
            prompt = get_prompt(name)
            prompt_lower = prompt.lower()

            # Should warn against artifact markers
            assert "_old" in prompt_lower or "markers" in prompt_lower, \
                f"{name} should warn against markers"


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_legacy_aliases_exist(self):
        """Test legacy prompt aliases work."""
        # These should not raise
        assert BLINDED_PAIR_SYSTEM is not None
        assert BLINDED_PAIR_USER is not None

    def test_legacy_equals_v1(self):
        """Test legacy aliases equal V1 prompts."""
        assert BLINDED_PAIR_SYSTEM == BLINDED_PAIR_SYSTEM_V1

    def test_prompts_are_unique(self):
        """Test V1 and V2 prompts are different."""
        assert BLINDED_PAIR_SYSTEM_V1 != BLINDED_PAIR_SYSTEM_V2


class TestPromptFormatting:
    """Tests for prompt formatting/templating."""

    def test_perturbation_prompts_have_placeholders(self):
        """Test perturbation prompts have format placeholders."""
        prompt = WRONG_PARAMETER_PROMPT_V1
        assert "{tool_name}" in prompt
        assert "{param_name}" in prompt
        assert "{current_value}" in prompt

    def test_prompt_formatting_works(self):
        """Test prompts can be formatted."""
        prompt = WRONG_PARAMETER_PROMPT_V1
        formatted = prompt.format(
            tool_name="search",
            param_name="query",
            current_value="test",
            value_type="string",
            task_context="Testing",
        )
        assert "search" in formatted
        assert "query" in formatted
        assert "{" not in formatted.split("JSON")[0]  # No unformatted placeholders
