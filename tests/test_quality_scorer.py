"""
Tests for PerturbationQualityScorer module.

Tests cover:
- Score parsing and validation
- Tier assignment
- Error handling
- Batch processing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from src.scoring.quality_scorer import PerturbationQualityScorer, create_quality_scorer
from src.scoring.prompts import format_quality_scoring_prompt, PERTURBATION_TYPE_DESCRIPTIONS


class TestQualityScorerPrompt:
    """Tests for quality scoring prompt formatting."""

    def test_format_prompt_basic(self):
        """Test basic prompt formatting."""
        prompt = format_quality_scoring_prompt(
            original_step_content="Search for weather",
            perturbed_step_content="Search for sports",
            perturbation_type="planning",
            perturbation_position="early"
        )

        assert "Search for weather" in prompt
        assert "Search for sports" in prompt
        assert "planning" in prompt
        assert "early" in prompt

    def test_format_prompt_with_description(self):
        """Test prompt with custom description."""
        prompt = format_quality_scoring_prompt(
            original_step_content="Original",
            perturbed_step_content="Perturbed",
            perturbation_type="tool_selection",
            perturbation_position="middle",
            perturbation_description="Custom error description"
        )

        assert "Custom error description" in prompt

    def test_default_descriptions(self):
        """Test default descriptions for all types."""
        for ptype in ["planning", "tool_selection", "parameter", "data_reference"]:
            assert ptype in PERTURBATION_TYPE_DESCRIPTIONS


class TestQualityScorerScoreParsing:
    """Tests for score parsing logic."""

    @pytest.fixture
    def scorer(self):
        """Create scorer with mocked LLM client."""
        with patch('boto3.client'):
            return PerturbationQualityScorer(
                model_config={
                    "name": "test-model",
                    "model_id": "test-model-id",
                    "config": {"temperature": 0.3, "max_tokens": 500}
                }
            )

    def test_parse_valid_response(self, scorer):
        """Test parsing valid JSON response."""
        response_text = json.dumps({
            "content_changed": 1,
            "syntactically_valid": 1,
            "semantically_meaningful": 1,
            "type_matches_intent": 1,
            "realistic_error": 2,
            "total_score": 6,
            "reasoning": "Good perturbation"
        })

        score = scorer._parse_score_response(response_text)

        assert score["content_changed"] == 1
        assert score["syntactically_valid"] == 1
        assert score["semantically_meaningful"] == 1
        assert score["type_matches_intent"] == 1
        assert score["realistic_error"] == 2
        assert score["total_score"] == 6
        assert score["reasoning"] == "Good perturbation"

    def test_parse_response_with_extra_text(self, scorer):
        """Test parsing JSON with surrounding text."""
        response_text = """Here is my analysis:
        {
            "content_changed": 1,
            "syntactically_valid": 1,
            "semantically_meaningful": 0,
            "type_matches_intent": 1,
            "realistic_error": 1,
            "reasoning": "Minor change"
        }
        """

        score = scorer._parse_score_response(response_text)

        assert score["content_changed"] == 1
        assert score["semantically_meaningful"] == 0
        assert score["total_score"] == 4  # 1+1+0+1+1

    def test_parse_clamps_values(self, scorer):
        """Test that out-of-range values are clamped."""
        response_text = json.dumps({
            "content_changed": 5,  # Should clamp to 1
            "syntactically_valid": -1,  # Should clamp to 0
            "semantically_meaningful": 1,
            "type_matches_intent": 1,
            "realistic_error": 10,  # Should clamp to 3
            "reasoning": "Test"
        })

        score = scorer._parse_score_response(response_text)

        assert score["content_changed"] == 1
        assert score["syntactically_valid"] == 0
        assert score["realistic_error"] == 3

    def test_parse_missing_field_raises(self, scorer):
        """Test that missing required field raises ValueError."""
        response_text = json.dumps({
            "content_changed": 1,
            # Missing other fields
        })

        with pytest.raises(ValueError, match="Missing required field"):
            scorer._parse_score_response(response_text)

    def test_parse_no_json_raises(self, scorer):
        """Test that response without JSON raises ValueError."""
        response_text = "This is not JSON at all"

        with pytest.raises(ValueError, match="No JSON found"):
            scorer._parse_score_response(response_text)


class TestQualityScorerTierAssignment:
    """Tests for quality tier assignment."""

    @pytest.fixture
    def scorer(self):
        """Create scorer with custom thresholds."""
        with patch('boto3.client'):
            return PerturbationQualityScorer(
                model_config={
                    "name": "test-model",
                    "model_id": "test-model-id",
                    "config": {}
                },
                tier_thresholds={
                    "high": 6,
                    "medium": 4,
                    "low": 1,
                    "invalid": 0
                }
            )

    def test_tier_high(self, scorer):
        """Test high tier assignment (score >= 6)."""
        assert scorer._assign_tier(7) == "high"
        assert scorer._assign_tier(6) == "high"

    def test_tier_medium(self, scorer):
        """Test medium tier assignment (4 <= score < 6)."""
        assert scorer._assign_tier(5) == "medium"
        assert scorer._assign_tier(4) == "medium"

    def test_tier_low(self, scorer):
        """Test low tier assignment (1 <= score < 4)."""
        assert scorer._assign_tier(3) == "low"
        assert scorer._assign_tier(2) == "low"
        assert scorer._assign_tier(1) == "low"

    def test_tier_invalid(self, scorer):
        """Test invalid tier assignment (score < 1)."""
        assert scorer._assign_tier(0) == "invalid"


class TestQualityScorerIntegration:
    """Integration tests for quality scorer."""

    @pytest.fixture
    def mock_scorer(self):
        """Create scorer with mocked LLM calls."""
        with patch('boto3.client') as mock_client:
            # Mock Bedrock response
            mock_response = {
                'body': MagicMock()
            }
            mock_response['body'].read.return_value = json.dumps({
                'content': [{
                    'text': json.dumps({
                        "content_changed": 1,
                        "syntactically_valid": 1,
                        "semantically_meaningful": 1,
                        "type_matches_intent": 1,
                        "realistic_error": 2,
                        "reasoning": "Test"
                    })
                }],
                'usage': {'input_tokens': 100, 'output_tokens': 50}
            }).encode()

            mock_client.return_value.invoke_model.return_value = mock_response

            return PerturbationQualityScorer(
                model_config={
                    "name": "test-model",
                    "model_id": "test-model-id",
                    "config": {"temperature": 0.3}
                }
            )

    def test_score_perturbation_returns_complete_score(self, mock_scorer):
        """Test that score_perturbation returns complete score dict."""
        perturbation = {
            "original_step_content": "Search for weather",
            "perturbed_step_content": "Search for sports",
            "perturbation_type": "planning",
            "perturbation_position": "early",
            "perturbation_metadata": {}
        }

        score = mock_scorer.score_perturbation(perturbation)

        # Check all required fields
        assert "content_changed" in score
        assert "syntactically_valid" in score
        assert "semantically_meaningful" in score
        assert "type_matches_intent" in score
        assert "realistic_error" in score
        assert "total_score" in score
        assert "reasoning" in score
        assert "quality_tier" in score
        assert "scored_by" in score
        assert "scored_at" in score

        # Check values
        assert score["total_score"] == 6
        assert score["quality_tier"] == "high"

    def test_default_error_score(self, mock_scorer):
        """Test error score when scoring fails."""
        error_score = mock_scorer._default_error_score("Test error")

        assert error_score["total_score"] == 0
        assert error_score["quality_tier"] == "invalid"
        assert "error" in error_score
        assert "Test error" in error_score["error"]


class TestCreateQualityScorer:
    """Tests for factory function."""

    def test_create_from_config(self):
        """Test creating scorer from config dict."""
        config = {
            "judge_model": {
                "name": "claude-sonnet-4-5",
                "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "config": {"temperature": 0.3, "max_tokens": 500}
            },
            "metrics": {},
            "tier_thresholds": {"high": 6, "medium": 4, "low": 1, "invalid": 0}
        }

        with patch('boto3.client'):
            scorer = create_quality_scorer(config)

        assert scorer.model_name == "claude-sonnet-4-5"
        assert scorer.temperature == 0.3
        assert scorer.max_tokens == 500
