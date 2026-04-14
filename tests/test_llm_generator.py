"""
Tests for LLM perturbation generator.

Tests mutation generation with mocked LLM client.
"""

import json
import pytest
from unittest.mock import Mock, patch
from src.perturbations.llm_generator import (
    LLMPerturbationGenerator,
    LLMMutationResult,
    get_llm_generator,
)


class MockBedrockClient:
    """Mock Bedrock client for testing."""

    def __init__(self, response: str = None):
        self.response = response or '{"mutated_value": "mock_value", "mutation_type": "mock", "reasoning": "test"}'
        self.call_count = 0

    def invoke(self, **kwargs):
        self.call_count += 1
        return {"response": self.response}


class TestLLMMutationResult:
    """Tests for LLMMutationResult dataclass."""

    def test_result_creation(self):
        """Test creating a mutation result."""
        result = LLMMutationResult(
            mutated_value="new_value",
            mutation_type="typo",
            reasoning="Test reason",
            raw_llm_response='{"mutated_value": "new_value"}',
            parse_success=True,
            prompt_name="TEST_PROMPT",
        )
        assert result.mutated_value == "new_value"
        assert result.parse_success is True

    def test_result_defaults(self):
        """Test result with default values."""
        result = LLMMutationResult(
            mutated_value="value",
            mutation_type="test",
        )
        assert result.reasoning is None
        assert result.parse_success is True


class TestLLMPerturbationGenerator:
    """Tests for LLMPerturbationGenerator."""

    @patch("src.perturbations.llm_generator.get_bedrock_client")
    def test_generate_wrong_parameter(self, mock_get_client):
        """Test generating wrong parameter value."""
        mock_client = MockBedrockClient()
        mock_get_client.return_value = mock_client

        generator = LLMPerturbationGenerator()
        result = generator.generate_wrong_parameter(
            tool_name="search",
            param_name="query",
            current_value="test",
            value_type="string",
        )

        assert result is not None
        assert result.parse_success is True
        assert result.mutated_value == "mock_value"
        assert result.prompt_name == "WRONG_PARAMETER_PROMPT_V1"

    @patch("src.perturbations.llm_generator.get_bedrock_client")
    def test_generate_value_mutation(self, mock_get_client):
        """Test generating value mutation."""
        mock_client = MockBedrockClient()
        mock_get_client.return_value = mock_client

        generator = LLMPerturbationGenerator()
        result = generator.generate_value_mutation(
            original_value="original",
            value_type="string",
        )

        assert result is not None
        assert result.parse_success is True
        assert mock_client.call_count == 1

    @patch("src.perturbations.llm_generator.get_bedrock_client")
    def test_generate_wrong_date(self, mock_get_client):
        """Test generating wrong date."""
        mock_response = '{"mutated_date": "2024-01-02", "mutation_type": "shift_day", "reasoning": "off by one"}'
        mock_client = MockBedrockClient(mock_response)
        mock_get_client.return_value = mock_client

        generator = LLMPerturbationGenerator()
        result = generator.generate_wrong_date(
            original_date="2024-01-01",
            date_format="YYYY-MM-DD",
        )

        assert result is not None
        assert result.parse_success is True
        assert result.mutated_value == "2024-01-02"
        assert result.prompt_name == "WRONG_DATE_PROMPT_V1"

    @patch("src.perturbations.llm_generator.get_bedrock_client")
    def test_generate_wrong_identifier(self, mock_get_client):
        """Test generating wrong identifier."""
        mock_response = '{"mutated_id": "user_id_typo", "mutation_type": "typo", "reasoning": "common typo"}'
        mock_client = MockBedrockClient(mock_response)
        mock_get_client.return_value = mock_client

        generator = LLMPerturbationGenerator()
        result = generator.generate_wrong_identifier(
            original_id="user_id",
            id_type="variable",
        )

        assert result is not None
        assert result.parse_success is True
        assert result.mutated_value == "user_id_typo"


class TestJSONParsing:
    """Tests for JSON response parsing."""

    @patch("src.perturbations.llm_generator.get_bedrock_client")
    def test_parse_valid_json(self, mock_get_client):
        """Test parsing valid JSON response."""
        mock_response = '{"mutated_value": "test", "mutation_type": "mock"}'
        mock_client = MockBedrockClient(mock_response)
        mock_get_client.return_value = mock_client

        generator = LLMPerturbationGenerator()
        result = generator.generate_value_mutation("orig", "string")

        assert result.parse_success is True

    @patch("src.perturbations.llm_generator.get_bedrock_client")
    def test_parse_markdown_json(self, mock_get_client):
        """Test parsing JSON in markdown code block."""
        mock_response = '```json\n{"mutated_value": "test", "mutation_type": "mock"}\n```'
        mock_client = MockBedrockClient(mock_response)
        mock_get_client.return_value = mock_client

        generator = LLMPerturbationGenerator()
        result = generator.generate_value_mutation("orig", "string")

        assert result.parse_success is True
        assert result.mutated_value == "test"

    @patch("src.perturbations.llm_generator.get_bedrock_client")
    def test_parse_json_with_text(self, mock_get_client):
        """Test parsing JSON embedded in text."""
        mock_response = 'Here is the result: {"mutated_value": "embedded", "mutation_type": "test"} Hope this helps!'
        mock_client = MockBedrockClient(mock_response)
        mock_get_client.return_value = mock_client

        generator = LLMPerturbationGenerator()
        result = generator.generate_value_mutation("orig", "string")

        assert result.parse_success is True
        assert result.mutated_value == "embedded"

    @patch("src.perturbations.llm_generator.get_bedrock_client")
    def test_parse_failure_returns_result(self, mock_get_client):
        """Test parse failure still returns result with flag."""
        mock_response = "This is not valid JSON at all!"
        mock_client = MockBedrockClient(mock_response)
        mock_get_client.return_value = mock_client

        generator = LLMPerturbationGenerator(max_retries=0)
        result = generator.generate_value_mutation("orig", "string")

        # Should return result with parse_success=False
        assert result is not None
        assert result.parse_success is False
        assert result.mutation_type == "llm_parse_failed"

    @patch("src.perturbations.llm_generator.get_bedrock_client")
    def test_raw_response_stored(self, mock_get_client):
        """Test raw LLM response is always stored."""
        mock_response = '{"mutated_value": "test", "mutation_type": "mock"}'
        mock_client = MockBedrockClient(mock_response)
        mock_get_client.return_value = mock_client

        generator = LLMPerturbationGenerator()
        result = generator.generate_value_mutation("orig", "string")

        assert result.raw_llm_response == mock_response


class TestRetryBehavior:
    """Tests for retry behavior."""

    @patch("src.perturbations.llm_generator.get_bedrock_client")
    def test_retries_on_parse_failure(self, mock_get_client):
        """Test retries on JSON parse failure."""
        # First call fails, second succeeds
        responses = ["invalid", '{"mutated_value": "retry_success", "mutation_type": "test"}']
        call_count = [0]

        class MockClientWithRetry:
            def invoke(self, **kwargs):
                idx = min(call_count[0], len(responses) - 1)
                call_count[0] += 1
                return {"response": responses[idx]}

        mock_get_client.return_value = MockClientWithRetry()

        generator = LLMPerturbationGenerator(max_retries=1)
        result = generator.generate_value_mutation("orig", "string")

        assert result.parse_success is True
        assert result.mutated_value == "retry_success"
        assert call_count[0] == 2  # Two calls made


class TestSingletonGenerator:
    """Tests for singleton generator instance."""

    def test_get_llm_generator_returns_same(self):
        """Test get_llm_generator returns singleton."""
        # Reset singleton
        import src.perturbations.llm_generator as module
        module._default_generator = None

        with patch("src.perturbations.llm_generator.get_bedrock_client"):
            gen1 = get_llm_generator()
            gen2 = get_llm_generator()
            assert gen1 is gen2
