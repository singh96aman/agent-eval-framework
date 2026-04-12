"""
Tests for judge evaluation system.
"""

import pytest
import json
from datetime import datetime

from src.data.schema import Trajectory, Step, StepType, GroundTruth
from src.judges.schema import JudgeOutput, ErrorSeverity, EvaluationResults
from src.judges import Judge, parse_json_response
from src.judges.prompts import build_evaluation_prompt, format_trajectory_for_judge

# === Fixtures ===


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    ground_truth = GroundTruth(
        task_description="Find the weather in Paris",
        expected_answer="The weather in Paris is sunny with 20°C",
    )

    return Trajectory(
        trajectory_id="test_001",
        benchmark="toolbench",
        steps=[
            Step(
                step_id="step_1",
                step_number=1,
                step_type=StepType.REASONING,
                content="I need to call a weather API",
            ),
            Step(
                step_id="step_2",
                step_number=2,
                step_type=StepType.TOOL_EXECUTION,
                content="Calling weather API",
                tool_name="get_weather",
                tool_input={"city": "Paris"},
                tool_output='{"temp": 20, "condition": "sunny"}',
            ),
            Step(
                step_id="step_3",
                step_number=3,
                step_type=StepType.FINAL_ANSWER,
                content="The weather in Paris is sunny with 20°C",
            ),
        ],
        ground_truth=ground_truth,
        metadata={},
    )


@pytest.fixture
def sample_judge_response():
    """Sample valid JSON response from a judge."""
    return """```json
{
  "task_success": 1,
  "completeness": 95,
  "hallucination": 0,
  "sycophancy": 0,
  "efficiency_errors": 0,
  "overall_score": 90,
  "step_errors": [
    {
      "step_index": 2,
      "error_type": "parameter",
      "severity": "minor",
      "description": "Could have included more details",
      "impacts_task_success": false
    }
  ],
  "reasoning": "Agent successfully completed the task with minor issues."
}
```"""


# === Test Prompts ===


def test_format_trajectory_for_judge(sample_trajectory):
    """Test trajectory formatting for judge prompts."""
    formatted = format_trajectory_for_judge(sample_trajectory)

    assert "AGENT TRAJECTORY" in formatted
    assert "Find the weather in Paris" in formatted
    assert "test_001" in formatted
    assert "Step 1" in formatted
    assert "get_weather" in formatted
    assert "sunny" in formatted


def test_build_evaluation_prompt(sample_trajectory):
    """Test full evaluation prompt construction."""
    prompt = build_evaluation_prompt(sample_trajectory)

    # Check structure
    assert "AGENT TRAJECTORY" in prompt
    assert "Task Success" in prompt
    assert "Completeness" in prompt
    assert "Hallucination" in prompt
    assert "JSON format" in prompt

    # Check trajectory content
    assert "Find the weather in Paris" in prompt


# === Test JSON Parsing ===


def test_parse_json_response_valid(sample_judge_response):
    """Test parsing a valid JSON response."""
    output = parse_json_response(
        response_text=sample_judge_response,
        trajectory_id="test_001",
        judge_name="test-judge",
        model_id="test-model",
    )

    assert output.trajectory_id == "test_001"
    assert output.judge_name == "test-judge"
    assert output.task_success == 1
    assert output.completeness == 95
    assert output.overall_score == 90
    assert len(output.step_errors) == 1
    assert output.step_errors[0].step_index == 2
    assert output.step_errors[0].severity == ErrorSeverity.MINOR


def test_parse_json_response_no_markdown():
    """Test parsing JSON without markdown wrapper."""
    response = json.dumps(
        {
            "task_success": 0,
            "completeness": 50,
            "hallucination": 1,
            "sycophancy": 0,
            "efficiency_errors": 2,
            "overall_score": 40,
            "step_errors": [],
            "reasoning": "Task failed due to hallucination",
        }
    )

    output = parse_json_response(
        response_text=response,
        trajectory_id="test_002",
        judge_name="test-judge",
        model_id="test-model",
    )

    assert output.task_success == 0
    assert output.hallucination == 1
    assert output.efficiency_errors == 2
    assert "hallucination" in output.reasoning


def test_parse_json_response_invalid():
    """Test parsing invalid JSON response."""
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        parse_json_response(
            response_text="This is not JSON",
            trajectory_id="test_003",
            judge_name="test-judge",
            model_id="test-model",
        )


# === Test Judge Base Class ===


class MockJudge(Judge):
    """Mock judge for testing."""

    def _call_llm(self, system_prompt, user_prompt):
        """Mock LLM call."""
        return {
            "response": json.dumps(
                {
                    "task_success": 1,
                    "completeness": 100,
                    "hallucination": 0,
                    "sycophancy": 0,
                    "efficiency_errors": 0,
                    "overall_score": 100,
                    "step_errors": [],
                    "reasoning": "Perfect execution",
                }
            ),
            "tokens_used": 100,
            "time_ms": 500,
        }

    def _parse_response(self, response_text, trajectory_id):
        """Mock response parsing."""
        return parse_json_response(
            response_text=response_text,
            trajectory_id=trajectory_id,
            judge_name=self.name,
            model_id=self.model_id,
        )


def test_judge_evaluate_success(sample_trajectory):
    """Test successful judge evaluation."""
    judge = MockJudge(
        name="mock-judge", model_id="mock-model", temperature=0.7, max_tokens=2000
    )

    output = judge.evaluate(sample_trajectory)

    assert output is not None
    assert output.trajectory_id == "test_001"
    assert output.overall_score == 100
    assert output.tokens_used == 100
    assert output.evaluation_time_ms == 500

    # Check stats
    stats = judge.get_stats()
    assert stats["total_calls"] == 1
    assert stats["failed_calls"] == 0
    assert stats["total_tokens"] == 100


def test_judge_evaluate_with_retry(sample_trajectory):
    """Test judge evaluation with retry on failure."""

    class FailingJudge(MockJudge):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.call_count = 0

        def _call_llm(self, system_prompt, user_prompt):
            self.call_count += 1
            if self.call_count < 2:
                raise Exception("Temporary failure")
            return super()._call_llm(system_prompt, user_prompt)

    judge = FailingJudge(name="failing-judge", model_id="failing-model")

    output = judge.evaluate(sample_trajectory, max_retries=3)

    assert output is not None
    assert judge.call_count == 2  # Failed once, succeeded second time


# === Test Schema ===


def test_judge_output_to_dict():
    """Test JudgeOutput serialization to dict."""
    output = JudgeOutput(
        trajectory_id="test_001",
        judge_name="test-judge",
        model_id="test-model",
        task_success=1,
        completeness=90,
        efficiency_errors=1,
        hallucination=0,
        sycophancy=0,
        overall_score=85,
        reasoning="Good execution with minor inefficiency",
    )

    output_dict = output.to_dict()

    assert output_dict["trajectory_id"] == "test_001"
    assert output_dict["task_success"] == 1
    assert output_dict["overall_score"] == 85
    assert isinstance(output_dict["timestamp"], datetime)


def test_judge_output_from_dict():
    """Test JudgeOutput deserialization from dict."""
    data = {
        "trajectory_id": "test_001",
        "judge_name": "test-judge",
        "model_id": "test-model",
        "task_success": 0,
        "completeness": 60,
        "efficiency_errors": 3,
        "hallucination": 1,
        "sycophancy": 0,
        "overall_score": 45,
        "step_errors": [
            {
                "step_index": 3,
                "error_type": "reasoning",
                "severity": "critical",
                "description": "Incorrect conclusion",
                "impacts_task_success": True,
            }
        ],
        "reasoning": "Task failed due to reasoning error",
        "timestamp": datetime.utcnow(),
        "evaluation_time_ms": 1200,
        "tokens_used": 250,
    }

    output = JudgeOutput.from_dict(data)

    assert output.trajectory_id == "test_001"
    assert output.task_success == 0
    assert output.hallucination == 1
    assert len(output.step_errors) == 1
    assert output.step_errors[0].severity == ErrorSeverity.CRITICAL


def test_evaluation_results_to_dict():
    """Test EvaluationResults serialization."""
    results = EvaluationResults(
        experiment_id="exp_001",
        judge_name="claude-sonnet-4.5",
        total_evaluated=50,
        successful=48,
        failed=2,
        total_time_seconds=120.5,
        total_tokens=12000,
        average_score=75.3,
        evaluation_errors=["Error 1", "Error 2"],
    )

    results_dict = results.to_dict()

    assert results_dict["experiment_id"] == "exp_001"
    assert results_dict["successful"] == 48
    assert results_dict["failed"] == 2
    assert len(results_dict["evaluation_errors"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
