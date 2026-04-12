"""
Judge evaluation system for LLM-based trajectory assessment.

Provides:
- Base Judge interface and ClaudeJudge implementation
- Prompt templates for various evaluation modes
- Response parsing and validation
- Aggregation across samples
- Storage utilities
- Unit runner for batch processing
"""

# Schema classes
from src.judges.schema import (
    JudgeOutput,
    StepError,
    ErrorSeverity,
    EvaluationBatch,
    EvaluationResults,
    JudgeMode,
    ErrorType,
    ParseStatus,
    JudgeConfig,
    InputView,
    DetectionOutput,
    LocalizationOutput,
    ImpactOutput,
    PairComparisonOutput,
    Section5JudgeOutput,
    AggregatedDetection,
    AggregatedLocalization,
    AggregatedImpact,
    AggregatedJudgeOutput,
)

# Prompts
from src.judges.prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_evaluation_prompt,
    build_unit_prompt,
    build_view_for_single_trajectory,
    build_view_for_blinded_pair,
    build_view_for_labeled_pair,
    format_trajectory_for_judge,
)

# Utils - Judge classes and utilities
from src.judges.utils import (
    # Base class and implementations
    Judge,
    ClaudeJudge,
    create_claude_judge,
    # Parsing
    parse_json_response,
    parse_judge_response,
    map_step_to_canonical,
    validate_judge_output,
    # Aggregation
    aggregate_across_samples,
    batch_aggregate_across_samples,
    # Storage
    save_judge_output,
    load_judge_outputs,
    save_checkpoint,
    load_checkpoint,
    save_judge_outputs_to_mongodb,
    load_judge_outputs_from_mongodb,
    # Unit runner
    UnitJudgeRunner,
    create_unit_runner,
)

__all__ = [
    # Schema
    "JudgeOutput",
    "StepError",
    "ErrorSeverity",
    "EvaluationBatch",
    "EvaluationResults",
    "JudgeMode",
    "ErrorType",
    "ParseStatus",
    "JudgeConfig",
    "InputView",
    "DetectionOutput",
    "LocalizationOutput",
    "ImpactOutput",
    "PairComparisonOutput",
    "Section5JudgeOutput",
    "AggregatedDetection",
    "AggregatedLocalization",
    "AggregatedImpact",
    "AggregatedJudgeOutput",
    # Prompts
    "JUDGE_SYSTEM_PROMPT",
    "build_evaluation_prompt",
    "build_unit_prompt",
    "build_view_for_single_trajectory",
    "build_view_for_blinded_pair",
    "build_view_for_labeled_pair",
    "format_trajectory_for_judge",
    # Judge classes
    "Judge",
    "ClaudeJudge",
    "create_claude_judge",
    # Parsing
    "parse_json_response",
    "parse_judge_response",
    "map_step_to_canonical",
    "validate_judge_output",
    # Aggregation
    "aggregate_across_samples",
    "batch_aggregate_across_samples",
    # Storage
    "save_judge_output",
    "load_judge_outputs",
    "save_checkpoint",
    "load_checkpoint",
    "save_judge_outputs_to_mongodb",
    "load_judge_outputs_from_mongodb",
    # Runner
    "UnitJudgeRunner",
    "create_unit_runner",
]
