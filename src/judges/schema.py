"""
Judge output schema and data structures.

Defines the structured format for LLM judge evaluations of agent trajectories.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class ErrorSeverity(Enum):
    """Severity levels for step errors."""
    CRITICAL = "critical"  # Blocks task completion
    MAJOR = "major"  # Significant impact on quality
    MINOR = "minor"  # Small issue, task still succeeds
    NONE = "none"  # No error


@dataclass
class StepError:
    """
    Represents an error identified in a specific step.
    """
    step_index: int
    error_type: str  # e.g., "planning", "tool_selection", "parameter", "reasoning"
    severity: ErrorSeverity
    description: str
    impacts_task_success: bool  # Does this error prevent task completion?

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "step_index": self.step_index,
            "error_type": self.error_type,
            "severity": self.severity.value,
            "description": self.description,
            "impacts_task_success": self.impacts_task_success
        }


@dataclass
class JudgeOutput:
    """
    Complete output from a judge evaluation.

    This aligns with the 5 metrics defined in the POC experiment config:
    - task_success: binary (0 or 1)
    - completeness: percentage (0-100)
    - efficiency: count of unnecessary tool calls
    - hallucination: binary (0 or 1)
    - sycophancy: binary (0 or 1) - if agent agreed with false user premises
    """
    trajectory_id: str
    judge_name: str
    model_id: str

    # Core metrics (from config)
    task_success: int  # 0 or 1
    completeness: float  # 0-100
    efficiency_errors: int  # Count of unnecessary/redundant calls
    hallucination: int  # 0 or 1
    sycophancy: int  # 0 or 1

    # Overall score (0-100, higher is better)
    overall_score: float

    # Detailed analysis
    step_errors: List[StepError] = field(default_factory=list)
    reasoning: str = ""

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    evaluation_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "trajectory_id": self.trajectory_id,
            "judge_name": self.judge_name,
            "model_id": self.model_id,
            "task_success": self.task_success,
            "completeness": self.completeness,
            "efficiency_errors": self.efficiency_errors,
            "hallucination": self.hallucination,
            "sycophancy": self.sycophancy,
            "overall_score": self.overall_score,
            "step_errors": [error.to_dict() for error in self.step_errors],
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "evaluation_time_ms": self.evaluation_time_ms,
            "tokens_used": self.tokens_used
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JudgeOutput':
        """Create from dictionary (e.g., from MongoDB)."""
        step_errors = [
            StepError(
                step_index=err["step_index"],
                error_type=err["error_type"],
                severity=ErrorSeverity(err["severity"]),
                description=err["description"],
                impacts_task_success=err["impacts_task_success"]
            )
            for err in data.get("step_errors", [])
        ]

        return cls(
            trajectory_id=data["trajectory_id"],
            judge_name=data["judge_name"],
            model_id=data["model_id"],
            task_success=data["task_success"],
            completeness=data["completeness"],
            efficiency_errors=data["efficiency_errors"],
            hallucination=data["hallucination"],
            sycophancy=data["sycophancy"],
            overall_score=data["overall_score"],
            step_errors=step_errors,
            reasoning=data.get("reasoning", ""),
            timestamp=data.get("timestamp", datetime.utcnow()),
            evaluation_time_ms=data.get("evaluation_time_ms"),
            tokens_used=data.get("tokens_used")
        )


@dataclass
class EvaluationBatch:
    """
    Represents a batch of evaluations to process.
    """
    experiment_id: str
    perturbations: List[Dict[str, Any]]  # List of perturbation dicts
    judge_name: str
    batch_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvaluationResults:
    """
    Summary results from a judge evaluation run.
    """
    experiment_id: str
    judge_name: str
    total_evaluated: int
    successful: int
    failed: int
    total_time_seconds: float
    total_tokens: int
    average_score: float
    evaluation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
