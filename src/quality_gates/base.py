"""
Base classes for quality gates.

Quality gates are validation checks that must pass for data to proceed
through the pipeline. They use ONLY regex/parsing - NO LLM calls.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GateStatus(Enum):
    """Status of a quality gate check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class GateResult:
    """Result of a single quality gate check."""

    gate_name: str
    status: GateStatus
    message: str
    value: Optional[Any] = None
    threshold: Optional[Any] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gate_name": self.gate_name,
            "status": self.status.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "details": self.details,
        }

    @property
    def passed(self) -> bool:
        """Check if gate passed (PASS or WARN)."""
        return self.status in (GateStatus.PASS, GateStatus.WARN)


@dataclass
class GateReport:
    """Aggregated report from multiple quality gates."""

    phase: str
    timestamp: str
    results: List[GateResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Check if all gates passed (no FAIL status)."""
        return all(r.status != GateStatus.FAIL for r in self.results)

    @property
    def pass_count(self) -> int:
        """Count of passed gates."""
        return sum(1 for r in self.results if r.status == GateStatus.PASS)

    @property
    def fail_count(self) -> int:
        """Count of failed gates."""
        return sum(1 for r in self.results if r.status == GateStatus.FAIL)

    @property
    def warn_count(self) -> int:
        """Count of warning gates."""
        return sum(1 for r in self.results if r.status == GateStatus.WARN)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "phase": self.phase,
            "timestamp": self.timestamp,
            "all_passed": self.all_passed,
            "summary": {
                "total": len(self.results),
                "pass": self.pass_count,
                "fail": self.fail_count,
                "warn": self.warn_count,
            },
            "results": [r.to_dict() for r in self.results],
        }


class BaseGate(ABC):
    """
    Abstract base class for quality gates.

    All gates must:
    - Use only regex/parsing (no LLM calls)
    - Return a GateResult
    - Be idempotent
    """

    name: str = "base_gate"
    description: str = "Base quality gate"

    @abstractmethod
    def check(self, data: Any, config: Optional[Dict[str, Any]] = None) -> GateResult:
        """
        Run the quality gate check.

        Args:
            data: Data to check (varies by gate type)
            config: Optional configuration overrides

        Returns:
            GateResult with pass/fail status
        """
        pass

    def _pass(
        self,
        message: str,
        value: Any = None,
        threshold: Any = None,
        details: Dict = None,
    ) -> GateResult:
        """Create a passing result."""
        return GateResult(
            gate_name=self.name,
            status=GateStatus.PASS,
            message=message,
            value=value,
            threshold=threshold,
            details=details,
        )

    def _fail(
        self,
        message: str,
        value: Any = None,
        threshold: Any = None,
        details: Dict = None,
    ) -> GateResult:
        """Create a failing result."""
        return GateResult(
            gate_name=self.name,
            status=GateStatus.FAIL,
            message=message,
            value=value,
            threshold=threshold,
            details=details,
        )

    def _warn(
        self,
        message: str,
        value: Any = None,
        threshold: Any = None,
        details: Dict = None,
    ) -> GateResult:
        """Create a warning result."""
        return GateResult(
            gate_name=self.name,
            status=GateStatus.WARN,
            message=message,
            value=value,
            threshold=threshold,
            details=details,
        )

    def _skip(
        self,
        message: str,
        details: Dict = None,
    ) -> GateResult:
        """Create a skipped result."""
        return GateResult(
            gate_name=self.name,
            status=GateStatus.SKIP,
            message=message,
            details=details,
        )


class GateRunner:
    """
    Runner for executing multiple quality gates.

    Usage:
        runner = GateRunner(phase="perturb")
        runner.add_gate(NoSyntheticMarkersGate())
        runner.add_gate(JSONValidityGate())
        report = runner.run(perturbations)
    """

    def __init__(self, phase: str):
        """
        Initialize gate runner.

        Args:
            phase: Pipeline phase name (load, perturb, create_units, etc.)
        """
        self.phase = phase
        self.gates: List[BaseGate] = []

    def add_gate(self, gate: BaseGate):
        """Add a gate to run."""
        self.gates.append(gate)

    def run(
        self, data: Any, config: Optional[Dict[str, Any]] = None
    ) -> GateReport:
        """
        Run all gates and return report.

        Args:
            data: Data to check
            config: Optional configuration

        Returns:
            GateReport with all results
        """
        from datetime import datetime

        results = []
        for gate in self.gates:
            try:
                result = gate.check(data, config)
                results.append(result)
            except Exception as e:
                # Gate crashed - treat as fail
                results.append(
                    GateResult(
                        gate_name=gate.name,
                        status=GateStatus.FAIL,
                        message=f"Gate crashed: {e}",
                        details={"exception": str(e)},
                    )
                )

        return GateReport(
            phase=self.phase,
            timestamp=datetime.utcnow().isoformat() + "Z",
            results=results,
        )
