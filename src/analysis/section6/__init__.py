"""
Section 6 Analysis Module: Detection Metrics (6A), Calibration (6B), Agreement + Claim (6C).

This module computes per-unit analysis results that combine:
- Ground truth from evaluation units, outcome evidence, and human labels
- Judge outputs from LLM evaluations
- Detection evaluation metrics (6A)
- Calibration evaluation metrics (6B)

Results are stored in MongoDB for flexible aggregation via ops scripts.
"""

from src.analysis.section6.schema import (
    AnalysisResult,
    GroundTruth,
    JudgeOutputSummary,
    DetectionEval,
    CalibrationEval,
    HumanComparison,
)
from src.analysis.section6.evaluator import Section6Evaluator
from src.analysis.section6.storage import Section6Storage

__all__ = [
    "AnalysisResult",
    "GroundTruth",
    "JudgeOutputSummary",
    "DetectionEval",
    "CalibrationEval",
    "HumanComparison",
    "Section6Evaluator",
    "Section6Storage",
]
