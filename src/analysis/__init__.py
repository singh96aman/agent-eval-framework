"""
Analysis module for RQ1 Consequentiality Calibration.

Computes calibration metrics between Judge Penalty Score (JPS) and
Outcome Degradation (OD).
"""

from src.analysis.calibration import CalibrationAnalyzer, CalibrationReport

__all__ = ["CalibrationAnalyzer", "CalibrationReport"]
