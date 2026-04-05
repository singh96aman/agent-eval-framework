"""
Metrics for judge calibration analysis.

This module provides functions to compute the Criticality-Calibration Gap (CCG)
and related statistical analyses.
"""

from .ccg import (
    compute_tcs,
    compute_jps,
    compute_ccg,
    CCGAnalysis,
    aggregate_by_condition,
    statistical_analysis,
    compute_ccg_analysis
)
from .criticality_scorer import CriticalityScorer
from .ccg_calculator import CCGCalculator

__all__ = [
    'compute_tcs',
    'compute_jps',
    'compute_ccg',
    'CCGAnalysis',
    'aggregate_by_condition',
    'statistical_analysis',
    'compute_ccg_analysis',
    'CriticalityScorer',
    'CCGCalculator'
]
