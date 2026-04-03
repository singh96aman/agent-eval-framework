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
    statistical_analysis
)

__all__ = [
    'compute_tcs',
    'compute_jps',
    'compute_ccg',
    'CCGAnalysis',
    'aggregate_by_condition',
    'statistical_analysis'
]
