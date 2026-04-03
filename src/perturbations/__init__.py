"""
Perturbation generation framework for agent trajectories.

This module generates realistic errors in agent trajectories to test
judge calibration to error criticality.
"""

from .generator import PerturbationGenerator
from .strategies import (
    PlanningErrorStrategy,
    ToolSelectionErrorStrategy,
    ParameterErrorStrategy,
)
from .tool_similarity import ToolSimilarityMatcher

__all__ = [
    "PerturbationGenerator",
    "PlanningErrorStrategy",
    "ToolSelectionErrorStrategy",
    "ParameterErrorStrategy",
    "ToolSimilarityMatcher",
]
