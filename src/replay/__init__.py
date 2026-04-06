"""
Replay module for computing Outcome Degradation (OD).

OD = (baseline_outcome - perturbed_outcome) / 100

This module grades final answers from baseline and perturbed trajectories
to measure actual task outcome impact.
"""

from src.replay.od_scorer import ODScorer, ODResult

__all__ = ["ODScorer", "ODResult"]
