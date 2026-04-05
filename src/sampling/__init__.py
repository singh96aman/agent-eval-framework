"""
Sampling module for primary perturbation selection.

Provides stratified sampling to select primary perturbations
for human annotation based on quality scores and condition quotas.
"""

from .primary_selector import PrimarySelector

__all__ = ["PrimarySelector"]
