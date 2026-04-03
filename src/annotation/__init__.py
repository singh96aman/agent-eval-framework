"""
Annotation tools for ground truth criticality scoring.

This module provides interfaces for human researchers to annotate
the criticality of perturbations in agent trajectories.
"""

from .tools import (
    AnnotationInterface,
    AnnotationReviewer,
    Annotation,
    load_annotation,
    save_annotation
)

__all__ = [
    'AnnotationInterface',
    'AnnotationReviewer',
    'Annotation',
    'load_annotation',
    'save_annotation'
]
