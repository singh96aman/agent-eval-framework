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
    save_annotation,
)
from .stratified_sampler import (
    StratifiedAnnotationSampler,
    load_annotations_from_file,
    validate_annotations,
)

__all__ = [
    "AnnotationInterface",
    "AnnotationReviewer",
    "Annotation",
    "load_annotation",
    "save_annotation",
    "StratifiedAnnotationSampler",
    "load_annotations_from_file",
    "validate_annotations",
]
