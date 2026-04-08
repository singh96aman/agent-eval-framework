"""
Typed representation module for agent trajectories.

This module enriches raw trajectories with typed fields for controlled perturbation:
- Step roles and terminal flags
- Entity extraction
- Dependency graphs (direct + transitive)
- Artifact tracking
- Perturbable slots
- Critical path scoring
"""

from src.typing.schema import (
    TypedStep,
    TypedTrajectory,
    DependencyEdge,
    Artifact,
    PerturbableSlot,
    ProvenanceField,
    StepRole,
    ArtifactType,
    DependencyType,
    ValueType,
)
from src.typing.typer import TrajectoryTyper

__all__ = [
    "TypedStep",
    "TypedTrajectory",
    "DependencyEdge",
    "Artifact",
    "PerturbableSlot",
    "ProvenanceField",
    "StepRole",
    "ArtifactType",
    "DependencyType",
    "ValueType",
    "TrajectoryTyper",
]
