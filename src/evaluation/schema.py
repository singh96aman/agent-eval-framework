"""
Schema definitions for Section 4: Evaluation Unit.

Defines EvaluationUnit, BaselineData, PerturbedData, and supporting dataclasses
for the atomic unit of analysis per 4_Requirements_EvaluationUnit.MD.

The Evaluation Unit packages a baseline trajectory with its perturbed variant,
along with all metadata needed for evaluation by humans, LLM judges, and
outcome verification.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EvaluationCapabilities:
    """
    Flags for what evaluation methods are possible for this unit.

    These capability flags determine the replay tier and what kinds
    of outcome evidence can be collected.

    Attributes:
        has_objective_verifier: Automated verification exists (test suite, exact match, etc.)
        can_replay: Can re-execute the trajectory from scratch
        can_regenerate_downstream: Can continue generation from a perturbed prefix
        environment_accessible: External APIs/tools are available and deterministic enough
        ground_truth_available: Expected answer or success label exists
    """

    has_objective_verifier: bool
    can_replay: bool
    can_regenerate_downstream: bool
    environment_accessible: bool
    ground_truth_available: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_objective_verifier": self.has_objective_verifier,
            "can_replay": self.can_replay,
            "can_regenerate_downstream": self.can_regenerate_downstream,
            "environment_accessible": self.environment_accessible,
            "ground_truth_available": self.ground_truth_available,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationCapabilities":
        """Create EvaluationCapabilities from dictionary."""
        return cls(
            has_objective_verifier=data["has_objective_verifier"],
            can_replay=data["can_replay"],
            can_regenerate_downstream=data["can_regenerate_downstream"],
            environment_accessible=data["environment_accessible"],
            ground_truth_available=data["ground_truth_available"],
        )

    def compute_replay_tier(self) -> Optional[int]:
        """
        Compute replay tier from capabilities.

        Returns:
            1: Full replay possible
            2: Downstream regeneration possible
            3: Final-answer grading only
            None: No outcome evidence possible
        """
        # Tier 1: Full replay possible
        if (
            self.can_replay
            and self.environment_accessible
            and self.has_objective_verifier
        ):
            return 1

        # Tier 2: Can regenerate downstream from perturbed prefix
        if self.can_regenerate_downstream and self.has_objective_verifier:
            return 2

        # Tier 3: Final-answer grading only
        if self.ground_truth_available or self.has_objective_verifier:
            return 3

        # Fallback: no outcome evidence possible
        return None


@dataclass
class DerivedCache:
    """
    Denormalized fields for quick filtering.

    These fields are derived from canonical objects (baseline trajectory
    and perturbation record). They should be regenerated via build script,
    never hand-edited.

    Attributes:
        _warning: Reminder that these fields are derived
        baseline_outcome: From baseline.trajectory.baseline_outcome
        baseline_num_steps: From len(baseline.trajectory.steps)
        perturbed_num_steps: From len(perturbed.trajectory.steps)
        perturbation_class: From perturbed.perturbation_record.perturbation_class
        perturbation_family: From perturbed.perturbation_record.perturbation_family
        perturbation_type: From perturbed.perturbation_record.perturbation_type
        target_step_canonical_id: From perturbed.perturbation_record
        expected_impact: From perturbed.perturbation_record.expected_impact
        expected_detectability: From perturbed.perturbation_record.expected_detectability
    """

    baseline_outcome: float
    baseline_num_steps: int
    perturbed_num_steps: int
    perturbation_class: str
    perturbation_family: str
    perturbation_type: str
    target_step_canonical_id: str
    expected_impact: int
    expected_detectability: int
    _warning: str = "Derived from canonical objects. Regenerate, never hand-edit."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "_warning": self._warning,
            "baseline_outcome": self.baseline_outcome,
            "baseline_num_steps": self.baseline_num_steps,
            "perturbed_num_steps": self.perturbed_num_steps,
            "perturbation_class": self.perturbation_class,
            "perturbation_family": self.perturbation_family,
            "perturbation_type": self.perturbation_type,
            "target_step_canonical_id": self.target_step_canonical_id,
            "expected_impact": self.expected_impact,
            "expected_detectability": self.expected_detectability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DerivedCache":
        """Create DerivedCache from dictionary."""
        return cls(
            _warning=data.get(
                "_warning",
                "Derived from canonical objects. Regenerate, never hand-edit.",
            ),
            baseline_outcome=data["baseline_outcome"],
            baseline_num_steps=data["baseline_num_steps"],
            perturbed_num_steps=data["perturbed_num_steps"],
            perturbation_class=data["perturbation_class"],
            perturbation_family=data["perturbation_family"],
            perturbation_type=data["perturbation_type"],
            target_step_canonical_id=data["target_step_canonical_id"],
            expected_impact=data["expected_impact"],
            expected_detectability=data["expected_detectability"],
        )


@dataclass
class BlindingAssignment:
    """
    A/B assignment for blinded evaluation.

    This object lives ONLY in the blinding key file (private/blinding_key.json),
    never in canonical evaluation units or evaluator views.

    Attributes:
        evaluation_unit_id: The evaluation unit this assignment belongs to
        trajectory_a_variant_id: Variant ID assigned to position A
        trajectory_b_variant_id: Variant ID assigned to position B
        is_a_baseline: True if trajectory A is the baseline (PRIVATE)
        is_a_perturbed: True if trajectory A is the perturbed version (PRIVATE)
    """

    evaluation_unit_id: str
    trajectory_a_variant_id: str
    trajectory_b_variant_id: str
    is_a_baseline: bool
    is_a_perturbed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "evaluation_unit_id": self.evaluation_unit_id,
            "trajectory_a_variant_id": self.trajectory_a_variant_id,
            "trajectory_b_variant_id": self.trajectory_b_variant_id,
            "is_a_baseline": self.is_a_baseline,
            "is_a_perturbed": self.is_a_perturbed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlindingAssignment":
        """Create BlindingAssignment from dictionary."""
        return cls(
            evaluation_unit_id=data["evaluation_unit_id"],
            trajectory_a_variant_id=data["trajectory_a_variant_id"],
            trajectory_b_variant_id=data["trajectory_b_variant_id"],
            is_a_baseline=data["is_a_baseline"],
            is_a_perturbed=data["is_a_perturbed"],
        )


@dataclass
class BaselineData:
    """
    Wrapper for baseline trajectory data.

    Contains the unperturbed typed trajectory which serves as the
    control/reference for comparison.

    Attributes:
        trajectory_variant_id: Unique ID for this baseline instance (e.g., gaia_122::base)
        trajectory: The full TypedTrajectory as a dict (SOURCE OF TRUTH)
    """

    trajectory_variant_id: str
    trajectory: Dict[str, Any]  # TypedTrajectory as dict

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trajectory_variant_id": self.trajectory_variant_id,
            "trajectory": self.trajectory,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineData":
        """Create BaselineData from dictionary."""
        return cls(
            trajectory_variant_id=data["trajectory_variant_id"],
            trajectory=data["trajectory"],
        )


@dataclass
class PerturbedData:
    """
    Wrapper for perturbed trajectory data.

    Contains the typed trajectory with exactly one perturbation applied,
    plus the full perturbation record metadata.

    Attributes:
        trajectory_variant_id: Unique ID for this perturbed instance (e.g., gaia_122::pert::001)
        trajectory: The full TypedTrajectory with perturbation applied as a dict (SOURCE OF TRUTH)
        perturbation_record: The full PerturbationRecord as a dict (SOURCE OF TRUTH)
    """

    trajectory_variant_id: str
    trajectory: Dict[str, Any]  # TypedTrajectory as dict
    perturbation_record: Dict[str, Any]  # PerturbationRecord as dict

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trajectory_variant_id": self.trajectory_variant_id,
            "trajectory": self.trajectory,
            "perturbation_record": self.perturbation_record,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerturbedData":
        """Create PerturbedData from dictionary."""
        return cls(
            trajectory_variant_id=data["trajectory_variant_id"],
            trajectory=data["trajectory"],
            perturbation_record=data["perturbation_record"],
        )


@dataclass
class EvaluationUnit:
    """
    The atomic unit of analysis for the study.

    Packages a baseline trajectory with its perturbed variant, along with
    all metadata needed for evaluation by humans, LLM judges, and outcome
    verification.

    Design Principles:
    1. Self-contained: Contains everything needed to evaluate it
    2. Paired: Always links baseline and perturbed versions for direct comparison
    3. Blinded: Supports blinded evaluation (blinding info stored separately)
    4. Traceable: Full provenance from original trajectory through perturbation
    5. Single source of truth: Canonical data in one place; derived fields regenerated

    Attributes:
        evaluation_unit_id: Unique identifier (e.g., eval::gaia_122::001)
        experiment_id: ID of the experiment this unit belongs to
        created_at: ISO 8601 timestamp of creation
        source_trajectory_id: Immutable reference to original trajectory (e.g., gaia_122)
        benchmark: Benchmark name (toolbench | gaia | swebench)
        task_id: Task identifier within benchmark
        task_text: The task description text
        baseline: Baseline trajectory data (SOURCE OF TRUTH)
        perturbed: Perturbed trajectory data with perturbation record (SOURCE OF TRUTH)
        derived_cache: Denormalized fields for filtering (regenerate, never hand-edit)
        evaluation_capabilities: Flags for what evaluation is possible
        replay_tier: Derived from capabilities (1, 2, 3, or None)
        blinding: A/B assignment for blinded evaluation
    """

    # === Identity ===
    evaluation_unit_id: str
    experiment_id: str
    created_at: str  # ISO 8601

    # === Source Reference ===
    source_trajectory_id: str
    benchmark: str
    task_id: str
    task_text: str

    # === Trajectory Data (CANONICAL) ===
    baseline: BaselineData
    perturbed: PerturbedData

    # === Derived Data ===
    derived_cache: DerivedCache
    evaluation_capabilities: EvaluationCapabilities

    # === Evaluation Configuration ===
    replay_tier: Optional[int]
    blinding: BlindingAssignment

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "evaluation_unit_id": self.evaluation_unit_id,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "source_trajectory_id": self.source_trajectory_id,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "task_text": self.task_text,
            "baseline": self.baseline.to_dict(),
            "perturbed": self.perturbed.to_dict(),
            "derived_cache": self.derived_cache.to_dict(),
            "evaluation_capabilities": self.evaluation_capabilities.to_dict(),
            "replay_tier": self.replay_tier,
            "blinding": self.blinding.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationUnit":
        """Create EvaluationUnit from dictionary."""
        return cls(
            evaluation_unit_id=data["evaluation_unit_id"],
            experiment_id=data["experiment_id"],
            created_at=data["created_at"],
            source_trajectory_id=data["source_trajectory_id"],
            benchmark=data["benchmark"],
            task_id=data["task_id"],
            task_text=data["task_text"],
            baseline=BaselineData.from_dict(data["baseline"]),
            perturbed=PerturbedData.from_dict(data["perturbed"]),
            derived_cache=DerivedCache.from_dict(data["derived_cache"]),
            evaluation_capabilities=EvaluationCapabilities.from_dict(
                data["evaluation_capabilities"]
            ),
            replay_tier=data.get("replay_tier"),
            blinding=BlindingAssignment.from_dict(data["blinding"]),
        )

    def get_baseline_trajectory_id(self) -> str:
        """Get the baseline trajectory variant ID."""
        return self.baseline.trajectory_variant_id

    def get_perturbed_trajectory_id(self) -> str:
        """Get the perturbed trajectory variant ID."""
        return self.perturbed.trajectory_variant_id

    def get_perturbation_id(self) -> str:
        """Get the perturbation ID from the perturbation record."""
        return self.perturbed.perturbation_record.get("perturbation_id", "")

    def get_target_step_index(self) -> int:
        """Get the target step index from the perturbation record."""
        return self.perturbed.perturbation_record.get("target_step_index", -1)

    def is_placebo(self) -> bool:
        """Check if this is a placebo perturbation."""
        return self.derived_cache.perturbation_class == "placebo"

    def is_high_impact(self) -> bool:
        """Check if this is a high-impact perturbation (expected_impact=3)."""
        return self.derived_cache.expected_impact == 3

    def is_subtle(self) -> bool:
        """Check if this is a subtle perturbation (expected_detectability=0 or 1)."""
        return self.derived_cache.expected_detectability <= 1
