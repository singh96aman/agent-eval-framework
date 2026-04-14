"""
PerturbationGeneratorV2: Orchestrator for Section 3 Controlled Perturbations.

This module ties together all generators (placebo, fine-grained, coarse-grained)
and provides unified interface for:
- Slot selection based on diversity, criticality, position
- Impact derivation from Section 2 typed representation
- Routing to appropriate generators
- QC validation

Usage:
    generator = PerturbationGeneratorV2(random_seed=42)
    perturbations = generator.generate_for_trajectory(
        typed_trajectory,
        target_count=3
    )
"""

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.typing.schema import (
    TypedTrajectory,
    TypedStep,
    PerturbableSlot,
)
from src.perturbations.schema import (
    PerturbationClass,
    PerturbationFamily,
    PerturbationType,
    PerturbationRecord,
    PerturbationIndex,
)
from src.perturbations.placebo import (
    get_placebo_generator,
)
from src.perturbations.fine_grained import (
    get_fine_grained_generator,
)
from src.perturbations.coarse_grained import (
    get_coarse_grained_generator,
)
from src.perturbations.qc import PerturbationQC


@dataclass
class SlotCandidate:
    """A candidate slot for perturbation with scoring metadata."""

    step_index: int
    slot: PerturbableSlot
    position: str  # early, middle, late
    critical_path_score: float
    eligible_classes: List[PerturbationClass]
    eligible_families: List[PerturbationFamily]
    score: float = 0.0


class PerturbationGeneratorV2:
    """
    Unified perturbation generator orchestrating all perturbation types.

    Responsibilities:
    - Enumerate perturbable slots from typed trajectories
    - Score and select slots for diverse perturbation coverage
    - Route to appropriate generators
    - Run QC validation
    - Track perturbation index
    """

    def __init__(
        self,
        random_seed: Optional[int] = None,
        llm_client=None,
        enable_qc: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the generator.

        Args:
            random_seed: Random seed for reproducibility
            llm_client: Optional LLM client for Claude-based generators
            enable_qc: Whether to run QC validation
            config: Optional experiment config for new ID scheme
        """
        self.random_seed = random_seed
        self.random = random.Random(random_seed)
        self.llm_client = llm_client
        self.enable_qc = enable_qc
        self.config = config

        # Initialize QC
        self.qc = PerturbationQC() if enable_qc else None

        # Track generation statistics
        self.stats = defaultdict(int)
        self.index = PerturbationIndex()

    def _regenerate_id(self, record: PerturbationRecord) -> PerturbationRecord:
        """Regenerate perturbation_id using new config-based scheme."""
        if self.config:
            from src.utils import generate_perturbation_id
            experiment_id = self.config["experiment"]["id"]
            generator_config = (
                self.config.get("phases", {})
                .get("perturb", {})
                .get("generators", {})
                .get(record.perturbation_class, {})
            )
            record.perturbation_id = generate_perturbation_id(
                experiment_id=experiment_id,
                trajectory_id=record.original_trajectory_id,
                step_idx=record.target_step_index,
                perturbation_type=record.perturbation_type,
                generator_config=generator_config,
            )
        return record

    def generate_for_trajectory(
        self,
        typed_trajectory: TypedTrajectory,
        target_count: int = 3,
        class_weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[PerturbationRecord, TypedTrajectory]]:
        """
        Generate perturbations for a single trajectory.

        Args:
            typed_trajectory: Typed trajectory from Section 2
            target_count: Target number of perturbations
            class_weights: Optional weights for class distribution
                          Default: placebo=0.2, fine_grained=0.5, coarse_grained=0.3

        Returns:
            List of (PerturbationRecord, perturbed_trajectory) tuples
        """
        if class_weights is None:
            class_weights = {
                "placebo": 0.2,
                "fine_grained": 0.5,
                "coarse_grained": 0.3,
            }

        # Enumerate all perturbable slots
        candidates = self._enumerate_slot_candidates(typed_trajectory)

        if not candidates:
            return []

        # Score candidates for diversity
        self._score_candidates(candidates, typed_trajectory)

        # Select perturbations based on target count and distribution
        selected = self._select_perturbations(candidates, target_count, class_weights)

        # Generate perturbations
        results = []
        for candidate, target_class in selected:
            result = self._generate_perturbation(
                typed_trajectory, candidate, target_class
            )
            if result:
                record, perturbed_traj = result

                # Run QC if enabled
                if self.qc:
                    original_step = typed_trajectory.get_step(candidate.step_index)
                    if original_step:
                        record = self.qc.validate(record, perturbed_traj, original_step)

                results.append((record, perturbed_traj))
                self.stats[record.perturbation_class] += 1
                self.stats[record.perturbation_family] += 1
                self.stats["total"] += 1

        return results

    def _enumerate_slot_candidates(
        self, trajectory: TypedTrajectory
    ) -> List[SlotCandidate]:
        """Enumerate all perturbable slots with metadata."""
        candidates = []
        num_steps = len(trajectory.steps)

        for step in trajectory.steps:
            # Determine position
            position = self._get_position(step.step_index, num_steps)

            # Get critical path score
            cps = 0.5
            if step.critical_path_score:
                cps = step.critical_path_score.value

            # Process each perturbable slot
            for slot in step.perturbable_slots:
                eligible_classes, eligible_families = self._get_eligible_perturbations(
                    slot, step
                )

                if eligible_classes:
                    candidates.append(
                        SlotCandidate(
                            step_index=step.step_index,
                            slot=slot,
                            position=position,
                            critical_path_score=cps,
                            eligible_classes=eligible_classes,
                            eligible_families=eligible_families,
                        )
                    )

        # Also add trajectory-level candidates for coarse-grained
        coarse_candidates = self._get_coarse_grained_candidates(trajectory)
        candidates.extend(coarse_candidates)

        return candidates

    def _get_eligible_perturbations(
        self, slot: PerturbableSlot, step: TypedStep
    ) -> Tuple[List[PerturbationClass], List[PerturbationFamily]]:
        """Determine which perturbation classes/families apply to a slot."""
        classes = []
        families = []

        allowed = slot.allowed_perturbation_types

        # Map allowed types to classes and families
        if "data_reference" in allowed:
            classes.append(PerturbationClass.FINE_GRAINED)
            families.append(PerturbationFamily.DATA_REFERENCE)
            # Placebo can also apply to data_reference slots
            classes.append(PerturbationClass.PLACEBO)

        if "parameter" in allowed:
            classes.append(PerturbationClass.FINE_GRAINED)
            families.append(PerturbationFamily.PARAMETER)
            classes.append(PerturbationClass.PLACEBO)

        if "tool_selection" in allowed:
            # Fine-grained: near-neighbor
            classes.append(PerturbationClass.FINE_GRAINED)
            # Coarse-grained: wrong family
            classes.append(PerturbationClass.COARSE_GRAINED)
            families.append(PerturbationFamily.TOOL_SELECTION)

        if "structural" in allowed:
            classes.append(PerturbationClass.COARSE_GRAINED)
            families.append(PerturbationFamily.STRUCTURAL)

        # Deduplicate
        classes = list(set(classes))
        families = list(set(families))

        return classes, families

    def _get_coarse_grained_candidates(
        self, trajectory: TypedTrajectory
    ) -> List[SlotCandidate]:
        """Get trajectory-level candidates for coarse-grained perturbations."""
        candidates = []
        num_steps = len(trajectory.steps)

        # Terminal flag candidates: non-terminal steps with artifacts
        for step in trajectory.steps:
            if step.is_terminal_step:
                continue

            # Check if step can be false terminal
            if step.produced_artifacts and step.step_role in [
                "reasoning",
                "extraction",
                "decision",
            ]:

                position = self._get_position(step.step_index, num_steps)
                cps = (
                    step.critical_path_score.value if step.critical_path_score else 0.5
                )

                # Create a pseudo-slot for terminal_flag
                pseudo_slot = PerturbableSlot(
                    slot="terminal_flags",
                    value_type="boolean",
                    current_value={"is_terminal_step": False},
                    allowed_perturbation_types=["terminal_flag"],
                )

                candidates.append(
                    SlotCandidate(
                        step_index=step.step_index,
                        slot=pseudo_slot,
                        position=position,
                        critical_path_score=cps,
                        eligible_classes=[PerturbationClass.COARSE_GRAINED],
                        eligible_families=[PerturbationFamily.TERMINAL_FLAG],
                    )
                )

        # Structural candidates: steps with high critical path score
        for step in trajectory.steps:
            cps = step.critical_path_score.value if step.critical_path_score else 0.5
            if cps >= 0.6 and step.step_index > 0:
                position = self._get_position(step.step_index, num_steps)

                pseudo_slot = PerturbableSlot(
                    slot="step_structure",
                    value_type="structural",
                    current_value={"step_index": step.step_index},
                    allowed_perturbation_types=["structural"],
                )

                candidates.append(
                    SlotCandidate(
                        step_index=step.step_index,
                        slot=pseudo_slot,
                        position=position,
                        critical_path_score=cps,
                        eligible_classes=[PerturbationClass.COARSE_GRAINED],
                        eligible_families=[PerturbationFamily.STRUCTURAL],
                    )
                )

        return candidates

    def _get_position(self, step_index: int, num_steps: int) -> str:
        """Determine position label for a step."""
        if num_steps <= 3:
            if step_index == 0:
                return "early"
            elif step_index == num_steps - 1:
                return "late"
            else:
                return "middle"

        third = num_steps / 3
        if step_index < third:
            return "early"
        elif step_index < 2 * third:
            return "middle"
        else:
            return "late"

    def _score_candidates(
        self,
        candidates: List[SlotCandidate],
        trajectory: TypedTrajectory,
    ):
        """Score candidates for selection priority."""
        # Track what we've already selected to encourage diversity
        position_counts = defaultdict(int)
        family_counts = defaultdict(int)
        step_counts = defaultdict(int)

        for candidate in candidates:
            # Base score from critical path
            score = candidate.critical_path_score

            # Diversity bonus - prefer under-represented positions
            pos_count = position_counts[candidate.position]
            score += max(0, 0.3 - pos_count * 0.1)

            # Diversity bonus - prefer under-represented families
            for family in candidate.eligible_families:
                fam_count = family_counts[family]
                score += max(0, 0.2 - fam_count * 0.05)

            # Penalty for same step (encourage spreading across steps)
            step_count = step_counts[candidate.step_index]
            score -= step_count * 0.2

            candidate.score = score

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

    def _select_perturbations(
        self,
        candidates: List[SlotCandidate],
        target_count: int,
        class_weights: Dict[str, float],
    ) -> List[Tuple[SlotCandidate, PerturbationClass]]:
        """Select perturbations to meet target distribution."""
        # Calculate target counts per class
        targets = {
            PerturbationClass.PLACEBO: int(
                target_count * class_weights.get("placebo", 0.2)
            ),
            PerturbationClass.FINE_GRAINED: int(
                target_count * class_weights.get("fine_grained", 0.5)
            ),
            PerturbationClass.COARSE_GRAINED: int(
                target_count * class_weights.get("coarse_grained", 0.3)
            ),
        }

        # Ensure at least 1 of each if target_count >= 3
        if target_count >= 3:
            for cls in targets:
                targets[cls] = max(1, targets[cls])

        # Adjust to match total
        total_target = sum(targets.values())
        if total_target < target_count:
            # Add remainder to fine_grained
            targets[PerturbationClass.FINE_GRAINED] += target_count - total_target

        selected = []
        used_steps = set()
        counts = defaultdict(int)

        for candidate in candidates:
            if len(selected) >= target_count:
                break

            # Find a class that needs more perturbations
            target_class = None
            for cls in candidate.eligible_classes:
                if counts[cls] < targets[cls]:
                    target_class = cls
                    break

            if target_class is None:
                continue

            # Prefer not reusing same step
            if candidate.step_index in used_steps:
                # Allow if we really need this class
                if counts[target_class] < targets[target_class] - 1:
                    continue

            selected.append((candidate, target_class))
            used_steps.add(candidate.step_index)
            counts[target_class] += 1

        return selected

    def _generate_perturbation(
        self,
        trajectory: TypedTrajectory,
        candidate: SlotCandidate,
        target_class: PerturbationClass,
    ) -> Optional[Tuple[PerturbationRecord, TypedTrajectory]]:
        """Generate a single perturbation."""
        step = trajectory.get_step(candidate.step_index)
        if not step:
            return None

        try:
            if target_class == PerturbationClass.PLACEBO:
                return self._generate_placebo(trajectory, step, candidate)
            elif target_class == PerturbationClass.FINE_GRAINED:
                return self._generate_fine_grained(trajectory, step, candidate)
            elif target_class == PerturbationClass.COARSE_GRAINED:
                return self._generate_coarse_grained(trajectory, candidate)
        except Exception:
            self.stats["errors"] += 1
            return None

        return None

    def _generate_placebo(
        self,
        trajectory: TypedTrajectory,
        step: TypedStep,
        candidate: SlotCandidate,
    ) -> Optional[Tuple[PerturbationRecord, TypedTrajectory]]:
        """Generate placebo perturbation."""
        # Choose placebo type based on slot
        value_type = candidate.slot.value_type

        if value_type in ["string", "search_query"]:
            ptype = self.random.choice(
                [
                    PerturbationType.PARAPHRASE,
                    PerturbationType.SYNONYM,
                ]
            )
        elif value_type in ["object", "json_object"]:
            ptype = self.random.choice(
                [
                    PerturbationType.FORMATTING,
                    PerturbationType.REORDER_ARGS,
                ]
            )
        else:
            ptype = PerturbationType.FORMATTING

        generator = get_placebo_generator(ptype)
        record = generator.generate(step, trajectory.trajectory_id)

        if record is None:
            return None

        # Regenerate ID using new scheme
        record = self._regenerate_id(record)

        # Create perturbed trajectory
        perturbed_traj = self._apply_step_perturbation(
            trajectory, step.step_index, record
        )

        return record, perturbed_traj

    def _generate_fine_grained(
        self,
        trajectory: TypedTrajectory,
        step: TypedStep,
        candidate: SlotCandidate,
    ) -> Optional[Tuple[PerturbationRecord, TypedTrajectory]]:
        """Generate fine-grained perturbation."""
        # Choose family based on slot
        family = self.random.choice(candidate.eligible_families)

        # Choose type based on family
        if family == PerturbationFamily.DATA_REFERENCE:
            ptype = self.random.choice(
                [
                    PerturbationType.WRONG_VALUE,
                    PerturbationType.OFF_BY_ONE,
                    PerturbationType.TYPO_IN_ID,
                ]
            )
        elif family == PerturbationFamily.PARAMETER:
            ptype = self.random.choice(
                [
                    PerturbationType.THRESHOLD_SHIFT,
                    PerturbationType.QUERY_DRIFT,
                    PerturbationType.WRONG_PARAMETER,
                ]
            )
        elif family == PerturbationFamily.TOOL_SELECTION:
            ptype = PerturbationType.NEAR_NEIGHBOR_TOOL
        else:
            return None

        generator = get_fine_grained_generator(
            family, ptype, random_seed=self.random_seed, llm_client=self.llm_client
        )
        record = generator.generate(step, trajectory.trajectory_id, trajectory)

        if record is None:
            return None

        # Regenerate ID using new scheme
        record = self._regenerate_id(record)

        # Create perturbed trajectory
        perturbed_traj = self._apply_step_perturbation(
            trajectory, step.step_index, record
        )

        return record, perturbed_traj

    def _generate_coarse_grained(
        self,
        trajectory: TypedTrajectory,
        candidate: SlotCandidate,
    ) -> Optional[Tuple[PerturbationRecord, TypedTrajectory]]:
        """Generate coarse-grained perturbation."""
        family = self.random.choice(candidate.eligible_families)

        if family == PerturbationFamily.STRUCTURAL:
            # NOTE: SKIPPED_PREREQUISITE removed - changes trajectory length
            ptype = PerturbationType.WRONG_PLAN
        elif family == PerturbationFamily.TERMINAL_FLAG:
            ptype = self.random.choice(
                [
                    PerturbationType.FALSE_TERMINAL,
                    PerturbationType.PREMATURE_TERMINATION,
                ]
            )
        elif family == PerturbationFamily.TOOL_SELECTION:
            ptype = PerturbationType.WRONG_TOOL_FAMILY
        else:
            return None

        # Get prompt from config if available
        prompt_template = None
        if self.config and ptype == PerturbationType.WRONG_PLAN:
            prompts = (
                self.config.get("phases", {})
                .get("perturb", {})
                .get("generators", {})
                .get("coarse_grained", {})
                .get("prompts", {})
            )
            prompt_template = prompts.get("wrong_plan")

        generator = get_coarse_grained_generator(
            family,
            ptype,
            random_seed=self.random_seed,
            llm_client=self.llm_client,
            prompt_template=prompt_template,
        )

        result = generator.generate(trajectory)
        if result:
            record, perturbed_traj = result
            # Regenerate ID using new scheme
            record = self._regenerate_id(record)
            return record, perturbed_traj
        return result

    def _apply_step_perturbation(
        self,
        trajectory: TypedTrajectory,
        step_index: int,
        record: PerturbationRecord,
    ) -> TypedTrajectory:
        """Apply a step-level perturbation to create perturbed trajectory."""
        perturbed = deepcopy(trajectory)

        # Find the step and apply the perturbation
        for i, step in enumerate(perturbed.steps):
            if step.step_index == step_index:
                # Update the slot value
                slot_path = record.target_slot
                self._set_nested_value(step, slot_path, record.perturbed_value)

                # If raw_text was the target, update it directly
                if slot_path == "raw_text":
                    step.raw_text = str(record.perturbed_value)

                break

        # Update trajectory ID
        perturbed.trajectory_id = (
            f"{trajectory.trajectory_id}_"
            f"{record.perturbation_class}_"
            f"{record.perturbation_type}"
        )

        return perturbed

    def _set_nested_value(self, obj: Any, path: str, value: Any):
        """Set a value at a nested path (e.g., 'tool_arguments.query')."""
        parts = path.split(".")
        current = obj

        for i, part in enumerate(parts[:-1]):
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return  # Path not found

        # Set the final value
        final_part = parts[-1]
        if hasattr(current, final_part):
            setattr(current, final_part, value)
        elif isinstance(current, dict):
            current[final_part] = value

    def get_stats(self) -> Dict[str, int]:
        """Get generation statistics."""
        return dict(self.stats)

    def get_index(self) -> PerturbationIndex:
        """Get the perturbation index."""
        return self.index


def generate_perturbations_for_batch(
    trajectories: List[TypedTrajectory],
    target_per_trajectory: int = 3,
    random_seed: Optional[int] = None,
    llm_client=None,
    verbose: bool = False,
    parallelism: int = 1,
    on_batch_save: Optional[Callable[[List[Tuple[PerturbationRecord, TypedTrajectory]]], None]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Tuple[PerturbationRecord, TypedTrajectory]], PerturbationIndex]:
    """
    Generate perturbations for a batch of trajectories.

    Args:
        trajectories: List of typed trajectories
        target_per_trajectory: Target perturbations per trajectory
        random_seed: Random seed for reproducibility
        llm_client: Optional LLM client
        verbose: Whether to print progress
        parallelism: Number of parallel workers (default 1 = sequential)
        on_batch_save: Optional callback to save perturbations after each batch.
                       Called with list of (record, perturbed_trajectory) tuples.
        config: Optional experiment config for new ID scheme

    Returns:
        Tuple of (list of (record, perturbed_trajectory), PerturbationIndex)
    """
    from src.utils.parallel import parallel_map_with_index, ParallelConfig

    index = PerturbationIndex()
    all_results = []
    # Map trajectory_id to trajectory for callback use
    traj_by_id = {t.trajectory_id: t for t in trajectories}

    def process_trajectory(idx: int, trajectory: TypedTrajectory):
        """Process a single trajectory."""
        # Each call gets unique seed for reproducibility
        thread_seed = (random_seed or 42) + idx if random_seed else None
        generator = PerturbationGeneratorV2(
            random_seed=thread_seed,
            llm_client=llm_client,
            config=config,
        )
        return generator.generate_for_trajectory(
            trajectory,
            target_count=target_per_trajectory,
        )

    def handle_batch_complete(batch_results):
        """Handle batch completion - save incrementally."""
        batch_perturbations = []
        for (idx, trajectory), result, error in batch_results:
            if error or not result:
                continue
            # result is (idx, actual_result) from indexed_func wrapper
            _, actual_result = result
            for record, perturbed_traj in actual_result:
                batch_perturbations.append((record, perturbed_traj))
                # Also update index
                index.add_perturbation(
                    record,
                    trajectory.benchmark,
                    f"data/perturbed/{trajectory.benchmark}_perturbed.json",
                )

        if batch_perturbations:
            all_results.extend(batch_perturbations)
            if on_batch_save:
                on_batch_save(batch_perturbations)
                if verbose:
                    print(f"    Saved {len(batch_perturbations)} perturbations")

    parallel_config = ParallelConfig(
        workers=parallelism,
        batch_size=50,
        rate_limit_delay=0.1,
        verbose=verbose,
        on_batch_complete=handle_batch_complete,
    )

    parallel_map_with_index(
        process_trajectory,
        trajectories,
        config=parallel_config,
        desc="Generating perturbations",
    )

    if verbose:
        print("\nGeneration complete:")
        print(index.get_distribution_report())

    return all_results, index
