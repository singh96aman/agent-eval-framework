"""
Batch-level balancing for perturbation distribution.

Ensures that across all trajectories, the overall distribution
matches target percentages (20% placebo, 50% fine-grained, 30% coarse-grained).
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.perturbations.schema import (
    PerturbationRecord,
)


class BatchDistribution:
    """Tracks and manages batch-level distribution."""

    def __init__(
        self,
        total_target: int,
        class_weights: Optional[Dict[str, float]] = None,
        family_weights: Optional[Dict[str, Dict[str, float]]] = None,
        position_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize distribution tracker.

        Args:
            total_target: Total perturbations target
            class_weights: Target weights by class (default: 20/50/30)
            family_weights: Target weights by family within class
            position_weights: Target weights by position
        """
        self.total_target = total_target

        self.class_weights = class_weights or {
            "placebo": 0.20,
            "fine_grained": 0.50,
            "coarse_grained": 0.30,
        }

        self.family_weights = family_weights or {
            "fine_grained": {
                "data_reference": 0.35,
                "parameter": 0.35,
                "tool_selection": 0.30,
            },
            "coarse_grained": {
                "structural": 0.60,
                "terminal_flag": 0.15,
                "tool_selection": 0.25,
            },
        }

        self.position_weights = position_weights or {
            "early": 0.30,
            "middle": 0.50,
            "late": 0.20,
        }

        # Compute targets
        self.class_targets = {
            cls: int(total_target * weight)
            for cls, weight in self.class_weights.items()
        }

        # Track current counts
        self.class_counts = defaultdict(int)
        self.family_counts = defaultdict(lambda: defaultdict(int))
        self.position_counts = defaultdict(int)

    def get_class_needs(self) -> Dict[str, int]:
        """Get remaining needed count per class."""
        return {
            cls: max(0, target - self.class_counts[cls])
            for cls, target in self.class_targets.items()
        }

    def get_most_needed_class(self) -> str:
        """Get the class that is furthest behind its target."""
        needs = self.get_class_needs()
        # Return class with highest remaining need
        return max(needs.keys(), key=lambda c: needs[c])

    def suggest_class_for_trajectory(
        self,
        slot_idx: int,
        eligible_classes: List[str],
    ) -> Optional[str]:
        """
        Suggest which class to generate for a trajectory slot.

        Balances across the batch by preferring under-represented classes.
        """
        if not eligible_classes:
            return None

        needs = self.get_class_needs()

        # Filter to eligible classes
        eligible_needs = {cls: needs.get(cls, 0) for cls in eligible_classes}

        if not eligible_needs:
            return None

        # Return class with highest remaining need
        return max(eligible_needs.keys(), key=lambda c: eligible_needs[c])

    def record(
        self,
        record: PerturbationRecord,
        position: Optional[str] = None,
    ):
        """Record a generated perturbation."""
        self.class_counts[record.perturbation_class] += 1

        if record.perturbation_family:
            self.family_counts[record.perturbation_class][
                record.perturbation_family
            ] += 1

        if position:
            self.position_counts[position] += 1

    def get_distribution_report(self) -> Dict[str, Any]:
        """Get current distribution as a report."""
        total = sum(self.class_counts.values())

        class_dist = {}
        for cls, count in self.class_counts.items():
            target_pct = self.class_weights.get(cls, 0) * 100
            actual_pct = (count / total * 100) if total > 0 else 0
            class_dist[cls] = {
                "count": count,
                "target_pct": target_pct,
                "actual_pct": actual_pct,
                "gap": actual_pct - target_pct,
            }

        return {
            "total": total,
            "target": self.total_target,
            "by_class": class_dist,
            "by_family": dict(self.family_counts),
            "by_position": dict(self.position_counts),
        }

    def is_balanced(self, tolerance: float = 0.05) -> bool:
        """
        Check if distribution is within tolerance of targets.

        Args:
            tolerance: Acceptable deviation from target (e.g., 0.05 = 5%)
        """
        total = sum(self.class_counts.values())
        if total == 0:
            return False

        for cls, target_weight in self.class_weights.items():
            actual_weight = self.class_counts[cls] / total
            if abs(actual_weight - target_weight) > tolerance:
                return False

        return True


class PerturbationBalancer:
    """
    Balances perturbation generation across a batch of trajectories.

    Strategies:
    1. Pre-allocation: Assign target classes to trajectories before generation
    2. Dynamic balancing: Adjust class selection during generation
    3. Post-hoc rebalancing: Sample/oversample after generation
    """

    def __init__(
        self,
        total_target: int,
        class_weights: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize balancer.

        Args:
            total_target: Total perturbations to generate
            class_weights: Target distribution by class
            random_seed: Random seed for reproducibility
        """
        self.total_target = total_target
        self.class_weights = class_weights or {
            "placebo": 0.20,
            "fine_grained": 0.50,
            "coarse_grained": 0.30,
        }
        self.random = random.Random(random_seed)
        self.distribution = BatchDistribution(total_target, class_weights)

    def pre_allocate(
        self,
        num_trajectories: int,
        per_trajectory: int = 3,
    ) -> List[List[str]]:
        """
        Pre-allocate class assignments for trajectories.

        Returns list of class assignments for each trajectory.
        Each inner list has `per_trajectory` class names.
        """
        total = num_trajectories * per_trajectory
        if total > self.total_target:
            total = self.total_target

        # Calculate how many of each class we need
        class_counts = {
            cls: int(total * weight) for cls, weight in self.class_weights.items()
        }

        # Adjust to match total exactly
        diff = total - sum(class_counts.values())
        if diff > 0:
            # Add remainder to fine_grained
            class_counts["fine_grained"] += diff

        # Create pool of class assignments
        pool = []
        for cls, count in class_counts.items():
            pool.extend([cls] * count)

        # Shuffle
        self.random.shuffle(pool)

        # Distribute across trajectories
        allocations = []
        idx = 0
        for _ in range(num_trajectories):
            traj_alloc = []
            for _ in range(per_trajectory):
                if idx < len(pool):
                    traj_alloc.append(pool[idx])
                    idx += 1
            allocations.append(traj_alloc)

        return allocations

    def rebalance_sample(
        self,
        records: List[PerturbationRecord],
        target_count: Optional[int] = None,
    ) -> List[PerturbationRecord]:
        """
        Post-hoc rebalancing by sampling.

        Samples from over-represented classes and keeps all from
        under-represented classes to achieve target distribution.
        """
        target = target_count or self.total_target

        # Group by class
        by_class = defaultdict(list)
        for record in records:
            by_class[record.perturbation_class].append(record)

        # Calculate target counts
        class_targets = {
            cls: int(target * weight) for cls, weight in self.class_weights.items()
        }

        # Sample from each class
        result = []
        for cls, cls_records in by_class.items():
            cls_target = class_targets.get(cls, 0)
            if len(cls_records) <= cls_target:
                # Under-represented: keep all
                result.extend(cls_records)
            else:
                # Over-represented: sample
                sampled = self.random.sample(cls_records, cls_target)
                result.extend(sampled)

        return result

    def create_batched_generator(
        self,
        trajectories: List[Any],
        generator: Any,
        per_trajectory: int = 3,
    ):
        """
        Create a generator that yields balanced perturbations.

        Uses pre-allocation strategy for batch-level balance.
        """
        # Pre-allocate classes
        allocations = self.pre_allocate(len(trajectories), per_trajectory)

        for trajectory, class_alloc in zip(trajectories, allocations):
            # Generate perturbations with specified class targets
            for target_class in class_alloc:
                yield trajectory, target_class


def balance_perturbation_batch(
    records: List[PerturbationRecord],
    total_target: int,
    class_weights: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
) -> Tuple[List[PerturbationRecord], Dict[str, Any]]:
    """
    Balance a batch of perturbations to match target distribution.

    Args:
        records: Generated perturbation records
        total_target: Target total count
        class_weights: Target distribution
        random_seed: Random seed

    Returns:
        Tuple of (balanced records, balance report)
    """
    balancer = PerturbationBalancer(
        total_target=total_target,
        class_weights=class_weights,
        random_seed=random_seed,
    )

    # Get initial distribution
    initial_dist = BatchDistribution(total_target, class_weights)
    for record in records:
        initial_dist.record(record)

    # Rebalance
    balanced = balancer.rebalance_sample(records, total_target)

    # Get final distribution
    final_dist = BatchDistribution(total_target, class_weights)
    for record in balanced:
        final_dist.record(record)

    report = {
        "initial": initial_dist.get_distribution_report(),
        "final": final_dist.get_distribution_report(),
        "removed": len(records) - len(balanced),
        "is_balanced": final_dist.is_balanced(),
    }

    return balanced, report
