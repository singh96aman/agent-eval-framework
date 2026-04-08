"""
Primary Selector - selects primary perturbations for human annotation.

Uses stratified sampling with quality tiers and condition quotas to
ensure balanced coverage across perturbation types and positions.
"""

import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class PrimarySelector:
    """
    Selects primary perturbations using stratified quota-based sampling.

    Ensures:
    - Quality tier distribution (high/medium/low)
    - Minimum samples per condition (type x position)
    - No duplicate trajectories in primary sample
    """

    def __init__(
        self,
        total_primary: int = 600,
        tier_distribution: Optional[Dict[str, float]] = None,
        min_per_condition: int = 40,
        tier_thresholds: Optional[Dict[str, int]] = None,
        random_seed: int = 42
    ):
        """
        Initialize primary selector.

        Args:
            total_primary: Total number of primary perturbations to select
            tier_distribution: Target % for each tier (high, medium, low)
            min_per_condition: Minimum samples per (type, position) condition
            tier_thresholds: Score thresholds for tier assignment
            random_seed: Random seed for reproducibility
        """
        self.total_primary = total_primary
        self.tier_distribution = tier_distribution or {
            "high": 0.33,
            "medium": 0.50,
            "low": 0.17
        }
        self.min_per_condition = min_per_condition
        self.tier_thresholds = tier_thresholds or {
            "high": 6,
            "medium": 4,
            "low": 1,
            "invalid": 0
        }
        self.random_seed = random_seed

    def select(
        self,
        perturbations: List[Dict[str, Any]],
        conditions: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """
        Select primary perturbations using stratified sampling.

        Args:
            perturbations: List of perturbation documents with quality_score
            conditions: List of conditions to ensure coverage
                       If None, derived from perturbations

        Returns:
            List of perturbation_ids selected as primary
        """
        # Reset random seed for reproducibility
        random.seed(self.random_seed)

        # Step 1: Assign quality tiers if not already assigned
        for p in perturbations:
            if "quality_tier" not in p or not p.get("quality_tier"):
                score = p.get("quality_score", {}).get("total_score", 0)
                p["quality_tier"] = self._assign_tier(score)

        # Step 2: Filter out invalid (score = 0)
        valid = [p for p in perturbations if p["quality_tier"] != "invalid"]

        if not valid:
            print("Warning: No valid perturbations to select from")
            return []

        print(f"Selecting from {len(valid)} valid perturbations "
              f"(filtered {len(perturbations) - len(valid)} invalid)")

        # Step 3: Group by condition (type x position)
        by_condition = defaultdict(list)
        for p in valid:
            cond = (p["perturbation_type"], p["perturbation_position"])
            by_condition[cond].append(p)

        # Determine conditions
        if conditions:
            all_conditions = [(c["type"], c["position"]) for c in conditions]
        else:
            all_conditions = list(by_condition.keys())

        print(f"Found {len(all_conditions)} conditions")

        # Step 4: Calculate quotas
        quota_per_condition = max(
            self.min_per_condition,
            self.total_primary // len(all_conditions)
        )

        # Step 5: Select using stratified sampling with quotas
        selected = []
        assigned_trajectories = set()
        selection_stats = defaultdict(lambda: {"count": 0, "by_tier": defaultdict(int)})

        for cond in all_conditions:
            cond_perturbations = by_condition.get(cond, [])

            if not cond_perturbations:
                print(f"Warning: No perturbations for condition {cond}")
                continue

            # Sort by tier priority (high > medium > low), then by score
            cond_perturbations.sort(
                key=lambda p: (
                    {"high": 3, "medium": 2, "low": 1}.get(p["quality_tier"], 0),
                    p.get("quality_score", {}).get("total_score", 0)
                ),
                reverse=True
            )

            # Shuffle within tiers for randomness while preserving tier order
            shuffled = self._shuffle_within_tiers(cond_perturbations)

            selected_for_cond = 0

            for p in shuffled:
                traj_id = p.get("original_trajectory_id")

                # Skip if trajectory already assigned
                if traj_id in assigned_trajectories:
                    continue

                selected.append(p["perturbation_id"])
                assigned_trajectories.add(traj_id)
                selected_for_cond += 1

                # Track stats
                selection_stats[cond]["count"] += 1
                selection_stats[cond]["by_tier"][p["quality_tier"]] += 1

                if selected_for_cond >= quota_per_condition:
                    break

            print(f"  {cond}: selected {selected_for_cond}/{quota_per_condition} "
                  f"(available: {len(cond_perturbations)})")

        # Step 6: If we haven't reached total_primary, fill remaining
        if len(selected) < self.total_primary:
            remaining_needed = self.total_primary - len(selected)
            print(f"Filling {remaining_needed} remaining slots...")

            # Get unselected perturbations
            selected_set = set(selected)
            unselected = [
                p for p in valid
                if p["perturbation_id"] not in selected_set
                and p.get("original_trajectory_id") not in assigned_trajectories
            ]

            # Sort by quality score descending
            unselected.sort(
                key=lambda p: p.get("quality_score", {}).get("total_score", 0),
                reverse=True
            )

            for p in unselected[:remaining_needed]:
                selected.append(p["perturbation_id"])
                assigned_trajectories.add(p.get("original_trajectory_id"))

        print(f"\nSelected {len(selected)} primary perturbations")
        return selected

    def _shuffle_within_tiers(
        self,
        perturbations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Shuffle perturbations within each tier for randomness.

        Args:
            perturbations: Sorted perturbations (by tier)

        Returns:
            Perturbations with random order within tiers
        """
        by_tier = defaultdict(list)
        for p in perturbations:
            by_tier[p["quality_tier"]].append(p)

        result = []
        for tier in ["high", "medium", "low"]:
            tier_perts = by_tier.get(tier, [])
            random.shuffle(tier_perts)
            result.extend(tier_perts)

        return result

    def _assign_tier(self, total_score: int) -> str:
        """
        Assign quality tier based on total score.

        Args:
            total_score: Total quality score (0-7)

        Returns:
            Quality tier: "high", "medium", "low", or "invalid"
        """
        if total_score >= self.tier_thresholds["high"]:
            return "high"
        elif total_score >= self.tier_thresholds["medium"]:
            return "medium"
        elif total_score >= self.tier_thresholds["low"]:
            return "low"
        else:
            return "invalid"

    def get_selection_stats(
        self,
        perturbations: List[Dict[str, Any]],
        selected_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Generate statistics about the primary selection.

        Args:
            perturbations: All perturbations
            selected_ids: List of selected perturbation IDs

        Returns:
            Stats dict with distribution information
        """
        selected_set = set(selected_ids)
        selected_perts = [
            p for p in perturbations
            if p["perturbation_id"] in selected_set
        ]

        # By type
        by_type = defaultdict(int)
        for p in selected_perts:
            by_type[p["perturbation_type"]] += 1

        # By position
        by_position = defaultdict(int)
        for p in selected_perts:
            by_position[p["perturbation_position"]] += 1

        # By condition
        by_condition = defaultdict(int)
        for p in selected_perts:
            cond = f"{p['perturbation_type']}_{p['perturbation_position']}"
            by_condition[cond] += 1

        # By tier
        by_tier = defaultdict(int)
        for p in selected_perts:
            tier = p.get("quality_tier", "unknown")
            by_tier[tier] += 1

        # Unique trajectories
        unique_trajectories = len(set(
            p.get("original_trajectory_id") for p in selected_perts
        ))

        return {
            "total_selected": len(selected_ids),
            "unique_trajectories": unique_trajectories,
            "by_type": dict(by_type),
            "by_position": dict(by_position),
            "by_condition": dict(by_condition),
            "by_tier": dict(by_tier),
            "selection_timestamp": datetime.now(timezone.utc).isoformat()
        }


def create_primary_selector(config: Dict[str, Any]) -> PrimarySelector:
    """
    Factory function to create primary selector from config.

    Args:
        config: Primary selection configuration dict

    Returns:
        Configured PrimarySelector instance
    """
    return PrimarySelector(
        total_primary=config.get("total_primary", 600),
        tier_distribution=config.get("tier_distribution"),
        min_per_condition=config.get("min_per_condition", 40),
        random_seed=config.get("random_seed", 42)
    )
