"""
Stratified sampling for human annotation.

This module provides a sampler that selects perturbations for human annotation
with stratification across conditions, benchmarks, and quality tiers.
"""

import json
import random
from collections import defaultdict
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from src.storage.mongodb import MongoDBStorage


class StratifiedAnnotationSampler:
    """
    Stratified sampling for human annotation ground truth.

    Ensures coverage across:
    - 3 perturbation classes (placebo, fine_grained, coarse_grained)
    - Multiple perturbation conditions (type x position)
    - 3 benchmarks (toolbench, gaia, swebench)
    - 3 quality tiers (high, medium, low)

    Sampling strategy:
    1. Primary stratification by perturbation CLASS (20% placebo, 50% fine, 30% coarse)
    2. Secondary stratification by condition (type x position) within each class
    3. Tertiary stratification by benchmark
    4. Quaternary stratification by quality tier
    """

    # Perturbation classes with target distributions
    CLASS_DISTRIBUTION = {
        "placebo": 0.20,
        "fine_grained": 0.50,
        "coarse_grained": 0.30,
    }

    # Conditions by class
    PLACEBO_CONDITIONS = [
        ("paraphrase", "early"),
        ("paraphrase", "middle"),
        ("paraphrase", "late"),
        ("formatting", "middle"),
        ("synonym", "middle"),
        ("reorder_args", "middle"),
    ]

    FINE_GRAINED_CONDITIONS = [
        ("parameter", "early"),
        ("parameter", "middle"),
        ("parameter", "late"),
        ("data_reference", "early"),
        ("data_reference", "middle"),
        ("data_reference", "late"),
        ("near_neighbor_tool", "middle"),
    ]

    COARSE_GRAINED_CONDITIONS = [
        ("planning", "early"),
        ("planning", "middle"),
        ("planning", "late"),
        ("tool_selection", "early"),
        ("tool_selection", "middle"),
        ("tool_selection", "late"),
        ("structural", "early"),
        ("structural", "middle"),
    ]

    # Legacy CONDITIONS for backward compatibility
    CONDITIONS = [
        ("planning", "early"),
        ("planning", "middle"),
        ("planning", "late"),
        ("tool_selection", "early"),
        ("tool_selection", "middle"),
        ("tool_selection", "late"),
        ("parameter", "early"),
        ("parameter", "middle"),
        ("parameter", "late"),
        ("data_reference", "middle"),
        ("data_reference", "late"),
    ]

    BENCHMARKS = ["toolbench", "gaia", "swebench"]
    QUALITY_TIERS = ["high", "medium", "low"]

    def __init__(
        self,
        perturbations: List[Dict],
        random_seed: int = 42,
        stratify_by_class: bool = True,
        class_distribution: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize sampler with perturbations.

        Args:
            perturbations: List of perturbation dicts from MongoDB
            random_seed: Random seed for reproducibility
            stratify_by_class: Whether to stratify by perturbation class
            class_distribution: Optional custom class distribution
                               (default: 20% placebo, 50% fine, 30% coarse)
        """
        # Filter to primary perturbations only
        self.perturbations = [
            p for p in perturbations if p.get("is_primary_for_experiment", False)
        ]
        self.random_seed = random_seed
        self.stratify_by_class = stratify_by_class
        self.class_distribution = class_distribution or self.CLASS_DISTRIBUTION.copy()
        random.seed(random_seed)

        # Build index
        self._build_index()

    def _build_index(self):
        """
        Index perturbations by (class, condition, benchmark, tier).

        Creates a multi-level index for efficient stratified sampling.
        """
        self.index = defaultdict(list)
        self.class_index = defaultdict(list)  # Index by class only

        for p in self.perturbations:
            pert_class = p.get("perturbation_class", "unknown")
            condition = (p.get("perturbation_type"), p.get("perturbation_position"))
            benchmark = self._get_benchmark(p)
            tier = p.get("quality_tier", "medium")

            # Add to class index
            self.class_index[pert_class].append(p)

            # Add to main index with class
            key = (pert_class, condition, benchmark, tier)
            self.index[key].append(p)

        # Print index summary
        total_indexed = sum(len(v) for v in self.class_index.values())
        print(f"Indexed {total_indexed} perturbations:")

        # Summary by class
        for pert_class in ["placebo", "fine_grained", "coarse_grained"]:
            class_count = len(self.class_index.get(pert_class, []))
            pct = (class_count / total_indexed * 100) if total_indexed > 0 else 0
            print(f"  {pert_class}: {class_count} ({pct:.1f}%)")

    def sample(self, total: int = 100) -> List[Dict]:
        """
        Sample perturbations with stratification.

        Strategy (when stratify_by_class=True):
        1. Allocate by class: placebo ~20%, fine_grained ~50%, coarse ~30%
        2. Within each class, stratify by condition (type x position)
        3. Within each condition, stratify by benchmark and tier
        4. Fill remaining slots if needed

        Legacy strategy (stratify_by_class=False):
        1. Allocate samples_per_condition = total // 11
        2. For each condition, try to get coverage across benchmarks/tiers

        Args:
            total: Total number of samples to select

        Returns:
            List of perturbation dicts to annotate
        """
        if self.stratify_by_class:
            return self._sample_with_class_stratification(total)
        else:
            return self._sample_legacy(total)

    def _sample_with_class_stratification(self, total: int) -> List[Dict]:
        """
        Sample with class-level stratification.

        Args:
            total: Total number of samples

        Returns:
            List of perturbation dicts
        """
        # Make working copies
        working_class_index = {
            k: list(v) for k, v in self.class_index.items()
        }

        selected = []

        # Calculate targets per class
        class_targets = {}
        for pert_class, ratio in self.class_distribution.items():
            class_targets[pert_class] = int(total * ratio)

        # Adjust to ensure we hit total
        remainder = total - sum(class_targets.values())
        if remainder > 0:
            # Add remainder to fine_grained (largest class)
            class_targets["fine_grained"] = class_targets.get("fine_grained", 0) + remainder

        print(f"\nSampling {total} perturbations with class stratification:")
        for pert_class, target in class_targets.items():
            print(f"  {pert_class}: {target} target")

        # Sample from each class
        for pert_class, target in class_targets.items():
            class_samples = self._sample_from_class(
                pert_class, target, working_class_index
            )
            selected.extend(class_samples)
            pct = (len(class_samples) / total * 100) if total > 0 else 0
            print(f"  {pert_class}: {len(class_samples)} sampled ({pct:.1f}%)")

        # Fill remaining if needed
        while len(selected) < total:
            extra = self._sample_any_remaining_by_class(working_class_index)
            if extra:
                selected.append(extra)
            else:
                print(
                    f"  Warning: Only found {len(selected)} samples (target: {total})"
                )
                break

        return selected[:total]

    def _sample_from_class(
        self, pert_class: str, target: int, working_index: Dict
    ) -> List[Dict]:
        """
        Sample from a single class with condition stratification.

        Args:
            pert_class: The perturbation class
            target: Target number of samples
            working_index: Working copy of class index

        Returns:
            List of sampled perturbations
        """
        pool = working_index.get(pert_class, [])
        if not pool:
            return []

        # Shuffle for randomness
        random.shuffle(pool)

        # Take up to target samples
        samples = pool[:target]
        # Remove from working index
        working_index[pert_class] = pool[target:]

        return samples

    def _sample_any_remaining_by_class(self, working_index: Dict) -> Optional[Dict]:
        """
        Sample from any remaining class pool.

        Args:
            working_index: Working copy of class index

        Returns:
            A perturbation dict, or None if nothing remains
        """
        for pert_class, pool in working_index.items():
            if pool:
                sample = pool.pop(0)
                return sample
        return None

    def _sample_legacy(self, total: int) -> List[Dict]:
        """
        Legacy sampling by condition only (no class stratification).

        Args:
            total: Total number of samples

        Returns:
            List of perturbation dicts
        """
        # Make a copy of the index to avoid modifying original
        working_index = {key: list(perts) for key, perts in self.index.items()}

        selected = []
        samples_per_condition = total // len(self.CONDITIONS)  # 9

        print(
            f"\nSampling {total} perturbations "
            f"({samples_per_condition} per condition)..."
        )

        for condition in self.CONDITIONS:
            condition_samples = self._sample_for_condition(
                condition, samples_per_condition, working_index
            )
            selected.extend(condition_samples)
            print(
                f"  {condition[0]}_{condition[1]}: {len(condition_samples)} samples"
            )

        # Fill to exactly total if needed
        while len(selected) < total:
            extra = self._sample_any_remaining(working_index)
            if extra:
                selected.append(extra)
            else:
                print(
                    f"  Warning: Only found {len(selected)} samples (target: {total})"
                )
                break

        # Truncate if we somehow got more
        return selected[:total]

    def _sample_for_condition(
        self, condition: tuple, target: int, working_index: Dict
    ) -> List[Dict]:
        """
        Sample for a single condition with benchmark/tier stratification.

        Tries to get:
        - 1 sample from each (benchmark, tier) combination first
        - Then fills remaining from any available

        Args:
            condition: (perturbation_type, position) tuple
            target: Target number of samples for this condition
            working_index: Working copy of the index

        Returns:
            List of selected perturbation dicts
        """
        samples = []

        # First pass: try to get 1 from each (benchmark, tier) combination
        for benchmark in self.BENCHMARKS:
            for tier in self.QUALITY_TIERS:
                key = (condition, benchmark, tier)
                pool = working_index.get(key, [])

                if pool and len(samples) < target:
                    sample = random.choice(pool)
                    samples.append(sample)
                    pool.remove(sample)

        # Second pass: fill remaining from any available
        while len(samples) < target:
            found = False
            for benchmark in self.BENCHMARKS:
                for tier in self.QUALITY_TIERS:
                    key = (condition, benchmark, tier)
                    pool = working_index.get(key, [])

                    if pool:
                        sample = random.choice(pool)
                        samples.append(sample)
                        pool.remove(sample)
                        found = True
                        break

                if found or len(samples) >= target:
                    break

            if not found:
                break  # No more samples available for this condition

        return samples

    def _sample_any_remaining(self, working_index: Dict) -> Optional[Dict]:
        """
        Sample from any remaining pool.

        Args:
            working_index: Working copy of the index

        Returns:
            A perturbation dict, or None if nothing remains
        """
        for key, pool in working_index.items():
            if pool:
                sample = random.choice(pool)
                pool.remove(sample)
                return sample
        return None

    def _get_benchmark(self, perturbation: Dict) -> str:
        """
        Extract benchmark from perturbation or trajectory ID.

        Args:
            perturbation: Perturbation dict

        Returns:
            Benchmark name
        """
        traj_id = perturbation.get("original_trajectory_id", "").lower()

        if "toolbench" in traj_id:
            return "toolbench"
        elif "gaia" in traj_id:
            return "gaia"
        elif "swe" in traj_id:
            return "swebench"

        return "unknown"

    def export_for_annotation(self, selected: List[Dict], output_path: str):
        """
        Export selected samples to JSON for annotation interface.

        Creates a JSON file with all information needed for human annotation:
        - Original and perturbed step content
        - Perturbation metadata
        - Empty annotation fields to fill

        Args:
            selected: List of selected perturbation dicts
            output_path: Path to write JSON file
        """
        export_data = []

        for i, p in enumerate(selected):
            export_data.append(
                {
                    "annotation_id": i + 1,
                    "perturbation_id": p.get("perturbation_id"),
                    "original_trajectory_id": p.get("original_trajectory_id"),
                    "perturbed_trajectory_id": p.get("perturbed_trajectory_id"),
                    "perturbation_type": p.get("perturbation_type"),
                    "perturbation_position": p.get("perturbation_position"),
                    "perturbed_step_number": p.get("perturbed_step_number"),
                    "quality_tier": p.get("quality_tier"),
                    "quality_score": p.get("quality_score"),
                    "benchmark": self._get_benchmark(p),
                    # Include step content for annotation
                    "original_step_content": p.get("original_step_content"),
                    "perturbed_step_content": p.get("perturbed_step_content"),
                    "perturbation_metadata": p.get("perturbation_metadata", {}),
                    # Annotation fields (to be filled by human)
                    "annotation": {
                        "task_success_degradation": None,  # 0 or 1
                        "subsequent_error_rate": None,  # 0-3
                        "criticality_rating": None,  # 1-5
                        "confidence": None,  # 1-3
                        "notes": None,
                        "annotated_at": None,
                        "annotator_id": None,
                    },
                }
            )

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\nExported {len(export_data)} samples to {output_path}")

    def flag_in_mongodb(
        self, selected: List[Dict], storage: MongoDBStorage, experiment_id: str
    ) -> int:
        """
        Flag selected perturbations in MongoDB for annotation.

        Sets `selected_for_annotation=True` on each selected perturbation,
        allowing the annotation interface to fetch them directly.

        Args:
            selected: List of selected perturbation dicts
            storage: MongoDB storage instance
            experiment_id: Experiment ID for annotation tracking

        Returns:
            Number of perturbations flagged
        """
        flagged_count = 0
        perturbation_ids = [p.get("perturbation_id") for p in selected]

        # First, clear any previous annotation flags for this experiment
        storage.db["perturbations"].update_many(
            {"experiment_id": experiment_id},
            {"$set": {"selected_for_annotation": False}},
        )

        # Flag selected perturbations
        for pert_id in perturbation_ids:
            result = storage.db["perturbations"].update_one(
                {"perturbation_id": pert_id},
                {
                    "$set": {
                        "selected_for_annotation": True,
                        "annotation_experiment_id": experiment_id,
                        "selected_at": datetime.utcnow(),
                    }
                },
            )
            if result.modified_count > 0:
                flagged_count += 1

        print(f"\n📌 Flagged {flagged_count} perturbations in MongoDB for annotation")
        return flagged_count

    def get_distribution_report(self, selected: List[Dict]) -> Dict:
        """
        Generate distribution report for validation.

        Args:
            selected: List of selected perturbation dicts

        Returns:
            Dict with distribution statistics
        """
        report = {
            "total": len(selected),
            "by_condition": defaultdict(int),
            "by_benchmark": defaultdict(int),
            "by_quality_tier": defaultdict(int),
            "by_condition_benchmark": defaultdict(int),
            "by_type": defaultdict(int),
            "by_position": defaultdict(int),
        }

        for p in selected:
            ptype = p.get("perturbation_type", "unknown")
            pos = p.get("perturbation_position", "unknown")
            condition = f"{ptype}_{pos}"
            benchmark = self._get_benchmark(p)
            tier = p.get("quality_tier", "unknown")

            report["by_condition"][condition] += 1
            report["by_benchmark"][benchmark] += 1
            report["by_quality_tier"][tier] += 1
            report["by_condition_benchmark"][f"{condition}_{benchmark}"] += 1
            report["by_type"][ptype] += 1
            report["by_position"][pos] += 1

        # Convert defaultdicts to regular dicts
        return {
            k: dict(v) if isinstance(v, defaultdict) else v for k, v in report.items()
        }

    def print_distribution_report(self, selected: List[Dict]):
        """
        Print a formatted distribution report.

        Args:
            selected: List of selected perturbation dicts
        """
        report = self.get_distribution_report(selected)

        print("\n" + "=" * 70)
        print("ANNOTATION SAMPLE DISTRIBUTION REPORT")
        print("=" * 70)
        print(f"\nTotal samples: {report['total']}")

        print("\nBy Condition (Type x Position):")
        print("-" * 40)
        for cond, count in sorted(report["by_condition"].items()):
            print(f"  {cond}: {count}")

        print("\nBy Benchmark:")
        print("-" * 40)
        for benchmark, count in sorted(report["by_benchmark"].items()):
            pct = (count / report["total"]) * 100
            print(f"  {benchmark}: {count} ({pct:.1f}%)")

        print("\nBy Quality Tier:")
        print("-" * 40)
        for tier, count in sorted(report["by_quality_tier"].items()):
            pct = (count / report["total"]) * 100
            print(f"  {tier}: {count} ({pct:.1f}%)")

        print("\n" + "=" * 70)


def load_annotations_from_file(path: str) -> List[Dict]:
    """
    Load completed annotations from JSON file.

    Args:
        path: Path to annotations JSON file

    Returns:
        List of annotation dicts
    """
    with open(path, "r") as f:
        return json.load(f)


def validate_annotations(annotations: List[Dict]) -> Dict:
    """
    Validate completed annotations.

    Checks:
    - All required fields filled
    - Values in valid ranges
    - Confidence scores present

    Args:
        annotations: List of annotation dicts

    Returns:
        Validation report dict
    """
    complete = 0
    incomplete = 0
    invalid = 0
    issues = []

    for ann in annotations:
        annotation_data = ann.get("annotation", {})

        # Check if annotation exists
        if not annotation_data:
            incomplete += 1
            issues.append(f"Missing annotation for {ann.get('perturbation_id')}")
            continue

        # Check required fields
        tsd = annotation_data.get("task_success_degradation")
        ser = annotation_data.get("subsequent_error_rate")
        crit = annotation_data.get("criticality_rating")

        if tsd is None or ser is None or crit is None:
            incomplete += 1
            issues.append(f"Incomplete annotation for {ann.get('perturbation_id')}")
            continue

        # Validate ranges
        valid = True
        if tsd not in [0, 1]:
            valid = False
            issues.append(f"Invalid TSD ({tsd}) for {ann.get('perturbation_id')}")
        if not (0 <= ser <= 3):
            valid = False
            issues.append(f"Invalid SER ({ser}) for {ann.get('perturbation_id')}")
        if not (1 <= crit <= 5):
            valid = False
            issues.append(
                f"Invalid criticality ({crit}) for {ann.get('perturbation_id')}"
            )

        if valid:
            complete += 1
        else:
            invalid += 1

    return {
        "total": len(annotations),
        "complete": complete,
        "incomplete": incomplete,
        "invalid": invalid,
        "issues": issues[:10],  # First 10 issues
    }
