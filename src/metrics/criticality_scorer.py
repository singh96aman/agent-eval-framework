"""
Criticality Scorer - Computes True Criticality Score (TCS) for perturbations.

TCS represents the ground-truth impact of an error on task completion.

Modes:
- heuristic: Assign TCS based on (type, position) lookup table
- human: Load from human annotation file
- hybrid: Human TCS for annotated samples, heuristic for rest

Formula (human annotation):
    TCS = (task_success_degradation * 50) + (subsequent_error_rate * 10) + (criticality_rating * 8)

Heuristic TCS values are based on prior research on error impact patterns.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class CriticalityScorer:
    """
    Computes True Criticality Score (TCS) for perturbations.

    The TCS represents ground-truth error criticality, used as the
    denominator in CCG calculation.

    Supported modes:
    - "heuristic": Fast assignment based on type/position lookup
    - "human": Load from annotation file
    - "hybrid": Human for annotated, heuristic for rest (recommended)
    """

    # Heuristic TCS values based on (type, position)
    # Higher values for early structural errors, lower for late surface errors
    HEURISTIC_TCS = {
        "planning_early": 90,
        "planning_middle": 70,
        "planning_late": 40,
        "tool_selection_early": 80,
        "tool_selection_middle": 60,
        "tool_selection_late": 30,
        "parameter_early": 50,
        "parameter_middle": 40,
        "parameter_late": 20,
        "data_reference_early": 40,
        "data_reference_middle": 30,
        "data_reference_late": 20,
    }

    def __init__(self, config: Dict):
        """
        Initialize criticality scorer.

        Args:
            config: Configuration dict with:
                - mode: "heuristic" | "human" | "hybrid"
                - human_annotation_path: Path to annotations JSON (for human/hybrid)
                - heuristic_tcs: Optional override for heuristic values
        """
        self.mode = config.get("mode", "heuristic")
        self.human_annotations = {}

        # Allow config to override default heuristic values
        custom_heuristic = config.get("heuristic_tcs", {})
        if custom_heuristic:
            self.HEURISTIC_TCS.update(custom_heuristic)

        # Load human annotations if needed
        if self.mode in ["human", "hybrid"]:
            annotation_path = config.get("human_annotation_path")
            if annotation_path and Path(annotation_path).exists():
                self._load_human_annotations(annotation_path)
            elif self.mode == "human":
                print("Warning: human mode specified but no annotations found")

    def _load_human_annotations(self, path: str):
        """
        Load human annotations from JSON file.

        Expects format from StratifiedAnnotationSampler.export_for_annotation():
        [
            {
                "perturbation_id": "...",
                "perturbation_type": "...",
                "perturbation_position": "...",
                "annotation": {
                    "task_success_degradation": 0 or 1,
                    "subsequent_error_rate": 0-3,
                    "criticality_rating": 1-5,
                    ...
                }
            },
            ...
        ]

        Args:
            path: Path to annotations JSON file
        """
        with open(path, 'r') as f:
            annotations = json.load(f)

        for ann in annotations:
            annotation_data = ann.get("annotation", {})

            # Check for pre-computed tcs_score first (from MongoDB export)
            precomputed_tcs = annotation_data.get("tcs_score")

            # Get individual components
            tsd = annotation_data.get("task_success_degradation")
            ser = annotation_data.get("subsequent_error_rate")
            crit = annotation_data.get("criticality_rating")

            # Accept if we have precomputed TCS OR all components
            if precomputed_tcs is not None or (tsd is not None and ser is not None):
                self.human_annotations[ann["perturbation_id"]] = {
                    "task_success_degradation": tsd,
                    "subsequent_error_rate": ser,
                    "criticality_rating": crit,
                    "tcs_score": precomputed_tcs,  # May be None
                    "perturbation_type": ann.get("perturbation_type"),
                    "perturbation_position": ann.get("perturbation_position")
                }

        print(f"Loaded {len(self.human_annotations)} human annotations from {path}")

    def compute_tcs(self, perturbation: Dict) -> float:
        """
        Compute TCS for a single perturbation.

        Follows mode priority:
        - hybrid: Try human first, fall back to heuristic
        - human: Human only (raises if not found)
        - heuristic: Heuristic only

        Args:
            perturbation: Perturbation dict with:
                - perturbation_id
                - perturbation_type
                - perturbation_position

        Returns:
            TCS value (0-100)

        Raises:
            ValueError: If mode="human" and no annotation exists
        """
        pert_id = perturbation.get("perturbation_id")

        # Try human annotation first (if hybrid or human mode)
        if self.mode in ["human", "hybrid"] and pert_id in self.human_annotations:
            return self._compute_human_tcs(self.human_annotations[pert_id])

        # Fall back to heuristic
        if self.mode in ["heuristic", "hybrid"]:
            return self._compute_heuristic_tcs(perturbation)

        # If mode is "human" and no annotation exists
        raise ValueError(f"No human annotation available for {pert_id} in mode '{self.mode}'")

    def _compute_human_tcs(self, annotation: Dict) -> float:
        """
        Compute TCS from human annotation.

        If precomputed tcs_score exists (from MongoDB), use it directly.
        Otherwise compute from components using formula:
        TCS = (TSD * 50) + (SER * 10) + (criticality * 8)

        Note: MongoDB annotations use formula: (TSD * 100) + (SER * 10)
        which is stored in tcs_score field.

        Args:
            annotation: Dict with tcs_score or components

        Returns:
            TCS value
        """
        # Use precomputed TCS if available (from MongoDB)
        if annotation.get("tcs_score") is not None:
            return min(annotation["tcs_score"], 100)

        # Otherwise compute from components
        tsd = annotation.get("task_success_degradation", 0)
        ser = min(annotation.get("subsequent_error_rate", 0), 3)  # Cap at 3
        criticality = annotation.get("criticality_rating", 3)

        tcs = (tsd * 50) + (ser * 10) + (criticality * 8)

        return min(tcs, 100)  # Cap at 100

    def _compute_heuristic_tcs(self, perturbation: Dict) -> float:
        """
        Compute TCS from heuristic lookup table.

        Args:
            perturbation: Dict with perturbation_type and perturbation_position

        Returns:
            TCS value from lookup table, or 50 as default
        """
        ptype = perturbation.get("perturbation_type", "")
        pos = perturbation.get("perturbation_position", "")
        key = f"{ptype}_{pos}"

        return float(self.HEURISTIC_TCS.get(key, 50))  # Default to 50 if unknown

    def compute_batch(self, perturbations: List[Dict]) -> List[float]:
        """
        Compute TCS for multiple perturbations.

        Args:
            perturbations: List of perturbation dicts

        Returns:
            List of TCS values (same order as input)
        """
        return [self.compute_tcs(p) for p in perturbations]

    def compute_batch_with_ids(self, perturbations: List[Dict]) -> Dict[str, float]:
        """
        Compute TCS for multiple perturbations, returning as dict.

        Args:
            perturbations: List of perturbation dicts

        Returns:
            Dict mapping perturbation_id -> TCS
        """
        return {
            p.get("perturbation_id"): self.compute_tcs(p)
            for p in perturbations
        }

    def validate_heuristic(self) -> Dict:
        """
        Validate heuristic TCS against human annotations.

        Computes Pearson and Spearman correlation between
        heuristic assignments and human-computed TCS.

        Requires human annotations to be loaded.

        Returns:
            Validation report with correlation metrics
        """
        if not self.human_annotations:
            return {"error": "No human annotations loaded"}

        try:
            from scipy.stats import pearsonr, spearmanr
        except ImportError:
            return {"error": "scipy not installed"}

        human_tcs_values = []
        heuristic_tcs_values = []

        for pert_id, annotation in self.human_annotations.items():
            # Compute human TCS
            human_tcs = self._compute_human_tcs(annotation)
            human_tcs_values.append(human_tcs)

            # Compute heuristic TCS from annotation metadata
            ptype = annotation.get("perturbation_type", "")
            pos = annotation.get("perturbation_position", "")
            key = f"{ptype}_{pos}"
            heuristic_tcs = float(self.HEURISTIC_TCS.get(key, 50))
            heuristic_tcs_values.append(heuristic_tcs)

        # Compute correlations
        if len(human_tcs_values) < 3:
            return {
                "n_samples": len(human_tcs_values),
                "error": "Need at least 3 samples for correlation"
            }

        pearson_r, pearson_p = pearsonr(human_tcs_values, heuristic_tcs_values)
        spearman_r, spearman_p = spearmanr(human_tcs_values, heuristic_tcs_values)

        # Interpret correlation strength
        abs_r = abs(pearson_r)
        if abs_r > 0.7:
            strength = "strong"
        elif abs_r > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        return {
            "n_samples": len(human_tcs_values),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "mean_human_tcs": float(np.mean(human_tcs_values)),
            "std_human_tcs": float(np.std(human_tcs_values)),
            "mean_heuristic_tcs": float(np.mean(heuristic_tcs_values)),
            "std_heuristic_tcs": float(np.std(heuristic_tcs_values)),
            "correlation_strength": strength,
            "heuristic_valid": abs_r > 0.4  # Moderate or better
        }

    def get_tcs_distribution_by_condition(self) -> Dict[str, Dict]:
        """
        Get TCS distribution by condition from human annotations.

        Returns:
            Dict mapping condition -> stats (mean, std, count)
        """
        if not self.human_annotations:
            return {}

        from collections import defaultdict

        by_condition = defaultdict(list)

        for pert_id, annotation in self.human_annotations.items():
            ptype = annotation.get("perturbation_type", "")
            pos = annotation.get("perturbation_position", "")
            condition = f"{ptype}_{pos}"

            tcs = self._compute_human_tcs(annotation)
            by_condition[condition].append(tcs)

        return {
            condition: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values),
                "heuristic": float(self.HEURISTIC_TCS.get(condition, 50))
            }
            for condition, values in by_condition.items()
        }

    def print_validation_report(self):
        """Print a formatted validation report."""
        validation = self.validate_heuristic()

        if "error" in validation:
            print(f"Validation error: {validation['error']}")
            return

        print("\n" + "=" * 70)
        print("HEURISTIC TCS VALIDATION REPORT")
        print("=" * 70)

        print(f"\nSample size: {validation['n_samples']}")

        print(f"\nCorrelation with Human TCS:")
        print(f"  Pearson r:  {validation['pearson_r']:.3f} (p={validation['pearson_p']:.4f})")
        print(f"  Spearman r: {validation['spearman_r']:.3f} (p={validation['spearman_p']:.4f})")
        print(f"  Strength:   {validation['correlation_strength']}")

        print(f"\nTCS Statistics:")
        print(f"  Human mean:     {validation['mean_human_tcs']:.1f} +/- {validation['std_human_tcs']:.1f}")
        print(f"  Heuristic mean: {validation['mean_heuristic_tcs']:.1f} +/- {validation['std_heuristic_tcs']:.1f}")

        if validation['heuristic_valid']:
            print(f"\nResult: VALID (correlation >= 0.4)")
        else:
            print(f"\nResult: INVALID (correlation < 0.4)")
            print("  Consider using human TCS instead of heuristic for CCG calculation.")

        # Distribution by condition
        by_condition = self.get_tcs_distribution_by_condition()
        if by_condition:
            print(f"\nBy Condition (Human vs Heuristic):")
            print("-" * 50)
            for condition in sorted(by_condition.keys()):
                stats = by_condition[condition]
                human_mean = stats['mean']
                heuristic = stats['heuristic']
                diff = human_mean - heuristic
                print(f"  {condition:25s}: Human={human_mean:.1f}, Heuristic={heuristic:.1f}, Diff={diff:+.1f}")

        print("=" * 70)
