"""
ID Generation Utilities.

All IDs are deterministic and config-bound. The encoded config portion
can be decoded to trace back to the exact configuration used.

ID Scheme:
- experiment_id: {config.experiment.id}
- trajectory_id: {benchmark}_{idx}_{b64(dataset_config)}
- step_id: step_{idx}
- perturbation_id: {experiment_id}_{trajectory_id}_step_{step_idx}_{type}_{b64(generator_config)}
- evaluation_unit_id: {trajectory_id}_{perturbation_id}
- annotation_id: {evaluation_unit_id}_{annotator_id}
- outcome_id: {evaluation_unit_id}_{model_name}
"""

import base64
import json
from typing import Any, Dict


# =============================================================================
# Base64 Encoding/Decoding
# =============================================================================

def encode_config(config_dict: Dict[str, Any]) -> str:
    """
    Encode a config dict to URL-safe Base64 string.

    Args:
        config_dict: Configuration dictionary to encode

    Returns:
        URL-safe Base64 encoded string (no padding)
    """
    json_str = json.dumps(config_dict, separators=(",", ":"), sort_keys=True)
    encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
    return encoded.rstrip("=")  # Remove padding for cleaner IDs


def decode_config(encoded: str) -> Dict[str, Any]:
    """
    Decode a Base64 string back to config dict.

    Args:
        encoded: URL-safe Base64 encoded string

    Returns:
        Original configuration dictionary
    """
    # Add padding back if needed
    padding = 4 - (len(encoded) % 4)
    if padding != 4:
        encoded += "=" * padding
    json_str = base64.urlsafe_b64decode(encoded).decode()
    return json.loads(json_str)


# =============================================================================
# ID Generators
# =============================================================================

def generate_experiment_id(config: Dict[str, Any]) -> str:
    """
    Get experiment ID from config.

    Args:
        config: Full experiment config

    Returns:
        Experiment ID string
    """
    return config["experiment"]["id"]


def generate_trajectory_id(
    benchmark: str,
    idx: int,
    dataset_config: Dict[str, Any],
) -> str:
    """
    Generate trajectory ID with encoded dataset config.

    Format: {benchmark}_{idx}_{b64(dataset_config)}

    Args:
        benchmark: Benchmark name (toolbench, gaia, swebench)
        idx: Trajectory index within benchmark
        dataset_config: Dataset configuration from config["experiment"]["datasets"][benchmark]

    Returns:
        Trajectory ID string
    """
    config_encoded = encode_config(dataset_config)
    return f"{benchmark}_{idx}_{config_encoded}"


def generate_step_id(idx: int) -> str:
    """
    Generate step ID.

    Format: step_{idx}

    Args:
        idx: Step index (0-based)

    Returns:
        Step ID string
    """
    return f"step_{idx}"


def generate_perturbation_id(
    experiment_id: str,
    trajectory_id: str,
    step_idx: int,
    perturbation_type: str,
    generator_config: Dict[str, Any],
) -> str:
    """
    Generate perturbation ID with encoded generator config.

    Format: {experiment_id}_{trajectory_id}_step_{step_idx}_{type}_{b64(generator_config)}

    Args:
        experiment_id: Experiment ID
        trajectory_id: Trajectory ID
        step_idx: Target step index
        perturbation_type: Type of perturbation (e.g., "wrong_value", "paraphrase")
        generator_config: Generator config from config["phases"]["perturb"]["generators"][class]

    Returns:
        Perturbation ID string
    """
    config_encoded = encode_config(generator_config)
    return f"{experiment_id}_{trajectory_id}_step_{step_idx}_{perturbation_type}_{config_encoded}"


def generate_evaluation_unit_id(
    trajectory_id: str,
    perturbation_id: str,
) -> str:
    """
    Generate evaluation unit ID.

    Format: {trajectory_id}_{perturbation_id}

    Args:
        trajectory_id: Trajectory ID
        perturbation_id: Perturbation ID

    Returns:
        Evaluation unit ID string
    """
    return f"{trajectory_id}_{perturbation_id}"


def generate_annotation_id(
    evaluation_unit_id: str,
    annotator_id: str,
) -> str:
    """
    Generate annotation ID.

    Format: {evaluation_unit_id}_{annotator_id}

    Args:
        evaluation_unit_id: Evaluation unit ID
        annotator_id: Annotator identifier from config["phases"]["annotate"]["annotator_id"]

    Returns:
        Annotation ID string
    """
    return f"{evaluation_unit_id}_{annotator_id}"


def generate_outcome_id(
    evaluation_unit_id: str,
    model_name: str,
) -> str:
    """
    Generate outcome ID for judge evaluation.

    Format: {evaluation_unit_id}_{model_name}

    Args:
        evaluation_unit_id: Evaluation unit ID
        model_name: Judge model name from config["phases"]["judge"]["models"][idx]["name"]

    Returns:
        Outcome ID string
    """
    return f"{evaluation_unit_id}_{model_name}"


# =============================================================================
# ID Parsers (extract components from IDs)
# =============================================================================

def parse_trajectory_id(trajectory_id: str) -> Dict[str, Any]:
    """
    Parse trajectory ID to extract components.

    Args:
        trajectory_id: Trajectory ID string

    Returns:
        Dict with benchmark, idx, and decoded dataset_config
    """
    parts = trajectory_id.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid trajectory_id format: {trajectory_id}")

    benchmark, idx_str, config_encoded = parts
    return {
        "benchmark": benchmark,
        "idx": int(idx_str),
        "dataset_config": decode_config(config_encoded),
    }


def parse_perturbation_id(perturbation_id: str) -> Dict[str, Any]:
    """
    Parse perturbation ID to extract components.

    Args:
        perturbation_id: Perturbation ID string

    Returns:
        Dict with experiment_id, trajectory_id, step_idx, perturbation_type, generator_config
    """
    # Format: {experiment_id}_{trajectory_id}_step_{step_idx}_{type}_{config}
    # This is complex because trajectory_id itself contains underscores

    # Find "_step_" to split
    step_marker = "_step_"
    step_pos = perturbation_id.find(step_marker)
    if step_pos == -1:
        raise ValueError(f"Invalid perturbation_id format: {perturbation_id}")

    # Everything before _step_ is experiment_id + trajectory_id
    prefix = perturbation_id[:step_pos]
    suffix = perturbation_id[step_pos + len(step_marker):]

    # Split suffix: {step_idx}_{type}_{config}
    suffix_parts = suffix.split("_", 2)
    if len(suffix_parts) != 3:
        raise ValueError(f"Invalid perturbation_id format: {perturbation_id}")

    step_idx_str, perturbation_type, config_encoded = suffix_parts

    # Split prefix to get experiment_id and trajectory_id
    # experiment_id is first part, trajectory_id is the rest
    # This assumes experiment_id doesn't contain the trajectory pattern
    prefix_parts = prefix.split("_", 1)
    if len(prefix_parts) == 2:
        experiment_id = prefix_parts[0]
        trajectory_id = prefix_parts[1]
    else:
        experiment_id = prefix
        trajectory_id = ""

    return {
        "experiment_id": experiment_id,
        "trajectory_id": trajectory_id,
        "step_idx": int(step_idx_str),
        "perturbation_type": perturbation_type,
        "generator_config": decode_config(config_encoded),
    }


def parse_evaluation_unit_id(evaluation_unit_id: str) -> Dict[str, Any]:
    """
    Parse evaluation unit ID to extract components.

    Args:
        evaluation_unit_id: Evaluation unit ID string

    Returns:
        Dict with trajectory_id and perturbation_id
    """
    # Format: {trajectory_id}_{perturbation_id}
    # Find the experiment_id boundary in perturbation_id part
    # perturbation_id starts with experiment_id
    # trajectory_id format: {benchmark}_{idx}_{b64}
    # We need to find where trajectory_id ends and perturbation_id begins

    # Look for the pattern where perturbation_id starts (experiment_id prefix)
    # This is complex, so we split by finding "_step_" marker in perturbation
    step_marker = "_step_"
    step_pos = evaluation_unit_id.find(step_marker)
    if step_pos == -1:
        raise ValueError(f"Invalid evaluation_unit_id: {evaluation_unit_id}")

    # Everything before _step_ includes trajectory_id and experiment_id
    # We need to find the boundary
    # For now, return the raw split - caller can parse further if needed
    return {
        "trajectory_id": None,  # Complex to extract, use full ID
        "perturbation_id": None,  # Complex to extract
        "raw": evaluation_unit_id,
    }


def parse_annotation_id(annotation_id: str) -> Dict[str, Any]:
    """
    Parse annotation ID to extract components.

    Args:
        annotation_id: Annotation ID string

    Returns:
        Dict with evaluation_unit_id and annotator_id
    """
    # Format: {evaluation_unit_id}_{annotator_id}
    # annotator_id is the last component
    last_underscore = annotation_id.rfind("_")
    if last_underscore == -1:
        raise ValueError(f"Invalid annotation_id format: {annotation_id}")

    return {
        "evaluation_unit_id": annotation_id[:last_underscore],
        "annotator_id": annotation_id[last_underscore + 1:],
    }


def parse_outcome_id(outcome_id: str) -> Dict[str, Any]:
    """
    Parse outcome ID to extract components.

    Args:
        outcome_id: Outcome ID string

    Returns:
        Dict with evaluation_unit_id and model_name
    """
    # Format: {evaluation_unit_id}_{model_name}
    # model_name is the last component
    last_underscore = outcome_id.rfind("_")
    if last_underscore == -1:
        raise ValueError(f"Invalid outcome_id format: {outcome_id}")

    return {
        "evaluation_unit_id": outcome_id[:last_underscore],
        "model_name": outcome_id[last_underscore + 1:],
    }


# =============================================================================
# Convenience class for ID generation from config
# =============================================================================

class IDGenerator:
    """
    ID generator bound to a specific experiment config.

    Usage:
        generator = IDGenerator(config)
        traj_id = generator.trajectory_id("toolbench", 42)
        pert_id = generator.perturbation_id(traj_id, 3, "wrong_value", "fine_grained")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with experiment config.

        Args:
            config: Full experiment configuration (schema 3.0.0)
        """
        self.config = config
        self.experiment_id = config["experiment"]["id"]

    def trajectory_id(self, benchmark: str, idx: int) -> str:
        """Generate trajectory ID for given benchmark and index."""
        dataset_config = self.config["experiment"]["datasets"].get(benchmark, {})
        return generate_trajectory_id(benchmark, idx, dataset_config)

    def step_id(self, idx: int) -> str:
        """Generate step ID."""
        return generate_step_id(idx)

    def perturbation_id(
        self,
        trajectory_id: str,
        step_idx: int,
        perturbation_type: str,
        perturbation_class: str,
    ) -> str:
        """
        Generate perturbation ID.

        Args:
            trajectory_id: Trajectory ID
            step_idx: Target step index
            perturbation_type: Type (e.g., "wrong_value")
            perturbation_class: Class (e.g., "fine_grained", "placebo", "coarse_grained")
        """
        generator_config = (
            self.config.get("phases", {})
            .get("perturb", {})
            .get("generators", {})
            .get(perturbation_class, {})
        )
        return generate_perturbation_id(
            self.experiment_id,
            trajectory_id,
            step_idx,
            perturbation_type,
            generator_config,
        )

    def evaluation_unit_id(self, trajectory_id: str, perturbation_id: str) -> str:
        """Generate evaluation unit ID from trajectory and perturbation IDs."""
        return generate_evaluation_unit_id(trajectory_id, perturbation_id)

    def annotation_id(self, evaluation_unit_id: str) -> str:
        """Generate annotation ID using annotator from config."""
        annotator_id = (
            self.config.get("phases", {})
            .get("annotate", {})
            .get("annotator_id", "unknown")
        )
        return generate_annotation_id(evaluation_unit_id, annotator_id)

    def outcome_id(self, evaluation_unit_id: str, model_idx: int = 0) -> str:
        """Generate outcome ID for judge model at given index."""
        models = (
            self.config.get("phases", {})
            .get("judge", {})
            .get("models", [])
        )
        if model_idx < len(models):
            model_name = models[model_idx].get("name", f"model_{model_idx}")
        else:
            model_name = f"model_{model_idx}"
        return generate_outcome_id(evaluation_unit_id, model_name)
