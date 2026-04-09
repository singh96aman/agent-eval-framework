"""
Compute evaluation capabilities from trajectory and benchmark properties.

This module provides functions to determine what evaluation methods are available
for a given trajectory based on its benchmark source and trajectory-specific properties.
"""

from typing import Dict, Any

# Benchmark default capability configurations
BENCHMARK_DEFAULTS: Dict[str, Dict[str, bool]] = {
    "swebench": {
        "can_replay": True,
        "environment_accessible": True,
        "has_objective_verifier": True,
        "can_regenerate_downstream": True,
        "ground_truth_available": True,
    },
    "gaia": {
        "can_replay": False,
        "environment_accessible": False,
        "has_objective_verifier": True,
        "can_regenerate_downstream": False,  # varies, default to False
        "ground_truth_available": True,
    },
    "toolbench": {
        "can_replay": False,
        "environment_accessible": False,
        "has_objective_verifier": False,  # partial, default to False
        "can_regenerate_downstream": False,
        "ground_truth_available": False,  # partial, default to False
    },
}


def get_benchmark_defaults(benchmark: str) -> Dict[str, bool]:
    """
    Returns default capability flags for a given benchmark.

    Args:
        benchmark: The benchmark name (e.g., 'swebench', 'gaia', 'toolbench').

    Returns:
        A dictionary with the following boolean capability flags:
        - can_replay: Whether the trajectory can be replayed in the environment
        - environment_accessible: Whether the execution environment is accessible
        - has_objective_verifier: Whether an objective verifier exists for outcomes
        - can_regenerate_downstream: Whether downstream steps can be regenerated
        - ground_truth_available: Whether ground truth answers/outcomes exist

    Raises:
        ValueError: If the benchmark is not recognized.
    """
    benchmark_lower = benchmark.lower()

    if benchmark_lower not in BENCHMARK_DEFAULTS:
        raise ValueError(
            f"Unknown benchmark: '{benchmark}'. "
            f"Supported benchmarks: {list(BENCHMARK_DEFAULTS.keys())}"
        )

    # Return a copy to prevent mutation of defaults
    return BENCHMARK_DEFAULTS[benchmark_lower].copy()


def compute_capabilities(trajectory: Dict[str, Any], benchmark: str) -> Dict[str, bool]:
    """
    Compute evaluation capabilities for a trajectory based on benchmark defaults
    and trajectory-specific overrides.

    The function starts with benchmark defaults and then overrides specific
    capabilities if the trajectory contains explicit values for them.

    Args:
        trajectory: A dictionary containing trajectory data. May include optional
            fields that override benchmark defaults:
            - has_objective_verifier: bool
            - ground_truth_available: bool
            - can_replay: bool
        benchmark: The benchmark name (e.g., 'swebench', 'gaia', 'toolbench').

    Returns:
        A dictionary with all 5 capability flags:
        - can_replay
        - environment_accessible
        - has_objective_verifier
        - can_regenerate_downstream
        - ground_truth_available

    Raises:
        ValueError: If the benchmark is not recognized.
    """
    # Start with benchmark defaults
    capabilities = get_benchmark_defaults(benchmark)

    # Override fields based on trajectory properties if they exist
    override_fields = [
        "has_objective_verifier",
        "ground_truth_available",
        "can_replay",
    ]

    for field in override_fields:
        if field in trajectory and isinstance(trajectory[field], bool):
            capabilities[field] = trajectory[field]

    return capabilities
