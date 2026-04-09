"""
Assign replay tier based on evaluation capabilities.

This module implements the tier assignment logic from Section 4.4 of the paper,
determining which level of outcome evidence is achievable for a given trajectory
based on its capabilities.
"""

from typing import Dict, Optional


def assign_replay_tier(capabilities: Dict[str, bool]) -> Optional[int]:
    """
    Assign a replay tier based on the provided capabilities.

    The tier system (from Section 4.4) determines what level of outcome evidence
    can be obtained for evaluating a trajectory:

    - **Tier 1**: Full replay with objective verification
      Requires: can_replay AND environment_accessible AND has_objective_verifier
      Provides: Complete causal evidence of error impact

    - **Tier 2**: Downstream regeneration with objective verification
      Requires: can_regenerate_downstream AND has_objective_verifier
      Provides: Evidence of downstream impact without full replay

    - **Tier 3**: Ground truth or objective verification available
      Requires: ground_truth_available OR has_objective_verifier
      Provides: Reference comparison without causal replay

    - **None**: No outcome evidence possible
      When no tier conditions are met

    Args:
        capabilities: A dictionary containing boolean capability flags:
            - can_replay: Whether the trajectory can be replayed
            - environment_accessible: Whether the environment is accessible
            - has_objective_verifier: Whether an objective verifier exists
            - can_regenerate_downstream: Whether downstream can be regenerated
            - ground_truth_available: Whether ground truth exists

    Returns:
        The assigned tier (1, 2, or 3) or None if no outcome evidence is possible.

    Raises:
        KeyError: If required capability fields are missing from the input.
    """
    # Extract capabilities with explicit key access to ensure all required fields exist
    can_replay = capabilities["can_replay"]
    environment_accessible = capabilities["environment_accessible"]
    has_objective_verifier = capabilities["has_objective_verifier"]
    can_regenerate_downstream = capabilities["can_regenerate_downstream"]
    ground_truth_available = capabilities["ground_truth_available"]

    # Tier 1: Full replay capability
    # Can replay in accessible environment with objective verification
    if can_replay and environment_accessible and has_objective_verifier:
        return 1

    # Tier 2: Downstream regeneration capability
    # Can regenerate downstream steps with objective verification
    if can_regenerate_downstream and has_objective_verifier:
        return 2

    # Tier 3: Reference-based evaluation
    # Has either ground truth or objective verifier for comparison
    if ground_truth_available or has_objective_verifier:
        return 3

    # No outcome evidence possible
    return None
