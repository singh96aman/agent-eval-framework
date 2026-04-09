"""
ID generation and parsing functions for trajectory evaluation system.

This module implements the ID conventions defined in Section 4.2 of
4_Requirements_EvaluationUnit.MD:

| ID Type               | Format                           | Example                    |
|-----------------------|----------------------------------|----------------------------|
| source_trajectory_id  | {benchmark}_{task_id}            | gaia_122, toolbench_171104 |
| trajectory_variant_id | {source}::base or                | gaia_122::base             |
|                       | {source}::pert::{index:03d}      | gaia_122::pert::001        |
| evaluation_unit_id    | eval::{source}::{index:03d}      | eval::gaia_122::001        |
| canonical_step_id     | {source}::step::{position}       | gaia_122::step::2          |
"""

import re
from typing import Optional, Tuple

# Regex patterns for ID validation
# Source IDs can have various formats depending on benchmark:
# - toolbench: toolbench_123456
# - gaia: gaia_122
# - swebench: swebench_pandas-dev__pandas.95280573.func_pm_ctrl_shuffle__6gupnvfd.8qa84d2e
# The pattern allows any characters except :: (the delimiter)
SOURCE_ID_PATTERN = re.compile(r"^[a-zA-Z][^:]+$")
VARIANT_BASE_PATTERN = re.compile(r"^([^:]+)::base$")
VARIANT_PERT_PATTERN = re.compile(r"^([^:]+)::pert::(\d{3})$")
EVALUATION_UNIT_PATTERN = re.compile(r"^eval::([^:]+)::(\d{3})$")
CANONICAL_STEP_PATTERN = re.compile(r"^([^:]+)::step::(\d+)$")


def generate_trajectory_variant_id(
    source_id: str, variant_type: str, index: Optional[int] = None
) -> str:
    """
    Generate a trajectory variant ID.

    Args:
        source_id: The source trajectory ID (e.g., 'gaia_122')
        variant_type: Either 'base' for baseline or 'pert' for perturbed
        index: Required for 'pert' variant_type, the perturbation index (0-999)

    Returns:
        The trajectory variant ID string

    Raises:
        ValueError: If variant_type is invalid, or index is missing/invalid for 'pert'

    Examples:
        >>> generate_trajectory_variant_id('gaia_122', 'base')
        'gaia_122::base'
        >>> generate_trajectory_variant_id('gaia_122', 'pert', 1)
        'gaia_122::pert::001'
    """
    if variant_type not in ("base", "pert"):
        raise ValueError(f"variant_type must be 'base' or 'pert', got '{variant_type}'")

    if variant_type == "base":
        return f"{source_id}::base"
    else:
        if index is None:
            raise ValueError("index is required for 'pert' variant_type")
        if not isinstance(index, int) or index < 0 or index > 999:
            raise ValueError(f"index must be an integer between 0 and 999, got {index}")
        return f"{source_id}::pert::{index:03d}"


def generate_evaluation_unit_id(source_id: str, perturbation_index: int) -> str:
    """
    Generate an evaluation unit ID.

    Args:
        source_id: The source trajectory ID (e.g., 'gaia_122')
        perturbation_index: The perturbation index (0-999)

    Returns:
        The evaluation unit ID string

    Raises:
        ValueError: If perturbation_index is out of valid range

    Examples:
        >>> generate_evaluation_unit_id('gaia_122', 1)
        'eval::gaia_122::001'
    """
    if (
        not isinstance(perturbation_index, int)
        or perturbation_index < 0
        or perturbation_index > 999
    ):
        raise ValueError(
            f"perturbation_index must be an integer between 0 and 999, got {perturbation_index}"
        )
    return f"eval::{source_id}::{perturbation_index:03d}"


def generate_canonical_step_id(source_trajectory_id: str, step_position: int) -> str:
    """
    Generate a canonical step ID.

    Args:
        source_trajectory_id: The source trajectory ID (e.g., 'gaia_122')
        step_position: The step position (0 or positive integer)

    Returns:
        The canonical step ID string

    Raises:
        ValueError: If step_position is negative

    Examples:
        >>> generate_canonical_step_id('gaia_122', 2)
        'gaia_122::step::2'
    """
    if not isinstance(step_position, int) or step_position < 0:
        raise ValueError(
            f"step_position must be a non-negative integer, got {step_position}"
        )
    return f"{source_trajectory_id}::step::{step_position}"


def parse_trajectory_variant_id(variant_id: str) -> Tuple[str, str, Optional[int]]:
    """
    Parse a trajectory variant ID into its components.

    Args:
        variant_id: The trajectory variant ID to parse

    Returns:
        A tuple of (source_id, variant_type, index) where:
        - source_id: The source trajectory ID
        - variant_type: 'base' or 'pert'
        - index: The perturbation index (int) for 'pert', None for 'base'

    Raises:
        ValueError: If the variant_id format is invalid

    Examples:
        >>> parse_trajectory_variant_id('gaia_122::base')
        ('gaia_122', 'base', None)
        >>> parse_trajectory_variant_id('gaia_122::pert::001')
        ('gaia_122', 'pert', 1)
    """
    base_match = VARIANT_BASE_PATTERN.match(variant_id)
    if base_match:
        return (base_match.group(1), "base", None)

    pert_match = VARIANT_PERT_PATTERN.match(variant_id)
    if pert_match:
        return (pert_match.group(1), "pert", int(pert_match.group(2)))

    raise ValueError(f"Invalid trajectory variant ID format: '{variant_id}'")


def parse_evaluation_unit_id(unit_id: str) -> Tuple[str, int]:
    """
    Parse an evaluation unit ID into its components.

    Args:
        unit_id: The evaluation unit ID to parse

    Returns:
        A tuple of (source_id, perturbation_index)

    Raises:
        ValueError: If the unit_id format is invalid

    Examples:
        >>> parse_evaluation_unit_id('eval::gaia_122::001')
        ('gaia_122', 1)
    """
    match = EVALUATION_UNIT_PATTERN.match(unit_id)
    if not match:
        raise ValueError(f"Invalid evaluation unit ID format: '{unit_id}'")
    return (match.group(1), int(match.group(2)))


def parse_canonical_step_id(step_id: str) -> Tuple[str, int]:
    """
    Parse a canonical step ID into its components.

    Args:
        step_id: The canonical step ID to parse

    Returns:
        A tuple of (source_trajectory_id, step_position)

    Raises:
        ValueError: If the step_id format is invalid

    Examples:
        >>> parse_canonical_step_id('gaia_122::step::2')
        ('gaia_122', 2)
    """
    match = CANONICAL_STEP_PATTERN.match(step_id)
    if not match:
        raise ValueError(f"Invalid canonical step ID format: '{step_id}'")
    return (match.group(1), int(match.group(2)))


def validate_id_format(id_str: str, id_type: str) -> bool:
    """
    Validate that an ID string matches the expected format for its type.

    Args:
        id_str: The ID string to validate
        id_type: One of 'source', 'variant', 'evaluation_unit', or 'canonical_step'

    Returns:
        True if the ID format is valid, False otherwise

    Raises:
        ValueError: If id_type is not recognized

    Examples:
        >>> validate_id_format('gaia_122', 'source')
        True
        >>> validate_id_format('gaia_122::base', 'variant')
        True
        >>> validate_id_format('gaia_122::pert::001', 'variant')
        True
        >>> validate_id_format('eval::gaia_122::001', 'evaluation_unit')
        True
        >>> validate_id_format('gaia_122::step::2', 'canonical_step')
        True
        >>> validate_id_format('invalid', 'source')
        False
    """
    if id_type == "source":
        return bool(SOURCE_ID_PATTERN.match(id_str))
    elif id_type == "variant":
        return bool(
            VARIANT_BASE_PATTERN.match(id_str) or VARIANT_PERT_PATTERN.match(id_str)
        )
    elif id_type == "evaluation_unit":
        return bool(EVALUATION_UNIT_PATTERN.match(id_str))
    elif id_type == "canonical_step":
        return bool(CANONICAL_STEP_PATTERN.match(id_str))
    else:
        raise ValueError(
            f"id_type must be one of 'source', 'variant', 'evaluation_unit', "
            f"'canonical_step', got '{id_type}'"
        )
