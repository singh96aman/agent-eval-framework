"""
Prompt Registry for config-driven prompt management.

Provides centralized lookup of prompts by name, enabling:
- A/B testing of prompt variations
- Reproducible prompt versioning
- Config-driven prompt selection

Usage:
    from src.prompts.registry import get_prompt

    prompt = get_prompt("BLINDED_PAIR_SYSTEM_V2")
"""

from typing import Dict, Optional

from src.prompts.judge_prompts import JUDGE_PROMPTS
from src.prompts.perturbation_prompts import PERTURBATION_PROMPTS


# Combined registry of all prompts
PROMPT_REGISTRY: Dict[str, str] = {
    **JUDGE_PROMPTS,
    **PERTURBATION_PROMPTS,
}


def get_prompt(name: str) -> str:
    """
    Get a prompt by name from the registry.

    Args:
        name: Prompt name (e.g., "BLINDED_PAIR_SYSTEM_V2")

    Returns:
        Prompt template string

    Raises:
        KeyError: If prompt name not found in registry
    """
    if name not in PROMPT_REGISTRY:
        available = ", ".join(sorted(PROMPT_REGISTRY.keys()))
        raise KeyError(
            f"Prompt '{name}' not found in registry. "
            f"Available prompts: {available}"
        )
    return PROMPT_REGISTRY[name]


def get_prompt_safe(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a prompt by name, returning default if not found.

    Args:
        name: Prompt name
        default: Value to return if not found

    Returns:
        Prompt template string or default
    """
    return PROMPT_REGISTRY.get(name, default)


def list_prompts(category: Optional[str] = None) -> Dict[str, str]:
    """
    List available prompts, optionally filtered by category.

    Args:
        category: Optional filter ("judge", "perturbation", or None for all)

    Returns:
        Dict of prompt names to templates
    """
    if category is None:
        return PROMPT_REGISTRY.copy()
    elif category.lower() == "judge":
        return JUDGE_PROMPTS.copy()
    elif category.lower() == "perturbation":
        return PERTURBATION_PROMPTS.copy()
    else:
        raise ValueError(f"Unknown category: {category}. Use 'judge' or 'perturbation'.")


def validate_prompt_config(config: Dict) -> bool:
    """
    Validate that all prompt references in a config exist in the registry.

    Args:
        config: Experiment config dict

    Returns:
        True if all prompts are valid

    Raises:
        ValueError: If any prompt reference is invalid
    """
    errors = []

    # Check prompts section
    prompts_config = config.get("prompts", {})

    # Check judge prompts
    judge_config = prompts_config.get("judge", {})
    for mode, mode_prompts in judge_config.items():
        if isinstance(mode_prompts, dict):
            for prompt_type, prompt_name in mode_prompts.items():
                if prompt_name not in PROMPT_REGISTRY:
                    errors.append(f"judge.{mode}.{prompt_type}: {prompt_name}")

    # Check perturbation prompts
    pert_config = prompts_config.get("perturbation", {})
    for pert_type, prompt_name in pert_config.items():
        if prompt_name not in PROMPT_REGISTRY:
            errors.append(f"perturbation.{pert_type}: {prompt_name}")

    if errors:
        raise ValueError(
            f"Invalid prompt references in config: {', '.join(errors)}. "
            f"Available prompts: {', '.join(sorted(PROMPT_REGISTRY.keys()))}"
        )

    return True


# Aliases for backward compatibility
BLINDED_PAIR_SYSTEM = JUDGE_PROMPTS["BLINDED_PAIR_SYSTEM_V1"]
BLINDED_PAIR_USER = JUDGE_PROMPTS["BLINDED_PAIR_USER_V1"]
SINGLE_TRAJECTORY_SYSTEM = JUDGE_PROMPTS["SINGLE_TRAJECTORY_SYSTEM_V1"]
SINGLE_TRAJECTORY_USER = JUDGE_PROMPTS["SINGLE_TRAJECTORY_USER_V1"]
LABELED_PAIR_SYSTEM = JUDGE_PROMPTS["LABELED_PAIR_SYSTEM_V1"]
LABELED_PAIR_USER = JUDGE_PROMPTS["LABELED_PAIR_USER_V1"]
