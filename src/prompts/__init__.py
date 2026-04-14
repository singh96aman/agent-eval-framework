"""
Prompt Registry Module.

Provides centralized, versioned, config-driven prompt management.

Usage:
    from src.prompts import get_prompt

    # Get a specific prompt version
    prompt = get_prompt("BLINDED_PAIR_SYSTEM_V2")

    # List available prompts
    from src.prompts import list_prompts
    all_prompts = list_prompts()
    judge_prompts = list_prompts(category="judge")

    # Validate config prompt references
    from src.prompts import validate_prompt_config
    validate_prompt_config(experiment_config)
"""

from src.prompts.registry import (
    get_prompt,
    get_prompt_safe,
    list_prompts,
    validate_prompt_config,
    PROMPT_REGISTRY,
    # Backward compatibility aliases
    BLINDED_PAIR_SYSTEM,
    BLINDED_PAIR_USER,
    SINGLE_TRAJECTORY_SYSTEM,
    SINGLE_TRAJECTORY_USER,
    LABELED_PAIR_SYSTEM,
    LABELED_PAIR_USER,
)

from src.prompts.judge_prompts import (
    JUDGE_PROMPTS,
    SINGLE_TRAJECTORY_SYSTEM_V1,
    SINGLE_TRAJECTORY_USER_V1,
    BLINDED_PAIR_SYSTEM_V1,
    BLINDED_PAIR_SYSTEM_V2,
    BLINDED_PAIR_USER_V1,
    BLINDED_PAIR_USER_V2,
    LABELED_PAIR_SYSTEM_V1,
    LABELED_PAIR_USER_V1,
)

from src.prompts.perturbation_prompts import (
    PERTURBATION_PROMPTS,
    PARAPHRASE_PROMPT_V1,
    WRONG_PLAN_PROMPT_V1,
    WRONG_PARAMETER_PROMPT_V1,
    VALUE_MUTATION_PROMPT_V1,
    WRONG_DATE_PROMPT_V1,
    WRONG_IDENTIFIER_PROMPT_V1,
)

__all__ = [
    # Core registry functions
    "get_prompt",
    "get_prompt_safe",
    "list_prompts",
    "validate_prompt_config",
    "PROMPT_REGISTRY",
    # Judge prompts
    "JUDGE_PROMPTS",
    "SINGLE_TRAJECTORY_SYSTEM_V1",
    "SINGLE_TRAJECTORY_USER_V1",
    "BLINDED_PAIR_SYSTEM_V1",
    "BLINDED_PAIR_SYSTEM_V2",
    "BLINDED_PAIR_USER_V1",
    "BLINDED_PAIR_USER_V2",
    "LABELED_PAIR_SYSTEM_V1",
    "LABELED_PAIR_USER_V1",
    # Perturbation prompts
    "PERTURBATION_PROMPTS",
    "PARAPHRASE_PROMPT_V1",
    "WRONG_PLAN_PROMPT_V1",
    "WRONG_PARAMETER_PROMPT_V1",
    "VALUE_MUTATION_PROMPT_V1",
    "WRONG_DATE_PROMPT_V1",
    "WRONG_IDENTIFIER_PROMPT_V1",
    # Backward compatibility
    "BLINDED_PAIR_SYSTEM",
    "BLINDED_PAIR_USER",
    "SINGLE_TRAJECTORY_SYSTEM",
    "SINGLE_TRAJECTORY_USER",
    "LABELED_PAIR_SYSTEM",
    "LABELED_PAIR_USER",
]
