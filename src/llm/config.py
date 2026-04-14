"""
Central LLM configuration.

Loads settings from experiment config or uses defaults.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# =============================================================================
# Default experiment config path
# =============================================================================

DEFAULT_CONFIG_PATH = "config/experiments/v2/pocv2/trajectory_sampling_v8.json"  # noqa: E501

# Cached config
_config_cache: Optional[Dict[str, Any]] = None


def load_experiment_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load experiment config from JSON file."""
    global _config_cache

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # Use cache if available and same path
    if _config_cache is not None:
        return _config_cache

    # Find config file relative to project root
    path = Path(config_path)
    if not path.is_absolute():
        # Try relative to cwd first, then project root
        if not path.exists():
            project_root = Path(__file__).parent.parent.parent
            path = project_root / config_path

    if path.exists():
        with open(path) as f:
            _config_cache = json.load(f)
            return _config_cache

    # Return empty dict if not found
    return {}


def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration from experiment config."""
    config = load_experiment_config()
    return config.get("generation", {}).get("llm", {})


# =============================================================================
# Model IDs and settings (from config or defaults)
# =============================================================================

def get_default_model_id() -> str:
    """Get default model ID from config."""
    llm_config = get_llm_config()
    return llm_config.get(
        "model",
        "global.anthropic.claude-sonnet-4-20250514-v1:0"
    )


def get_default_region() -> str:
    """Get AWS region from config or environment."""
    llm_config = get_llm_config()
    return llm_config.get(
        "region",
        os.environ.get("AWS_REGION", "us-east-1")
    )


def get_default_max_tokens() -> int:
    """Get max tokens from config."""
    llm_config = get_llm_config()
    return llm_config.get("max_tokens", 500)


def get_default_temperature() -> float:
    """Get temperature from config."""
    llm_config = get_llm_config()
    return llm_config.get("temperature", 0.3)


# Convenience exports (evaluated at import time)
DEFAULT_MODEL_ID = get_default_model_id()
DEFAULT_REGION = get_default_region()
DEFAULT_MAX_TOKENS = get_default_max_tokens()
DEFAULT_TEMPERATURE = get_default_temperature()

# Alternative models (for manual override)
MODELS = {
    "sonnet": "global.anthropic.claude-sonnet-4-20250514-v1:0",
    "haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "opus": "us.anthropic.claude-opus-4-20250514-v1:0",
}


def get_model_id(model_name: Optional[str] = None) -> str:
    """
    Get model ID by name or return default from config.

    Args:
        model_name: Optional model name ("sonnet", "haiku", "opus")
                   If None, returns model from experiment config

    Returns:
        Bedrock inference profile ID
    """
    if model_name is None:
        return get_default_model_id()
    return MODELS.get(model_name.lower(), get_default_model_id())
