"""
View generation modules for evaluation units.

This package provides functions to generate different views of evaluation units
for various evaluation contexts:

- human: Views for human evaluators (detectability, consequence, preference)
- llm_judge: Views for LLM judge evaluation (single, blinded_pair, labeled_pair)
- outcome: Views for outcome verification (replay, regeneration, grading)
"""

from src.evaluation.views.human import (
    create_simplified_step,
    generate_consequence_view,
    generate_detectability_view,
    generate_human_view,
    generate_preference_view,
)
from src.evaluation.views.llm_judge import (
    generate_blinded_pair_view,
    generate_labeled_pair_view,
    generate_single_trajectory_view,
    generate_view,
    get_supported_modes,
    validate_mode_consistency,
)
from src.evaluation.views.outcome import (
    generate_outcome_view,
    generate_outcome_views_batch,
    get_verification_config,
)

__all__ = [
    # Human evaluator views
    "create_simplified_step",
    "generate_detectability_view",
    "generate_consequence_view",
    "generate_preference_view",
    "generate_human_view",
    # LLM Judge views
    "generate_blinded_pair_view",
    "generate_labeled_pair_view",
    "generate_single_trajectory_view",
    "generate_view",
    "get_supported_modes",
    "validate_mode_consistency",
    # Outcome verification views
    "generate_outcome_view",
    "generate_outcome_views_batch",
    "get_verification_config",
]
