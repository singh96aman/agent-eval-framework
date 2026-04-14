"""
Pydantic schema for experiment configuration (v3.0.0).

Validates that config has exactly: schema, experiment, phases.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

CURRENT_SCHEMA_VERSION = "3.0.0"


# =============================================================================
# Storage & Output
# =============================================================================
class StorageConfig(BaseModel):
    """MongoDB storage configuration."""

    backend: Literal["mongodb"] = "mongodb"
    database: str


class MongoDBOutputConfig(BaseModel):
    """MongoDB output collections."""

    collection: str = "perturbed_trajectories"
    index_collection: str = "perturbation_index"


class JsonOutputConfig(BaseModel):
    """JSON file output configuration."""

    dir: str = "data/perturbed"


class ResultsOutputConfig(BaseModel):
    """Results output configuration."""

    dir: str = "results/v8"
    checkpoints: bool = True


class OutputConfig(BaseModel):
    """Output configuration."""

    mongodb: Optional[MongoDBOutputConfig] = None
    json_output: Optional[JsonOutputConfig] = Field(None, alias="json")
    results: Optional[ResultsOutputConfig] = None

    class Config:
        populate_by_name = True


# =============================================================================
# Targets & Datasets
# =============================================================================
class PerTrajectoryTarget(BaseModel):
    """Per-trajectory perturbation target."""

    target: int = 3


class ClassDistribution(BaseModel):
    """Perturbation class distribution."""

    placebo: float = 0.20
    fine_grained: float = 0.50
    coarse_grained: float = 0.30

    @model_validator(mode="after")
    def validate_sum(self):
        total = self.placebo + self.fine_grained + self.coarse_grained
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Class distribution must sum to 1.0, got {total}")
        return self


class TargetsConfig(BaseModel):
    """Perturbation targets configuration."""

    total_perturbations: int = 1500
    per_trajectory: PerTrajectoryTarget = Field(default_factory=PerTrajectoryTarget)
    by_class: ClassDistribution = Field(default_factory=ClassDistribution)


class DatasetFilters(BaseModel):
    """Dataset filtering options."""

    min_steps: int = 3
    max_steps: int = 50
    require_task_success: bool = True
    require_grader_aligned: Optional[bool] = None


class DatasetConfig(BaseModel):
    """Individual dataset configuration."""

    enabled: bool = True
    source: Optional[str] = None
    path: Optional[str] = None
    split: Optional[str] = None
    filters: Optional[DatasetFilters] = None


class DatasetsConfig(BaseModel):
    """All datasets configuration."""

    toolbench: Optional[DatasetConfig] = None
    gaia: Optional[DatasetConfig] = None
    swebench: Optional[DatasetConfig] = None


# =============================================================================
# Experiment
# =============================================================================
class ExperimentSection(BaseModel):
    """Experiment metadata and global settings."""

    id: str
    name: str
    description: Optional[str] = None
    random_seed: int = 42
    storage: StorageConfig
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# =============================================================================
# Phase: Load
# =============================================================================
class GraderConfig(BaseModel):
    """Grader configuration."""

    type: str = "HeuristicGrader"
    min_pass_rate: float = 0.30


class SamplingFilters(BaseModel):
    """Sampling filter configuration."""

    min_steps: int = 3
    max_steps: int = 50
    require_task_success: bool = True
    require_grader_passed: bool = True


class SamplingConfig(BaseModel):
    """Sampling configuration for load phase."""

    seed: int = 42
    targets: Dict[str, int] = Field(default_factory=dict)
    require_grader_aligned: bool = True
    grader: GraderConfig = Field(default_factory=GraderConfig)
    filters: SamplingFilters = Field(default_factory=SamplingFilters)


class QualityGateConfig(BaseModel):
    """Generic quality gate configuration."""

    min: Optional[float] = None
    max: Optional[float] = None
    require: Optional[bool] = None
    max_rate: Optional[float] = None
    min_rate: Optional[float] = None
    max_last_step_rate: Optional[float] = None
    min_std: Optional[float] = None

    class Config:
        extra = "allow"


class BaselineVerifyPromptsConfig(BaseModel):
    """Baseline verification prompts configuration."""

    system: str = "SINGLE_TRAJECTORY_SYSTEM_V2"
    user: str = "SINGLE_TRAJECTORY_USER_V2"


class LoadPhaseConfig(BaseModel):
    """Load phase configuration."""

    enabled: bool = True
    sampling: Optional[SamplingConfig] = None
    quality_gates: Optional[Dict[str, QualityGateConfig]] = None

    # Baseline verification settings (run Claude on trajectories before storing)
    baseline_verify: bool = False
    baseline_verify_model: Optional["JudgeModelConfig"] = None
    baseline_verify_prompts: BaselineVerifyPromptsConfig = Field(
        default_factory=BaselineVerifyPromptsConfig
    )
    baseline_verify_parallelism: int = 4


# =============================================================================
# Phase: Perturb
# =============================================================================
class LLMConfig(BaseModel):
    """LLM configuration for perturbation generation."""

    provider: Literal["bedrock", "openai", "anthropic"] = "bedrock"
    region: Optional[str] = None
    model: str
    max_tokens: int = 2000
    temperature: float = 0.3
    use_for: List[str] = Field(default_factory=list)


class GeneratorClassConfig(BaseModel):
    """Generator configuration for a perturbation class."""

    types: List[str] = Field(default_factory=list)
    llm_required: Optional[List[str]] = None
    excluded: Optional[List[str]] = None


class GeneratorsConfig(BaseModel):
    """All generators configuration."""

    placebo: Optional[GeneratorClassConfig] = None
    fine_grained: Optional[GeneratorClassConfig] = None
    coarse_grained: Optional[GeneratorClassConfig] = None


class ClassValidationModelConfig(BaseModel):
    """Model configuration for class validation."""

    provider: Literal["bedrock", "openai", "anthropic"] = "bedrock"
    model: str
    max_tokens: int = 500
    temperature: float = 0.0


class ClassValidationConfig(BaseModel):
    """Class validation configuration for perturb phase."""

    enabled: bool = False
    model: Optional[ClassValidationModelConfig] = None
    prompt: Optional[str] = None
    min_match_rate: float = 0.90


class PerturbPhaseConfig(BaseModel):
    """Perturb phase configuration."""

    enabled: bool = True
    parallelism: int = 4
    llm: Optional[LLMConfig] = None
    prompts: Optional[Dict[str, str]] = None
    generators: Optional[GeneratorsConfig] = None
    class_validation: Optional[ClassValidationConfig] = None
    quality_gates: Optional[Dict[str, QualityGateConfig]] = None


# =============================================================================
# Phase: Evaluation Unit
# =============================================================================
class BlindingConfig(BaseModel):
    """Blinding configuration."""

    algorithm: Literal["hash_based", "random"] = "hash_based"
    seed_source: str = "experiment.random_seed"
    verify_balance: bool = True
    balance_range: List[float] = Field(default=[0.45, 0.55])


class EvaluationUnitPhaseConfig(BaseModel):
    """Evaluation unit phase configuration."""

    enabled: bool = True
    blinding: Optional[BlindingConfig] = None
    quality_gates: Optional[Dict[str, QualityGateConfig]] = None


# =============================================================================
# Phase: Annotate
# =============================================================================
class StratificationConfig(BaseModel):
    """Stratification configuration for annotation sampling."""

    by_class: Optional[ClassDistribution] = None
    by_position: bool = True
    by_benchmark: bool = True
    tolerance: float = 0.05


class AnnotatePhaseConfig(BaseModel):
    """Annotate phase configuration."""

    enabled: bool = True
    schema_type: Optional[str] = Field(None, alias="schema")
    annotator_id: Optional[str] = None
    mode: Literal["blinded_pair", "labeled_pair", "single_trajectory"] = "blinded_pair"
    sample_size: int = 50
    output_dir: str = "data/human_labels"
    storage_backend: str = "mongodb"
    resume: bool = True
    stratification: Optional[StratificationConfig] = None

    class Config:
        populate_by_name = True


# =============================================================================
# Phase: Judge
# =============================================================================
class JudgeModelConfig(BaseModel):
    """Judge model configuration."""

    name: str
    provider: Literal["bedrock", "openai", "anthropic"] = "bedrock"
    model: str
    max_tokens: int = 1500
    temperature: float = 0.0


class JudgePromptConfig(BaseModel):
    """Judge prompt configuration."""

    system: str
    user: str


class JudgePromptsConfig(BaseModel):
    """All judge prompts configuration."""

    single_trajectory: Optional[JudgePromptConfig] = None
    blinded_pair: Optional[JudgePromptConfig] = None
    labeled_pair: Optional[JudgePromptConfig] = None


class JudgeFilterConfig(BaseModel):
    """Judge filter configuration."""

    annotated_only: bool = False


class JudgePhaseConfig(BaseModel):
    """Judge phase configuration."""

    enabled: bool = True
    version: Optional[str] = None  # Version for output isolation (e.g., "v2")
    models: List[JudgeModelConfig] = Field(default_factory=list)
    mode: Literal["blinded_pair", "labeled_pair", "single_trajectory"] = "blinded_pair"
    modes: Optional[List[str]] = None  # List of modes to run
    prompts: Optional[JudgePromptsConfig] = None
    parallelism: int = 4
    filter: Optional[JudgeFilterConfig] = None


# =============================================================================
# Phase: Compute
# =============================================================================
class JudgeEvalConfig(BaseModel):
    """Judge evaluation configuration."""

    judge: str = "claude-sonnet-4"
    mode: str = "blinded_pair"
    samples_per_unit: int = 1
    parallelization: int = 2


class OutcomeEvidenceConfig(BaseModel):
    """Outcome evidence configuration."""

    resume: bool = True
    quality_gates: Optional[Dict[str, QualityGateConfig]] = None


class ComputePhaseConfig(BaseModel):
    """Compute phase configuration."""

    enabled: bool = True
    version: Optional[str] = None  # Version for filtering judge outputs
    targets: List[str] = Field(default_factory=list)
    judge_eval: Optional[JudgeEvalConfig] = None
    outcome_evidence: Optional[OutcomeEvidenceConfig] = None


# =============================================================================
# Phases
# =============================================================================
class PhasesConfig(BaseModel):
    """All phases configuration."""

    load: Optional[LoadPhaseConfig] = None
    perturb: Optional[PerturbPhaseConfig] = None
    evaluation_unit: Optional[EvaluationUnitPhaseConfig] = None
    annotate: Optional[AnnotatePhaseConfig] = None
    judge: Optional[JudgePhaseConfig] = None
    compute: Optional[ComputePhaseConfig] = None

    class Config:
        extra = "forbid"


# =============================================================================
# Root Config
# =============================================================================
class ExperimentConfig(BaseModel):
    """
    Root experiment configuration (schema 3.0.0).

    Only allows: schema, experiment, phases as top-level keys.
    """

    schema_version: str = Field(alias="schema")
    experiment: ExperimentSection
    phases: PhasesConfig

    class Config:
        extra = "forbid"
        populate_by_name = True

    @model_validator(mode="after")
    def validate_schema_version(self):
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: expected {CURRENT_SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )
        return self


def load_and_validate_config(config_path: str | Path) -> ExperimentConfig:
    """
    Load and validate an experiment configuration file.

    Args:
        config_path: Path to the JSON config file

    Returns:
        Validated ExperimentConfig

    Raises:
        ValidationError: If config is invalid
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        data = json.load(f)

    return ExperimentConfig.model_validate(data)


def validate_config_dict(data: Dict[str, Any]) -> ExperimentConfig:
    """
    Validate a config dictionary.

    Args:
        data: Config dictionary

    Returns:
        Validated ExperimentConfig

    Raises:
        ValidationError: If config is invalid
    """
    return ExperimentConfig.model_validate(data)
