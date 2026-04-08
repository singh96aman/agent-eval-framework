"""
Schema definitions for typed trajectory representation.

Defines TypedStep, TypedTrajectory, and supporting dataclasses
for the enriched trajectory format per 2_Requirements_TypedRepresentation.MD.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class StepRole(Enum):
    """Role classification for trajectory steps."""
    PLANNING = "planning"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    REASONING = "reasoning"
    EXTRACTION = "extraction"
    DECISION = "decision"
    FINAL_RESPONSE = "final_response"


class ArtifactType(Enum):
    """Type of artifact produced by a step."""
    SEARCH_RESULTS = "search_results"
    FILEPATH = "filepath"
    PATCH = "patch"
    LINE_NUMBER = "line_number"
    NUMERIC_ANSWER = "numeric_answer"
    CODE_SNIPPET = "code_snippet"
    DIAGNOSIS = "diagnosis"
    API_RESPONSE = "api_response"
    TEXT = "text"


class DependencyType(Enum):
    """Type of dependency between steps."""
    USES_OBSERVATION_FROM = "uses_observation_from"
    USES_EXTRACTED_VALUE_FROM = "uses_extracted_value_from"
    FOLLOWS_PLAN_FROM = "follows_plan_from"
    ACTS_ON_ENTITY_FROM = "acts_on_entity_from"


class ValueType(Enum):
    """Type of value in perturbable slots and extractions."""
    # Tool argument types
    FILEPATH = "filepath"
    LINE_NUMBER = "line_number"
    INTEGER = "integer"
    FLOAT = "float"
    SEARCH_QUERY = "search_query"
    CODE_SNIPPET = "code_snippet"
    API_ENDPOINT = "api_endpoint"
    IDENTIFIER = "identifier"
    ENTITY_NAME = "entity_name"
    DATE = "date"
    URL = "url"
    # Network types
    IPV4 = "ipv4"
    DOMAIN = "domain"
    EMAIL = "email"
    # Coordinate types
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    # Tool types
    TOOL = "tool"
    # Generic
    STRING = "string"
    OBJECT = "object"
    BOOLEAN = "boolean"
    JSON_OBJECT = "json_object"
    # Domain-specific
    FOOD_ITEM = "food_item"
    LOCATION = "location"
    PERSON = "person"
    ORGANIZATION = "organization"


@dataclass
class ProvenanceField:
    """
    Field with provenance tracking for derived annotations.

    Attributes:
        value: The actual value (bool, float, etc.)
        source: How the value was derived ("heuristic", "llm", "human")
        confidence: Confidence score (0-1) or None if not applicable
    """
    value: Union[bool, float, int, str]
    source: str  # "heuristic" | "llm" | "human"
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceField":
        return cls(**data)


@dataclass
class ExtractionProvenance:
    """
    Provenance information for extraction steps.

    Tracks how a value was extracted and validates the extraction.

    Attributes:
        extraction_method: How the value was extracted
        evidence_in_content: Whether the pattern actually matched in content
        source_tool_name: Tool that produced the source value
        confidence: Confidence score (0-1)
    """
    extraction_method: str  # "regex_numeric", "regex_filepath", "pattern_match"
    evidence_in_content: bool
    source_tool_name: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extraction_method": self.extraction_method,
            "evidence_in_content": self.evidence_in_content,
            "source_tool_name": self.source_tool_name,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionProvenance":
        return cls(**data)


@dataclass
class DependencyEdge:
    """
    Typed dependency between steps.

    Attributes:
        type: Kind of dependency (uses_observation_from, etc.)
        source_step: Index of the step this depends on
        reason: Human-readable explanation
        evidence: Grounded evidence for this dependency (artifact name, file path, etc.)
    """
    type: str  # DependencyType value
    source_step: int
    reason: str
    evidence: Optional[str] = None  # e.g., "artifact:api_response_3", "file:/src/main.py"

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type,
            "source_step": self.source_step,
            "reason": self.reason,
        }
        if self.evidence:
            result["evidence"] = self.evidence
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DependencyEdge":
        return cls(**data)


@dataclass
class Artifact:
    """
    An artifact produced by a step.

    Attributes:
        name: Identifier for the artifact
        artifact_type: Type of artifact (search_results, filepath, etc.)
    """
    name: str
    artifact_type: str  # ArtifactType value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "artifact_type": self.artifact_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        return cls(**data)


@dataclass
class PerturbableSlot:
    """
    A slot that can be perturbed in a step.

    Attributes:
        slot: JSON path to the value (e.g., "tool_arguments.path")
        value_type: Type of the value (filepath, integer, etc.)
        current_value: The current value at this slot
        allowed_perturbation_types: List of perturbation categories
    """
    slot: str
    value_type: str  # ValueType value
    current_value: Any
    allowed_perturbation_types: List[str]  # ["placebo", "data_reference", "parameter", "tool_selection", "structural"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "value_type": self.value_type,
            "current_value": self.current_value,
            "allowed_perturbation_types": self.allowed_perturbation_types,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerturbableSlot":
        return cls(**data)


@dataclass
class TypedStep:
    """
    A fully typed step in an agent trajectory.

    Core fields are filled in Pass 1 (parsing/heuristics).
    Derived fields are filled in Pass 2-3 (heuristics).
    """
    # === Identity (from raw, renamed) ===
    step_index: int
    raw_text: str

    # === Core Typing (Pass 1) ===
    step_role: str  # StepRole value

    # === Terminal Flags (Pass 1) ===
    is_terminal_step: bool
    produces_final_answer: bool
    produces_patch: bool

    # === Tool Fields (Pass 1) ===
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None

    # === Normalized Operation (Pass 1) ===
    # Semantic operation type, independent of tool_name
    # Values: "read", "write", "execute", "submit", "reason"
    normalized_operation: Optional[str] = None

    # === Extraction Fields (Pass 1) ===
    extracted_value: Optional[Any] = None
    value_type: Optional[str] = None  # ValueType value
    source_step: Optional[int] = None
    source_description: Optional[str] = None
    extraction_provenance: Optional[ExtractionProvenance] = None

    # === Dependency Graph (Pass 1-2) ===
    depends_on_steps: List[int] = field(default_factory=list)
    dependency_edges: List[DependencyEdge] = field(default_factory=list)
    # Transitive closure of dependencies
    transitive_depends_on: List[int] = field(default_factory=list)

    # === Artifacts (Pass 1-2) ===
    entities: List[str] = field(default_factory=list)
    produced_artifacts: List[Artifact] = field(default_factory=list)
    consumed_artifacts: List[str] = field(default_factory=list)

    # === Perturbation Support (Pass 2) ===
    perturbable_slots: List[PerturbableSlot] = field(default_factory=list)

    # === Derived Annotations (Pass 2-3) ===
    critical_path_score: Optional[ProvenanceField] = None
    affects_final_answer: Optional[ProvenanceField] = None
    affects_patch: Optional[ProvenanceField] = None
    affects_tool_execution: Optional[ProvenanceField] = None
    recoverable_if_wrong: Optional[ProvenanceField] = None
    observable_if_wrong: Optional[ProvenanceField] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "step_index": self.step_index,
            "raw_text": self.raw_text,
            "step_role": self.step_role,
            "is_terminal_step": self.is_terminal_step,
            "produces_final_answer": self.produces_final_answer,
            "produces_patch": self.produces_patch,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "observation": self.observation,
            "normalized_operation": self.normalized_operation,
            "extracted_value": self.extracted_value,
            "value_type": self.value_type,
            "source_step": self.source_step,
            "source_description": self.source_description,
        }
        # Add extraction provenance if present
        if self.extraction_provenance:
            result["extraction_provenance"] = self.extraction_provenance.to_dict()
        result.update({
            "depends_on_steps": self.depends_on_steps,
            "dependency_edges": [e.to_dict() for e in self.dependency_edges],
            "transitive_depends_on": self.transitive_depends_on,
            "entities": self.entities,
            "produced_artifacts": [a.to_dict() for a in self.produced_artifacts],
            "consumed_artifacts": self.consumed_artifacts,
            "perturbable_slots": [s.to_dict() for s in self.perturbable_slots],
        })
        # Add derived fields if present
        if self.critical_path_score:
            result["critical_path_score"] = self.critical_path_score.to_dict()
        if self.affects_final_answer:
            result["affects_final_answer"] = self.affects_final_answer.to_dict()
        if self.affects_patch:
            result["affects_patch"] = self.affects_patch.to_dict()
        if self.affects_tool_execution:
            result["affects_tool_execution"] = self.affects_tool_execution.to_dict()
        if self.recoverable_if_wrong:
            result["recoverable_if_wrong"] = self.recoverable_if_wrong.to_dict()
        if self.observable_if_wrong:
            result["observable_if_wrong"] = self.observable_if_wrong.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TypedStep":
        """Create TypedStep from dictionary."""
        data = data.copy()
        # Convert nested objects
        data["dependency_edges"] = [
            DependencyEdge.from_dict(e) for e in data.get("dependency_edges", [])
        ]
        data["produced_artifacts"] = [
            Artifact.from_dict(a) for a in data.get("produced_artifacts", [])
        ]
        data["perturbable_slots"] = [
            PerturbableSlot.from_dict(s) for s in data.get("perturbable_slots", [])
        ]
        # Convert extraction provenance
        if "extraction_provenance" in data and data["extraction_provenance"]:
            data["extraction_provenance"] = ExtractionProvenance.from_dict(
                data["extraction_provenance"]
            )
        # Convert provenance fields
        for field_name in [
            "critical_path_score",
            "affects_final_answer",
            "affects_patch",
            "affects_tool_execution",
            "recoverable_if_wrong",
            "observable_if_wrong",
        ]:
            if field_name in data and data[field_name]:
                data[field_name] = ProvenanceField.from_dict(data[field_name])
        return cls(**data)


@dataclass
class TypedTrajectory:
    """
    A fully typed agent trajectory.

    Contains trajectory-level metadata and a list of typed steps.
    """
    # === Identity (from raw) ===
    trajectory_id: str
    benchmark: str

    # === Task (from raw, normalized) ===
    task_id: str
    task_text: str
    expected_answer: Optional[str] = None

    # === Metadata (from raw) ===
    domain: Optional[str] = None
    difficulty: Optional[str] = None

    # === Enrichment (added by typing pass) ===
    num_steps: int = 0
    environment_type: str = "tool_use"  # tool_use | qa | code_edit | browser
    ground_truth_available: bool = False
    baseline_outcome: float = 0.0
    has_objective_verifier: bool = False
    can_replay: bool = False
    can_regenerate_downstream: bool = False

    # === Steps ===
    steps: List[TypedStep] = field(default_factory=list)

    # === Provenance (from raw) ===
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trajectory_id": self.trajectory_id,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "task_text": self.task_text,
            "expected_answer": self.expected_answer,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "num_steps": self.num_steps,
            "environment_type": self.environment_type,
            "ground_truth_available": self.ground_truth_available,
            "baseline_outcome": self.baseline_outcome,
            "has_objective_verifier": self.has_objective_verifier,
            "can_replay": self.can_replay,
            "can_regenerate_downstream": self.can_regenerate_downstream,
            "steps": [step.to_dict() for step in self.steps],
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TypedTrajectory":
        """Create TypedTrajectory from dictionary."""
        data = data.copy()
        data["steps"] = [TypedStep.from_dict(s) for s in data.get("steps", [])]
        return cls(**data)

    def get_step(self, index: int) -> Optional[TypedStep]:
        """Get step by index."""
        for step in self.steps:
            if step.step_index == index:
                return step
        return None

    def get_terminal_step(self) -> Optional[TypedStep]:
        """Get the terminal step if present."""
        for step in self.steps:
            if step.is_terminal_step:
                return step
        return None

    def get_critical_steps(self, threshold: float = 0.7) -> List[TypedStep]:
        """Get steps with critical_path_score above threshold."""
        return [
            step for step in self.steps
            if step.critical_path_score and step.critical_path_score.value >= threshold
        ]
