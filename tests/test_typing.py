"""
Unit tests for the typing module.

Tests all components:
- StepTyper
- EntityExtractor
- DependencyAnalyzer
- ArtifactTracker
- SlotTyper
- CriticalPathScorer
- TrajectoryTyper (integration)
"""

import pytest
from src.typing.schema import (
    TypedStep,
    TypedTrajectory,
    DependencyEdge,
    Artifact,
    PerturbableSlot,
    ProvenanceField,
    StepRole,
    ValueType,
)
from src.typing.step_typer import StepTyper
from src.typing.entity_extractor import EntityExtractor
from src.typing.dependency_analyzer import DependencyAnalyzer
from src.typing.artifact_tracker import ArtifactTracker
from src.typing.slot_typer import SlotTyper
from src.typing.critical_path import CriticalPathScorer
from src.typing.typer import TrajectoryTyper

# === Fixtures ===


@pytest.fixture
def toolbench_step():
    """Sample ToolBench step with tool call."""
    return {
        "step_id": "step_1",
        "step_number": 1,
        "step_type": "tool_execution",
        "content": 'Thought: I will search for nutrition data.\nAction: api_nutrition\nAction Input: {"ingr": "oatmeal"}',
        "tool_name": "api_nutrition_data_for_edamam",
        "tool_input": {"ingr": "oatmeal"},
        "tool_output": '{"calories": 150}',
        "metadata": {
            "thought": "I will search for nutrition data.",
            "action": "api_nutrition",
        },
    }


@pytest.fixture
def gaia_extraction_step():
    """Sample GAIA step with extraction."""
    return {
        "step_id": "step_2",
        "step_number": 2,
        "step_type": "reasoning",
        "content": "Examine the search results page to locate the answer (12)",
        "tool_name": None,
        "tool_input": None,
        "tool_output": None,
    }


@pytest.fixture
def swebench_patch_step():
    """Sample SWE-bench step with patch."""
    return {
        "step_id": "step_13",
        "step_number": 13,
        "step_type": "tool_execution",
        "content": "Let's fix this by removing the reversed() call",
        "tool_name": "str_replace_editor",
        "tool_input": {
            "path": "/testbed/tenacity/wait.py",
            "old_str": "self.strategies = list(reversed(strategies))",
            "new_str": "self.strategies = list(strategies)",
        },
        "tool_output": "File edited successfully",
    }


@pytest.fixture
def finish_step():
    """Sample Finish step."""
    return {
        "step_id": "step_5",
        "step_number": 5,
        "step_type": "tool_execution",
        "content": "Thought: I have the answer.\nAction: Finish",
        "tool_name": "Finish",
        "tool_input": {
            "return_type": "give_answer",
            "final_answer": "The answer is 6",
        },
        "tool_output": None,
    }


@pytest.fixture
def sample_trajectory():
    """Sample trajectory for integration testing."""
    return {
        "trajectory_id": "toolbench_12345",
        "benchmark": "toolbench",
        "domain": "data_information",
        "steps": [
            {
                "step_id": "step_1",
                "step_number": 1,
                "step_type": "tool_execution",
                "content": "Thought: I will search for F1 news.\nAction: get_f1_news\nAction Input: {}",
                "tool_name": "get_f1_news",
                "tool_input": {},
                "tool_output": '{"articles": [{"title": "F1 Update"}]}',
            },
            {
                "step_id": "step_2",
                "step_number": 2,
                "step_type": "tool_execution",
                "content": "Thought: I got the news. Now I will finish.\nAction: Finish",
                "tool_name": "Finish",
                "tool_input": {
                    "return_type": "give_answer",
                    "final_answer": "Here is the F1 news",
                },
                "tool_output": None,
            },
        ],
        "ground_truth": {
            "task_description": "Get F1 news",
            "expected_answer": None,
            "task_success": True,
        },
        "provenance": {"sampling_seed": 42},
    }


# === StepTyper Tests ===


class TestStepTyper:
    def test_classify_tool_call(self, toolbench_step):
        typer = StepTyper()
        role = typer.classify_step_role(toolbench_step, 1, 5, "toolbench")
        assert role == StepRole.TOOL_CALL.value

    def test_classify_extraction(self, gaia_extraction_step):
        typer = StepTyper()
        role = typer.classify_step_role(gaia_extraction_step, 2, 5, "gaia")
        assert role == StepRole.EXTRACTION.value

    def test_classify_final_response(self, finish_step):
        typer = StepTyper()
        role = typer.classify_step_role(finish_step, 5, 5, "toolbench")
        assert role == StepRole.FINAL_RESPONSE.value

    def test_terminal_flags_finish(self, finish_step):
        typer = StepTyper()
        is_terminal, produces_answer, produces_patch = typer.determine_terminal_flags(
            finish_step, 5, 5, "toolbench"
        )
        assert is_terminal is True
        assert produces_answer is True
        assert produces_patch is False

    def test_terminal_flags_patch(self, swebench_patch_step):
        typer = StepTyper()
        is_terminal, produces_answer, produces_patch = typer.determine_terminal_flags(
            swebench_patch_step, 13, 14, "swebench"
        )
        assert produces_patch is True

    def test_extraction_fields(self, gaia_extraction_step):
        typer = StepTyper()
        prev_steps = [
            {
                "step_number": 1,
                "tool_output": "search results",
                "tool_name": "web_search",
            }
        ]
        extraction = typer.extract_extraction_fields(
            gaia_extraction_step, StepRole.EXTRACTION.value, prev_steps
        )
        assert extraction["extracted_value"] == 12
        assert extraction["value_type"] == ValueType.INTEGER.value
        assert extraction["source_step"] == 1


# === EntityExtractor Tests ===


class TestEntityExtractor:
    def test_extract_file_paths(self):
        extractor = EntityExtractor()
        step = {"content": "Looking at /testbed/tenacity/wait.py for the bug"}
        entities = extractor.extract_entities(step)
        assert any("/testbed/tenacity/wait.py" in e for e in entities)

    def test_extract_urls(self):
        extractor = EntityExtractor()
        step = {"content": "Navigate to https://example.com/api/data"}
        entities = extractor.extract_entities(step)
        assert "https://example.com/api/data" in entities

    def test_extract_line_numbers(self):
        extractor = EntityExtractor()
        step = {"content": "The bug is on line 108"}
        entities = extractor.extract_entities(step)
        assert "line 108" in entities

    def test_extract_tool_name_parts(self):
        extractor = EntityExtractor()
        step = {
            "content": "",
            "tool_name": "api_nutrition_data_for_edamam_analysis",
        }
        entities = extractor.extract_entities(step)
        assert "api_nutrition_data_for_edamam_analysis" in entities


# === DependencyAnalyzer Tests ===


class TestDependencyAnalyzer:
    def test_direct_dependencies(self):
        analyzer = DependencyAnalyzer()
        typed_steps = [
            {
                "step_index": 1,
                "step_role": "tool_call",
                "raw_text": "First step",
                "produced_artifacts": [{"name": "result_1"}],
                "consumed_artifacts": [],
            },
            {
                "step_index": 2,
                "step_role": "tool_call",
                "raw_text": "Uses result from previous step",
                "produced_artifacts": [],
                "consumed_artifacts": ["result_1"],
            },
        ]
        entity_map = {1: {"entity1"}, 2: {"entity1"}}
        result = analyzer.analyze_dependencies(typed_steps, entity_map)
        assert 1 in result[1]["depends_on_steps"]

    def test_transitive_closure(self):
        analyzer = DependencyAnalyzer()
        typed_steps = [
            {
                "step_index": 1,
                "step_role": "tool_call",
                "raw_text": "Step 1",
                "produced_artifacts": [{"name": "a"}],
                "consumed_artifacts": [],
            },
            {
                "step_index": 2,
                "step_role": "tool_call",
                "raw_text": "Step 2",
                "produced_artifacts": [{"name": "b"}],
                "consumed_artifacts": ["a"],
            },
            {
                "step_index": 3,
                "step_role": "tool_call",
                "raw_text": "Step 3",
                "produced_artifacts": [],
                "consumed_artifacts": ["b"],
            },
        ]
        entity_map = {1: set(), 2: set(), 3: set()}
        result = analyzer.analyze_dependencies(typed_steps, entity_map)
        # Step 3 transitively depends on step 1
        assert 1 in result[2]["transitive_depends_on"]


# === ArtifactTracker Tests ===


class TestArtifactTracker:
    def test_produces_api_response(self):
        """Test that tool calls with output produce api_response artifacts."""
        tracker = ArtifactTracker()
        # Use a tool name not in READ_ONLY_TOOLS to produce an artifact
        typed_steps = [
            {
                "step_index": 1,
                "step_role": "tool_call",
                "tool_name": "api_call",
                "raw_text": "",
                "observation": "results",
                "tool_arguments": {},
            },
        ]
        result = tracker.track_artifacts(typed_steps)
        artifacts = result[0]["produced_artifacts"]
        assert any("api_response" in a["artifact_type"] for a in artifacts)

    def test_produces_patch(self):
        """Test that edit tools with produces_patch flag produce patch artifacts."""
        tracker = ArtifactTracker()
        typed_steps = [
            {
                "step_index": 1,
                "step_role": "tool_call",
                "tool_name": "str_replace_editor",
                "raw_text": "",
                "observation": "",
                "tool_arguments": {"old_str": "x", "new_str": "y"},
                "produces_patch": True,
            },
        ]
        result = tracker.track_artifacts(typed_steps)
        artifacts = result[0]["produced_artifacts"]
        assert any("patch" in a["artifact_type"] for a in artifacts)

    def test_consumed_artifacts(self):
        """Test that extraction steps consume artifacts from dependencies."""
        tracker = ArtifactTracker()
        typed_steps = [
            # Use bash tool (not read-only) which produces text output
            {
                "step_index": 1,
                "step_role": "tool_call",
                "tool_name": "bash",
                "raw_text": "",
                "observation": "data",
                "tool_arguments": {},
                "depends_on_steps": [],
            },
            {
                "step_index": 2,
                "step_role": "extraction",
                "tool_name": None,
                "raw_text": "Extract (42)",
                "observation": None,
                "tool_arguments": None,
                "depends_on_steps": [1],
                "extracted_value": 42,
                "value_type": "integer",
            },
        ]
        result = tracker.track_artifacts(typed_steps)
        assert len(result[1]["consumed_artifacts"]) > 0


# === SlotTyper Tests ===


class TestSlotTyper:
    def test_tool_name_slot(self, toolbench_step):
        typer = SlotTyper()
        step = {
            "step_role": "tool_call",
            "tool_name": "api_nutrition",
            "tool_arguments": {"ingr": "oatmeal"},
        }
        slots = typer.identify_slots(step)
        tool_slots = [s for s in slots if s["slot"] == "tool_name"]
        assert len(tool_slots) == 1
        assert "tool_selection" in tool_slots[0]["allowed_perturbation_types"]

    def test_filepath_slot(self, swebench_patch_step):
        typer = SlotTyper()
        step = {
            "step_role": "tool_call",
            "tool_name": "str_replace_editor",
            "tool_arguments": {
                "path": "/testbed/tenacity/wait.py",
                "old_str": "x",
                "new_str": "y",
            },
        }
        slots = typer.identify_slots(step)
        path_slots = [s for s in slots if "path" in s["slot"]]
        assert len(path_slots) >= 1
        assert path_slots[0]["value_type"] == ValueType.FILEPATH.value

    def test_extracted_value_slot(self):
        typer = SlotTyper()
        step = {
            "step_role": "extraction",
            "extracted_value": 12,
            "value_type": "integer",
            "tool_arguments": None,
        }
        slots = typer.identify_slots(step)
        extracted_slots = [s for s in slots if s["slot"] == "extracted_value"]
        assert len(extracted_slots) == 1
        assert "data_reference" in extracted_slots[0]["allowed_perturbation_types"]


# === CriticalPathScorer Tests ===


class TestCriticalPathScorer:
    def test_terminal_step_high_score(self):
        scorer = CriticalPathScorer()
        typed_steps = [
            {
                "step_index": 1,
                "step_role": "tool_call",
                "is_terminal_step": False,
                "produces_final_answer": False,
                "depends_on_steps": [],
                "transitive_depends_on": [],
                "produced_artifacts": [],
            },
            {
                "step_index": 2,
                "step_role": "final_response",
                "is_terminal_step": True,
                "produces_final_answer": True,
                "depends_on_steps": [1],
                "transitive_depends_on": [1],
                "produced_artifacts": [],
            },
        ]
        result = scorer.score_trajectory(typed_steps, "toolbench")
        # Terminal/final_response steps should have high criticality (>= 0.9)
        assert result[1]["critical_path_score"]["value"] >= 0.9

    def test_extraction_not_recoverable(self):
        scorer = CriticalPathScorer()
        typed_steps = [
            {
                "step_index": 1,
                "step_role": "tool_call",
                "is_terminal_step": False,
                "produces_final_answer": False,
                "depends_on_steps": [],
                "transitive_depends_on": [],
                "produced_artifacts": [],
            },
            {
                "step_index": 2,
                "step_role": "extraction",
                "is_terminal_step": False,
                "produces_final_answer": False,
                "depends_on_steps": [1],
                "transitive_depends_on": [1],
                "produced_artifacts": [],
            },
        ]
        result = scorer.score_trajectory(typed_steps, "gaia")
        assert result[1]["recoverable_if_wrong"]["value"] is False

    def test_extraction_not_observable(self):
        scorer = CriticalPathScorer()
        typed_steps = [
            {
                "step_index": 1,
                "step_role": "extraction",
                "is_terminal_step": False,
                "produces_final_answer": False,
                "observation": None,
                "depends_on_steps": [],
                "transitive_depends_on": [],
                "produced_artifacts": [],
            },
        ]
        result = scorer.score_trajectory(typed_steps, "gaia")
        assert result[0]["observable_if_wrong"]["value"] is False


# === TrajectoryTyper Integration Tests ===


class TestTrajectoryTyper:
    def test_type_full_trajectory(self, sample_trajectory):
        typer = TrajectoryTyper(use_spacy=False)
        typed = typer.type_trajectory(sample_trajectory)

        assert typed.trajectory_id == "toolbench_12345"
        assert typed.benchmark == "toolbench"
        assert typed.num_steps == 2
        assert len(typed.steps) == 2

    def test_step_roles_assigned(self, sample_trajectory):
        typer = TrajectoryTyper(use_spacy=False)
        typed = typer.type_trajectory(sample_trajectory)

        assert typed.steps[0].step_role == StepRole.TOOL_CALL.value
        assert typed.steps[1].step_role == StepRole.FINAL_RESPONSE.value

    def test_terminal_flags_set(self, sample_trajectory):
        typer = TrajectoryTyper(use_spacy=False)
        typed = typer.type_trajectory(sample_trajectory)

        assert typed.steps[1].is_terminal_step is True
        assert typed.steps[1].produces_final_answer is True

    def test_perturbable_slots_identified(self, sample_trajectory):
        typer = TrajectoryTyper(use_spacy=False)
        typed = typer.type_trajectory(sample_trajectory)

        # First step should have tool_name slot
        tool_call_step = typed.steps[0]
        assert len(tool_call_step.perturbable_slots) > 0

    def test_critical_scores_computed(self, sample_trajectory):
        typer = TrajectoryTyper(use_spacy=False)
        typed = typer.type_trajectory(sample_trajectory)

        for step in typed.steps:
            assert step.critical_path_score is not None
            assert 0 <= step.critical_path_score.value <= 1

    def test_serialization_roundtrip(self, sample_trajectory):
        typer = TrajectoryTyper(use_spacy=False)
        typed = typer.type_trajectory(sample_trajectory)

        # Serialize to dict
        as_dict = typed.to_dict()

        # Deserialize back
        restored = TypedTrajectory.from_dict(as_dict)

        assert restored.trajectory_id == typed.trajectory_id
        assert len(restored.steps) == len(typed.steps)
        assert restored.steps[0].step_role == typed.steps[0].step_role


# === Schema Tests ===


class TestSchema:
    def test_typed_step_to_dict(self):
        step = TypedStep(
            step_index=1,
            raw_text="Test step",
            step_role="tool_call",
            is_terminal_step=False,
            produces_final_answer=False,
            produces_patch=False,
            tool_name="test_tool",
            critical_path_score=ProvenanceField(value=0.8, source="heuristic"),
        )
        d = step.to_dict()
        assert d["step_index"] == 1
        assert d["critical_path_score"]["value"] == 0.8

    def test_typed_step_from_dict(self):
        d = {
            "step_index": 1,
            "raw_text": "Test",
            "step_role": "tool_call",
            "is_terminal_step": False,
            "produces_final_answer": False,
            "produces_patch": False,
            "depends_on_steps": [],
            "dependency_edges": [],
            "transitive_depends_on": [],
            "entities": [],
            "produced_artifacts": [],
            "consumed_artifacts": [],
            "perturbable_slots": [],
        }
        step = TypedStep.from_dict(d)
        assert step.step_index == 1
        assert step.step_role == "tool_call"

    def test_provenance_field(self):
        pf = ProvenanceField(value=True, source="heuristic", confidence=0.9)
        d = pf.to_dict()
        restored = ProvenanceField.from_dict(d)
        assert restored.value is True
        assert restored.confidence == 0.9
