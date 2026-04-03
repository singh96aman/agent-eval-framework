"""
Tests for perturbation generation system.

This module tests:
- Tool similarity matching
- Individual perturbation strategies (A, B, C)
- Perturbation generator
- Integration with experiment runner
"""

import pytest
from src.data.schema import Trajectory, Step, StepType, GroundTruth
from src.perturbations.generator import PerturbationGenerator
from src.perturbations.strategies import (
    PlanningErrorStrategy,
    ToolSelectionErrorStrategy,
    ParameterErrorStrategy,
)
from src.perturbations.tool_similarity import ToolSimilarityMatcher


# Sample system prompt for testing
SAMPLE_SYSTEM_PROMPT = """
You are AutoGPT. You have access of the following tools:
1.greyhound_racing_uk: Greyhound Racing API

Specifically, you have access to the following APIs:
[{'name': 'racecards_for_greyhound_racing_uk', 'description': 'Get races list'},
 {'name': 'race_detail_info_for_greyhound_racing_uk', 'description': 'Get race detailed info by ID'},
 {'name': 'results_for_greyhound_racing_uk', 'description': 'Get results races by date'},
 {'name': 'latest_coupons_for_get_27coupons', 'description': 'Get latest coupons'},
 {'name': 'trending_coupons_for_get_27coupons', 'description': 'Get trending coupons'},
 {'name': 'popular_coupons_for_get_27coupons', 'description': 'Get popular coupons'},
 {'name': 'Finish', 'description': 'Finish the task'}]
"""


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    steps = [
        Step(
            step_id="step_1",
            step_number=1,
            step_type=StepType.TOOL_EXECUTION,
            content=(
                "Thought: I need to call the racecards_for_greyhound_racing_uk function "
                "to get the race schedule.\n"
                "Action: racecards_for_greyhound_racing_uk\n"
                "Action Input: {}"
            ),
            tool_name="racecards_for_greyhound_racing_uk",
            tool_input={},
            tool_output='{"races": []}'
        ),
        Step(
            step_id="step_2",
            step_number=2,
            step_type=StepType.TOOL_EXECUTION,
            content=(
                "Thought: Now I need to get details for race ID 53128.\n"
                "Action: race_detail_info_for_greyhound_racing_uk\n"
                "Action Input: {\"id_race\": \"53128\"}"
            ),
            tool_name="race_detail_info_for_greyhound_racing_uk",
            tool_input={"id_race": "53128"},
            tool_output='{"greyhounds": []}'
        ),
        Step(
            step_id="step_3",
            step_number=3,
            step_type=StepType.TOOL_EXECUTION,
            content=(
                "Thought: Let me try the results function.\n"
                "Action: results_for_greyhound_racing_uk\n"
                "Action Input: {}"
            ),
            tool_name="results_for_greyhound_racing_uk",
            tool_input={},
            tool_output='{"error": "timeout"}'
        ),
    ]

    ground_truth = GroundTruth(
        task_description=(
            "Get the race schedule for this week and details for race ID 53128"
        ),
        expected_answer="Race schedule and details for race 53128",
        task_success=False
    )

    metadata = {
        "system_prompt": SAMPLE_SYSTEM_PROMPT
    }

    return Trajectory(
        trajectory_id="test_traj_001",
        benchmark="toolbench",
        steps=steps,
        ground_truth=ground_truth,
        metadata=metadata
    )


class TestToolSimilarityMatcher:
    """Test tool similarity matching."""

    def test_extract_tools_from_system_prompt(self):
        """Test extracting tools from system prompt."""
        matcher = ToolSimilarityMatcher()
        tools = matcher.extract_tools_from_system_prompt(SAMPLE_SYSTEM_PROMPT)

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert "racecards_for_greyhound_racing_uk" in tool_names
        assert "results_for_greyhound_racing_uk" in tool_names
        assert "Finish" not in tool_names  # Should be filtered out

    def test_extract_purpose_keywords(self):
        """Test extracting purpose keywords from tool names."""
        matcher = ToolSimilarityMatcher()

        keywords = matcher._extract_purpose_keywords("latest_coupons")
        assert "latest" in keywords
        assert "coupons" in keywords

        keywords = matcher._extract_purpose_keywords("get_user_info_by_id")
        assert "get" in keywords
        assert "user" in keywords
        assert "info" in keywords
        assert "id" in keywords

    def test_index_tools(self):
        """Test indexing tools by API family."""
        matcher = ToolSimilarityMatcher()
        matcher.index_tools(SAMPLE_SYSTEM_PROMPT)

        families = matcher.get_api_families()
        assert "greyhound_racing_uk" in families
        assert "get_27coupons" in families

        tools_in_family = matcher.get_tools_in_family("greyhound_racing_uk")
        assert len(tools_in_family) == 3
        assert "racecards_for_greyhound_racing_uk" in tools_in_family
        assert "results_for_greyhound_racing_uk" in tools_in_family

    def test_find_plausible_substitutes(self):
        """Test finding plausible tool substitutes."""
        matcher = ToolSimilarityMatcher()
        matcher.index_tools(SAMPLE_SYSTEM_PROMPT)

        # racecards should be confused with results (same API, different temporal)
        substitutes = matcher.find_plausible_substitutes("racecards_for_greyhound_racing_uk")
        assert len(substitutes) > 0
        assert "results_for_greyhound_racing_uk" in substitutes

        # latest should be confused with trending/popular
        substitutes = matcher.find_plausible_substitutes("latest_coupons_for_get_27coupons")
        assert len(substitutes) > 0
        # Should find trending or popular
        assert any("trending" in s or "popular" in s for s in substitutes)


class TestPlanningErrorStrategy:
    """Test Type A: Planning errors."""

    def test_perturb_step(self, sample_trajectory):
        """Test applying planning error to a step."""
        strategy = PlanningErrorStrategy(random_seed=42)

        original_step = sample_trajectory.steps[0]
        perturbed_step = strategy.perturb_step(
            step=original_step,
            trajectory=sample_trajectory
        )

        # Should modify content
        assert perturbed_step.content != original_step.content

        # Should have perturbation metadata
        assert "perturbation" in perturbed_step.metadata
        assert perturbed_step.metadata["perturbation"]["type"] == "planning_error"

        # Thought should be different
        original_thought = "I need to call the racecards_for_greyhound_racing_uk function"
        assert original_thought in original_step.content
        # Perturbed thought should be modified
        assert perturbed_step.content != original_step.content


class TestToolSelectionErrorStrategy:
    """Test Type B: Tool selection errors."""

    def test_perturb_step(self, sample_trajectory):
        """Test replacing tool with plausible substitute."""
        strategy = ToolSelectionErrorStrategy(random_seed=42)

        original_step = sample_trajectory.steps[0]
        perturbed_step = strategy.perturb_step(
            step=original_step,
            trajectory=sample_trajectory,
            system_prompt=SAMPLE_SYSTEM_PROMPT
        )

        # Should change tool name
        assert perturbed_step.tool_name != original_step.tool_name

        # Should be a plausible substitute from same API family
        assert "greyhound_racing_uk" in perturbed_step.tool_name

        # Should have perturbation metadata
        assert "perturbation" in perturbed_step.metadata
        assert perturbed_step.metadata["perturbation"]["type"] == "tool_selection_error"
        assert perturbed_step.metadata["perturbation"]["original_tool"] == original_step.tool_name

    def test_no_substitutes_available(self, sample_trajectory):
        """Test when no plausible substitutes exist."""
        strategy = ToolSelectionErrorStrategy(random_seed=42)

        # Step with unique tool
        step = sample_trajectory.steps[1]  # race_detail_info (no close substitutes)

        perturbed_step = strategy.perturb_step(
            step=step,
            trajectory=sample_trajectory,
            system_prompt=SAMPLE_SYSTEM_PROMPT
        )

        # Might return original if no substitutes
        # (depends on similarity matching logic)
        assert perturbed_step is not None


class TestParameterErrorStrategy:
    """Test Type C: Parameter errors."""

    def test_remove_param(self, sample_trajectory):
        """Test removing a required parameter."""
        strategy = ParameterErrorStrategy(random_seed=42)

        original_step = sample_trajectory.steps[1]  # Has {"id_race": "53128"}
        perturbed_step = strategy.perturb_step(
            step=original_step,
            trajectory=sample_trajectory
        )

        # Should modify parameters
        assert perturbed_step.tool_input != original_step.tool_input

        # Should have perturbation metadata
        assert "perturbation" in perturbed_step.metadata
        assert perturbed_step.metadata["perturbation"]["type"] == "parameter_error"

    def test_corrupt_param_value(self, sample_trajectory):
        """Test corrupting parameter value."""
        strategy = ParameterErrorStrategy(random_seed=42)
        strategy.ERROR_TYPES = ["wrong_param_value"]  # Force this error type

        original_step = sample_trajectory.steps[1]
        original_value = original_step.tool_input["id_race"]

        perturbed_step = strategy.perturb_step(
            step=original_step,
            trajectory=sample_trajectory
        )

        # Value should be changed
        if perturbed_step.tool_input:  # Might be empty if param removed
            perturbed_value = perturbed_step.tool_input.get("id_race")
            if perturbed_value:  # If key still exists
                assert perturbed_value != original_value


class TestPerturbationGenerator:
    """Test main perturbation generator."""

    def test_generate_planning_perturbation(self, sample_trajectory):
        """Test generating a planning error perturbation."""
        generator = PerturbationGenerator(random_seed=42)

        perturbed = generator.generate_perturbation(
            trajectory=sample_trajectory,
            perturbation_type="planning",
            position="early"
        )

        assert perturbed is not None
        assert perturbed.perturbation_type == "planning"
        assert perturbed.perturbation_position == "early"
        assert perturbed.perturbed_step_number <= 2  # early = 1-2

    def test_generate_tool_selection_perturbation(self, sample_trajectory):
        """Test generating a tool selection error."""
        generator = PerturbationGenerator(random_seed=42)

        perturbed = generator.generate_perturbation(
            trajectory=sample_trajectory,
            perturbation_type="tool_selection",
            position="early",
            system_prompt=SAMPLE_SYSTEM_PROMPT
        )

        assert perturbed is not None
        assert perturbed.perturbation_type == "tool_selection"
        # Should have changed the tool name
        original_step = sample_trajectory.steps[0]
        perturbed_step = perturbed.perturbed_trajectory.steps[0]
        assert perturbed_step.tool_name != original_step.tool_name

    def test_generate_parameter_perturbation(self, sample_trajectory):
        """Test generating a parameter error."""
        generator = PerturbationGenerator(random_seed=42)

        perturbed = generator.generate_perturbation(
            trajectory=sample_trajectory,
            perturbation_type="parameter",
            position="early"
        )

        assert perturbed is not None
        assert perturbed.perturbation_type == "parameter"

    def test_position_mapping(self, sample_trajectory):
        """Test position mapping (early, middle, late)."""
        generator = PerturbationGenerator(random_seed=42)

        # Test early position
        perturbed_early = generator.generate_perturbation(
            trajectory=sample_trajectory,
            perturbation_type="planning",
            position="early"
        )
        assert perturbed_early.perturbed_step_number <= 2

        # Test middle position
        perturbed_middle = generator.generate_perturbation(
            trajectory=sample_trajectory,
            perturbation_type="planning",
            position="middle"
        )
        assert 3 <= perturbed_middle.perturbed_step_number <= 5

    def test_generate_all_perturbations(self, sample_trajectory):
        """Test generating all 9 perturbation conditions."""
        generator = PerturbationGenerator(random_seed=42)

        perturbations = generator.generate_all_perturbations(
            trajectory=sample_trajectory,
            system_prompt=SAMPLE_SYSTEM_PROMPT
        )

        # Should have up to 9 perturbations (some might fail if trajectory too short)
        assert len(perturbations) > 0
        assert len(perturbations) <= 9

        # Check diversity
        types = set(p.perturbation_type for p in perturbations)
        positions = set(p.perturbation_position for p in perturbations)

        assert len(types) > 0
        assert len(positions) > 0

    def test_static_mode_preserves_subsequent_steps(self, sample_trajectory):
        """Test that static mode keeps original subsequent steps."""
        generator = PerturbationGenerator(random_seed=42)

        perturbed = generator.generate_perturbation(
            trajectory=sample_trajectory,
            perturbation_type="planning",
            position="early",
            mode="static"
        )

        # Original has 3 steps, perturbed should also have 3
        assert len(perturbed.perturbed_trajectory.steps) == len(sample_trajectory.steps)

        # Steps after perturbation should be marked
        perturbed_step_num = perturbed.perturbed_step_number
        for step in perturbed.perturbed_trajectory.steps:
            if step.step_number > perturbed_step_num:
                assert step.metadata.get("conditioned_on_error") is True

    def test_perturbation_id_generation(self):
        """Test generating standardized perturbation IDs."""
        generator = PerturbationGenerator()

        pert_id = generator.get_perturbation_id(
            trajectory_id="toolbench_12345",
            perturbation_type="planning",
            position="early"
        )

        assert pert_id == "toolbench_12345_planning_early"

    def test_invalid_perturbation_type(self, sample_trajectory):
        """Test error handling for invalid perturbation type."""
        generator = PerturbationGenerator()

        with pytest.raises(ValueError, match="Unknown perturbation type"):
            generator.generate_perturbation(
                trajectory=sample_trajectory,
                perturbation_type="invalid_type",
                position="early"
            )

    def test_invalid_position(self, sample_trajectory):
        """Test error handling for invalid position."""
        generator = PerturbationGenerator()

        with pytest.raises(ValueError, match="Unknown position"):
            generator.generate_perturbation(
                trajectory=sample_trajectory,
                perturbation_type="planning",
                position="invalid_position"
            )


class TestPerturbationIntegration:
    """Integration tests for perturbation system."""

    def test_end_to_end_perturbation_flow(self, sample_trajectory):
        """Test complete perturbation workflow."""
        # Initialize generator
        generator = PerturbationGenerator(random_seed=42)

        # Generate a perturbation
        perturbed = generator.generate_perturbation(
            trajectory=sample_trajectory,
            perturbation_type="planning",
            position="early",
            system_prompt=SAMPLE_SYSTEM_PROMPT
        )

        # Verify structure
        assert perturbed is not None
        assert perturbed.original_trajectory == sample_trajectory
        assert perturbed.perturbed_trajectory != sample_trajectory
        assert perturbed.perturbed_trajectory.trajectory_id != sample_trajectory.trajectory_id

        # Verify serialization
        perturbed_dict = perturbed.to_dict()
        assert "original_trajectory" in perturbed_dict
        assert "perturbed_trajectory" in perturbed_dict
        assert "perturbation_type" in perturbed_dict

        # Verify can be reconstructed
        from src.data.schema import PerturbedTrajectory
        reconstructed = PerturbedTrajectory.from_dict(perturbed_dict)
        assert reconstructed.perturbation_type == perturbed.perturbation_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
