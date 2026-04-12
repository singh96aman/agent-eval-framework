"""
Tests for Section 6 Analysis Module.

Tests evaluator logic, schema serialization, and key metric computations.
"""

import pytest
from unittest.mock import MagicMock

from src.analysis.section6.schema import (
    AnalysisResult,
    CalibrationEval,
    DetectionEval,
    GroundTruth,
    JudgeOutputSummary,
    derive_impact_level,
    impact_score_to_tier,
    map_error_type_to_family,
)
from src.analysis.section6.evaluator import Section6Evaluator

# ============================================================================
# Utility Function Tests
# ============================================================================


class TestDeriveImpactLevel:
    """Tests for derive_impact_level function."""

    def test_none_returns_none(self):
        assert derive_impact_level(None) is None

    def test_zero_degradation(self):
        assert derive_impact_level(0.0) == 0

    def test_low_degradation(self):
        assert derive_impact_level(0.1) == 1
        assert derive_impact_level(0.25) == 1

    def test_medium_degradation(self):
        assert derive_impact_level(0.3) == 2
        assert derive_impact_level(0.5) == 2

    def test_high_degradation(self):
        assert derive_impact_level(0.6) == 3
        assert derive_impact_level(1.0) == 3

    def test_negative_degradation(self):
        # Negative OD means improvement, should be 0
        assert derive_impact_level(-0.5) == 0


class TestImpactScoreToTier:
    """Tests for impact_score_to_tier function."""

    def test_tier_0(self):
        assert impact_score_to_tier(0.0) == 0
        assert impact_score_to_tier(0.25) == 0

    def test_tier_1(self):
        assert impact_score_to_tier(0.26) == 1
        assert impact_score_to_tier(0.5) == 1

    def test_tier_2(self):
        assert impact_score_to_tier(0.51) == 2
        assert impact_score_to_tier(0.75) == 2

    def test_tier_3(self):
        assert impact_score_to_tier(0.76) == 3
        assert impact_score_to_tier(1.0) == 3


class TestMapErrorTypeToFamily:
    """Tests for map_error_type_to_family function."""

    def test_data_reference_mapping(self):
        assert map_error_type_to_family("data_reference") == "data_reference"
        assert map_error_type_to_family("DATA_REFERENCE") == "data_reference"

    def test_parameter_mapping(self):
        assert map_error_type_to_family("parameter") == "parameter"
        assert map_error_type_to_family("PARAMETER") == "parameter"

    def test_tool_selection_mapping(self):
        assert map_error_type_to_family("tool_selection") == "tool_selection"

    def test_planning_maps_to_structural(self):
        assert map_error_type_to_family("planning") == "structural"
        assert map_error_type_to_family("PLANNING") == "structural"

    def test_other_returns_none(self):
        assert map_error_type_to_family("other") is None

    def test_unknown_returns_none(self):
        assert map_error_type_to_family("unknown_type") is None
        assert map_error_type_to_family(None) is None


# ============================================================================
# Schema Tests
# ============================================================================


class TestGroundTruth:
    """Tests for GroundTruth dataclass."""

    def test_to_dict_and_from_dict(self):
        gt = GroundTruth(
            perturbation_class="fine_grained",
            perturbation_family="data_reference",
            perturbation_type="wrong_variable",
            target_step_canonical_id="step_3",
            expected_impact=2,
            expected_detectability=1,
            benchmark="toolbench",
            outcome_degradation=0.4,
            true_impact_level=2,
            baseline_outcome_binary=True,
            perturbed_outcome_binary=False,
            human_error_detected=True,
            human_error_step_id="step_3",
            human_impact_tier=2.0,
            human_error_type="data_reference",
        )

        d = gt.to_dict()
        gt2 = GroundTruth.from_dict(d)

        assert gt2.perturbation_class == "fine_grained"
        assert gt2.outcome_degradation == 0.4
        assert gt2.human_error_detected is True

    def test_optional_fields(self):
        gt = GroundTruth(
            perturbation_class="placebo",
            perturbation_family="none",
            perturbation_type="none",
            target_step_canonical_id="none",
            expected_impact=0,
            expected_detectability=0,
            benchmark="gaia",
            outcome_degradation=None,
            true_impact_level=None,
            baseline_outcome_binary=None,
            perturbed_outcome_binary=None,
        )

        d = gt.to_dict()
        gt2 = GroundTruth.from_dict(d)

        assert gt2.outcome_degradation is None
        assert gt2.human_error_detected is None


class TestDetectionEval:
    """Tests for DetectionEval dataclass."""

    def test_true_positive_case(self):
        de = DetectionEval(
            detection_correct=True,
            is_true_positive=True,
            is_false_positive=False,
            is_true_negative=False,
            is_false_negative=False,
            localization_correct=True,
            localization_distance=0,
            localization_near=True,
            type_correct=True,
            is_critical_detected=True,
        )

        d = de.to_dict()
        de2 = DetectionEval.from_dict(d)

        assert de2.is_true_positive is True
        assert de2.localization_correct is True

    def test_false_negative_case(self):
        de = DetectionEval(
            detection_correct=False,
            is_true_positive=False,
            is_false_positive=False,
            is_true_negative=False,
            is_false_negative=True,
        )

        assert de.is_false_negative is True
        assert de.localization_correct is None


class TestCalibrationEval:
    """Tests for CalibrationEval dataclass."""

    def test_over_reaction(self):
        ce = CalibrationEval(
            cce=0.4,
            abs_cce=0.4,
            over_reaction=True,
            under_reaction=False,
            failure_predicted=True,
            failure_actual=False,
            failure_correct=False,
            impact_tier_predicted=3,
            impact_tier_error=2,
        )

        d = ce.to_dict()
        ce2 = CalibrationEval.from_dict(d)

        assert ce2.over_reaction is True
        assert ce2.failure_correct is False


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_create_and_serialize(self):
        gt = GroundTruth(
            perturbation_class="fine_grained",
            perturbation_family="parameter",
            perturbation_type="wrong_param",
            target_step_canonical_id="step_2",
            expected_impact=2,
            expected_detectability=1,
            benchmark="swebench",
            outcome_degradation=0.3,
            true_impact_level=2,
            baseline_outcome_binary=True,
            perturbed_outcome_binary=False,
        )

        jo = JudgeOutputSummary(
            error_detected=True,
            error_confidence=0.8,
            predicted_step_canonical_id="step_2",
            predicted_error_type="wrong_parameter",
            localization_confidence=0.7,
            predicted_impact_score=0.6,
            predicted_failure_prob=0.5,
        )

        de = DetectionEval(
            detection_correct=True,
            is_true_positive=True,
            is_false_positive=False,
            is_true_negative=False,
            is_false_negative=False,
            localization_correct=True,
            localization_distance=0,
            localization_near=True,
            type_correct=True,
        )

        ce = CalibrationEval(
            cce=0.1,
            abs_cce=0.1,
            over_reaction=False,
            under_reaction=False,
            failure_predicted=False,
            failure_actual=False,
            failure_correct=True,
            impact_tier_predicted=2,
            impact_tier_error=0,
        )

        result = AnalysisResult.create(
            experiment_id="test_exp",
            evaluation_unit_id="unit_123",
            judge_model="gpt-4",
            ground_truth=gt,
            judge_output=jo,
            detection_eval=de,
            calibration_eval=ce,
        )

        assert result.experiment_id == "test_exp"
        assert result.analysis_id.startswith("analysis_")

        # Round-trip
        d = result.to_dict()
        result2 = AnalysisResult.from_dict(d)

        assert result2.evaluation_unit_id == "unit_123"
        assert result2.ground_truth.perturbation_class == "fine_grained"
        assert result2.detection_eval.is_true_positive is True


# ============================================================================
# Evaluator Tests
# ============================================================================


class TestSection6Evaluator:
    """Tests for Section6Evaluator class."""

    @pytest.fixture
    def evaluator(self):
        return Section6Evaluator("test_experiment")

    @pytest.fixture
    def mock_eval_unit(self):
        """Create a mock EvaluationUnit."""
        unit = MagicMock()
        unit.evaluation_unit_id = "unit_test_1"
        unit.benchmark = "toolbench"

        # derived_cache
        dc = MagicMock()
        dc.perturbation_class = "fine_grained"
        dc.perturbation_family = "data_reference"
        dc.perturbation_type = "wrong_variable"
        dc.target_step_canonical_id = "step_3"
        dc.expected_impact = 2
        dc.expected_detectability = 1
        unit.derived_cache = dc

        # perturbed trajectory for localization distance
        unit.perturbed = MagicMock()
        unit.perturbed.trajectory = {
            "steps": [
                {"canonical_step_id": "step_1"},
                {"canonical_step_id": "step_2"},
                {"canonical_step_id": "step_3"},
                {"canonical_step_id": "step_4"},
            ]
        }

        return unit

    @pytest.fixture
    def mock_placebo_unit(self):
        """Create a mock placebo EvaluationUnit."""
        unit = MagicMock()
        unit.evaluation_unit_id = "unit_placebo_1"
        unit.benchmark = "gaia"

        dc = MagicMock()
        dc.perturbation_class = "placebo"
        dc.perturbation_family = None
        dc.perturbation_type = None
        dc.target_step_canonical_id = None
        dc.expected_impact = 0
        dc.expected_detectability = 0
        unit.derived_cache = dc

        unit.perturbed = MagicMock()
        unit.perturbed.trajectory = {"steps": []}

        return unit

    @pytest.fixture
    def mock_judge_output_detected(self):
        """Create a mock judge output that detected an error."""
        jo = MagicMock()
        jo.judge_model = "gpt-4"

        jo.detection = MagicMock()
        jo.detection.error_detected = True
        jo.detection.error_confidence = 0.85

        jo.localization = MagicMock()
        jo.localization.predicted_error_step_canonical = "step_3"
        jo.localization.predicted_error_type = "wrong_variable"
        jo.localization.localization_confidence = 0.75

        jo.impact = MagicMock()
        jo.impact.predicted_impact_score = 0.6
        jo.impact.predicted_failure_prob = 0.4

        return jo

    @pytest.fixture
    def mock_judge_output_not_detected(self):
        """Create a mock judge output that did not detect an error."""
        jo = MagicMock()
        jo.judge_model = "claude-3"

        jo.detection = MagicMock()
        jo.detection.error_detected = False
        jo.detection.error_confidence = 0.2

        jo.localization = None
        jo.impact = MagicMock()
        jo.impact.predicted_impact_score = 0.1
        jo.impact.predicted_failure_prob = 0.1

        return jo

    @pytest.fixture
    def mock_outcome_record(self):
        """Create a mock OutcomeRecord."""
        rec = MagicMock()
        rec.evaluation_unit_id = "unit_test_1"
        rec.metrics = MagicMock()
        rec.metrics.outcome_degradation = 0.4

        rec.baseline = MagicMock()
        rec.baseline.outcome_binary = True

        rec.perturbed = MagicMock()
        rec.perturbed.outcome_binary = False

        return rec

    @pytest.fixture
    def mock_human_label(self):
        """Create a mock AggregatedLabel."""
        label = MagicMock()
        label.evaluation_unit_id = "unit_test_1"

        label.aggregated_detectability = MagicMock()
        label.aggregated_detectability.error_detected_majority = True
        label.aggregated_detectability.error_step_id_majority = "step_3"

        label.aggregated_consequence = MagicMock()
        label.aggregated_consequence.mean_impact_tier = 2.0
        label.aggregated_consequence.error_type_majority = "data_reference"

        return label

    def test_evaluate_true_positive(
        self, evaluator, mock_eval_unit, mock_judge_output_detected, mock_outcome_record
    ):
        """Test evaluation of a true positive case."""
        result = evaluator.evaluate_unit(
            eval_unit=mock_eval_unit,
            judge_output=mock_judge_output_detected,
            outcome_record=mock_outcome_record,
        )

        # Ground truth
        assert result.ground_truth.perturbation_class == "fine_grained"
        assert result.ground_truth.outcome_degradation == 0.4
        assert result.ground_truth.true_impact_level == 2

        # Detection
        assert result.detection_eval.is_true_positive is True
        assert result.detection_eval.is_false_positive is False
        assert result.detection_eval.detection_correct is True
        assert result.detection_eval.localization_correct is True
        assert result.detection_eval.localization_distance == 0

        # Calibration
        assert result.calibration_eval.cce == pytest.approx(0.2, rel=0.01)  # 0.6 - 0.4
        assert result.calibration_eval.failure_actual is True  # perturbed failed
        assert result.calibration_eval.failure_predicted is False  # 0.4 < 0.5

    def test_evaluate_false_negative(
        self,
        evaluator,
        mock_eval_unit,
        mock_judge_output_not_detected,
        mock_outcome_record,
    ):
        """Test evaluation of a false negative case."""
        result = evaluator.evaluate_unit(
            eval_unit=mock_eval_unit,
            judge_output=mock_judge_output_not_detected,
            outcome_record=mock_outcome_record,
        )

        assert result.detection_eval.is_false_negative is True
        assert result.detection_eval.detection_correct is False
        assert result.detection_eval.localization_correct is None  # not detected

    def test_evaluate_true_negative_placebo(
        self, evaluator, mock_placebo_unit, mock_judge_output_not_detected
    ):
        """Test evaluation of a true negative (placebo correctly not detected)."""
        result = evaluator.evaluate_unit(
            eval_unit=mock_placebo_unit,
            judge_output=mock_judge_output_not_detected,
        )

        assert result.detection_eval.is_true_negative is True
        assert result.detection_eval.detection_correct is True

    def test_evaluate_false_positive_placebo(
        self, evaluator, mock_placebo_unit, mock_judge_output_detected
    ):
        """Test evaluation of a false positive (placebo incorrectly detected)."""
        result = evaluator.evaluate_unit(
            eval_unit=mock_placebo_unit,
            judge_output=mock_judge_output_detected,
        )

        assert result.detection_eval.is_false_positive is True
        assert result.detection_eval.detection_correct is False

    def test_evaluate_with_human_labels(
        self,
        evaluator,
        mock_eval_unit,
        mock_judge_output_detected,
        mock_outcome_record,
        mock_human_label,
    ):
        """Test evaluation includes human comparison when labels provided."""
        result = evaluator.evaluate_unit(
            eval_unit=mock_eval_unit,
            judge_output=mock_judge_output_detected,
            outcome_record=mock_outcome_record,
            human_label=mock_human_label,
        )

        assert result.human_comparison is not None
        assert result.human_comparison.detection_agrees is True
        assert result.human_comparison.localization_agrees is True

    def test_localization_distance_computed(
        self, evaluator, mock_eval_unit, mock_judge_output_detected
    ):
        """Test localization distance is computed correctly."""
        # Modify judge to predict wrong step
        mock_judge_output_detected.localization.predicted_error_step_canonical = (
            "step_1"
        )

        result = evaluator.evaluate_unit(
            eval_unit=mock_eval_unit,
            judge_output=mock_judge_output_detected,
        )

        # step_1 is index 0, step_3 is index 2, distance = 2
        assert result.detection_eval.localization_distance == 2
        assert result.detection_eval.localization_correct is False
        assert result.detection_eval.localization_near is False

    def test_over_reaction_detected(
        self, evaluator, mock_eval_unit, mock_judge_output_detected
    ):
        """Test over-reaction is detected when high prediction on low impact."""
        # Set low true impact
        mock_outcome = MagicMock()
        mock_outcome.metrics.outcome_degradation = 0.1  # Low impact
        mock_outcome.baseline.outcome_binary = True
        mock_outcome.perturbed.outcome_binary = True

        # Set high predicted impact
        mock_judge_output_detected.impact.predicted_impact_score = 0.8

        result = evaluator.evaluate_unit(
            eval_unit=mock_eval_unit,
            judge_output=mock_judge_output_detected,
            outcome_record=mock_outcome,
        )

        assert result.calibration_eval.over_reaction is True

    def test_under_reaction_detected(
        self, evaluator, mock_eval_unit, mock_judge_output_detected
    ):
        """Test under-reaction is detected when low prediction on high impact."""
        # Set high true impact
        mock_outcome = MagicMock()
        mock_outcome.metrics.outcome_degradation = 0.8  # High impact -> level 3
        mock_outcome.baseline.outcome_binary = True
        mock_outcome.perturbed.outcome_binary = False

        # Set low predicted impact
        mock_judge_output_detected.impact.predicted_impact_score = 0.3

        result = evaluator.evaluate_unit(
            eval_unit=mock_eval_unit,
            judge_output=mock_judge_output_detected,
            outcome_record=mock_outcome,
        )

        assert result.calibration_eval.under_reaction is True


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestEvaluatorIntegration:
    """Integration tests that verify end-to-end evaluation logic."""

    def test_critical_error_tracking(self):
        """Test that critical errors (expected_impact=3) are tracked."""
        evaluator = Section6Evaluator("test_exp")

        # Create unit with critical expected impact
        unit = MagicMock()
        unit.evaluation_unit_id = "critical_unit"
        unit.benchmark = "toolbench"

        dc = MagicMock()
        dc.perturbation_class = "coarse_grained"
        dc.perturbation_family = "structural"
        dc.perturbation_type = "missing_step"
        dc.target_step_canonical_id = "step_5"
        dc.expected_impact = 3  # Critical
        dc.expected_detectability = 2
        unit.derived_cache = dc
        unit.perturbed = MagicMock()
        unit.perturbed.trajectory = {"steps": []}

        # Judge that detected the error
        jo = MagicMock()
        jo.judge_model = "gpt-4"
        jo.detection = MagicMock()
        jo.detection.error_detected = True
        jo.detection.error_confidence = 0.9
        jo.localization = MagicMock()
        jo.localization.predicted_error_step_canonical = "step_5"
        jo.localization.predicted_error_type = "planning"  # Maps to structural
        jo.localization.localization_confidence = 0.8
        jo.impact = MagicMock()
        jo.impact.predicted_impact_score = 0.85
        jo.impact.predicted_failure_prob = 0.8

        result = evaluator.evaluate_unit(eval_unit=unit, judge_output=jo)

        assert result.detection_eval.is_critical_detected is True

    def test_type_mapping_across_families(self):
        """Test that error type is correctly mapped to family."""
        evaluator = Section6Evaluator("test_exp")

        # Create unit with data_reference family
        unit = MagicMock()
        unit.evaluation_unit_id = "type_test"
        unit.benchmark = "gaia"

        dc = MagicMock()
        dc.perturbation_class = "fine_grained"
        dc.perturbation_family = "data_reference"
        dc.perturbation_type = "wrong_variable"
        dc.target_step_canonical_id = "step_1"
        dc.expected_impact = 1
        dc.expected_detectability = 1
        unit.derived_cache = dc
        unit.perturbed = MagicMock()
        unit.perturbed.trajectory = {"steps": [{"canonical_step_id": "step_1"}]}

        # Judge predicts a type that maps to data_reference
        jo = MagicMock()
        jo.judge_model = "claude-3"
        jo.detection = MagicMock()
        jo.detection.error_detected = True
        jo.detection.error_confidence = 0.7
        jo.localization = MagicMock()
        jo.localization.predicted_error_step_canonical = "step_1"
        jo.localization.predicted_error_type = "data_reference"
        jo.localization.localization_confidence = 0.6
        jo.impact = MagicMock()
        jo.impact.predicted_impact_score = 0.3
        jo.impact.predicted_failure_prob = 0.2

        result = evaluator.evaluate_unit(eval_unit=unit, judge_output=jo)

        assert result.detection_eval.type_correct is True
