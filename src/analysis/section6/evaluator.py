"""
Section 6 Evaluator: Compute detection and calibration evaluations.

This module provides the Section6Evaluator class which computes all
per-unit analysis metrics for Tasks 6A (Detection) and 6B (Calibration).
"""

from typing import Any, Dict, Optional

from src.analysis.section6.schema import (
    AnalysisResult,
    CalibrationEval,
    DetectionEval,
    GroundTruth,
    HumanComparison,
    JudgeOutputSummary,
    derive_impact_level,
    impact_score_to_tier,
    map_error_type_to_family,
)
from src.evaluation.schema import EvaluationUnit
from src.human_labels.schema import AggregatedLabel
from src.judges.schema import Section5JudgeOutput
from src.outcome_evidence.schema import OutcomeRecord


class Section6Evaluator:
    """
    Compute detection and calibration evaluations for evaluation units.

    This evaluator processes (evaluation_unit, judge_output) pairs and computes:
    - Ground truth from evaluation unit, outcome evidence, and human labels
    - Detection evaluation (6A): TP/FP/TN/FN, localization, type identification
    - Calibration evaluation (6B): CCE, over/under-reaction, failure prediction
    - Human comparison (when labels available)
    """

    def __init__(self, experiment_id: str):
        """
        Initialize the evaluator.

        Args:
            experiment_id: The experiment ID for analysis results
        """
        self.experiment_id = experiment_id

    def evaluate_unit(
        self,
        eval_unit: EvaluationUnit,
        judge_output: Section5JudgeOutput,
        outcome_record: Optional[OutcomeRecord] = None,
        human_label: Optional[AggregatedLabel] = None,
    ) -> AnalysisResult:
        """
        Compute all 6A and 6B evaluations for one (unit, judge) pair.

        Args:
            eval_unit: The evaluation unit with perturbation metadata
            judge_output: The judge's evaluation output
            outcome_record: Optional outcome evidence for calibration
            human_label: Optional human labels for comparison

        Returns:
            AnalysisResult with all computed evaluations
        """
        # Build ground truth
        ground_truth = self._build_ground_truth(eval_unit, outcome_record, human_label)

        # Extract judge output summary
        judge_summary = self._extract_judge_output(judge_output)

        # Compute detection evaluation (6A)
        detection_eval = self._evaluate_detection(
            judge_summary, ground_truth, eval_unit
        )

        # Compute calibration evaluation (6B)
        calibration_eval = self._evaluate_calibration(judge_summary, ground_truth)

        # Compute human comparison (if labels available)
        human_comparison = self._compare_to_human(judge_summary, human_label)

        return AnalysisResult.create(
            experiment_id=self.experiment_id,
            evaluation_unit_id=eval_unit.evaluation_unit_id,
            judge_model=judge_output.judge_model,
            ground_truth=ground_truth,
            judge_output=judge_summary,
            detection_eval=detection_eval,
            calibration_eval=calibration_eval,
            human_comparison=human_comparison,
        )

    def _build_ground_truth(
        self,
        eval_unit: EvaluationUnit,
        outcome_record: Optional[OutcomeRecord],
        human_label: Optional[AggregatedLabel],
    ) -> GroundTruth:
        """
        Build ground truth from multiple sources.

        Args:
            eval_unit: Evaluation unit with perturbation metadata
            outcome_record: Optional outcome evidence
            human_label: Optional human labels

        Returns:
            GroundTruth with combined data
        """
        dc = eval_unit.derived_cache

        # Outcome evidence data
        outcome_degradation = None
        true_impact_level = None
        baseline_outcome_binary = None
        perturbed_outcome_binary = None

        if outcome_record:
            outcome_degradation = outcome_record.metrics.outcome_degradation
            true_impact_level = derive_impact_level(outcome_degradation)
            baseline_outcome_binary = outcome_record.baseline.outcome_binary
            perturbed_outcome_binary = outcome_record.perturbed.outcome_binary

        # Human labels data
        human_error_detected = None
        human_error_step_id = None
        human_impact_tier = None
        human_error_type = None

        if human_label:
            if human_label.aggregated_detectability:
                human_error_detected = (
                    human_label.aggregated_detectability.error_detected_majority
                )
                human_error_step_id = (
                    human_label.aggregated_detectability.error_step_id_majority
                )

            if human_label.aggregated_consequence:
                human_impact_tier = human_label.aggregated_consequence.mean_impact_tier
                human_error_type = (
                    human_label.aggregated_consequence.error_type_majority
                )

        return GroundTruth(
            perturbation_class=dc.perturbation_class,
            perturbation_family=dc.perturbation_family,
            perturbation_type=dc.perturbation_type,
            target_step_canonical_id=dc.target_step_canonical_id,
            expected_impact=dc.expected_impact,
            expected_detectability=dc.expected_detectability,
            benchmark=eval_unit.benchmark,
            outcome_degradation=outcome_degradation,
            true_impact_level=true_impact_level,
            baseline_outcome_binary=baseline_outcome_binary,
            perturbed_outcome_binary=perturbed_outcome_binary,
            human_error_detected=human_error_detected,
            human_error_step_id=human_error_step_id,
            human_impact_tier=human_impact_tier,
            human_error_type=human_error_type,
        )

    def _extract_judge_output(
        self, judge_output: Section5JudgeOutput
    ) -> JudgeOutputSummary:
        """
        Extract relevant fields from judge output.

        Args:
            judge_output: The full judge output

        Returns:
            JudgeOutputSummary with relevant fields
        """
        detection = judge_output.detection
        localization = judge_output.localization
        impact = judge_output.impact

        return JudgeOutputSummary(
            error_detected=detection.error_detected if detection else False,
            error_confidence=detection.error_confidence if detection else 0.0,
            predicted_step_canonical_id=(
                localization.predicted_error_step_canonical if localization else None
            ),
            predicted_error_type=(
                localization.predicted_error_type if localization else None
            ),
            localization_confidence=(
                localization.localization_confidence if localization else None
            ),
            predicted_impact_score=impact.predicted_impact_score if impact else 0.0,
            predicted_failure_prob=impact.predicted_failure_prob if impact else 0.0,
        )

    def _evaluate_detection(
        self,
        judge_output: JudgeOutputSummary,
        ground_truth: GroundTruth,
        eval_unit: EvaluationUnit,
    ) -> DetectionEval:
        """
        Compute detection evaluation (6A metrics).

        Args:
            judge_output: Judge's predictions
            ground_truth: Ground truth data
            eval_unit: Evaluation unit for step index mapping

        Returns:
            DetectionEval with all detection metrics
        """
        is_placebo = ground_truth.perturbation_class == "placebo"
        detected = judge_output.error_detected

        # Binary classification
        is_true_positive = detected and not is_placebo
        is_false_positive = detected and is_placebo
        is_true_negative = not detected and is_placebo
        is_false_negative = not detected and not is_placebo

        # Detection correct: (detected AND non-placebo) OR (not detected AND placebo)
        detection_correct = is_true_positive or is_true_negative

        # Localization (only if detected and not placebo)
        localization_correct = None
        localization_distance = None
        localization_near = None

        if detected and not is_placebo:
            predicted_step = judge_output.predicted_step_canonical_id
            target_step = ground_truth.target_step_canonical_id

            if predicted_step and target_step:
                localization_correct = predicted_step == target_step

                # Try to compute distance using step indices
                localization_distance = self._compute_localization_distance(
                    predicted_step, target_step, eval_unit
                )

                if localization_distance is not None:
                    localization_near = localization_distance <= 1

        # Type identification (only if detected and not placebo)
        # Compare at both type level and family level for flexibility
        type_correct = None
        if detected and not is_placebo:
            predicted_type = judge_output.predicted_error_type
            if predicted_type:
                predicted_lower = predicted_type.lower()
                gt_type_lower = ground_truth.perturbation_type.lower()
                gt_family_lower = ground_truth.perturbation_family.lower()

                # Option 1: Direct type match (judge predicts exact type)
                if predicted_lower == gt_type_lower:
                    type_correct = True
                # Option 2: Family-level match (judge predicts at family level)
                elif predicted_lower == gt_family_lower:
                    type_correct = True
                # Option 3: Judge type maps to ground truth family
                else:
                    mapped_family = map_error_type_to_family(predicted_type)
                    if mapped_family:
                        type_correct = mapped_family.lower() == gt_family_lower
                    else:
                        type_correct = False

        # Critical error detection
        is_critical_detected = None
        if ground_truth.expected_impact == 3:
            is_critical_detected = detected

        return DetectionEval(
            detection_correct=detection_correct,
            is_true_positive=is_true_positive,
            is_false_positive=is_false_positive,
            is_true_negative=is_true_negative,
            is_false_negative=is_false_negative,
            localization_correct=localization_correct,
            localization_distance=localization_distance,
            localization_near=localization_near,
            type_correct=type_correct,
            is_critical_detected=is_critical_detected,
        )

    def _compute_localization_distance(
        self,
        predicted_step: str,
        target_step: str,
        eval_unit: EvaluationUnit,
    ) -> Optional[int]:
        """
        Compute the distance between predicted and target steps.

        Args:
            predicted_step: Predicted step canonical ID
            target_step: Target step canonical ID
            eval_unit: Evaluation unit for step index mapping

        Returns:
            Distance in steps, or None if cannot be computed
        """
        # Try to build step index map from trajectory
        try:
            perturbed_traj = eval_unit.perturbed.trajectory
            steps = perturbed_traj.get("steps", [])

            step_to_index = {}
            for idx, step in enumerate(steps):
                canonical_id = step.get("canonical_step_id")
                if canonical_id:
                    step_to_index[canonical_id] = idx

            if predicted_step in step_to_index and target_step in step_to_index:
                return abs(step_to_index[predicted_step] - step_to_index[target_step])
        except (AttributeError, KeyError, TypeError):
            pass

        return None

    def _evaluate_calibration(
        self,
        judge_output: JudgeOutputSummary,
        ground_truth: GroundTruth,
    ) -> CalibrationEval:
        """
        Compute calibration evaluation (6B metrics).

        Args:
            judge_output: Judge's predictions
            ground_truth: Ground truth data

        Returns:
            CalibrationEval with all calibration metrics
        """
        predicted_impact = judge_output.predicted_impact_score
        predicted_failure = judge_output.predicted_failure_prob

        # CCE requires outcome degradation
        cce = None
        abs_cce = None
        if ground_truth.outcome_degradation is not None:
            # Normalize OD to 0-1 (it's typically already in this range but can be negative)
            # We treat OD as the true impact, clipped to [0, 1]
            normalized_od = max(0.0, min(1.0, ground_truth.outcome_degradation))
            cce = predicted_impact - normalized_od
            abs_cce = abs(cce)

        # Over/under reaction requires true impact level
        over_reaction = None
        under_reaction = None
        if ground_truth.true_impact_level is not None:
            over_reaction = (
                predicted_impact > 0.5 and ground_truth.true_impact_level <= 1
            )
            under_reaction = (
                predicted_impact < 0.5 and ground_truth.true_impact_level == 3
            )

        # Failure prediction
        failure_predicted = predicted_failure > 0.5
        failure_actual = None
        failure_correct = None

        if ground_truth.perturbed_outcome_binary is not None:
            failure_actual = not ground_truth.perturbed_outcome_binary
            failure_correct = failure_predicted == failure_actual

        # Impact tier prediction
        impact_tier_predicted = impact_score_to_tier(predicted_impact)
        impact_tier_error = None
        if ground_truth.true_impact_level is not None:
            impact_tier_error = abs(
                impact_tier_predicted - ground_truth.true_impact_level
            )

        return CalibrationEval(
            cce=cce,
            abs_cce=abs_cce,
            over_reaction=over_reaction,
            under_reaction=under_reaction,
            failure_predicted=failure_predicted,
            failure_actual=failure_actual,
            failure_correct=failure_correct,
            impact_tier_predicted=impact_tier_predicted,
            impact_tier_error=impact_tier_error,
        )

    def _compare_to_human(
        self,
        judge_output: JudgeOutputSummary,
        human_label: Optional[AggregatedLabel],
    ) -> Optional[HumanComparison]:
        """
        Compare judge output to human labels.

        Args:
            judge_output: Judge's predictions
            human_label: Human labels (if available)

        Returns:
            HumanComparison or None if no human labels
        """
        if human_label is None:
            return None

        detection_agrees = None
        localization_agrees = None
        type_agrees = None
        impact_tier_diff = None

        # Detection agreement
        if human_label.aggregated_detectability:
            human_detected = (
                human_label.aggregated_detectability.error_detected_majority
            )
            detection_agrees = judge_output.error_detected == human_detected

            # Localization agreement (only if both detected error)
            if judge_output.error_detected and human_detected:
                human_step = human_label.aggregated_detectability.error_step_id_majority
                if human_step and judge_output.predicted_step_canonical_id:
                    localization_agrees = (
                        judge_output.predicted_step_canonical_id == human_step
                    )

        # Type agreement
        if human_label.aggregated_consequence:
            human_type = human_label.aggregated_consequence.error_type_majority
            if human_type and judge_output.predicted_error_type:
                # Map judge type to family and compare
                mapped_judge = map_error_type_to_family(
                    judge_output.predicted_error_type
                )
                # Human type is already the family
                type_agrees = mapped_judge == human_type

            # Impact tier difference
            human_impact = human_label.aggregated_consequence.mean_impact_tier
            if human_impact is not None:
                judge_tier = impact_score_to_tier(judge_output.predicted_impact_score)
                impact_tier_diff = judge_tier - human_impact

        return HumanComparison(
            detection_agrees=detection_agrees,
            localization_agrees=localization_agrees,
            type_agrees=type_agrees,
            impact_tier_diff=impact_tier_diff,
        )


def evaluate_from_dicts(
    experiment_id: str,
    eval_unit_dict: Dict[str, Any],
    judge_output_dict: Dict[str, Any],
    outcome_record_dict: Optional[Dict[str, Any]] = None,
    human_label_dict: Optional[Dict[str, Any]] = None,
) -> AnalysisResult:
    """
    Convenience function to evaluate from dictionary data.

    Args:
        experiment_id: Experiment ID
        eval_unit_dict: Evaluation unit as dictionary
        judge_output_dict: Judge output as dictionary
        outcome_record_dict: Optional outcome record as dictionary
        human_label_dict: Optional human label as dictionary

    Returns:
        AnalysisResult
    """
    eval_unit = EvaluationUnit.from_dict(eval_unit_dict)
    judge_output = Section5JudgeOutput.from_dict(judge_output_dict)

    outcome_record = None
    if outcome_record_dict:
        outcome_record = OutcomeRecord.from_dict(outcome_record_dict)

    human_label = None
    if human_label_dict:
        human_label = AggregatedLabel.from_dict(human_label_dict)

    evaluator = Section6Evaluator(experiment_id)
    return evaluator.evaluate_unit(eval_unit, judge_output, outcome_record, human_label)
