"""
Unit tests for the human_labels module.

Tests all components:
- Schema classes (AnnotationRecord, AggregatedLabel, etc.)
- Sampling functions
- Agreement computation
- Aggregation logic
- QC checks
- Storage functions
"""

import json
import os
import tempfile
from datetime import datetime

import pytest

from src.human_labels.schema import (
    AggregatedConsequence,
    AggregatedDetectability,
    AggregatedLabel,
    AggregatedPreference,
    AnnotationMode,
    AnnotationRecord,
    ConsequenceLabels,
    Correctness,
    DetectabilityLabels,
    ErrorTrajectory,
    ErrorType,
    LocalizationAccuracy,
    Preference,
    PreferenceLabels,
)
from src.human_labels.sampling import (
    assign_to_annotators,
    generate_annotation_batches,
    sample_for_annotation,
)
from src.human_labels.agreement import (
    compute_agreement_report,
    compute_krippendorff_alpha,
    compute_localization_accuracy,
    compute_pairwise_agreement,
)
from src.human_labels.aggregation import (
    aggregate_annotations,
    compute_aggregation_summary,
    derive_error_type_match,
    derive_localization_accuracy,
)
from src.human_labels.qc import (
    check_consecutive_same,
    check_gold_set_accuracy,
    check_time_bounds,
    check_time_bounds_batch,
    run_all_qc_checks,
)
from src.human_labels.storage import (
    create_human_labels_directories,
    export_aggregated_labels,
    load_aggregated_labels,
    load_annotations_from_json,
    save_annotations_to_json,
)

# === Fixtures ===


@pytest.fixture
def sample_detectability_labels():
    """Sample detectability labels."""
    return DetectabilityLabels(
        error_detected=True,
        error_trajectory="A",
        error_step_id="gaia_122::step::2",
        confidence=4,
    )


@pytest.fixture
def sample_consequence_labels():
    """Sample consequence labels."""
    return ConsequenceLabels(
        error_type="data_reference",
        impact_tier=3,
        propagation_depth=1,
        correctness_a="incorrect",
        correctness_b="correct",
    )


@pytest.fixture
def sample_preference_labels():
    """Sample preference labels."""
    return PreferenceLabels(
        preference="B",
        preference_reason="Trajectory B provides correct answer",
    )


@pytest.fixture
def sample_annotation_record(sample_detectability_labels):
    """Sample complete annotation record."""
    return AnnotationRecord(
        annotation_id="ann_001",
        evaluation_unit_id="eval::gaia_122::001",
        annotator_id="annotator_A",
        annotation_mode="detectability",
        created_at="2026-04-10T14:30:00Z",
        view_file="views/human/detectability/eval_gaia_122_001.json",
        trajectory_a_variant_id="gaia_122::pert::001",
        trajectory_b_variant_id="gaia_122::base",
        detectability=sample_detectability_labels,
        consequence=None,
        preference=None,
        time_spent_seconds=145,
        notes="Step 2 has wrong value",
        flagged_for_review=False,
    )


@pytest.fixture
def multiple_annotations():
    """Multiple annotations for same unit by different annotators."""
    annotations = []
    for i, annotator in enumerate(["annotator_A", "annotator_B", "annotator_C"]):
        ann = AnnotationRecord(
            annotation_id=f"ann_{i:03d}",
            evaluation_unit_id="eval::gaia_122::001",
            annotator_id=annotator,
            annotation_mode="detectability",
            created_at=f"2026-04-10T14:{30+i}:00Z",
            view_file="views/human/detectability/eval_gaia_122_001.json",
            trajectory_a_variant_id="gaia_122::pert::001",
            trajectory_b_variant_id="gaia_122::base",
            detectability=DetectabilityLabels(
                error_detected=True,
                error_trajectory="A",
                error_step_id="gaia_122::step::2",
                confidence=3 + i,
            ),
            consequence=None,
            preference=None,
            time_spent_seconds=100 + i * 30,
            notes=None,
            flagged_for_review=False,
        )
        annotations.append(ann)
    return annotations


@pytest.fixture
def sample_evaluation_units():
    """Sample evaluation units for sampling tests."""
    units = []
    benchmarks = ["toolbench", "swebench"]
    classes = ["placebo", "fine_grained", "coarse_grained"]
    families = ["data_reference", "parameter", "tool_selection"]

    for i in range(100):
        unit = {
            "evaluation_unit_id": f"eval::test_{i:03d}",
            "benchmark": benchmarks[i % 2],
            "derived_cache": {
                "perturbation_class": classes[i % 3],
                "perturbation_family": families[i % 3],
                "expected_detectability": i % 3,
                "expected_impact": i % 4,
            },
        }
        units.append(unit)
    return units


# === Schema Tests ===


class TestDetectabilityLabels:
    def test_to_dict(self, sample_detectability_labels):
        d = sample_detectability_labels.to_dict()
        assert d["error_detected"] is True
        assert d["error_trajectory"] == "A"
        assert d["error_step_id"] == "gaia_122::step::2"
        assert d["confidence"] == 4

    def test_from_dict(self):
        d = {
            "error_detected": False,
            "error_trajectory": "B",
            "error_step_id": None,
            "confidence": 2,
        }
        labels = DetectabilityLabels.from_dict(d)
        assert labels.error_detected is False
        assert labels.error_trajectory == "B"
        assert labels.error_step_id is None
        assert labels.confidence == 2


class TestConsequenceLabels:
    def test_to_dict(self, sample_consequence_labels):
        d = sample_consequence_labels.to_dict()
        assert d["error_type"] == "data_reference"
        assert d["impact_tier"] == 3
        assert d["propagation_depth"] == 1
        assert d["correctness_a"] == "incorrect"
        assert d["correctness_b"] == "correct"

    def test_from_dict(self):
        d = {
            "error_type": "planning",
            "impact_tier": 2,
            "propagation_depth": 2,
            "correctness_a": "partially_correct",
            "correctness_b": "correct",
        }
        labels = ConsequenceLabels.from_dict(d)
        assert labels.error_type == "planning"
        assert labels.impact_tier == 2


class TestPreferenceLabels:
    def test_to_dict(self, sample_preference_labels):
        d = sample_preference_labels.to_dict()
        assert d["preference"] == "B"
        assert d["preference_reason"] == "Trajectory B provides correct answer"

    def test_from_dict(self):
        d = {"preference": "tie", "preference_reason": None}
        labels = PreferenceLabels.from_dict(d)
        assert labels.preference == "tie"
        assert labels.preference_reason is None


class TestAnnotationRecord:
    def test_to_dict(self, sample_annotation_record):
        d = sample_annotation_record.to_dict()
        assert d["annotation_id"] == "ann_001"
        assert d["evaluation_unit_id"] == "eval::gaia_122::001"
        assert d["annotator_id"] == "annotator_A"
        assert d["detectability"]["error_detected"] is True
        assert d["consequence"] is None
        assert d["preference"] is None

    def test_from_dict(self, sample_annotation_record):
        d = sample_annotation_record.to_dict()
        restored = AnnotationRecord.from_dict(d)
        assert restored.annotation_id == sample_annotation_record.annotation_id
        assert restored.detectability.error_detected == True
        assert restored.detectability.error_trajectory == "A"

    def test_roundtrip(self, sample_annotation_record):
        d = sample_annotation_record.to_dict()
        restored = AnnotationRecord.from_dict(d)
        d2 = restored.to_dict()
        assert d == d2


class TestAggregatedLabel:
    def test_to_dict(self):
        agg_detect = AggregatedDetectability(
            error_detected_majority=True,
            error_detected_agreement=1.0,
            error_trajectory_majority="A",
            error_trajectory_agreement=1.0,
            error_step_id_majority="step_2",
            localization_agreement="exact",
            mean_confidence=3.5,
        )
        agg_label = AggregatedLabel(
            evaluation_unit_id="eval::test::001",
            annotation_ids=["ann_001", "ann_002"],
            num_annotators=2,
            aggregated_detectability=agg_detect,
            aggregated_consequence=None,
            aggregated_preference=None,
            low_agreement_flag=False,
            needs_adjudication=False,
        )
        d = agg_label.to_dict()
        assert d["evaluation_unit_id"] == "eval::test::001"
        assert d["num_annotators"] == 2
        assert d["aggregated_detectability"]["error_detected_majority"] is True

    def test_from_dict(self):
        d = {
            "evaluation_unit_id": "eval::test::001",
            "annotation_ids": ["ann_001"],
            "num_annotators": 1,
            "aggregated_detectability": {
                "error_detected_majority": False,
                "error_detected_agreement": 1.0,
                "error_trajectory_majority": "neither",
                "error_trajectory_agreement": 1.0,
                "error_step_id_majority": None,
                "localization_agreement": "mixed",
                "mean_confidence": 4.0,
            },
            "aggregated_consequence": None,
            "aggregated_preference": None,
            "low_agreement_flag": False,
            "needs_adjudication": False,
        }
        label = AggregatedLabel.from_dict(d)
        assert label.evaluation_unit_id == "eval::test::001"
        assert label.aggregated_detectability.error_detected_majority is False


# === Sampling Tests ===


class TestSampling:
    def test_sample_for_annotation_basic(self, sample_evaluation_units):
        result = sample_for_annotation(
            sample_evaluation_units,
            config={
                "mode_a_target": 50,
                "mode_b_target": 30,
                "mode_c_target": 20,
                "stratification": {},
                "coverage_minimums": {},
            },
            seed=42,
        )
        assert len(result["mode_a"]) == 50
        assert len(result["mode_b"]) == 30
        assert len(result["mode_c"]) == 20
        assert "sampling_report" in result

    def test_sample_reproducibility(self, sample_evaluation_units):
        result1 = sample_for_annotation(
            sample_evaluation_units,
            config={"mode_a_target": 30, "stratification": {}, "coverage_minimums": {}},
            seed=123,
        )
        result2 = sample_for_annotation(
            sample_evaluation_units,
            config={"mode_a_target": 30, "stratification": {}, "coverage_minimums": {}},
            seed=123,
        )
        ids1 = [u["evaluation_unit_id"] for u in result1["mode_a"]]
        ids2 = [u["evaluation_unit_id"] for u in result2["mode_a"]]
        assert ids1 == ids2

    def test_assign_to_annotators(self, sample_evaluation_units):
        sample = sample_evaluation_units[:50]
        annotators = ["A", "B", "C"]
        assignments = assign_to_annotators(
            sample,
            annotators,
            overlap_config={"full_overlap_count": 10, "pairwise_overlap_count": 15},
        )
        assert "assignments_by_annotator" in assignments
        assert "assignments_by_unit" in assignments
        assert len(assignments["full_overlap_units"]) == 10

        # All annotators should have the full overlap units
        for annotator in annotators:
            assert all(
                uid in assignments["assignments_by_annotator"][annotator]
                for uid in assignments["full_overlap_units"]
            )

    def test_generate_annotation_batches(self, sample_evaluation_units):
        sample = sample_evaluation_units[:50]
        annotators = ["A", "B"]
        assignments = assign_to_annotators(sample, annotators)
        batches = generate_annotation_batches(assignments, batch_size=10)

        assert "A" in batches
        assert "B" in batches
        # Each batch should have at most 10 items
        for annotator_batches in batches.values():
            for batch in annotator_batches:
                assert len(batch) <= 10


# === Agreement Tests ===


class TestAgreement:
    def test_compute_pairwise_agreement(self, multiple_annotations):
        result = compute_pairwise_agreement(
            multiple_annotations, dimension="error_detected"
        )
        # All annotators agreed (error_detected=True)
        assert result["overall_agreement"] == 1.0

    def test_compute_pairwise_agreement_with_disagreement(self):
        # Create annotations with disagreement
        annotations = [
            AnnotationRecord(
                annotation_id="ann_001",
                evaluation_unit_id="unit_1",
                annotator_id="A",
                annotation_mode="detectability",
                created_at="2026-04-10T14:30:00Z",
                view_file="view.json",
                trajectory_a_variant_id="a",
                trajectory_b_variant_id="b",
                detectability=DetectabilityLabels(
                    error_detected=True,
                    error_trajectory="A",
                    error_step_id="step_1",
                    confidence=3,
                ),
                consequence=None,
                preference=None,
                time_spent_seconds=100,
                notes=None,
                flagged_for_review=False,
            ),
            AnnotationRecord(
                annotation_id="ann_002",
                evaluation_unit_id="unit_1",
                annotator_id="B",
                annotation_mode="detectability",
                created_at="2026-04-10T14:35:00Z",
                view_file="view.json",
                trajectory_a_variant_id="a",
                trajectory_b_variant_id="b",
                detectability=DetectabilityLabels(
                    error_detected=False,
                    error_trajectory="neither",
                    error_step_id=None,
                    confidence=2,
                ),
                consequence=None,
                preference=None,
                time_spent_seconds=90,
                notes=None,
                flagged_for_review=False,
            ),
        ]
        result = compute_pairwise_agreement(annotations, dimension="error_detected")
        assert result["overall_agreement"] == 0.0

    def test_compute_localization_accuracy_exact(self):
        result = compute_localization_accuracy("step_2", "step_2")
        assert result == "exact"

    def test_compute_localization_accuracy_near(self):
        step_map = {"step_1": 0, "step_2": 1, "step_3": 2}
        result = compute_localization_accuracy("step_2", "step_3", step_map)
        assert result == "near"

    def test_compute_localization_accuracy_wrong(self):
        step_map = {"step_1": 0, "step_2": 1, "step_5": 4}
        result = compute_localization_accuracy("step_1", "step_5", step_map)
        assert result == "wrong"

    def test_compute_localization_accuracy_null_predicted(self):
        result = compute_localization_accuracy(None, "step_2")
        assert result == "wrong"


# === Aggregation Tests ===


class TestAggregation:
    def test_aggregate_annotations_perfect_agreement(self, multiple_annotations):
        aggregated = aggregate_annotations(multiple_annotations)
        assert len(aggregated) == 1
        agg = aggregated[0]
        assert agg.aggregated_detectability.error_detected_majority is True
        assert agg.aggregated_detectability.error_detected_agreement == 1.0
        assert agg.aggregated_detectability.error_trajectory_majority == "A"
        assert agg.low_agreement_flag is False

    def test_aggregate_annotations_with_disagreement(self):
        annotations = [
            AnnotationRecord(
                annotation_id="ann_001",
                evaluation_unit_id="unit_1",
                annotator_id="A",
                annotation_mode="detectability",
                created_at="2026-04-10T14:30:00Z",
                view_file="view.json",
                trajectory_a_variant_id="a",
                trajectory_b_variant_id="b",
                detectability=DetectabilityLabels(
                    error_detected=True,
                    error_trajectory="A",
                    error_step_id="step_1",
                    confidence=4,
                ),
                consequence=None,
                preference=None,
                time_spent_seconds=100,
                notes=None,
                flagged_for_review=False,
            ),
            AnnotationRecord(
                annotation_id="ann_002",
                evaluation_unit_id="unit_1",
                annotator_id="B",
                annotation_mode="detectability",
                created_at="2026-04-10T14:35:00Z",
                view_file="view.json",
                trajectory_a_variant_id="a",
                trajectory_b_variant_id="b",
                detectability=DetectabilityLabels(
                    error_detected=True,
                    error_trajectory="B",
                    error_step_id="step_2",
                    confidence=3,
                ),
                consequence=None,
                preference=None,
                time_spent_seconds=90,
                notes=None,
                flagged_for_review=False,
            ),
            AnnotationRecord(
                annotation_id="ann_003",
                evaluation_unit_id="unit_1",
                annotator_id="C",
                annotation_mode="detectability",
                created_at="2026-04-10T14:40:00Z",
                view_file="view.json",
                trajectory_a_variant_id="a",
                trajectory_b_variant_id="b",
                detectability=DetectabilityLabels(
                    error_detected=True,
                    error_trajectory="A",
                    error_step_id="step_1",
                    confidence=5,
                ),
                consequence=None,
                preference=None,
                time_spent_seconds=110,
                notes=None,
                flagged_for_review=False,
            ),
        ]
        aggregated = aggregate_annotations(annotations)
        agg = aggregated[0]
        # Majority on trajectory is A (2 vs 1)
        assert agg.aggregated_detectability.error_trajectory_majority == "A"
        # Agreement is 2/3
        assert (
            abs(agg.aggregated_detectability.error_trajectory_agreement - 2 / 3) < 0.01
        )

    def test_derive_localization_accuracy(self):
        assert derive_localization_accuracy("step_2", "step_2") == "exact"
        assert derive_localization_accuracy(None, "step_2") == "wrong"

    def test_derive_error_type_match(self):
        assert derive_error_type_match("data_reference", "data_reference") is True
        assert derive_error_type_match("planning", "structural") is True
        assert derive_error_type_match("tool_selection", "parameter") is False
        assert derive_error_type_match("unclear", "parameter") is False

    def test_compute_aggregation_summary(self, multiple_annotations):
        aggregated = aggregate_annotations(multiple_annotations)
        summary = compute_aggregation_summary(aggregated)
        assert summary["total_units"] == 1
        assert summary["units_with_detectability"] == 1
        assert summary["low_agreement_count"] == 0


# === QC Tests ===


class TestQC:
    def test_check_time_bounds_passed(self, sample_annotation_record):
        result = check_time_bounds(sample_annotation_record, min_sec=30, max_sec=600)
        assert result["passed"] is True
        assert result["issue"] is None

    def test_check_time_bounds_too_fast(self, sample_annotation_record):
        sample_annotation_record.time_spent_seconds = 10
        result = check_time_bounds(sample_annotation_record, min_sec=30, max_sec=600)
        assert result["passed"] is False
        assert result["issue"] == "too_fast"

    def test_check_time_bounds_too_slow(self, sample_annotation_record):
        sample_annotation_record.time_spent_seconds = 700
        result = check_time_bounds(sample_annotation_record, min_sec=30, max_sec=600)
        assert result["passed"] is False
        assert result["issue"] == "too_slow"

    def test_check_time_bounds_batch(self, multiple_annotations):
        result = check_time_bounds_batch(multiple_annotations)
        assert result["total_checked"] == 3
        assert result["passed_count"] == 3
        assert result["mean_time"] > 0

    def test_check_gold_set_accuracy(self):
        annotations = [
            AnnotationRecord(
                annotation_id="ann_001",
                evaluation_unit_id="gold_1",
                annotator_id="A",
                annotation_mode="detectability",
                created_at="2026-04-10T14:30:00Z",
                view_file="view.json",
                trajectory_a_variant_id="a",
                trajectory_b_variant_id="b",
                detectability=DetectabilityLabels(
                    error_detected=True,
                    error_trajectory="A",
                    error_step_id="step_1",
                    confidence=4,
                ),
                consequence=None,
                preference=None,
                time_spent_seconds=100,
                notes=None,
                flagged_for_review=False,
            ),
            AnnotationRecord(
                annotation_id="ann_002",
                evaluation_unit_id="gold_2",
                annotator_id="A",
                annotation_mode="detectability",
                created_at="2026-04-10T14:35:00Z",
                view_file="view.json",
                trajectory_a_variant_id="a",
                trajectory_b_variant_id="b",
                detectability=DetectabilityLabels(
                    error_detected=False,
                    error_trajectory="neither",
                    error_step_id=None,
                    confidence=3,
                ),
                consequence=None,
                preference=None,
                time_spent_seconds=90,
                notes=None,
                flagged_for_review=False,
            ),
        ]
        gold_set = [
            {"evaluation_unit_id": "gold_1", "gold_error_detected": True},
            {
                "evaluation_unit_id": "gold_2",
                "gold_error_detected": True,
            },  # Annotator got wrong
        ]
        result = check_gold_set_accuracy(annotations, gold_set)
        assert result["gold_checked"] == 2
        assert result["gold_correct"] == 1
        assert result["accuracy"] == 0.5
        assert result["status"] == "recalibration_needed"

    def test_check_consecutive_same(self):
        # Create annotations with consecutive same responses
        annotations = []
        for i in range(10):
            ann = AnnotationRecord(
                annotation_id=f"ann_{i:03d}",
                evaluation_unit_id=f"unit_{i:03d}",
                annotator_id="A",
                annotation_mode="detectability",
                created_at=f"2026-04-10T14:{30+i}:00Z",
                view_file="view.json",
                trajectory_a_variant_id="a",
                trajectory_b_variant_id="b",
                detectability=DetectabilityLabels(
                    error_detected=True,  # All same
                    error_trajectory="A",
                    error_step_id="step_1",
                    confidence=4,
                ),
                consequence=None,
                preference=None,
                time_spent_seconds=100,
                notes=None,
                flagged_for_review=False,
            )
            annotations.append(ann)

        result = check_consecutive_same(annotations, threshold=5)
        assert result["overall_flagged"] is True
        assert result["by_annotator"]["A"]["flagged"] is True
        assert result["by_annotator"]["A"]["max_consecutive"] == 10

    def test_check_consecutive_same_no_issue(self):
        # Create annotations with varying responses
        annotations = []
        for i in range(6):
            ann = AnnotationRecord(
                annotation_id=f"ann_{i:03d}",
                evaluation_unit_id=f"unit_{i:03d}",
                annotator_id="A",
                annotation_mode="detectability",
                created_at=f"2026-04-10T14:{30+i}:00Z",
                view_file="view.json",
                trajectory_a_variant_id="a",
                trajectory_b_variant_id="b",
                detectability=DetectabilityLabels(
                    error_detected=i % 2 == 0,  # Alternating
                    error_trajectory="A" if i % 2 == 0 else "B",
                    error_step_id="step_1",
                    confidence=4,
                ),
                consequence=None,
                preference=None,
                time_spent_seconds=100,
                notes=None,
                flagged_for_review=False,
            )
            annotations.append(ann)

        result = check_consecutive_same(annotations, threshold=5)
        assert result["overall_flagged"] is False


# === Storage Tests ===


class TestStorage:
    def test_save_and_load_annotations(self, sample_annotation_record):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            filepath = save_annotations_to_json(
                [sample_annotation_record], tmpdir, batch_id="test_001"
            )
            assert os.path.exists(filepath)

            # Load
            loaded = load_annotations_from_json(filepath)
            assert len(loaded) == 1
            assert loaded[0].annotation_id == sample_annotation_record.annotation_id
            assert loaded[0].detectability.error_detected is True

    def test_save_and_load_aggregated_labels(self):
        agg_label = AggregatedLabel(
            evaluation_unit_id="eval::test::001",
            annotation_ids=["ann_001"],
            num_annotators=1,
            aggregated_detectability=AggregatedDetectability(
                error_detected_majority=True,
                error_detected_agreement=1.0,
                error_trajectory_majority="A",
                error_trajectory_agreement=1.0,
                error_step_id_majority="step_1",
                localization_agreement="exact",
                mean_confidence=4.0,
            ),
            aggregated_consequence=None,
            aggregated_preference=None,
            low_agreement_flag=False,
            needs_adjudication=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "aggregated.json")
            export_aggregated_labels([agg_label], filepath)
            assert os.path.exists(filepath)

            loaded = load_aggregated_labels(filepath)
            assert len(loaded) == 1
            assert loaded[0].evaluation_unit_id == "eval::test::001"
            assert loaded[0].aggregated_detectability.error_detected_majority is True

    def test_create_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = create_human_labels_directories(tmpdir)
            assert "raw" in paths
            assert "aggregated" in paths
            assert "gold" in paths
            assert "metadata" in paths
            for path in paths.values():
                assert os.path.exists(path)

    def test_load_from_directory(self, sample_annotation_record):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save multiple files
            save_annotations_to_json(
                [sample_annotation_record], tmpdir, batch_id="batch_001"
            )

            ann2 = AnnotationRecord(
                annotation_id="ann_002",
                evaluation_unit_id="eval::gaia_123::001",
                annotator_id="annotator_B",
                annotation_mode="detectability",
                created_at="2026-04-10T15:30:00Z",
                view_file="view.json",
                trajectory_a_variant_id="a",
                trajectory_b_variant_id="b",
                detectability=DetectabilityLabels(
                    error_detected=False,
                    error_trajectory="neither",
                    error_step_id=None,
                    confidence=3,
                ),
                consequence=None,
                preference=None,
                time_spent_seconds=120,
                notes=None,
                flagged_for_review=False,
            )
            save_annotations_to_json([ann2], tmpdir, batch_id="batch_002")

            # Load from directory
            loaded = load_annotations_from_json(tmpdir)
            assert len(loaded) == 2


# === Enum Tests ===


class TestEnums:
    def test_annotation_mode(self):
        assert AnnotationMode.DETECTABILITY.value == "detectability"
        assert AnnotationMode.CONSEQUENCE.value == "consequence"
        assert AnnotationMode.PREFERENCE.value == "preference"

    def test_error_trajectory(self):
        assert ErrorTrajectory.A.value == "A"
        assert ErrorTrajectory.B.value == "B"
        assert ErrorTrajectory.NEITHER.value == "neither"
        assert ErrorTrajectory.BOTH.value == "both"

    def test_error_type(self):
        assert ErrorType.PLANNING.value == "planning"
        assert ErrorType.TOOL_SELECTION.value == "tool_selection"
        assert ErrorType.PARAMETER.value == "parameter"
        assert ErrorType.DATA_REFERENCE.value == "data_reference"
        assert ErrorType.UNCLEAR.value == "unclear"

    def test_correctness(self):
        assert Correctness.CORRECT.value == "correct"
        assert Correctness.PARTIALLY_CORRECT.value == "partially_correct"
        assert Correctness.INCORRECT.value == "incorrect"

    def test_preference(self):
        assert Preference.A.value == "A"
        assert Preference.B.value == "B"
        assert Preference.TIE.value == "tie"

    def test_localization_accuracy(self):
        assert LocalizationAccuracy.EXACT.value == "exact"
        assert LocalizationAccuracy.NEAR.value == "near"
        assert LocalizationAccuracy.WRONG.value == "wrong"


# === Integration Tests ===


class TestIntegration:
    def test_full_pipeline(self, sample_evaluation_units):
        """Test the full annotation pipeline from sampling to aggregation."""
        # Step 1: Sample units
        sample_result = sample_for_annotation(
            sample_evaluation_units,
            config={
                "mode_a_target": 20,
                "mode_b_target": 10,
                "mode_c_target": 5,
                "stratification": {},
                "coverage_minimums": {},
            },
            seed=42,
        )
        sampled_units = sample_result["mode_a"]
        assert len(sampled_units) == 20

        # Step 2: Assign to annotators
        annotators = ["annotator_A", "annotator_B"]
        assignments = assign_to_annotators(
            sampled_units,
            annotators,
            overlap_config={"full_overlap_count": 5, "pairwise_overlap_count": 5},
        )
        assert len(assignments["full_overlap_units"]) == 5

        # Step 3: Simulate annotations (for testing)
        annotations = []
        for unit in sampled_units:
            unit_id = unit["evaluation_unit_id"]
            assigned_annotators = assignments["assignments_by_unit"].get(unit_id, [])
            for annotator in assigned_annotators:
                ann = AnnotationRecord(
                    annotation_id=f"ann_{unit_id}_{annotator}",
                    evaluation_unit_id=unit_id,
                    annotator_id=annotator,
                    annotation_mode="detectability",
                    created_at="2026-04-10T14:30:00Z",
                    view_file="view.json",
                    trajectory_a_variant_id="a",
                    trajectory_b_variant_id="b",
                    detectability=DetectabilityLabels(
                        error_detected=True,
                        error_trajectory="A",
                        error_step_id="step_1",
                        confidence=4,
                    ),
                    consequence=None,
                    preference=None,
                    time_spent_seconds=120,
                    notes=None,
                    flagged_for_review=False,
                )
                annotations.append(ann)

        # Step 4: Run QC
        qc_report = run_all_qc_checks(annotations)
        assert qc_report["total_annotations"] == len(annotations)

        # Step 5: Aggregate
        aggregated = aggregate_annotations(annotations)
        assert len(aggregated) == 20  # One per unit

        # Step 6: Generate summary
        summary = compute_aggregation_summary(aggregated)
        assert summary["total_units"] == 20
        assert summary["units_with_detectability"] == 20


# === MongoDB Storage Tests ===


class MockCollection:
    """Mock MongoDB collection for testing."""

    def __init__(self):
        self.data = {}
        self.indexes = []

    def create_index(self, index, **kwargs):
        self.indexes.append(index)

    def update_one(self, filter_dict, update_dict, upsert=False):
        key = str(filter_dict)
        if upsert or key in self.data:
            self.data[key] = update_dict.get("$set", {})

    def find(self, query, projection=None):
        results = []
        for key, doc in self.data.items():
            # Simple matching: check if query fields exist in doc
            match = True
            for q_key, q_val in query.items():
                if doc.get(q_key) != q_val:
                    match = False
                    break
            if match:
                if projection:
                    results.append({k: doc.get(k) for k in projection.keys()})
                else:
                    # Return copy with _id
                    result = doc.copy()
                    result["_id"] = "mock_id"
                    results.append(result)
        return results


class MockDB:
    """Mock MongoDB database."""

    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection()
        return self.collections[name]


class MockStorage:
    """Mock MongoDBStorage for testing."""

    def __init__(self):
        self.db = MockDB()


class TestMongoDBStorage:
    """Tests for MongoDB storage functions."""

    @pytest.fixture
    def mock_storage(self):
        return MockStorage()

    @pytest.fixture
    def sample_annotation(self):
        return AnnotationRecord(
            annotation_id="ann_test_001",
            evaluation_unit_id="unit_001",
            annotator_id="researcher",
            annotation_mode="detectability",
            created_at="2026-04-10T14:30:00Z",
            view_file="view.json",
            trajectory_a_variant_id="a",
            trajectory_b_variant_id="b",
            detectability=DetectabilityLabels(
                error_detected=True,
                error_trajectory="A",
                error_step_id="step_1",
                confidence=4,
            ),
            consequence=None,
            preference=None,
            time_spent_seconds=120,
            notes=None,
            flagged_for_review=False,
        )

    @pytest.fixture
    def sample_aggregated_label(self):
        return AggregatedLabel(
            evaluation_unit_id="unit_001",
            annotation_ids=["ann_001", "ann_002"],
            num_annotators=2,
            aggregated_detectability=AggregatedDetectability(
                error_detected_majority=True,
                error_detected_agreement=1.0,
                error_trajectory_majority="A",
                error_trajectory_agreement=1.0,
                error_step_id_majority="step_1",
                localization_agreement="exact",
                mean_confidence=4.0,
            ),
            aggregated_consequence=None,
            aggregated_preference=None,
            low_agreement_flag=False,
            needs_adjudication=False,
        )

    def test_save_annotations_to_mongodb(self, mock_storage, sample_annotation):
        from src.human_labels.storage import save_annotations_to_mongodb

        saved = save_annotations_to_mongodb(
            [sample_annotation], mock_storage, "exp_test"
        )
        assert saved == 1
        assert len(mock_storage.db["human_labels"].data) == 1

    def test_load_annotations_from_mongodb(self, mock_storage, sample_annotation):
        from src.human_labels.storage import (
            save_annotations_to_mongodb,
            load_annotations_from_mongodb,
        )

        save_annotations_to_mongodb([sample_annotation], mock_storage, "exp_test")
        loaded = load_annotations_from_mongodb("exp_test", mock_storage)
        assert len(loaded) == 1
        assert loaded[0].annotation_id == "ann_test_001"
        assert loaded[0].evaluation_unit_id == "unit_001"

    def test_get_completed_annotation_unit_ids(self, mock_storage, sample_annotation):
        from src.human_labels.storage import (
            save_annotations_to_mongodb,
            get_completed_annotation_unit_ids,
        )

        save_annotations_to_mongodb([sample_annotation], mock_storage, "exp_test")
        completed = get_completed_annotation_unit_ids("exp_test", mock_storage)
        assert "unit_001" in completed

    def test_get_completed_annotation_unit_ids_with_annotator_filter(
        self, mock_storage, sample_annotation
    ):
        from src.human_labels.storage import (
            save_annotations_to_mongodb,
            get_completed_annotation_unit_ids,
        )

        save_annotations_to_mongodb([sample_annotation], mock_storage, "exp_test")
        # Filter by correct annotator
        completed = get_completed_annotation_unit_ids(
            "exp_test", mock_storage, annotator_id="researcher"
        )
        assert "unit_001" in completed
        # Filter by wrong annotator
        completed_other = get_completed_annotation_unit_ids(
            "exp_test", mock_storage, annotator_id="other_annotator"
        )
        assert "unit_001" not in completed_other

    def test_save_aggregated_labels_to_mongodb(
        self, mock_storage, sample_aggregated_label
    ):
        from src.human_labels.storage import save_aggregated_labels_to_mongodb

        saved = save_aggregated_labels_to_mongodb(
            [sample_aggregated_label], mock_storage, "exp_test"
        )
        assert saved == 1
        assert len(mock_storage.db["aggregated_human_labels"].data) == 1

    def test_load_aggregated_labels_from_mongodb(
        self, mock_storage, sample_aggregated_label
    ):
        from src.human_labels.storage import (
            save_aggregated_labels_to_mongodb,
            load_aggregated_labels_from_mongodb,
        )

        save_aggregated_labels_to_mongodb(
            [sample_aggregated_label], mock_storage, "exp_test"
        )
        loaded = load_aggregated_labels_from_mongodb("exp_test", mock_storage)
        assert len(loaded) == 1
        assert loaded[0].evaluation_unit_id == "unit_001"
        assert loaded[0].num_annotators == 2

    def test_upsert_prevents_duplicates(self, mock_storage, sample_annotation):
        from src.human_labels.storage import save_annotations_to_mongodb

        # Save twice
        save_annotations_to_mongodb([sample_annotation], mock_storage, "exp_test")
        save_annotations_to_mongodb([sample_annotation], mock_storage, "exp_test")

        # Should still only have 1 record (upserted, not duplicated)
        assert len(mock_storage.db["human_labels"].data) == 1
