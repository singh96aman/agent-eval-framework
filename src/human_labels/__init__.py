"""
Human Labels module for Section 5A.

Provides ground-truth human annotations for evaluating LLM judge performance.
Trained annotators assess detectability, localization, error type, impact,
and propagation across a stratified sample of evaluation units.

Usage:
    from src.human_labels import (
        AnnotationRecord,
        AggregatedLabel,
        sample_for_annotation,
        aggregate_annotations,
        compute_agreement_report,
    )
"""

# Schema classes
from src.human_labels.schema import (
    AnnotationMode,
    AnnotationRecord,
    AggregatedConsequence,
    AggregatedDetectability,
    AggregatedLabel,
    AggregatedPreference,
    ConsequenceLabels,
    Correctness,
    DetectabilityLabels,
    ErrorTrajectory,
    ErrorType,
    LocalizationAccuracy,
    Preference,
    PreferenceLabels,
)

# Sampling functions
from src.human_labels.sampling import (
    assign_to_annotators,
    generate_annotation_batches,
    sample_for_annotation,
)

# Agreement functions
from src.human_labels.agreement import (
    compute_agreement_report,
    compute_krippendorff_alpha,
    compute_localization_accuracy,
    compute_pairwise_agreement,
)

# Aggregation functions
from src.human_labels.aggregation import (
    aggregate_annotations,
    compute_aggregation_summary,
    derive_error_type_match,
    derive_localization_accuracy,
)

# QC functions
from src.human_labels.qc import (
    check_consecutive_same,
    check_gold_set_accuracy,
    check_time_bounds,
    check_time_bounds_batch,
    generate_annotator_quality_report,
    run_all_qc_checks,
)

# Storage functions
from src.human_labels.storage import (
    create_human_labels_directories,
    export_aggregated_labels,
    load_aggregated_labels,
    load_annotations_from_json,
    load_annotator_assignments,
    load_gold_set,
    save_agreement_report,
    save_annotations_to_json,
    save_annotator_assignments,
    save_gold_set,
    save_qc_report,
    # MongoDB storage functions
    get_completed_annotation_unit_ids,
    load_aggregated_labels_from_mongodb,
    load_annotations_from_mongodb,
    save_aggregated_labels_to_mongodb,
    save_annotations_to_mongodb,
)

__all__ = [
    # Schema classes
    "AnnotationMode",
    "AnnotationRecord",
    "AggregatedConsequence",
    "AggregatedDetectability",
    "AggregatedLabel",
    "AggregatedPreference",
    "ConsequenceLabels",
    "Correctness",
    "DetectabilityLabels",
    "ErrorTrajectory",
    "ErrorType",
    "LocalizationAccuracy",
    "Preference",
    "PreferenceLabels",
    # Sampling functions
    "assign_to_annotators",
    "generate_annotation_batches",
    "sample_for_annotation",
    # Agreement functions
    "compute_agreement_report",
    "compute_krippendorff_alpha",
    "compute_localization_accuracy",
    "compute_pairwise_agreement",
    # Aggregation functions
    "aggregate_annotations",
    "compute_aggregation_summary",
    "derive_error_type_match",
    "derive_localization_accuracy",
    # QC functions
    "check_consecutive_same",
    "check_gold_set_accuracy",
    "check_time_bounds",
    "check_time_bounds_batch",
    "generate_annotator_quality_report",
    "run_all_qc_checks",
    # Storage functions
    "create_human_labels_directories",
    "export_aggregated_labels",
    "load_aggregated_labels",
    "load_annotations_from_json",
    "load_annotator_assignments",
    "load_gold_set",
    "save_agreement_report",
    "save_annotations_to_json",
    "save_annotator_assignments",
    "save_gold_set",
    "save_qc_report",
    # MongoDB storage functions
    "get_completed_annotation_unit_ids",
    "load_aggregated_labels_from_mongodb",
    "load_annotations_from_mongodb",
    "save_aggregated_labels_to_mongodb",
    "save_annotations_to_mongodb",
]
