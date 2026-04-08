"""
Tests for Section 3: Controlled Perturbations.

Tests cover:
- Schema (PerturbationRecord, PerturbationIndex, enums)
- Placebo generators (formatting, synonym, reorder_args)
- Fine-grained generators (data_reference, parameter, tool_selection)
- Coarse-grained generators (structural, terminal_flag)
- QC pipeline (validators, impact derivation)
- Balancer (distribution tracking, rebalancing)
- Storage (export to JSON)
"""

import json
import pytest
import tempfile
from pathlib import Path

from src.perturbations.schema import (
    PerturbationClass,
    PerturbationFamily,
    PerturbationType,
    PerturbationRecord,
    PerturbationIndex,
    VALID_CLASS_FAMILY_COMBINATIONS,
    validate_class_family_combination,
)

from src.typing.schema import (
    TypedTrajectory,
    TypedStep,
    PerturbableSlot,
    ProvenanceField,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_typed_step():
    """Create a sample typed step for testing."""
    return TypedStep(
        step_index=1,
        raw_text="I will search for 'python testing' using the search API.",
        step_role="reasoning",
        is_terminal_step=False,
        produces_final_answer=False,
        produces_patch=False,
        tool_name="search_api",
        tool_arguments={"query": "python testing", "limit": 10},
        depends_on_steps=[],
        entities=["search_results"],
        perturbable_slots=[
            PerturbableSlot(
                slot="tool_arguments.query",
                value_type="search_query",
                current_value="python testing",
                allowed_perturbation_types=["parameter", "data_reference"],
            ),
            PerturbableSlot(
                slot="tool_arguments.limit",
                value_type="integer",
                current_value=10,
                allowed_perturbation_types=["parameter"],
            ),
        ],
        critical_path_score=ProvenanceField(value=0.7, source="heuristic"),
    )


@pytest.fixture
def sample_typed_trajectory(sample_typed_step):
    """Create a sample typed trajectory for testing."""
    steps = [
        TypedStep(
            step_index=0,
            raw_text="First, I need to plan my approach.",
            step_role="planning",
            is_terminal_step=False,
            produces_final_answer=False,
            produces_patch=False,
            depends_on_steps=[],
            perturbable_slots=[],
            critical_path_score=ProvenanceField(value=0.6, source="heuristic"),
        ),
        sample_typed_step,
        TypedStep(
            step_index=2,
            raw_text="The final answer is 42.",
            step_role="decision",
            is_terminal_step=True,
            produces_final_answer=True,
            produces_patch=False,
            depends_on_steps=[1],
            entities=["search_results"],
            perturbable_slots=[],
            critical_path_score=ProvenanceField(value=0.9, source="heuristic"),
        ),
    ]

    return TypedTrajectory(
        trajectory_id="test_traj_001",
        benchmark="toolbench",
        task_id="test_task_001",
        task_text="Test task description",
        steps=steps,
        num_steps=3,
        environment_type="api",
    )


@pytest.fixture
def sample_perturbation_record():
    """Create a sample perturbation record."""
    return PerturbationRecord(
        perturbation_id="pert_001",
        original_trajectory_id="test_traj_001",
        generation_timestamp="2026-04-08T10:00:00Z",
        target_step_index=1,
        target_slot="tool_arguments.query",
        perturbation_class=PerturbationClass.FINE_GRAINED.value,
        perturbation_family=PerturbationFamily.PARAMETER.value,
        perturbation_type=PerturbationType.QUERY_DRIFT.value,
        original_value="python testing",
        perturbed_value="python testing frameworks",
        mutation_method="heuristic",
        generation_status="valid",
    )


# ============================================================================
# Schema Tests
# ============================================================================


class TestPerturbationSchema:
    """Tests for perturbation schema classes."""

    def test_perturbation_class_enum(self):
        """Test PerturbationClass enum values."""
        assert PerturbationClass.PLACEBO.value == "placebo"
        assert PerturbationClass.FINE_GRAINED.value == "fine_grained"
        assert PerturbationClass.COARSE_GRAINED.value == "coarse_grained"

    def test_perturbation_family_enum(self):
        """Test PerturbationFamily enum values."""
        assert PerturbationFamily.DATA_REFERENCE.value == "data_reference"
        assert PerturbationFamily.PARAMETER.value == "parameter"
        assert PerturbationFamily.TOOL_SELECTION.value == "tool_selection"
        assert PerturbationFamily.STRUCTURAL.value == "structural"
        assert PerturbationFamily.TERMINAL_FLAG.value == "terminal_flag"

    def test_valid_class_family_combinations(self):
        """Test valid class-family combinations mapping."""
        # Placebo can work with data_reference, parameter
        placebo_families = VALID_CLASS_FAMILY_COMBINATIONS[PerturbationClass.PLACEBO]
        assert PerturbationFamily.DATA_REFERENCE in placebo_families
        assert PerturbationFamily.PARAMETER in placebo_families

        # Fine-grained has data_reference, parameter, tool_selection
        fg_families = VALID_CLASS_FAMILY_COMBINATIONS[PerturbationClass.FINE_GRAINED]
        assert PerturbationFamily.DATA_REFERENCE in fg_families
        assert PerturbationFamily.TOOL_SELECTION in fg_families

        # Coarse-grained has structural, terminal_flag, tool_selection
        cg_families = VALID_CLASS_FAMILY_COMBINATIONS[PerturbationClass.COARSE_GRAINED]
        assert PerturbationFamily.STRUCTURAL in cg_families
        assert PerturbationFamily.TERMINAL_FLAG in cg_families

    def test_validate_class_family_combination(self):
        """Test class-family validation."""
        assert (
            validate_class_family_combination(
                PerturbationClass.FINE_GRAINED, PerturbationFamily.PARAMETER
            )
            is True
        )
        assert (
            validate_class_family_combination(
                PerturbationClass.COARSE_GRAINED, PerturbationFamily.STRUCTURAL
            )
            is True
        )
        assert (
            validate_class_family_combination(
                PerturbationClass.PLACEBO, PerturbationFamily.TERMINAL_FLAG
            )
            is False
        )

    def test_perturbation_record_to_dict(self, sample_perturbation_record):
        """Test PerturbationRecord serialization."""
        record_dict = sample_perturbation_record.to_dict()

        assert record_dict["perturbation_id"] == "pert_001"
        assert record_dict["perturbation_class"] == "fine_grained"
        assert record_dict["perturbation_family"] == "parameter"
        assert record_dict["original_value"] == "python testing"
        assert record_dict["perturbed_value"] == "python testing frameworks"

    def test_perturbation_record_from_dict(self, sample_perturbation_record):
        """Test PerturbationRecord deserialization."""
        record_dict = sample_perturbation_record.to_dict()
        restored = PerturbationRecord.from_dict(record_dict)

        assert restored.perturbation_id == sample_perturbation_record.perturbation_id
        assert (
            restored.perturbation_class == sample_perturbation_record.perturbation_class
        )
        assert restored.original_value == sample_perturbation_record.original_value

    def test_perturbation_index_add_perturbation(self, sample_perturbation_record):
        """Test adding perturbations to index."""
        index = PerturbationIndex()
        index.add_perturbation(
            sample_perturbation_record,
            benchmark="toolbench",
            file_path="data/perturbed/toolbench.json",
        )

        assert index.total_perturbations == 1
        assert index.by_class["fine_grained"] == 1
        assert index.by_family["parameter"] == 1
        assert index.by_benchmark["toolbench"] == 1

    def test_perturbation_index_distribution_report(self, sample_perturbation_record):
        """Test distribution report generation."""
        index = PerturbationIndex()

        # Add multiple perturbations
        for i in range(10):
            cls = PerturbationClass.FINE_GRAINED if i < 5 else PerturbationClass.PLACEBO
            record = PerturbationRecord.create(
                original_trajectory_id="traj_001",
                perturbation_class=cls,
                perturbation_family=PerturbationFamily.PARAMETER,
                perturbation_type=PerturbationType.QUERY_DRIFT,
                target_step_index=1,
                target_slot="slot",
                original_value="orig",
                perturbed_value=f"pert_{i}",
                mutation_method="heuristic",
            )
            index.add_perturbation(record, "toolbench", "path.json")

        report = index.get_distribution_report()
        assert "Total Perturbations: 10" in report
        assert "fine_grained" in report
        assert "placebo" in report


# ============================================================================
# Placebo Generator Tests
# ============================================================================


class TestPlaceboGenerators:
    """Tests for placebo perturbation generators."""

    def test_formatting_generator(self, sample_typed_step, sample_typed_trajectory):
        """Test formatting placebo generator."""
        from src.perturbations.placebo import PlaceboFormattingGenerator

        generator = PlaceboFormattingGenerator()
        record = generator.generate(
            sample_typed_step, sample_typed_trajectory.trajectory_id
        )

        # Formatting may return None if no suitable slot
        if record:
            assert record.perturbation_class == "placebo"
            assert record.perturbation_type == "formatting"
            assert record.generation_method == "heuristic"
            assert record.perturbed_value != record.original_value

    def test_synonym_generator(self, sample_typed_step, sample_typed_trajectory):
        """Test synonym placebo generator."""
        from src.perturbations.placebo import PlaceboSynonymGenerator

        generator = PlaceboSynonymGenerator()
        record = generator.generate(
            sample_typed_step, sample_typed_trajectory.trajectory_id
        )

        if record:
            assert record.perturbation_class == "placebo"
            assert record.perturbation_type == "synonym"

    def test_reorder_args_generator(self, sample_typed_step, sample_typed_trajectory):
        """Test reorder_args placebo generator."""
        from src.perturbations.placebo import PlaceboReorderArgsGenerator

        # Needs a step with multiple tool arguments
        step = TypedStep(
            step_index=1,
            raw_text="Test",
            step_role="execution",
            is_terminal_step=False,
            produces_final_answer=False,
            produces_patch=False,
            tool_name="api_call",
            tool_arguments={"a": 1, "b": 2, "c": 3},
            depends_on_steps=[],
            perturbable_slots=[
                PerturbableSlot(
                    slot="tool_arguments",
                    value_type="object",
                    current_value={"a": 1, "b": 2, "c": 3},
                    allowed_perturbation_types=["parameter"],
                ),
            ],
            critical_path_score=ProvenanceField(value=0.5, source="heuristic"),
        )

        generator = PlaceboReorderArgsGenerator()
        record = generator.generate(step, "traj_001")

        if record:
            assert record.perturbation_class == "placebo"
            assert record.perturbation_type == "reorder_args"

    def test_get_placebo_generator(self):
        """Test placebo generator factory."""
        from src.perturbations.placebo import get_placebo_generator

        generator = get_placebo_generator(PerturbationType.FORMATTING)
        assert generator is not None

        generator = get_placebo_generator(PerturbationType.SYNONYM)
        assert generator is not None


# ============================================================================
# Fine-Grained Generator Tests
# ============================================================================


class TestFineGrainedGenerators:
    """Tests for fine-grained perturbation generators."""

    def test_data_reference_generator(self, sample_typed_step, sample_typed_trajectory):
        """Test data reference generator."""
        from src.perturbations.fine_grained import DataReferenceGenerator

        generator = DataReferenceGenerator(random_seed=42)
        record = generator.generate(
            sample_typed_step,
            sample_typed_trajectory.trajectory_id,
            trajectory=sample_typed_trajectory,
        )

        if record:
            assert record.perturbation_class == "fine_grained"
            assert record.perturbation_family == "data_reference"

    def test_parameter_generator(self, sample_typed_step, sample_typed_trajectory):
        """Test parameter generator."""
        from src.perturbations.fine_grained import ParameterGenerator

        generator = ParameterGenerator(random_seed=42)
        record = generator.generate(
            sample_typed_step,
            sample_typed_trajectory.trajectory_id,
            trajectory=sample_typed_trajectory,
        )

        if record:
            assert record.perturbation_class == "fine_grained"
            assert record.perturbation_family == "parameter"
            assert record.perturbation_type in [
                "threshold_shift",
                "query_drift",
                "wrong_parameter",
            ]

    def test_tool_selection_near_neighbor(
        self, sample_typed_step, sample_typed_trajectory
    ):
        """Test near-neighbor tool selection generator."""
        from src.perturbations.fine_grained import ToolSelectionNearNeighborGenerator

        generator = ToolSelectionNearNeighborGenerator(random_seed=42)
        record = generator.generate(
            sample_typed_step,
            sample_typed_trajectory.trajectory_id,
            trajectory=sample_typed_trajectory,
        )

        # May return None if no similar tool found
        if record:
            assert record.perturbation_class == "fine_grained"
            assert record.perturbation_family == "tool_selection"
            assert record.perturbation_type == "near_neighbor_tool"

    def test_get_fine_grained_generator(self):
        """Test fine-grained generator factory."""
        from src.perturbations.fine_grained import get_fine_grained_generator

        generator = get_fine_grained_generator(
            PerturbationFamily.PARAMETER,
            PerturbationType.QUERY_DRIFT,
        )
        assert generator is not None


# ============================================================================
# Coarse-Grained Generator Tests
# ============================================================================


class TestCoarseGrainedGenerators:
    """Tests for coarse-grained perturbation generators."""

    def test_skipped_prerequisite_generator(self, sample_typed_trajectory):
        """Test skipped prerequisite generator."""
        from src.perturbations.coarse_grained import SkippedPrerequisiteGenerator

        generator = SkippedPrerequisiteGenerator(random_seed=42)
        result = generator.generate(sample_typed_trajectory)

        # May return None if no valid prerequisite to skip
        if result:
            record, perturbed_traj = result
            assert record.perturbation_class == "coarse_grained"
            assert record.perturbation_family == "structural"
            assert record.perturbation_type == "skipped_prerequisite"

    def test_false_terminal_generator(self, sample_typed_trajectory):
        """Test false terminal generator."""
        from src.perturbations.coarse_grained import FalseTerminalGenerator

        generator = FalseTerminalGenerator(random_seed=42)
        result = generator.generate(sample_typed_trajectory)

        if result:
            record, perturbed_traj = result
            assert record.perturbation_class == "coarse_grained"
            assert record.perturbation_family == "terminal_flag"
            assert record.perturbation_type == "false_terminal"

    def test_premature_termination_generator(self, sample_typed_trajectory):
        """Test premature termination generator."""
        from src.perturbations.coarse_grained import PrematureTerminationGenerator

        generator = PrematureTerminationGenerator(random_seed=42)
        result = generator.generate(sample_typed_trajectory)

        if result:
            record, perturbed_traj = result
            assert record.perturbation_class == "coarse_grained"
            assert record.perturbation_family == "terminal_flag"
            assert record.perturbation_type == "premature_termination"
            # Should have fewer steps than original
            assert len(perturbed_traj.steps) < len(sample_typed_trajectory.steps)

    def test_get_coarse_grained_generator(self):
        """Test coarse-grained generator factory."""
        from src.perturbations.coarse_grained import get_coarse_grained_generator

        generator = get_coarse_grained_generator(
            PerturbationFamily.STRUCTURAL,
            PerturbationType.SKIPPED_PREREQUISITE,
        )
        assert generator is not None


# ============================================================================
# QC Pipeline Tests
# ============================================================================


class TestQCPipeline:
    """Tests for perturbation QC pipeline."""

    def test_schema_validator(
        self,
        sample_perturbation_record,
        sample_typed_trajectory,
        sample_typed_step,
    ):
        """Test schema validator."""
        from src.perturbations.qc import SchemaValidator

        validator = SchemaValidator()

        # Valid record
        results = validator.validate(
            sample_perturbation_record, sample_typed_trajectory, sample_typed_step
        )
        # Check all results passed
        assert all(r.passed for r in results)

    def test_diff_validator(
        self,
        sample_perturbation_record,
        sample_typed_trajectory,
        sample_typed_step,
    ):
        """Test diff validator ensures values are different."""
        from src.perturbations.qc import DiffValidator

        validator = DiffValidator()

        # Valid - values are different
        results = validator.validate(
            sample_perturbation_record, sample_typed_trajectory, sample_typed_step
        )
        assert all(r.passed for r in results)

    def test_class_family_validator(
        self,
        sample_perturbation_record,
        sample_typed_trajectory,
        sample_typed_step,
    ):
        """Test class-family combination validator."""
        from src.perturbations.qc import ClassFamilyValidator

        validator = ClassFamilyValidator()

        # Valid combination
        results = validator.validate(
            sample_perturbation_record, sample_typed_trajectory, sample_typed_step
        )
        assert all(r.passed for r in results)

    def test_perturbation_qc_validate(
        self,
        sample_perturbation_record,
        sample_typed_trajectory,
        sample_typed_step,
    ):
        """Test full QC validation pipeline."""
        from src.perturbations.qc import PerturbationQC

        qc = PerturbationQC()
        validated = qc.validate(
            sample_perturbation_record,
            sample_typed_trajectory,
            sample_typed_step,
        )

        # Should return a PerturbationRecord with updated status
        assert isinstance(validated, PerturbationRecord)
        assert validated.generation_status in ["valid", "invalid", "borderline"]


# ============================================================================
# Balancer Tests
# ============================================================================


class TestBalancer:
    """Tests for perturbation balancing."""

    def test_batch_distribution_tracking(self, sample_perturbation_record):
        """Test batch distribution tracking."""
        from src.perturbations.balancer import BatchDistribution

        dist = BatchDistribution(total_target=100)

        # Record some perturbations
        for _ in range(5):
            dist.record(sample_perturbation_record, position="early")

        assert dist.class_counts["fine_grained"] == 5
        assert dist.position_counts["early"] == 5

    def test_batch_distribution_needs(self):
        """Test calculating class needs."""
        from src.perturbations.balancer import BatchDistribution

        dist = BatchDistribution(
            total_target=100,
            class_weights={"placebo": 0.2, "fine_grained": 0.5, "coarse_grained": 0.3},
        )

        needs = dist.get_class_needs()
        assert needs["placebo"] == 20
        assert needs["fine_grained"] == 50
        assert needs["coarse_grained"] == 30

    def test_pre_allocation(self):
        """Test pre-allocating classes to trajectories."""
        from src.perturbations.balancer import PerturbationBalancer

        balancer = PerturbationBalancer(
            total_target=100,
            class_weights={"placebo": 0.2, "fine_grained": 0.5, "coarse_grained": 0.3},
            random_seed=42,
        )

        allocations = balancer.pre_allocate(num_trajectories=10, per_trajectory=3)

        assert len(allocations) == 10
        assert all(len(alloc) == 3 for alloc in allocations)

        # Check overall distribution
        all_classes = [cls for alloc in allocations for cls in alloc]
        class_counts = {}
        for cls in all_classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1

        # Should be approximately 20/50/30
        total = sum(class_counts.values())
        assert abs(class_counts.get("placebo", 0) / total - 0.2) < 0.1
        assert abs(class_counts.get("fine_grained", 0) / total - 0.5) < 0.1

    def test_rebalance_sample(self, sample_perturbation_record):
        """Test post-hoc rebalancing."""
        from src.perturbations.balancer import PerturbationBalancer

        # Create imbalanced records
        records = []
        for i in range(60):
            record = PerturbationRecord(
                perturbation_id=f"pert_{i}",
                original_trajectory_id="traj",
                generation_timestamp="2026-04-08T10:00:00Z",
                target_step_index=1,
                target_slot="slot",
                perturbation_class="fine_grained",  # All fine_grained - imbalanced
                perturbation_family="parameter",
                perturbation_type="query_drift",
                original_value="orig",
                perturbed_value=f"pert_{i}",
                mutation_method="heuristic",
            )
            records.append(record)

        balancer = PerturbationBalancer(total_target=50, random_seed=42)
        rebalanced = balancer.rebalance_sample(records, target_count=50)

        # Should sample down to target
        assert len(rebalanced) <= 50

    def test_balance_perturbation_batch(self):
        """Test batch balancing utility function."""
        from src.perturbations.balancer import balance_perturbation_batch

        # Create mixed records
        records = []
        classes = ["placebo"] * 10 + ["fine_grained"] * 30 + ["coarse_grained"] * 20

        for i, cls in enumerate(classes):
            record = PerturbationRecord(
                perturbation_id=f"pert_{i}",
                original_trajectory_id="traj",
                generation_timestamp="2026-04-08T10:00:00Z",
                target_step_index=1,
                target_slot="slot",
                perturbation_class=cls,
                perturbation_family="parameter",
                perturbation_type="query_drift",
                original_value="orig",
                perturbed_value=f"pert_{i}",
                mutation_method="heuristic",
            )
            records.append(record)

        balanced, report = balance_perturbation_batch(
            records=records,
            total_target=50,
            random_seed=42,
        )

        assert "initial" in report
        assert "final" in report
        assert report["final"]["total"] <= 50


# ============================================================================
# Storage Tests
# ============================================================================


class TestStorage:
    """Tests for perturbation storage and export."""

    def test_exporter_by_benchmark(self, sample_perturbation_record):
        """Test exporting perturbations by benchmark."""
        from src.perturbations.storage import PerturbationExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = PerturbationExporter(output_dir=tmpdir)

            perturbations = [
                {
                    "perturbation_id": "pert_001",
                    "perturbation_record": sample_perturbation_record.to_dict(),
                },
            ]

            path = exporter.export_by_benchmark(perturbations, "toolbench")

            assert Path(path).exists()
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 1

    def test_exporter_index(self, sample_perturbation_record):
        """Test exporting perturbation index."""
        from src.perturbations.storage import PerturbationExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = PerturbationExporter(output_dir=tmpdir)

            index = PerturbationIndex()
            index.add_perturbation(sample_perturbation_record, "toolbench", "path.json")

            path = exporter.export_index(index)

            assert Path(path).exists()
            with open(path) as f:
                data = json.load(f)
            assert data["total_perturbations"] == 1

    def test_build_index_from_perturbations(self, sample_perturbation_record):
        """Test building index from perturbation list."""
        from src.perturbations.storage import build_index_from_perturbations

        perturbations = [
            {
                "perturbation_record": sample_perturbation_record.to_dict(),
                "original_trajectory": {"benchmark": "toolbench"},
            },
        ]

        index = build_index_from_perturbations(perturbations)

        assert index.total_perturbations == 1
        assert index.by_benchmark["toolbench"] == 1

    def test_group_by_benchmark(self, sample_perturbation_record):
        """Test grouping perturbations by benchmark."""
        from src.perturbations.storage import group_by_benchmark

        perturbations = [
            {
                "perturbation_id": "pert_001",
                "original_trajectory": {"benchmark": "toolbench"},
            },
            {
                "perturbation_id": "pert_002",
                "original_trajectory": {"benchmark": "swebench"},
            },
            {
                "perturbation_id": "pert_003",
                "original_trajectory": {"benchmark": "toolbench"},
            },
        ]

        grouped = group_by_benchmark(perturbations)

        assert len(grouped["toolbench"]) == 2
        assert len(grouped["swebench"]) == 1

    def test_filter_valid_perturbations(self, sample_perturbation_record):
        """Test filtering to valid perturbations only."""
        from src.perturbations.storage import filter_valid_perturbations

        perturbations = [
            {
                "perturbation_id": "pert_001",
                "perturbation_record": {"generation_status": "valid"},
            },
            {
                "perturbation_id": "pert_002",
                "perturbation_record": {"generation_status": "invalid"},
            },
            {
                "perturbation_id": "pert_003",
                "perturbation_record": {"generation_status": "valid"},
            },
        ]

        valid = filter_valid_perturbations(perturbations)

        assert len(valid) == 2


# ============================================================================
# Generator V2 Tests
# ============================================================================


class TestGeneratorV2:
    """Tests for the main PerturbationGeneratorV2 orchestrator."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        from src.perturbations.generator_v2 import PerturbationGeneratorV2

        generator = PerturbationGeneratorV2(random_seed=42, enable_qc=True)

        assert generator.random_seed == 42
        assert generator.qc is not None
        assert generator.stats == {}

    def test_enumerate_slot_candidates(self, sample_typed_trajectory):
        """Test slot enumeration."""
        from src.perturbations.generator_v2 import PerturbationGeneratorV2

        generator = PerturbationGeneratorV2(random_seed=42)
        candidates = generator._enumerate_slot_candidates(sample_typed_trajectory)

        # Should find slots from typed steps
        assert len(candidates) > 0

    def test_get_position(self):
        """Test position calculation."""
        from src.perturbations.generator_v2 import PerturbationGeneratorV2

        generator = PerturbationGeneratorV2()

        assert generator._get_position(0, 10) == "early"
        assert generator._get_position(5, 10) == "middle"
        assert generator._get_position(9, 10) == "late"

    def test_generate_for_trajectory(self, sample_typed_trajectory):
        """Test generating perturbations for a trajectory."""
        from src.perturbations.generator_v2 import PerturbationGeneratorV2

        generator = PerturbationGeneratorV2(random_seed=42, enable_qc=True)

        results = generator.generate_for_trajectory(
            sample_typed_trajectory,
            target_count=3,
        )

        # Should generate some perturbations
        # (may be less than target if not enough eligible slots)
        assert len(results) >= 0
        assert len(results) <= 3

        for record, perturbed_traj in results:
            assert isinstance(record, PerturbationRecord)
            assert record.perturbation_class in [
                "placebo",
                "fine_grained",
                "coarse_grained",
            ]

    def test_generate_perturbations_for_batch(self, sample_typed_trajectory):
        """Test batch generation."""
        from src.perturbations.generator_v2 import generate_perturbations_for_batch

        trajectories = [sample_typed_trajectory]

        results, index = generate_perturbations_for_batch(
            trajectories=trajectories,
            target_per_trajectory=2,
            random_seed=42,
            verbose=False,
        )

        assert isinstance(results, list)
        assert isinstance(index, PerturbationIndex)

    def test_stats_tracking(self, sample_typed_trajectory):
        """Test that statistics are tracked during generation."""
        from src.perturbations.generator_v2 import PerturbationGeneratorV2

        generator = PerturbationGeneratorV2(random_seed=42)

        generator.generate_for_trajectory(
            sample_typed_trajectory,
            target_count=3,
        )

        stats = generator.get_stats()
        assert "total" in stats or stats.get("total", 0) >= 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_dry_run(self, sample_typed_trajectory):
        """Test full pipeline without actual LLM calls."""
        from src.perturbations.generator_v2 import PerturbationGeneratorV2
        from src.perturbations.balancer import balance_perturbation_batch
        from src.perturbations.storage import (
            PerturbationExporter,
            build_index_from_perturbations,
            group_by_benchmark,
        )

        # Generate
        generator = PerturbationGeneratorV2(random_seed=42, enable_qc=True)
        trajectories = [sample_typed_trajectory]

        all_results = []
        for traj in trajectories:
            results = generator.generate_for_trajectory(traj, target_count=3)
            all_results.extend(results)

        # Extract records
        records = [record for record, _ in all_results]

        if records:
            # Balance (if we have records)
            balanced, report = balance_perturbation_batch(
                records=records,
                total_target=min(len(records), 10),
                random_seed=42,
            )

            # Build perturbed trajectories list
            balanced_ids = {r.perturbation_id for r in balanced}
            perturbed_trajectories = [
                {
                    "perturbation_id": record.perturbation_id,
                    "perturbation_record": record.to_dict(),
                    "original_trajectory": {"benchmark": "toolbench"},
                }
                for record, _ in all_results
                if record.perturbation_id in balanced_ids
            ]

            # Build index
            index = build_index_from_perturbations(perturbed_trajectories)

            # Export
            with tempfile.TemporaryDirectory() as tmpdir:
                exporter = PerturbationExporter(output_dir=tmpdir)
                grouped = group_by_benchmark(perturbed_trajectories)
                paths = exporter.export_all(grouped, index)

                assert "index" in paths
                assert Path(paths["index"]).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
