"""
Tests for annotation tools.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.annotation.tools import (
    Annotation,
    AnnotationInterface,
    AnnotationReviewer,
    load_annotation,
    save_annotation,
)


class TestAnnotation:
    """Test Annotation dataclass."""

    def test_create_annotation(self):
        """Test creating an annotation."""
        ann = Annotation(
            perturbation_id="test_pert_1",
            annotator_id="user1",
            task_success_degradation=1,
            subsequent_error_rate=3,
            notes="Test annotation",
        )

        assert ann.perturbation_id == "test_pert_1"
        assert ann.annotator_id == "user1"
        assert ann.task_success_degradation == 1
        assert ann.subsequent_error_rate == 3
        assert ann.notes == "Test annotation"

    def test_compute_tcs(self):
        """Test TCS computation."""
        # Task failed, 3 downstream errors
        ann1 = Annotation(
            perturbation_id="test_1",
            annotator_id="user1",
            task_success_degradation=1,
            subsequent_error_rate=3,
        )
        assert ann1.compute_tcs() == 130.0  # (1 * 100) + (3 * 10)

        # Task succeeded, 2 downstream errors
        ann2 = Annotation(
            perturbation_id="test_2",
            annotator_id="user1",
            task_success_degradation=0,
            subsequent_error_rate=2,
        )
        assert ann2.compute_tcs() == 20.0  # (0 * 100) + (2 * 10)

        # No errors
        ann3 = Annotation(
            perturbation_id="test_3",
            annotator_id="user1",
            task_success_degradation=0,
            subsequent_error_rate=0,
        )
        assert ann3.compute_tcs() == 0.0

    def test_to_dict(self):
        """Test converting annotation to dict."""
        ann = Annotation(
            perturbation_id="test_1",
            annotator_id="user1",
            task_success_degradation=1,
            subsequent_error_rate=3,
        )

        data = ann.to_dict()

        assert data["perturbation_id"] == "test_1"
        assert data["annotator_id"] == "user1"
        assert data["task_success_degradation"] == 1
        assert data["subsequent_error_rate"] == 3
        assert "timestamp" in data

    def test_from_dict(self):
        """Test creating annotation from dict."""
        data = {
            "perturbation_id": "test_1",
            "annotator_id": "user1",
            "task_success_degradation": 1,
            "subsequent_error_rate": 3,
            "notes": "Test",
            "timestamp": datetime.utcnow().isoformat(),
            "annotation_time_seconds": 120.5,
        }

        ann = Annotation.from_dict(data)

        assert ann.perturbation_id == "test_1"
        assert ann.annotator_id == "user1"
        assert ann.task_success_degradation == 1


class TestAnnotationInterface:
    """Test AnnotationInterface."""

    @patch("src.annotation.tools.MongoDBStorage")
    def test_init(self, mock_storage):
        """Test initializing interface."""
        interface = AnnotationInterface(storage=mock_storage)
        assert interface.storage == mock_storage
        assert interface.annotations_dir == Path("data/annotations")

    @patch("src.annotation.tools.MongoDBStorage")
    def test_load_perturbation(self, mock_storage):
        """Test loading perturbation from MongoDB."""
        mock_db = MagicMock()
        mock_storage.db = mock_db

        # Mock perturbation record with all required fields
        mock_db["perturbations"].find_one.return_value = {
            "perturbation_id": "test_1",
            "original_trajectory_id": "orig_traj_1",
            "perturbed_trajectory_id": "pert_traj_1",
            "perturbation_type": "planning",
            "perturbation_position": "early",
            "perturbed_step_number": 1,
            "original_step_content": "original content",
            "perturbed_step_content": "perturbed content",
            "perturbation_metadata": {},
        }

        # Mock trajectory retrieval
        mock_orig_traj = {"trajectory_id": "orig_traj_1", "steps": []}
        mock_pert_traj = {"trajectory_id": "pert_traj_1", "steps": []}
        mock_storage.get_trajectory.side_effect = [mock_orig_traj, mock_pert_traj]

        interface = AnnotationInterface(storage=mock_storage)
        result = interface.load_perturbation("test_1")

        assert result is not None
        assert result["_perturbation_id"] == "test_1"
        assert result["perturbation_type"] == "planning"
        assert result["original_trajectory"] == mock_orig_traj
        mock_db["perturbations"].find_one.assert_called_once_with(
            {"perturbation_id": "test_1"}
        )


class TestAnnotationReviewer:
    """Test AnnotationReviewer."""

    def test_init(self, tmp_path):
        """Test initializing reviewer."""
        reviewer = AnnotationReviewer(annotations_dir=tmp_path)
        assert reviewer.annotations_dir == tmp_path

    def test_load_all_annotations_empty(self, tmp_path):
        """Test loading annotations from empty directory."""
        reviewer = AnnotationReviewer(annotations_dir=tmp_path)
        annotations = reviewer.load_all_annotations()
        assert annotations == []

    def test_load_all_annotations(self, tmp_path):
        """Test loading multiple annotations."""
        # Create test annotations
        ann1 = Annotation(
            perturbation_id="test_1",
            annotator_id="user1",
            task_success_degradation=1,
            subsequent_error_rate=3,
        )
        ann2 = Annotation(
            perturbation_id="test_2",
            annotator_id="user1",
            task_success_degradation=0,
            subsequent_error_rate=1,
        )

        # Save to temp directory
        with open(tmp_path / "test_1.json", "w") as f:
            json.dump(ann1.to_dict(), f)
        with open(tmp_path / "test_2.json", "w") as f:
            json.dump(ann2.to_dict(), f)

        # Load
        reviewer = AnnotationReviewer(annotations_dir=tmp_path)
        annotations = reviewer.load_all_annotations()

        assert len(annotations) == 2
        assert annotations[0].perturbation_id in ["test_1", "test_2"]

    def test_get_summary_stats_empty(self, tmp_path):
        """Test summary stats with no annotations."""
        reviewer = AnnotationReviewer(annotations_dir=tmp_path)
        stats = reviewer.get_summary_stats()
        assert stats == {"total": 0}

    def test_get_summary_stats(self, tmp_path):
        """Test summary statistics computation."""
        # Create test annotations
        annotations = [
            Annotation(
                perturbation_id=f"test_{i}",
                annotator_id="user1",
                task_success_degradation=1 if i % 2 == 0 else 0,
                subsequent_error_rate=i,
            )
            for i in range(5)
        ]

        # Save to temp directory
        for ann in annotations:
            with open(tmp_path / f"{ann.perturbation_id}.json", "w") as f:
                json.dump(ann.to_dict(), f)

        # Get stats
        reviewer = AnnotationReviewer(annotations_dir=tmp_path)
        stats = reviewer.get_summary_stats()

        assert stats["total"] == 5
        assert "tsd_mean" in stats
        assert "ser_mean" in stats
        assert "tcs_mean" in stats
        assert stats["tsd_task_failures"] == 3  # indices 0, 2, 4


class TestAnnotationFileOperations:
    """Test file load/save operations."""

    def test_save_and_load_annotation(self, tmp_path):
        """Test saving and loading annotation."""
        ann = Annotation(
            perturbation_id="test_1",
            annotator_id="user1",
            task_success_degradation=1,
            subsequent_error_rate=3,
            notes="Test annotation",
        )

        # Save
        success = save_annotation(ann, tmp_path)
        assert success

        # Check file exists
        filepath = tmp_path / "test_1.json"
        assert filepath.exists()

        # Load
        loaded = load_annotation("test_1", tmp_path)
        assert loaded is not None
        assert loaded.perturbation_id == "test_1"
        assert loaded.task_success_degradation == 1
        assert loaded.subsequent_error_rate == 3

    def test_load_nonexistent_annotation(self, tmp_path):
        """Test loading non-existent annotation."""
        loaded = load_annotation("nonexistent", tmp_path)
        assert loaded is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
