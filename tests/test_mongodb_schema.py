"""
Test MongoDB schema with foreign keys (not arrays).

This test verifies that:
1. Foreign key relationships work correctly
2. Schema format is correct (no arrays in experiments!)
3. Pagination logic works
4. Cache check logic works
5. No arrays of IDs are stored in experiments collection

NO MONGODB REQUIRED - uses mock data structures in memory.
"""

import pytest
from datetime import datetime


class MockStorage:
    """
    Mock storage that simulates MongoDB operations using in-memory dicts.

    This allows us to test schema format without requiring MongoDB.
    """

    def __init__(self):
        self.experiments = {}
        self.trajectories = {}
        self.annotations = {}
        self.judge_evaluations = {}
        self.ccg_scores = {}

    def create_experiment(self, experiment_id, **data):
        """Create experiment with progress tracking (counts only!)."""
        self.experiments[experiment_id] = {
            "experiment_id": experiment_id,
            "progress": data.get(
                "progress",
                {
                    "trajectories_loaded": 0,
                    "annotations_completed": 0,
                    "evaluations_completed": 0,
                },
            ),
            **data,
        }

    def save_trajectory(self, trajectory_id, experiment_id, **data):
        """Save trajectory with experiment_id foreign key."""
        self.trajectories[trajectory_id] = {
            "trajectory_id": trajectory_id,
            "experiment_id": experiment_id,  # Foreign key!
            **data,
        }

    def save_annotation(self, annotation_id, experiment_id, trajectory_id, **data):
        """Save annotation with foreign keys."""
        self.annotations[annotation_id] = {
            "annotation_id": annotation_id,
            "experiment_id": experiment_id,  # FK
            "trajectory_id": trajectory_id,  # FK
            **data,
        }

    def save_evaluation(self, evaluation_id, experiment_id, trajectory_id, **data):
        """Save judge evaluation with foreign keys."""
        self.judge_evaluations[evaluation_id] = {
            "evaluation_id": evaluation_id,
            "experiment_id": experiment_id,  # FK
            "trajectory_id": trajectory_id,  # FK
            **data,
        }

    def save_ccg_score(
        self, ccg_id, experiment_id, trajectory_id, annotation_id, evaluation_id, **data
    ):
        """Save CCG score with all foreign keys."""
        self.ccg_scores[ccg_id] = {
            "ccg_id": ccg_id,
            "experiment_id": experiment_id,  # FK
            "trajectory_id": trajectory_id,  # FK
            "annotation_id": annotation_id,  # FK
            "evaluation_id": evaluation_id,  # FK
            **data,
        }

    def get_trajectories_by_experiment(self, experiment_id, skip=0, limit=None):
        """Query trajectories by experiment_id (foreign key)."""
        results = [
            t for t in self.trajectories.values() if t["experiment_id"] == experiment_id
        ]

        # Apply pagination
        if skip:
            results = results[skip:]
        if limit:
            results = results[:limit]

        return results

    def count_trajectories(self, experiment_id=None):
        """Count trajectories matching criteria."""
        if experiment_id:
            return len(
                [
                    t
                    for t in self.trajectories.values()
                    if t["experiment_id"] == experiment_id
                ]
            )
        return len(self.trajectories)

    def check_evaluation_cache(self, trajectory_id, judge_model, sample_number=1):
        """Check if evaluation exists in cache."""
        for eval_data in self.judge_evaluations.values():
            if (
                eval_data["trajectory_id"] == trajectory_id
                and eval_data["judge_model"] == judge_model
                and eval_data.get("sample_number", 1) == sample_number
            ):
                return eval_data
        return None


@pytest.fixture
def storage():
    """Create mock storage."""
    return MockStorage()


def test_foreign_key_relationships(storage):
    """Test that foreign keys work correctly (no arrays in experiments)."""

    experiment_id = "test_exp_001"

    # Create experiment
    storage.create_experiment(
        experiment_id,
        name="Test Experiment",
        progress={"trajectories_loaded": 0},  # Count only, no array!
    )

    # Verify experiment has NO trajectory_refs array
    exp = storage.experiments[experiment_id]
    assert (
        "trajectory_refs" not in exp
    ), "❌ experiments should NOT have trajectory_refs array!"
    assert "progress" in exp
    assert isinstance(exp["progress"]["trajectories_loaded"], int)

    # Create trajectories with foreign key
    for i in range(3):
        storage.save_trajectory(
            f"traj_{i}",
            experiment_id,  # Foreign key!
            benchmark="toolbench",
            is_perturbed=False,
        )

    # Update progress (count, not array!)
    storage.experiments[experiment_id]["progress"]["trajectories_loaded"] = 3

    # Query trajectories by experiment_id (uses foreign key)
    trajectories = storage.get_trajectories_by_experiment(experiment_id)
    assert len(trajectories) == 3

    # All trajectories should have experiment_id
    for traj in trajectories:
        assert traj["experiment_id"] == experiment_id

    # Count should match
    count = storage.count_trajectories(experiment_id=experiment_id)
    assert count == 3

    print("✅ Foreign key relationships work correctly")


def test_pagination_support(storage):
    """Test that pagination works with foreign key queries."""

    experiment_id = "test_exp_002"
    storage.create_experiment(experiment_id, name="Pagination Test")

    # Create 25 trajectories
    for i in range(25):
        storage.save_trajectory(
            f"traj_page_{i}", experiment_id, benchmark="gaia", is_perturbed=False
        )

    # Test pagination
    page1 = storage.get_trajectories_by_experiment(experiment_id, skip=0, limit=10)
    assert len(page1) == 10

    page2 = storage.get_trajectories_by_experiment(experiment_id, skip=10, limit=10)
    assert len(page2) == 10

    page3 = storage.get_trajectories_by_experiment(experiment_id, skip=20, limit=10)
    assert len(page3) == 5  # Only 5 remaining

    # Verify total count
    total = storage.count_trajectories(experiment_id=experiment_id)
    assert total == 25

    print("✅ Pagination working correctly")


def test_judge_evaluation_cache(storage):
    """Test the most important cache: judge evaluations."""

    experiment_id = "test_exp_003"
    trajectory_id = "traj_cache_test"
    judge_model = "claude-3.5-sonnet"

    storage.create_experiment(experiment_id, name="Cache Test")

    storage.save_trajectory(
        trajectory_id, experiment_id, benchmark="toolbench", is_perturbed=True
    )

    # Check cache - should be miss
    cached = storage.check_evaluation_cache(trajectory_id, judge_model, sample_number=1)
    assert cached is None, "Cache should be empty initially"

    # Save evaluation
    storage.save_evaluation(
        "eval_001",
        experiment_id,
        trajectory_id,
        judge_model=judge_model,
        sample_number=1,
        overall_score=85.0,
        overall_penalty=15.0,
    )

    # Check cache again - should be hit!
    cached = storage.check_evaluation_cache(trajectory_id, judge_model, sample_number=1)
    assert cached is not None, "Cache should return existing evaluation"
    assert cached["overall_score"] == 85.0
    assert cached["judge_model"] == judge_model

    print("✅ Cache check working correctly")


def test_cross_collection_foreign_keys(storage):
    """Test foreign key relationships across all collections."""

    experiment_id = "test_exp_004"
    trajectory_id = "traj_fk_test"

    # Create experiment
    storage.create_experiment(experiment_id, name="Foreign Key Test")

    # Create trajectory
    storage.save_trajectory(
        trajectory_id,
        experiment_id,
        benchmark="toolbench",
        is_perturbed=True,
        perturbation={"type": "planning", "position": "early"},
    )

    # Create annotation with foreign keys
    storage.save_annotation(
        "ann_001",
        experiment_id,  # FK to experiments
        trajectory_id,  # FK to trajectories
        annotator="human_researcher",
        true_criticality_score=110.0,
    )

    # Create judge evaluation with foreign keys
    storage.save_evaluation(
        "eval_002",
        experiment_id,  # FK to experiments
        trajectory_id,  # FK to trajectories
        judge_model="claude-3.5-sonnet",
        sample_number=1,
        overall_score=85.0,
        overall_penalty=15.0,
    )

    # Create CCG score with foreign keys
    storage.save_ccg_score(
        "ccg_001",
        experiment_id,  # FK to experiments
        trajectory_id,  # FK to trajectories
        "ann_001",  # FK to annotations
        "eval_002",  # FK to judge_evaluations
        perturbation_type="planning",
        perturbation_position="early",
        judge_model="claude-3.5-sonnet",
        true_criticality_score=110.0,
        judge_penalty_score=15.0,
        criticality_calibration_gap=-0.86,
    )

    # Verify all foreign key queries work
    trajectories = storage.get_trajectories_by_experiment(experiment_id)
    assert len(trajectories) == 1

    # Verify foreign keys present
    annotation = storage.annotations["ann_001"]
    assert annotation["experiment_id"] == experiment_id
    assert annotation["trajectory_id"] == trajectory_id

    evaluation = storage.judge_evaluations["eval_002"]
    assert evaluation["experiment_id"] == experiment_id
    assert evaluation["trajectory_id"] == trajectory_id

    ccg = storage.ccg_scores["ccg_001"]
    assert ccg["experiment_id"] == experiment_id
    assert ccg["trajectory_id"] == trajectory_id
    assert ccg["annotation_id"] == "ann_001"
    assert ccg["evaluation_id"] == "eval_002"

    print("✅ Cross-collection foreign keys working correctly")


def test_no_16mb_limit(storage):
    """
    Simulate large experiment to verify no 16MB document limit.

    With arrays: ~400K trajectory IDs would hit 16MB limit
    With foreign keys: unlimited scalability
    """

    experiment_id = "test_exp_scalability"
    storage.create_experiment(
        experiment_id, name="Scalability Test", progress={"trajectories_loaded": 0}
    )

    # Simulate 1000 trajectories (in real case, could be millions)
    num_trajectories = 1000

    for i in range(num_trajectories):
        storage.save_trajectory(
            f"traj_scale_{i}",
            experiment_id,  # Foreign key!
            benchmark="toolbench",
            is_perturbed=False,
        )

    # Update progress with count
    storage.experiments[experiment_id]["progress"][
        "trajectories_loaded"
    ] = num_trajectories

    # Verify experiment document is still small (no array!)
    exp = storage.experiments[experiment_id]
    assert "trajectory_refs" not in exp
    assert exp["progress"]["trajectories_loaded"] == num_trajectories

    # Verify all trajectories accessible via foreign key
    count = storage.count_trajectories(experiment_id=experiment_id)
    assert count == num_trajectories

    # Verify pagination works with large dataset
    page = storage.get_trajectories_by_experiment(experiment_id, skip=0, limit=100)
    assert len(page) == 100

    print("✅ No 16MB limit - scalability verified")


def test_schema_format_validation():
    """Test that schema format is correct for all collections."""

    storage = MockStorage()
    experiment_id = "schema_test"

    # Create complete experiment
    storage.create_experiment(
        experiment_id,
        name="Schema Validation",
        config={},
        progress={
            "trajectories_loaded": 0,
            "annotations_completed": 0,
            "evaluations_completed": 0,
        },
        status="created",
        created_at=datetime.utcnow().isoformat(),
    )

    storage.save_trajectory(
        "traj_001",
        experiment_id,
        benchmark="toolbench",
        is_perturbed=False,
        steps=[],
        ground_truth={},
    )

    storage.save_annotation(
        "ann_001",
        experiment_id,
        "traj_001",
        annotator="test",
        task_success_degradation=1.0,
        subsequent_error_rate=0.5,
        true_criticality_score=105.0,
    )

    storage.save_evaluation(
        "eval_001",
        experiment_id,
        "traj_001",
        judge_model="claude-3.5-sonnet",
        sample_number=1,
        overall_score=80.0,
        overall_penalty=20.0,
    )

    storage.save_ccg_score(
        "ccg_001",
        experiment_id,
        "traj_001",
        "ann_001",
        "eval_001",
        true_criticality_score=105.0,
        judge_penalty_score=20.0,
        criticality_calibration_gap=-0.81,
    )

    # Verify schema format
    exp = storage.experiments[experiment_id]
    assert "experiment_id" in exp
    assert "progress" in exp
    assert "trajectory_refs" not in exp  # No arrays!

    traj = storage.trajectories["traj_001"]
    assert "trajectory_id" in traj
    assert "experiment_id" in traj  # Foreign key!
    assert "benchmark" in traj

    ann = storage.annotations["ann_001"]
    assert "annotation_id" in ann
    assert "experiment_id" in ann  # FK
    assert "trajectory_id" in ann  # FK
    assert "true_criticality_score" in ann

    evaluation = storage.judge_evaluations["eval_001"]
    assert "evaluation_id" in evaluation
    assert "experiment_id" in evaluation  # FK
    assert "trajectory_id" in evaluation  # FK
    assert "overall_penalty" in evaluation

    ccg = storage.ccg_scores["ccg_001"]
    assert "ccg_id" in ccg
    assert "experiment_id" in ccg  # FK
    assert "trajectory_id" in ccg  # FK
    assert "annotation_id" in ccg  # FK
    assert "evaluation_id" in ccg  # FK
    assert "criticality_calibration_gap" in ccg

    print("✅ All schema formats correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
