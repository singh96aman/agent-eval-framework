"""
Integration test for full pipeline:
config → trajectories → annotations → judge evaluations → CCG scores.

This test verifies:
1. Config is loaded correctly
2. Experiment is created with foreign key schema (no arrays!)
3. Trajectories are stored with experiment_id foreign key
4. Perturbations are generated correctly
5. Annotations are stored with foreign keys
6. Judge evaluations are stored with foreign keys (stubbed responses)
7. CCG scores are computed correctly with all foreign keys
8. All data follows the approved schema format

NO MONGODB REQUIRED - outputs saved to JSON files in tests/data/results/
"""

import pytest
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from src.data.schema import Trajectory, Step, GroundTruth


# Test data directories
TEST_DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = TEST_DATA_DIR / "results"


class IntegrationTestPipeline:
    """
    Full pipeline test that runs from config to CCG scores.

    Outputs saved to JSON files instead of MongoDB.

    Setup:
    - Loads test config
    - Creates experiment
    - Generates trajectories
    - Creates perturbations
    - Creates annotations (stubbed)
    - Creates judge evaluations (stubbed)
    - Computes CCG scores
    - Saves all to JSON files

    Tests verify schema format from JSON files.
    """

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.config = None
        self.experiment_id = None
        self.outputs = {
            "experiment": None,
            "trajectories": [],
            "annotations": [],
            "judge_evaluations": [],
            "ccg_scores": []
        }

        # Load stubbed responses
        self.stubbed_responses = self._load_stubbed_responses()

    def _load_stubbed_responses(self) -> Dict[str, Any]:
        """Load stubbed responses from JSON file."""
        with open(TEST_DATA_DIR / "stubbed_responses.json", "r") as f:
            return json.load(f)

    def _save_outputs(self):
        """Save all outputs to JSON files."""
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment
        with open(self.results_dir / "experiment.json", "w") as f:
            json.dump(self.outputs["experiment"], f, indent=2, default=str)

        # Save trajectories
        with open(self.results_dir / "trajectories.json", "w") as f:
            json.dump(self.outputs["trajectories"], f, indent=2, default=str)

        # Save annotations
        with open(self.results_dir / "annotations.json", "w") as f:
            json.dump(self.outputs["annotations"], f, indent=2, default=str)

        # Save judge evaluations
        with open(self.results_dir / "judge_evaluations.json", "w") as f:
            json.dump(self.outputs["judge_evaluations"], f, indent=2, default=str)

        # Save CCG scores
        with open(self.results_dir / "ccg_scores.json", "w") as f:
            json.dump(self.outputs["ccg_scores"], f, indent=2, default=str)

        print(f"✅ Results saved to {self.results_dir}")

    def load_config(self):
        """Step 1: Load test config."""
        with open(TEST_DATA_DIR / "test_config.json", "r") as f:
            self.config = json.load(f)
        self.experiment_id = self.config["experiment_id"]
        return self.config

    def create_experiment(self):
        """Step 2: Create experiment with progress tracking (counts only!)."""
        self.outputs["experiment"] = {
            "experiment_id": self.experiment_id,
            "name": self.config["name"],
            "description": self.config["description"],
            "config": self.config["config"],
            "progress": {
                "trajectories_loaded": 0,
                "perturbations_generated": 0,
                "annotations_completed": 0,
                "evaluations_completed": 0,
                "ccg_scores_computed": 0
            },
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        return self.experiment_id

    def load_trajectories(self):
        """
        Step 3: Load sample trajectory with experiment_id foreign key.
        """
        # Load sample trajectory
        with open(TEST_DATA_DIR / "sample_trajectory.json", "r") as f:
            traj_data = json.load(f)

        # Convert to Trajectory object
        steps = [Step.from_dict(s) for s in traj_data["steps"]]
        ground_truth = GroundTruth.from_dict(traj_data["ground_truth"])

        trajectory = Trajectory(
            trajectory_id=traj_data["trajectory_id"],
            benchmark=traj_data["benchmark"],
            steps=steps,
            ground_truth=ground_truth,
            metadata=traj_data.get("metadata", {})
        )

        # Store with experiment_id FOREIGN KEY
        traj_doc = trajectory.to_dict()
        traj_doc["experiment_id"] = self.experiment_id  # FOREIGN KEY!
        traj_doc["is_perturbed"] = False
        traj_doc["perturbation"] = None
        traj_doc["stored_at"] = datetime.utcnow().isoformat()

        self.outputs["trajectories"].append(traj_doc)

        # Update progress (count, not array!)
        self.outputs["experiment"]["progress"]["trajectories_loaded"] += 1
        self.outputs["experiment"]["updated_at"] = datetime.utcnow().isoformat()

        return trajectory

    def generate_perturbations(self, original_trajectory: Trajectory):
        """Step 4: Generate perturbed trajectory (stubbed perturbation)."""
        perturb_config = self.stubbed_responses["perturbations"]["planning_early"]

        # Copy original trajectory
        perturbed_steps = [Step.from_dict(s.to_dict()) for s in original_trajectory.steps]

        # Apply perturbation to step 1
        perturbed_steps[0].content = perturb_config["perturbed_content"]

        # Create perturbed trajectory
        perturbed_trajectory = Trajectory(
            trajectory_id=f"{original_trajectory.trajectory_id}_perturbed",
            benchmark=original_trajectory.benchmark,
            steps=perturbed_steps,
            ground_truth=GroundTruth.from_dict(original_trajectory.ground_truth.to_dict()),
            metadata=original_trajectory.metadata.copy()
        )

        # Store with experiment_id FOREIGN KEY
        perturbed_doc = perturbed_trajectory.to_dict()
        perturbed_doc["experiment_id"] = self.experiment_id  # FOREIGN KEY!
        perturbed_doc["is_perturbed"] = True
        perturbed_doc["original_trajectory_id"] = original_trajectory.trajectory_id
        perturbed_doc["perturbation"] = {
            "type": "planning",
            "position": "early",
            "perturbed_step_number": 1,
            "original_step_content": perturb_config["original_content"],
            "perturbed_step_content": perturb_config["perturbed_content"],
            "metadata": perturb_config["perturbation_metadata"]
        }
        perturbed_doc["stored_at"] = datetime.utcnow().isoformat()

        self.outputs["trajectories"].append(perturbed_doc)

        # Update progress
        self.outputs["experiment"]["progress"]["perturbations_generated"] += 1
        self.outputs["experiment"]["updated_at"] = datetime.utcnow().isoformat()

        return perturbed_trajectory

    def create_annotations(self, perturbed_trajectory: Trajectory):
        """Step 5: Create annotation (stubbed human annotation)."""
        annotation_data = self.stubbed_responses["annotations"][
            "toolbench_sample_001_perturbed"
        ]

        annotation_doc = {
            "annotation_id": f"ann_{self.experiment_id}_001",
            "experiment_id": self.experiment_id,  # FOREIGN KEY!
            "trajectory_id": perturbed_trajectory.trajectory_id,  # FOREIGN KEY!
            "annotator": annotation_data["annotator"],
            "task_success_degradation": annotation_data["task_success_degradation"],
            "task_success_before": annotation_data["task_success_before"],
            "task_success_after": annotation_data["task_success_after"],
            "subsequent_errors": annotation_data["subsequent_errors"],
            "subsequent_error_rate": annotation_data["subsequent_error_rate"],
            "true_criticality_score": annotation_data["true_criticality_score"],
            "confidence": annotation_data["confidence"],
            "notes": annotation_data["notes"],
            "annotated_at": datetime.utcnow().isoformat()
        }

        self.outputs["annotations"].append(annotation_doc)

        # Update progress
        self.outputs["experiment"]["progress"]["annotations_completed"] += 1
        self.outputs["experiment"]["updated_at"] = datetime.utcnow().isoformat()

        return annotation_doc

    def create_judge_evaluations(self, perturbed_trajectory: Trajectory):
        """
        Step 6: Create judge evaluation (stubbed - NO API CALL).
        """
        judge_data = self.stubbed_responses["judge_evaluations"]["claude-3.5-sonnet"]

        evaluation_doc = {
            "evaluation_id": f"eval_{self.experiment_id}_claude_001",
            "experiment_id": self.experiment_id,  # FOREIGN KEY!
            "trajectory_id": perturbed_trajectory.trajectory_id,  # FOREIGN KEY!
            "judge_model": "claude-3.5-sonnet",
            "judge_model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "judge_provider": "aws_bedrock",
            "sample_number": 1,
            "overall_score": judge_data["overall_score"],
            "overall_penalty": judge_data["overall_penalty"],
            "errors_identified": judge_data["errors_identified"],
            "raw_response": judge_data["raw_response"],
            "prompt_used": "You are evaluating an agent trajectory...",
            "api_call": judge_data["api_call"],
            "temperature": 0.7,
            "max_tokens": 2000,
            "evaluated_at": datetime.utcnow().isoformat()
        }

        self.outputs["judge_evaluations"].append(evaluation_doc)

        # Update progress
        self.outputs["experiment"]["progress"]["evaluations_completed"] += 1
        self.outputs["experiment"]["updated_at"] = datetime.utcnow().isoformat()

        return evaluation_doc

    def compute_ccg_scores(
        self,
        perturbed_trajectory: Trajectory,
        annotation: Dict[str, Any],
        evaluation: Dict[str, Any]
    ):
        """Step 7: Compute CCG score."""
        tcs = annotation["true_criticality_score"]
        jps = evaluation["overall_penalty"]
        ccg = (jps - tcs) / tcs if tcs != 0 else 0

        # Determine calibration status
        if ccg < -0.2:
            calibration_status = "under_penalized"
            gap_magnitude = "severe" if ccg < -0.5 else "moderate"
        elif ccg > 0.2:
            calibration_status = "over_penalized"
            gap_magnitude = "severe" if ccg > 0.5 else "moderate"
        else:
            calibration_status = "well_calibrated"
            gap_magnitude = "negligible"

        ccg_doc = {
            "ccg_id": f"ccg_{self.experiment_id}_001",
            "experiment_id": self.experiment_id,  # FOREIGN KEY!
            "trajectory_id": perturbed_trajectory.trajectory_id,  # FOREIGN KEY!
            "annotation_id": annotation["annotation_id"],  # FOREIGN KEY!
            "evaluation_id": evaluation["evaluation_id"],  # FOREIGN KEY!
            "perturbation_type": "planning",
            "perturbation_position": "early",
            "benchmark": perturbed_trajectory.benchmark,
            "judge_model": evaluation["judge_model"],
            "sample_number": evaluation["sample_number"],
            "true_criticality_score": tcs,
            "judge_penalty_score": jps,
            "criticality_calibration_gap": ccg,
            "calibration_status": calibration_status,
            "gap_magnitude": gap_magnitude,
            "computed_at": datetime.utcnow().isoformat()
        }

        self.outputs["ccg_scores"].append(ccg_doc)

        # Update progress
        self.outputs["experiment"]["progress"]["ccg_scores_computed"] += 1
        self.outputs["experiment"]["status"] = "completed"
        self.outputs["experiment"]["updated_at"] = datetime.utcnow().isoformat()

        return ccg_doc

    def run_full_pipeline(self):
        """Run entire pipeline from config to CCG scores and save to JSON."""
        print("\n🚀 Starting integration test pipeline (NO MONGODB)...")

        print("1️⃣  Loading config...")
        self.load_config()

        print("2️⃣  Creating experiment...")
        self.create_experiment()

        print("3️⃣  Loading trajectories...")
        original_traj = self.load_trajectories()

        print("4️⃣  Generating perturbations...")
        perturbed_traj = self.generate_perturbations(original_traj)

        print("5️⃣  Creating annotations (stubbed)...")
        annotation = self.create_annotations(perturbed_traj)

        print("6️⃣  Creating judge evaluations (stubbed - NO API CALL)...")
        evaluation = self.create_judge_evaluations(perturbed_traj)

        print("7️⃣  Computing CCG scores...")
        self.compute_ccg_scores(perturbed_traj, annotation, evaluation)

        print("8️⃣  Saving results to JSON files...")
        self._save_outputs()

        print("✅ Pipeline complete!\n")

        return self.outputs


@pytest.fixture(scope="class")
def pipeline_results():
    """
    Run pipeline once and save to JSON files.

    This fixture runs in setup, all tests read from JSON files.
    """
    # Clean up old results
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)

    # Run pipeline
    pipeline = IntegrationTestPipeline(RESULTS_DIR)
    results = pipeline.run_full_pipeline()

    return results


@pytest.fixture(scope="class")
def experiment_data():
    """Load experiment from JSON file."""
    with open(RESULTS_DIR / "experiment.json", "r") as f:
        return json.load(f)


@pytest.fixture(scope="class")
def trajectories_data():
    """Load trajectories from JSON file."""
    with open(RESULTS_DIR / "trajectories.json", "r") as f:
        return json.load(f)


@pytest.fixture(scope="class")
def annotations_data():
    """Load annotations from JSON file."""
    with open(RESULTS_DIR / "annotations.json", "r") as f:
        return json.load(f)


@pytest.fixture(scope="class")
def evaluations_data():
    """Load judge evaluations from JSON file."""
    with open(RESULTS_DIR / "judge_evaluations.json", "r") as f:
        return json.load(f)


@pytest.fixture(scope="class")
def ccg_data():
    """Load CCG scores from JSON file."""
    with open(RESULTS_DIR / "ccg_scores.json", "r") as f:
        return json.load(f)


class TestIntegrationPipeline:
    """
    Test class that verifies full pipeline with stubbed data.

    Setup runs the pipeline once, tests verify schema format from JSON files.
    NO MONGODB REQUIRED!
    """

    def test_experiment_schema_format(
        self,
        pipeline_results,
        experiment_data
    ):
        """Test experiment follows foreign key schema (no arrays!)."""
        # Verify no arrays of IDs
        msg = "❌ experiments should NOT have trajectory_refs array!"
        assert "trajectory_refs" not in experiment_data, msg
        assert "annotation_refs" not in experiment_data
        assert "judge_eval_refs" not in experiment_data
        assert "ccg_refs" not in experiment_data

        # Verify progress has counts only
        assert "progress" in experiment_data
        assert experiment_data["progress"]["trajectories_loaded"] == 1
        assert experiment_data["progress"]["perturbations_generated"] == 1
        assert experiment_data["progress"]["annotations_completed"] == 1
        assert experiment_data["progress"]["evaluations_completed"] == 1
        assert experiment_data["progress"]["ccg_scores_computed"] == 1

        # Verify status updated
        assert experiment_data["status"] == "completed"

        # Verify timestamps present
        assert "created_at" in experiment_data
        assert "updated_at" in experiment_data

        print("✅ Experiment schema correct (no arrays, only counts)")

    def test_trajectory_schema_format(
        self,
        pipeline_results,
        trajectories_data
    ):
        """Test trajectories have experiment_id foreign key."""
        assert len(trajectories_data) == 2  # 1 original + 1 perturbed

        experiment_id = trajectories_data[0]["experiment_id"]

        for traj in trajectories_data:
            # Verify foreign key
            msg = "❌ trajectories must have experiment_id!"
            assert "experiment_id" in traj, msg
            assert traj["experiment_id"] == experiment_id

            # Verify required fields
            assert "trajectory_id" in traj
            assert "benchmark" in traj
            assert "is_perturbed" in traj
            assert "steps" in traj
            assert "ground_truth" in traj
            assert "stored_at" in traj

        # Verify original trajectory
        original = [t for t in trajectories_data if not t["is_perturbed"]][0]
        assert original["perturbation"] is None
        assert "original_trajectory_id" not in original

        # Verify perturbed trajectory
        perturbed = [t for t in trajectories_data if t["is_perturbed"]][0]
        assert perturbed["perturbation"] is not None
        assert perturbed["perturbation"]["type"] == "planning"
        assert perturbed["perturbation"]["position"] == "early"
        assert perturbed["perturbation"]["perturbed_step_number"] == 1
        assert "original_trajectory_id" in perturbed
        assert perturbed["original_trajectory_id"] == original["trajectory_id"]

        print("✅ Trajectory schema correct (foreign keys present)")

    def test_annotation_schema_format(
        self,
        pipeline_results,
        annotations_data
    ):
        """Test annotations have foreign keys."""
        assert len(annotations_data) == 1

        annotation = annotations_data[0]

        # Verify foreign keys
        assert "experiment_id" in annotation
        assert "trajectory_id" in annotation

        # Verify required fields
        assert "annotation_id" in annotation
        assert "annotator" in annotation
        assert "task_success_degradation" in annotation
        assert "subsequent_error_rate" in annotation
        assert "subsequent_errors" in annotation
        assert "true_criticality_score" in annotation
        assert "annotated_at" in annotation

        # Verify TCS calculation: TCS = (TSD × 100) + (SER × 10)
        tsd = annotation["task_success_degradation"]
        ser = annotation["subsequent_error_rate"]
        expected_tcs = (tsd * 100) + (ser * 10)
        actual = annotation["true_criticality_score"]
        msg = f"TCS formula incorrect: {actual} != {expected_tcs}"
        assert actual == expected_tcs, msg

        # Verify subsequent errors format
        assert isinstance(annotation["subsequent_errors"], list)
        for error in annotation["subsequent_errors"]:
            assert "step_number" in error
            assert "error_type" in error
            assert "severity" in error
            assert "description" in error

        print("✅ Annotation schema correct (foreign keys + TCS)")

    def test_judge_evaluation_schema_format(
        self,
        pipeline_results,
        evaluations_data
    ):
        """Test judge evaluations have foreign keys."""
        assert len(evaluations_data) == 1

        evaluation = evaluations_data[0]

        # Verify foreign keys
        assert "experiment_id" in evaluation
        assert "trajectory_id" in evaluation

        # Verify required fields
        assert "evaluation_id" in evaluation
        assert "judge_model" in evaluation
        assert "judge_model_id" in evaluation
        assert "judge_provider" in evaluation
        assert "sample_number" in evaluation
        assert "overall_score" in evaluation
        assert "overall_penalty" in evaluation
        assert "errors_identified" in evaluation
        assert "raw_response" in evaluation
        assert "evaluated_at" in evaluation

        # Verify JPS calculation: JPS = 100 - overall_score
        expected_jps = 100 - evaluation["overall_score"]
        actual = evaluation["overall_penalty"]
        msg = f"JPS formula incorrect: {actual} != {expected_jps}"
        assert actual == expected_jps, msg

        # Verify errors format
        assert isinstance(evaluation["errors_identified"], list)
        for error in evaluation["errors_identified"]:
            assert "step_number" in error
            assert "error_description" in error
            assert "severity" in error
            assert "judge_reasoning" in error

        # Verify API call metadata
        assert "api_call" in evaluation
        assert "tokens_input" in evaluation["api_call"]
        assert "tokens_output" in evaluation["api_call"]
        assert "cost_usd" in evaluation["api_call"]

        print("✅ Judge evaluation schema correct (foreign keys + JPS)")

    def test_ccg_score_schema_format(
        self,
        pipeline_results,
        ccg_data
    ):
        """Test CCG scores have all foreign keys."""
        assert len(ccg_data) == 1

        ccg = ccg_data[0]

        # Verify ALL foreign keys
        assert "experiment_id" in ccg
        assert "trajectory_id" in ccg
        assert "annotation_id" in ccg
        assert "evaluation_id" in ccg

        # Verify required fields
        assert "ccg_id" in ccg
        assert "perturbation_type" in ccg
        assert "perturbation_position" in ccg
        assert "benchmark" in ccg
        assert "judge_model" in ccg
        assert "sample_number" in ccg
        assert "true_criticality_score" in ccg
        assert "judge_penalty_score" in ccg
        assert "criticality_calibration_gap" in ccg
        assert "calibration_status" in ccg
        assert "gap_magnitude" in ccg
        assert "computed_at" in ccg

        # Verify CCG calculation: CCG = (JPS - TCS) / TCS
        tcs = ccg["true_criticality_score"]
        jps = ccg["judge_penalty_score"]
        expected_ccg = (jps - tcs) / tcs
        actual = ccg["criticality_calibration_gap"]
        msg = f"CCG formula incorrect: {actual} != {expected_ccg}"
        assert abs(actual - expected_ccg) < 0.001, msg

        # Verify calibration status logic
        ccg_value = ccg["criticality_calibration_gap"]
        if ccg_value < -0.2:
            assert ccg["calibration_status"] == "under_penalized"
            if ccg_value < -0.5:
                assert ccg["gap_magnitude"] == "severe"
        elif ccg_value > 0.2:
            assert ccg["calibration_status"] == "over_penalized"
        else:
            assert ccg["calibration_status"] == "well_calibrated"

        print("✅ CCG score schema correct (all foreign keys + CCG formula)")

    def test_foreign_key_linkage(
        self,
        pipeline_results,
        trajectories_data,
        annotations_data,
        evaluations_data,
        ccg_data
    ):
        """Test that foreign keys correctly link across all collections."""
        # Get IDs
        perturbed_traj = [t for t in trajectories_data if t["is_perturbed"]][0]
        annotation = annotations_data[0]
        evaluation = evaluations_data[0]
        ccg = ccg_data[0]

        # Verify trajectory → annotation linkage
        assert annotation["trajectory_id"] == perturbed_traj["trajectory_id"]

        # Verify trajectory → evaluation linkage
        assert evaluation["trajectory_id"] == perturbed_traj["trajectory_id"]

        # Verify CCG → all other collections
        assert ccg["trajectory_id"] == perturbed_traj["trajectory_id"]
        assert ccg["annotation_id"] == annotation["annotation_id"]
        assert ccg["evaluation_id"] == evaluation["evaluation_id"]

        # Verify experiment_id consistent across all
        experiment_id = perturbed_traj["experiment_id"]
        assert annotation["experiment_id"] == experiment_id
        assert evaluation["experiment_id"] == experiment_id
        assert ccg["experiment_id"] == experiment_id

        print("✅ Foreign key linkage correct across all collections")

    def test_expected_ccg_value(
        self,
        pipeline_results,
        ccg_data
    ):
        """Test computed CCG matches expected from stubbed data."""
        ccg = ccg_data[0]

        # Load expected CCG
        with open(TEST_DATA_DIR / "stubbed_responses.json", "r") as f:
            stubbed = json.load(f)

        expected = stubbed["expected_ccg"]

        # Verify values match
        assert ccg["true_criticality_score"] == \
            expected["true_criticality_score"]
        assert ccg["judge_penalty_score"] == \
            expected["judge_penalty_score"]
        ccg_diff = abs(
            ccg["criticality_calibration_gap"] -
            expected["criticality_calibration_gap"]
        )
        assert ccg_diff < 0.01
        assert ccg["calibration_status"] == expected["calibration_status"]
        assert ccg["gap_magnitude"] == expected["gap_magnitude"]

        print("✅ CCG values match expected results from stubbed data")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
