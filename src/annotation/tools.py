"""
Annotation tools for ground truth criticality scoring.

Provides a CLI interface for human researchers to annotate the criticality
of perturbations in agent trajectories.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.data.schema import Trajectory, PerturbedTrajectory
from src.storage.mongodb import MongoDBStorage


@dataclass
class Annotation:
    """
    Ground truth annotation for a perturbed trajectory.

    Attributes:
        perturbation_id: ID of the perturbed trajectory
        annotator_id: ID of person doing annotation
        task_success_degradation: Binary - did perturbation cause task failure? (0 or 1)
        subsequent_error_rate: Count of errors that occurred after the perturbation
        notes: Free-form notes from annotator
        timestamp: When annotation was created
        annotation_time_seconds: How long annotation took
    """
    perturbation_id: str
    annotator_id: str
    task_success_degradation: int  # 0 or 1
    subsequent_error_rate: int  # Count of downstream errors
    notes: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    annotation_time_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Annotation':
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def compute_tcs(self) -> float:
        """
        Compute True Criticality Score (TCS).

        Formula: TCS = (TSD × 100) + (SER × 10)
        where:
          TSD = Task Success Degradation (0 or 1)
          SER = Subsequent Error Rate (count)

        Returns:
            TCS score (0-100+ range)
        """
        return (self.task_success_degradation * 100) + (self.subsequent_error_rate * 10)


class AnnotationInterface:
    """
    Interactive CLI interface for annotating perturbations.

    Displays baseline vs. perturbed trajectory side-by-side and prompts
    for criticality scores.
    """

    def __init__(self, storage: Optional[MongoDBStorage] = None):
        """
        Initialize annotation interface.

        Args:
            storage: MongoDB storage instance (if None, creates default)
        """
        self.storage = storage or MongoDBStorage()
        self.annotations_dir = Path("data/annotations")
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    def load_perturbation(self, perturbation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load perturbation from MongoDB.

        Args:
            perturbation_id: ID of perturbed trajectory

        Returns:
            Perturbation document or None if not found
        """
        return self.storage.db['perturbations'].find_one({'perturbation_id': perturbation_id})

    def display_comparison(self, perturbation_data: Dict[str, Any]) -> None:
        """
        Display baseline vs. perturbed trajectory side-by-side.

        Args:
            perturbation_data: Perturbation document from MongoDB
        """
        perturbed = PerturbedTrajectory.from_dict(perturbation_data)

        print("\n" + "=" * 80)
        print("TRAJECTORY COMPARISON")
        print("=" * 80)
        print(f"Perturbation ID: {perturbation_data['perturbation_id']}")
        print(f"Type: {perturbed.perturbation_type}")
        print(f"Position: {perturbed.perturbation_position}")
        print(f"Perturbed Step: {perturbed.perturbed_step_number}")
        print("=" * 80)

        # Task description
        print(f"\n📋 TASK: {perturbed.original_trajectory.ground_truth.task_description}")

        # Display steps up to and including perturbation
        print(f"\n🔍 STEPS (showing up to perturbed step {perturbed.perturbed_step_number}):\n")

        for step in perturbed.original_trajectory.steps[:perturbed.perturbed_step_number]:
            is_perturbed = (step.step_number == perturbed.perturbed_step_number)
            marker = "🔴 PERTURBED" if is_perturbed else "  "

            print(f"{marker} Step {step.step_number} [{step.step_type.value}]:")

            if is_perturbed:
                print(f"  ❌ ORIGINAL:  {perturbed.original_step_content[:200]}...")
                print(f"  ⚠️  PERTURBED: {perturbed.perturbed_step_content[:200]}...")
            else:
                print(f"     {step.content[:200]}...")

            if step.tool_name:
                print(f"     Tool: {step.tool_name}")
            print()

        # Show remaining steps from perturbed trajectory
        remaining_steps = perturbed.perturbed_trajectory.steps[perturbed.perturbed_step_number:]
        if remaining_steps:
            print(f"\n📊 SUBSEQUENT STEPS ({len(remaining_steps)} steps after perturbation):\n")
            for step in remaining_steps[:3]:  # Show first 3
                print(f"   Step {step.step_number}: {step.content[:100]}...")
            if len(remaining_steps) > 3:
                print(f"   ... ({len(remaining_steps) - 3} more steps)")

        print("\n" + "=" * 80)

    def prompt_annotation(self, perturbation_id: str, annotator_id: str = "default") -> Annotation:
        """
        Prompt user for annotation input.

        Args:
            perturbation_id: ID of perturbed trajectory
            annotator_id: ID of annotator (default: "default")

        Returns:
            Completed annotation
        """
        start_time = datetime.utcnow()

        print("\n📝 ANNOTATION QUESTIONS:")
        print("-" * 80)

        # Task Success Degradation
        while True:
            tsd_input = input("\n1. Did the perturbation cause the task to FAIL? (yes/no): ").strip().lower()
            if tsd_input in ['yes', 'y', '1']:
                tsd = 1
                break
            elif tsd_input in ['no', 'n', '0']:
                tsd = 0
                break
            else:
                print("   ⚠️  Please enter 'yes' or 'no'")

        # Subsequent Error Rate
        while True:
            try:
                ser_input = input("\n2. How many errors occurred AFTER this perturbation? (count): ").strip()
                ser = int(ser_input)
                if ser >= 0:
                    break
                else:
                    print("   ⚠️  Please enter a non-negative number")
            except ValueError:
                print("   ⚠️  Please enter a valid number")

        # Notes
        notes = input("\n3. Any additional notes? (optional, press Enter to skip): ").strip()

        end_time = datetime.utcnow()
        annotation_time = (end_time - start_time).total_seconds()

        annotation = Annotation(
            perturbation_id=perturbation_id,
            annotator_id=annotator_id,
            task_success_degradation=tsd,
            subsequent_error_rate=ser,
            notes=notes,
            timestamp=end_time,
            annotation_time_seconds=annotation_time
        )

        # Display computed TCS
        tcs = annotation.compute_tcs()
        print(f"\n✅ True Criticality Score (TCS): {tcs}")
        print(f"   Formula: ({tsd} × 100) + ({ser} × 10) = {tcs}")

        return annotation

    def annotate(self, perturbation_id: str, annotator_id: str = "default") -> Optional[Annotation]:
        """
        Full annotation workflow: load, display, prompt, save.

        Args:
            perturbation_id: ID of perturbed trajectory
            annotator_id: ID of annotator

        Returns:
            Completed annotation or None if failed
        """
        # Load perturbation
        perturbation_data = self.load_perturbation(perturbation_id)
        if not perturbation_data:
            print(f"❌ Perturbation not found: {perturbation_id}")
            return None

        # Check if already annotated
        existing_path = self.annotations_dir / f"{perturbation_id}.json"
        if existing_path.exists():
            print(f"\n⚠️  Annotation already exists: {existing_path}")
            overwrite = input("Overwrite? (yes/no): ").strip().lower()
            if overwrite not in ['yes', 'y']:
                print("Cancelled.")
                return None

        # Display comparison
        self.display_comparison(perturbation_data)

        # Prompt for annotation
        annotation = self.prompt_annotation(perturbation_id, annotator_id)

        # Save annotation
        save_annotation(annotation, self.annotations_dir)
        print(f"\n✅ Annotation saved: {existing_path}")

        return annotation

    def batch_annotate(self, perturbation_ids: List[str], annotator_id: str = "default") -> List[Annotation]:
        """
        Annotate multiple perturbations in sequence.

        Args:
            perturbation_ids: List of perturbation IDs
            annotator_id: ID of annotator

        Returns:
            List of completed annotations
        """
        annotations = []

        for i, pert_id in enumerate(perturbation_ids, 1):
            print(f"\n{'=' * 80}")
            print(f"ANNOTATION {i}/{len(perturbation_ids)}")
            print(f"{'=' * 80}")

            annotation = self.annotate(pert_id, annotator_id)
            if annotation:
                annotations.append(annotation)

            # Ask to continue
            if i < len(perturbation_ids):
                continue_input = input("\nContinue to next? (yes/no/quit): ").strip().lower()
                if continue_input in ['no', 'n', 'quit', 'q']:
                    print(f"Stopping. Annotated {len(annotations)}/{len(perturbation_ids)}")
                    break

        return annotations


class AnnotationReviewer:
    """
    Quality control and review of annotations.

    Provides tools to review annotations, flag inconsistencies,
    and compute agreement metrics.
    """

    def __init__(self, annotations_dir: Path = Path("data/annotations")):
        """
        Initialize reviewer.

        Args:
            annotations_dir: Directory containing annotation JSON files
        """
        self.annotations_dir = annotations_dir

    def load_all_annotations(self) -> List[Annotation]:
        """
        Load all annotations from directory.

        Returns:
            List of annotations
        """
        annotations = []

        if not self.annotations_dir.exists():
            return annotations

        for filepath in self.annotations_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    annotations.append(Annotation.from_dict(data))
            except Exception as e:
                print(f"⚠️  Error loading {filepath}: {e}")

        return annotations

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of annotations.

        Returns:
            Dictionary with summary stats
        """
        annotations = self.load_all_annotations()

        if not annotations:
            return {"total": 0}

        tcs_scores = [a.compute_tcs() for a in annotations]
        tsd_values = [a.task_success_degradation for a in annotations]
        ser_values = [a.subsequent_error_rate for a in annotations]

        return {
            "total": len(annotations),
            "tsd_mean": sum(tsd_values) / len(tsd_values),
            "tsd_task_failures": sum(tsd_values),
            "ser_mean": sum(ser_values) / len(ser_values),
            "ser_min": min(ser_values),
            "ser_max": max(ser_values),
            "tcs_mean": sum(tcs_scores) / len(tcs_scores),
            "tcs_min": min(tcs_scores),
            "tcs_max": max(tcs_scores),
            "annotators": list(set(a.annotator_id for a in annotations))
        }

    def print_summary(self) -> None:
        """Print summary statistics to console."""
        stats = self.get_summary_stats()

        print("\n" + "=" * 80)
        print("ANNOTATION SUMMARY")
        print("=" * 80)
        print(f"Total annotations: {stats['total']}")

        if stats['total'] > 0:
            print(f"\nTask Success Degradation:")
            print(f"  Mean: {stats['tsd_mean']:.2f}")
            print(f"  Task failures: {stats['tsd_task_failures']}/{stats['total']}")

            print(f"\nSubsequent Error Rate:")
            print(f"  Mean: {stats['ser_mean']:.2f}")
            print(f"  Range: [{stats['ser_min']}, {stats['ser_max']}]")

            print(f"\nTrue Criticality Score (TCS):")
            print(f"  Mean: {stats['tcs_mean']:.2f}")
            print(f"  Range: [{stats['tcs_min']:.2f}, {stats['tcs_max']:.2f}]")

            print(f"\nAnnotators: {', '.join(stats['annotators'])}")

        print("=" * 80)


def load_annotation(perturbation_id: str, annotations_dir: Path = Path("data/annotations")) -> Optional[Annotation]:
    """
    Load annotation from file.

    Args:
        perturbation_id: ID of perturbation
        annotations_dir: Directory containing annotations

    Returns:
        Annotation or None if not found
    """
    filepath = annotations_dir / f"{perturbation_id}.json"

    if not filepath.exists():
        return None

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return Annotation.from_dict(data)
    except Exception as e:
        print(f"Error loading annotation {filepath}: {e}")
        return None


def save_annotation(annotation: Annotation, annotations_dir: Path = Path("data/annotations")) -> bool:
    """
    Save annotation to file.

    Args:
        annotation: Annotation to save
        annotations_dir: Directory to save to

    Returns:
        True if successful, False otherwise
    """
    annotations_dir.mkdir(parents=True, exist_ok=True)
    filepath = annotations_dir / f"{annotation.perturbation_id}.json"

    try:
        with open(filepath, 'w') as f:
            json.dump(annotation.to_dict(), f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving annotation {filepath}: {e}")
        return False


def main():
    """
    CLI entry point for annotation tool.

    Usage:
        python -m src.annotation.tools <perturbation_id>
        python -m src.annotation.tools --review
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Annotate: python -m src.annotation.tools <perturbation_id>")
        print("  Review:   python -m src.annotation.tools --review")
        sys.exit(1)

    if sys.argv[1] == "--review":
        reviewer = AnnotationReviewer()
        reviewer.print_summary()
    else:
        perturbation_id = sys.argv[1]
        annotator_id = sys.argv[2] if len(sys.argv) > 2 else "default"

        interface = AnnotationInterface()
        interface.annotate(perturbation_id, annotator_id)


if __name__ == "__main__":
    main()
