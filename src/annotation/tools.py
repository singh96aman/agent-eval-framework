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
        Load perturbation from MongoDB and reconstruct full object with trajectories.

        Args:
            perturbation_id: ID of perturbed trajectory

        Returns:
            Perturbation document with embedded trajectories, or None if not found
        """
        # Load perturbation record (contains trajectory IDs)
        pert_record = self.storage.db['perturbations'].find_one({'perturbation_id': perturbation_id})
        if not pert_record:
            return None

        # Load the referenced trajectories
        # Note: get_trajectory() automatically strips storage metadata
        orig_traj = self.storage.get_trajectory(pert_record['original_trajectory_id'])
        pert_traj = self.storage.get_trajectory(pert_record['perturbed_trajectory_id'])

        if not orig_traj or not pert_traj:
            print(f"⚠️  Missing trajectories for perturbation {perturbation_id}")
            return None

        # Reconstruct full perturbation object with only schema-expected fields
        # Note: perturbation_id is NOT part of PerturbedTrajectory schema
        full_perturbation = {
            'original_trajectory': orig_traj,
            'perturbed_trajectory': pert_traj,
            'perturbation_type': pert_record['perturbation_type'],
            'perturbation_position': pert_record['perturbation_position'],
            'perturbed_step_number': pert_record['perturbed_step_number'],
            'original_step_content': pert_record['original_step_content'],
            'perturbed_step_content': pert_record['perturbed_step_content'],
            'perturbation_metadata': pert_record.get(
                'perturbation_metadata', {}
            )
        }

        # Store perturbation_id separately for annotation saving
        full_perturbation['_perturbation_id'] = pert_record['perturbation_id']

        return full_perturbation

    def display_comparison(self, perturbation_data: Dict[str, Any]) -> None:
        """
        Interactive step-by-step trajectory display with perturbation comparison.

        Args:
            perturbation_data: Perturbation document from MongoDB
        """
        # Extract perturbation_id before passing to from_dict
        pert_id = perturbation_data.get('_perturbation_id', 'unknown')

        # Remove _perturbation_id before deserializing (it's not in schema)
        data_for_schema = {
            k: v for k, v in perturbation_data.items()
            if k != '_perturbation_id'
        }

        perturbed = PerturbedTrajectory.from_dict(data_for_schema)

        # Header
        print("\n" + "=" * 80)
        print("TRAJECTORY REVIEW")
        print("=" * 80)
        print(f"Perturbation ID: {pert_id}")
        print(f"Perturbation Type: {perturbed.perturbation_type.upper()}")
        print(f"Position: {perturbed.perturbation_position.upper()}")
        print(f"Perturbed Step: #{perturbed.perturbed_step_number}")
        print("=" * 80)

        # Task
        print("\n" + "─" * 80)
        print("📋 TASK")
        print("─" * 80)
        task = perturbed.original_trajectory.ground_truth.task_description
        print(f"\n{task}\n")

        # Show planning if available (look for planning/reasoning steps before tools)
        planning_steps = [
            s for s in perturbed.original_trajectory.steps[:perturbed.perturbed_step_number]
            if s.step_type.value in ['planning', 'reasoning'] and not s.tool_name
        ]
        if planning_steps:
            print("─" * 80)
            print("🧠 INITIAL PLANNING")
            print("─" * 80)
            for step in planning_steps:
                print(f"\n{step.content}\n")

        # Walk through steps automatically (no pausing)
        print("─" * 80)
        print("👣 TRAJECTORY STEPS")
        print("─" * 80)
        print()

        for i, step in enumerate(perturbed.original_trajectory.steps, 1):
            is_perturbed = (step.step_number == perturbed.perturbed_step_number)

            if i <= perturbed.perturbed_step_number:
                if is_perturbed:
                    # SIDE-BY-SIDE COMPARISON
                    self._display_side_by_side_comparison(
                        step.step_number,
                        step.step_type.value,
                        perturbed.original_step_content,
                        perturbed.perturbed_step_content,
                        step.tool_name
                    )
                    break  # Stop after showing the perturbation
                else:
                    # Normal step display (complete content)
                    self._display_step(step)

        # Show what happened after perturbation (complete content)
        remaining_steps = perturbed.perturbed_trajectory.steps[
            perturbed.perturbed_step_number:
        ]
        if remaining_steps:
            print("\n" + "─" * 80)
            print("📊 WHAT HAPPENED AFTER THE PERTURBATION")
            print("─" * 80)
            print()

            # Show all remaining steps with complete content
            for step in remaining_steps:
                self._display_step(step)

        print("\n" + "=" * 80)

    def _display_step(self, step) -> None:
        """Display a single trajectory step with complete content."""
        print(f"  Step {step.step_number}: {step.step_type.value.upper()}")
        print()

        # Content (complete, no truncation)
        content = step.content.strip()
        # Indent each line
        for line in content.split('\n'):
            print(f"  {line}")

        # Tool info
        if step.tool_name:
            print()
            print(f"  🔧 Tool: {step.tool_name}")
            if step.tool_input:
                # Show complete input (no truncation)
                input_str = str(step.tool_input)
                print(f"  📥 Input: {input_str}")

        print()  # Blank line between steps

    def _display_side_by_side_comparison(
        self,
        step_num: int,
        step_type: str,
        original: str,
        perturbed: str,
        tool_name: Optional[str] = None
    ) -> None:
        """
        Display sequential comparison of original vs perturbed step.

        Shows full content without truncation for accurate annotation.
        """
        print("\n" + "=" * 80)
        print(f"🔴 PERTURBATION AT STEP {step_num} ({step_type.upper()})")
        print("=" * 80)

        if tool_name:
            print(f"🔧 Tool: {tool_name}")
        print()

        # Show ORIGINAL (full content)
        print("─" * 80)
        print("✅ ORIGINAL STEP CONTENT:")
        print("─" * 80)
        print()
        print(original)
        print()

        # Show PERTURBED (full content)
        print("─" * 80)
        print("⚠️  PERTURBED STEP CONTENT:")
        print("─" * 80)
        print()
        print(perturbed)
        print()

        print("=" * 80)

    def _wrap_text(self, text: str, width: int) -> List[str]:
        """
        Wrap text to specified width, breaking long words if needed.

        Args:
            text: Text to wrap
            width: Maximum width per line

        Returns:
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            # If word itself is longer than width, split it
            if len(word) > width:
                # Add current line if any
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = []
                    current_length = 0

                # Split the long word into chunks
                while len(word) > width:
                    lines.append(word[:width])
                    word = word[width:]

                # Add remaining part
                if word:
                    current_line.append(word)
                    current_length = len(word)

            elif current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + (1 if current_line else 0)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(' '.join(current_line))

        return lines if lines else [""]

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

    def annotation_exists(self, perturbation_id: str, annotator_id: str) -> bool:
        """
        Check if annotation already exists in MongoDB.

        Args:
            perturbation_id: ID of perturbation
            annotator_id: ID of annotator

        Returns:
            True if annotation exists, False otherwise
        """
        existing = self.storage.db['annotations'].find_one({
            'perturbation_id': perturbation_id,
            'annotator_id': annotator_id
        })
        return existing is not None

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

        # Check if already annotated (in MongoDB)
        if self.annotation_exists(perturbation_id, annotator_id):
            print(f"\n⚠️  Annotation already exists in MongoDB")
            overwrite = input("Overwrite? (yes/no): ").strip().lower()
            if overwrite not in ['yes', 'y']:
                print("Cancelled.")
                return None

        # Display comparison
        self.display_comparison(perturbation_data)

        # Prompt for annotation
        annotation = self.prompt_annotation(perturbation_id, annotator_id)

        # Save to MongoDB (primary storage)
        self._save_to_mongodb(annotation)

        # Also save to local file (backup/cache)
        local_path = self.annotations_dir / f"{perturbation_id}.json"
        save_annotation(annotation, self.annotations_dir)

        print(f"\n✅ Annotation saved to MongoDB")
        print(f"   Local backup: {local_path}")

        return annotation

    def _save_to_mongodb(self, annotation: Annotation) -> None:
        """
        Save annotation to MongoDB with proper schema.

        Args:
            annotation: Annotation to save
        """
        # Generate unique annotation_id: perturbation_id + annotator_id
        annotation_id = f"{annotation.perturbation_id}_{annotation.annotator_id}"

        # Get perturbation to extract experiment_id
        pert_record = self.storage.db['perturbations'].find_one({
            'perturbation_id': annotation.perturbation_id
        })
        experiment_id = pert_record['experiment_id'] if pert_record else None

        # Prepare document for MongoDB (matching schema)
        doc = {
            'annotation_id': annotation_id,  # Unique ID
            'perturbation_id': annotation.perturbation_id,
            'experiment_id': experiment_id,
            'annotator_id': annotation.annotator_id,
            'task_success_degradation': annotation.task_success_degradation,
            'subsequent_error_rate': annotation.subsequent_error_rate,
            'tcs_score': annotation.compute_tcs(),
            'notes': annotation.notes,
            'annotated_at': annotation.timestamp,
            'annotation_time_seconds': annotation.annotation_time_seconds
        }

        # Save to MongoDB (upsert to allow re-annotation)
        self.storage.db['annotations'].replace_one(
            {'annotation_id': annotation_id},
            doc,
            upsert=True
        )

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
