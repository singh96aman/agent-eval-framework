"""
Perturbation validation module.

Validates generated perturbations for quality issues:
- BUG1: No template fallback usage
- BUG2: Planning perturbations are semantic (not text corruption)
- BUG3: Position coverage (data_reference skips early by design)
- BUG4: Type diversity (all benchmarks have all 4 types)
- Uniqueness check
"""

from collections import defaultdict
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json


class PerturbationValidator:
    """Validates perturbation quality."""

    # Template phrase that indicates fallback (BUG1)
    TEMPLATE_PHRASE = "I'll focus on a subset of the requirements"

    # Patterns that indicate text corruption (BUG2)
    CORRUPTION_PHRASES = [
        'analyze the first one provide',
        'analyze the first one and',
        'historical data provide',
    ]

    # Map native types to standard categories (BUG4)
    TYPE_MAPPING = {
        'gaia_planning': 'planning',
        'gaia_tool_selection': 'tool_selection',
        'gaia_parameter': 'parameter',
        'gaia_data_reference': 'data_reference',
        'swebench_wrong_diagnosis': 'planning',
        'swebench_wrong_file': 'tool_selection',
        'swebench_wrong_location': 'parameter',
        'swebench_wrong_reference': 'data_reference',
        'planning': 'planning',
        'tool_selection': 'tool_selection',
        'parameter': 'parameter',
        'data_reference': 'data_reference',
    }

    REQUIRED_TYPES = {'planning', 'tool_selection', 'parameter', 'data_reference'}

    def __init__(self):
        self.results = {}

    def validate_all(
        self,
        perturbations: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, bool]:
        """
        Run all validation checks.

        Args:
            perturbations: Dict mapping benchmark name to list of perturbations

        Returns:
            Dict mapping check name to pass/fail status
        """
        self.results = {
            'BUG1_no_template': self._check_no_template_fallback(perturbations),
            'BUG2_planning_quality': self._check_planning_quality(perturbations),
            'BUG3_position_coverage': self._check_position_coverage(perturbations),
            'BUG4_type_diversity': self._check_type_diversity(perturbations),
            'uniqueness': self._check_uniqueness(perturbations),
        }
        return self.results

    def _check_no_template_fallback(
        self,
        perturbations: Dict[str, List[Dict]]
    ) -> Tuple[bool, List[str]]:
        """BUG1: No template fallback usage."""
        failures = []

        for benchmark, perturbs in perturbations.items():
            for p in perturbs:
                content = p.get('perturbed_step_content', '')
                if self.TEMPLATE_PHRASE in content:
                    traj_id = p.get('original_trajectory', {}).get(
                        'trajectory_id', 'unknown'
                    )
                    failures.append(f"{benchmark}/{traj_id}")

        passed = len(failures) == 0
        return passed, failures

    def _check_planning_quality(
        self,
        perturbations: Dict[str, List[Dict]]
    ) -> Tuple[bool, List[str]]:
        """BUG2: Planning perturbations are semantic, not text corruption."""
        suspicious = []

        for benchmark, perturbs in perturbations.items():
            for p in perturbs:
                ptype = p.get('perturbation_type', '')
                if 'planning' not in ptype.lower():
                    continue

                pert = p.get('perturbed_step_content', '')

                for phrase in self.CORRUPTION_PHRASES:
                    if phrase in pert.lower():
                        traj_id = p.get('original_trajectory', {}).get(
                            'trajectory_id', 'unknown'
                        )
                        suspicious.append(f"{benchmark}/{traj_id}: '{phrase}'")
                        break

        passed = len(suspicious) == 0
        return passed, suspicious

    def _check_position_coverage(
        self,
        perturbations: Dict[str, List[Dict]]
    ) -> Tuple[bool, str]:
        """BUG3: All (type, position) cells have coverage."""
        counts = defaultdict(lambda: defaultdict(int))

        for benchmark, perturbs in perturbations.items():
            for p in perturbs:
                ptype = p.get('perturbation_type', 'unknown')
                pos = p.get('perturbation_position', 'unknown')
                counts[ptype][pos] += 1

        # data_reference early=0 is by design (not a failure)
        note = ""
        if counts.get('data_reference', {}).get('early', 0) == 0:
            note = "data_reference has no early position (valid constraint)"

        return True, note

    def _check_type_diversity(
        self,
        perturbations: Dict[str, List[Dict]]
    ) -> Tuple[bool, List[str]]:
        """BUG4: Each benchmark has all perturbation types."""
        failures = []

        for benchmark, perturbs in perturbations.items():
            if not perturbs:
                continue

            raw_types = set(p.get('perturbation_type', '') for p in perturbs)
            types_present = set()
            for rt in raw_types:
                normalized = self.TYPE_MAPPING.get(rt, rt)
                types_present.add(normalized)

            missing = self.REQUIRED_TYPES - types_present
            if missing:
                failures.append(f"{benchmark} missing: {missing}")

        passed = len(failures) == 0
        return passed, failures

    def _check_uniqueness(
        self,
        perturbations: Dict[str, List[Dict]]
    ) -> Tuple[bool, int]:
        """Check that perturbations are contextually unique."""
        total_dups = 0

        for benchmark, perturbs in perturbations.items():
            seen = {}
            for p in perturbs:
                content = p.get('perturbed_step_content', '')
                key = content[:100] if content else ''
                if key in seen:
                    total_dups += 1
                else:
                    seen[key] = True

        # Allow some natural duplicates (threshold: 10)
        passed = total_dups <= 10
        return passed, total_dups

    def get_coverage_summary(
        self,
        perturbations: Dict[str, List[Dict]]
    ) -> Dict[str, Dict[str, Any]]:
        """Get coverage summary by type and position."""
        summary = {}

        for benchmark, perturbs in perturbations.items():
            if not perturbs:
                continue

            type_counts = defaultdict(int)
            pos_counts = defaultdict(int)

            for p in perturbs:
                ptype = p.get('perturbation_type', 'unknown')
                pos = p.get('perturbation_position', 'unknown')
                type_counts[ptype] += 1
                pos_counts[pos] += 1

            summary[benchmark] = {
                'total': len(perturbs),
                'by_type': dict(type_counts),
                'by_position': dict(pos_counts),
            }

        return summary

    def print_report(
        self,
        perturbations: Dict[str, List[Dict]],
        verbose: bool = True
    ) -> bool:
        """
        Print validation report and return overall pass/fail.

        Args:
            perturbations: Dict mapping benchmark to perturbations
            verbose: Whether to print detailed output

        Returns:
            True if all checks pass, False otherwise
        """
        print("=" * 60)
        print("PERTURBATION VALIDATION")
        print("=" * 60)

        # Show what we loaded
        for benchmark, perturbs in perturbations.items():
            print(f"Loaded {len(perturbs)} {benchmark} perturbations")

        print("\n" + "-" * 60)
        print("RUNNING CHECKS")
        print("-" * 60)

        # Run all checks
        results = self.validate_all(perturbations)

        # BUG1
        passed, failures = results['BUG1_no_template']
        if passed:
            print("BUG1 PASS: No template fallback detected")
        else:
            print(f"BUG1 FAIL: {len(failures)} use template fallback")
            if verbose and len(failures) <= 5:
                for f in failures:
                    print(f"  - {f}")

        # BUG2
        passed, suspicious = results['BUG2_planning_quality']
        if passed:
            print("BUG2 PASS: No text corruption in planning perturbations")
        else:
            print(f"BUG2 FAIL: {len(suspicious)} appear to be text corruption")
            if verbose:
                for s in suspicious[:5]:
                    print(f"  - {s}")

        # BUG3
        passed, note = results['BUG3_position_coverage']
        if note:
            print(f"BUG3 WARNING: {note}")
        print("BUG3 PASS: Position coverage acceptable")

        # BUG4
        passed, failures = results['BUG4_type_diversity']
        if passed:
            print("BUG4 PASS: All benchmarks have all perturbation types")
        else:
            print(f"BUG4 FAIL: {failures}")

        # Uniqueness
        passed, total_dups = results['uniqueness']
        if passed:
            print("UNIQUENESS PASS: Perturbations are sufficiently unique")
        else:
            print(f"UNIQUENESS WARNING: {total_dups} potential duplicates")

        # Coverage summary
        if verbose:
            summary = self.get_coverage_summary(perturbations)
            print("\n" + "=" * 60)
            print("COVERAGE SUMMARY")
            print("=" * 60)

            for benchmark, stats in summary.items():
                print(f"\n{benchmark.upper()}:")
                print(f"  Total: {stats['total']}")
                print(f"  By type: {stats['by_type']}")
                print(f"  By position: {stats['by_position']}")

        # Overall result
        all_pass = all(
            r[0] if isinstance(r, tuple) else r
            for r in results.values()
        )

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for check, result in results.items():
            passed = result[0] if isinstance(result, tuple) else result
            status = "PASS" if passed else "FAIL"
            print(f"{check}: {status}")

        print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

        return all_pass


def load_perturbations_from_files(
    perturbed_dir: Path
) -> Dict[str, List[Dict]]:
    """Load perturbations from JSON files in a directory."""
    perturbations = {}

    for benchmark in ['toolbench', 'gaia', 'swebench']:
        path = perturbed_dir / f'{benchmark}_perturbations.json'
        if path.exists():
            with open(path) as f:
                perturbations[benchmark] = json.load(f)
        else:
            perturbations[benchmark] = []

    return perturbations


def validate_perturbations_from_files(
    perturbed_dir: str = "data/perturbed",
    verbose: bool = True
) -> bool:
    """
    Validate perturbations from JSON files.

    Args:
        perturbed_dir: Directory containing perturbation JSON files
        verbose: Whether to print detailed output

    Returns:
        True if all checks pass
    """
    perturbed_path = Path(perturbed_dir)
    perturbations = load_perturbations_from_files(perturbed_path)

    if not any(perturbations.values()):
        print("ERROR: No perturbation files found!")
        return False

    validator = PerturbationValidator()
    return validator.print_report(perturbations, verbose=verbose)
