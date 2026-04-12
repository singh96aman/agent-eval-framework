"""
CriticalPathScorer: Compute heuristic-based derived annotations.

Pass 2-3 component that computes:
- critical_path_score: Likelihood of affecting task success if corrupted
- affects_final_answer: Whether error impacts final answer
- affects_patch: Whether error impacts code patch
- affects_tool_execution: Whether error impacts tool execution
- recoverable_if_wrong: Whether error can be recovered by later steps
- observable_if_wrong: Whether error would be visible in output
"""

from typing import Any, Dict, List

from src.typing.schema import ProvenanceField, StepRole


class CriticalPathScorer:
    """
    Compute critical path scores and derived annotations using heuristics.

    All derived fields include provenance (source="heuristic", confidence=None).
    """

    # Base criticality scores by step role (0-1 scale)
    ROLE_CRITICALITY = {
        StepRole.PLANNING.value: 0.8,
        StepRole.DECISION.value: 0.75,
        StepRole.TOOL_CALL.value: 0.7,
        StepRole.EXTRACTION.value: 0.85,
        StepRole.REASONING.value: 0.5,
        StepRole.OBSERVATION.value: 0.3,
        StepRole.FINAL_RESPONSE.value: 1.0,
    }

    # Role recovery likelihood (can later steps compensate?)
    ROLE_RECOVERABILITY = {
        StepRole.PLANNING.value: True,  # Can re-plan
        StepRole.DECISION.value: True,  # Can reconsider
        StepRole.TOOL_CALL.value: True,  # Can retry
        StepRole.EXTRACTION.value: False,  # Silent propagation
        StepRole.REASONING.value: True,  # Can reconsider
        StepRole.OBSERVATION.value: True,  # External data
        StepRole.FINAL_RESPONSE.value: False,  # End of trajectory
    }

    # Role observability (will error be visible?)
    ROLE_OBSERVABILITY = {
        StepRole.PLANNING.value: True,  # Plans are explicit
        StepRole.DECISION.value: True,  # Decisions are explicit
        StepRole.TOOL_CALL.value: True,  # Tool output visible
        StepRole.EXTRACTION.value: False,  # Silent wrong value
        StepRole.REASONING.value: True,  # Reasoning is explicit
        StepRole.OBSERVATION.value: True,  # Observations are explicit
        StepRole.FINAL_RESPONSE.value: True,  # Output visible
    }

    def __init__(self):
        pass

    def score_trajectory(
        self,
        typed_steps: List[Dict[str, Any]],
        benchmark: str,
    ) -> List[Dict[str, Any]]:
        """
        Compute derived annotations for all steps in a trajectory.

        Args:
            typed_steps: List of typed step dictionaries
            benchmark: Source benchmark (toolbench, gaia, swebench)

        Returns:
            Updated list with derived annotations
        """
        # First, compute dependency fanout for each step
        fanout_map = self._compute_fanout_map(typed_steps)

        # Find terminal step
        terminal_idx = self._find_terminal_step(typed_steps)

        for step in typed_steps:
            step_index = step["step_index"]

            # Compute critical path score
            critical_score = self._compute_critical_score(
                step, fanout_map.get(step_index, 0), len(typed_steps)
            )
            step["critical_path_score"] = ProvenanceField(
                value=critical_score,
                source="heuristic",
                confidence=None,
            ).to_dict()

            # Compute affects_* fields
            affects_answer, affects_patch, affects_tool = self._compute_affects(
                step, typed_steps, terminal_idx, benchmark
            )
            step["affects_final_answer"] = ProvenanceField(
                value=affects_answer,
                source="heuristic",
                confidence=None,
            ).to_dict()
            step["affects_patch"] = ProvenanceField(
                value=affects_patch,
                source="heuristic",
                confidence=None,
            ).to_dict()
            step["affects_tool_execution"] = ProvenanceField(
                value=affects_tool,
                source="heuristic",
                confidence=None,
            ).to_dict()

            # Compute recoverability
            recoverable = self._compute_recoverable(step, typed_steps)
            step["recoverable_if_wrong"] = ProvenanceField(
                value=recoverable,
                source="heuristic",
                confidence=None,
            ).to_dict()

            # Compute observability
            observable = self._compute_observable(step)
            step["observable_if_wrong"] = ProvenanceField(
                value=observable,
                source="heuristic",
                confidence=None,
            ).to_dict()

        return typed_steps

    def _compute_fanout_map(self, typed_steps: List[Dict[str, Any]]) -> Dict[int, int]:
        """Compute how many downstream steps depend on each step."""
        fanout: Dict[int, int] = {}

        for step in typed_steps:
            step_index = step["step_index"]
            fanout[step_index] = 0

        for step in typed_steps:
            # Count direct dependencies
            for dep_idx in step.get("depends_on_steps", []):
                if dep_idx in fanout:
                    fanout[dep_idx] += 1

            # Also count transitive dependencies (with lower weight)
            for dep_idx in step.get("transitive_depends_on", []):
                if dep_idx in fanout and dep_idx not in step.get(
                    "depends_on_steps", []
                ):
                    fanout[dep_idx] += 0.5

        return fanout

    def _find_terminal_step(self, typed_steps: List[Dict[str, Any]]) -> int:
        """Find the index of the terminal step."""
        for step in typed_steps:
            if step.get("is_terminal_step"):
                return step["step_index"]
        # Default to last step
        return typed_steps[-1]["step_index"] if typed_steps else 0

    def _compute_critical_score(
        self,
        step: Dict[str, Any],
        fanout: float,
        total_steps: int,
    ) -> float:
        """
        Compute critical path score for a step.

        Factors:
        1. Base role criticality
        2. Dependency fanout (more downstream consumers = higher)
        3. Position (early steps more critical for planning)
        4. Terminal step bonus
        """
        step_role = step.get("step_role", "reasoning")
        step_index = step["step_index"]

        # Base score from role
        base = self.ROLE_CRITICALITY.get(step_role, 0.5)

        # Fanout bonus (0-0.2)
        fanout_bonus = min(0.2, fanout * 0.05)

        # Position factor (early steps slightly more critical)
        position_factor = 1.0 - (step_index / max(total_steps, 1)) * 0.1

        # Terminal step bonus
        if step.get("is_terminal_step") or step.get("produces_final_answer"):
            base = max(base, 0.95)

        # Extraction steps with downstream dependencies are very critical
        if step_role == "extraction" and fanout > 0:
            base = max(base, 0.9)

        # Compute final score
        score = base * position_factor + fanout_bonus

        # Clamp to [0, 1]
        return round(min(1.0, max(0.0, score)), 3)

    def _compute_affects(
        self,
        step: Dict[str, Any],
        typed_steps: List[Dict[str, Any]],
        terminal_idx: int,
        benchmark: str,
    ) -> tuple:
        """
        Compute affects_final_answer, affects_patch, affects_tool_execution.

        Returns:
            Tuple of (affects_answer, affects_patch, affects_tool)
        """
        step_index = step["step_index"]
        step_role = step.get("step_role", "")

        # Check if this step is in the dependency chain of terminal step
        is_upstream_of_terminal = False
        for s in typed_steps:
            if s["step_index"] == terminal_idx:
                if step_index in s.get("transitive_depends_on", []):
                    is_upstream_of_terminal = True
                break

        # affects_final_answer
        affects_answer = (
            step.get("produces_final_answer", False)
            or is_upstream_of_terminal
            or step_role in ("extraction", "final_response")
        )

        # affects_patch (SWE-bench specific)
        affects_patch = False
        if benchmark == "swebench":
            affects_patch = (
                step.get("produces_patch", False)
                or step_role in ("extraction", "decision")
                or any(
                    "patch" in (a.get("artifact_type", "") or "")
                    for a in step.get("produced_artifacts", [])
                )
            )
            # Check if downstream of file-related steps
            if not affects_patch:
                for dep_idx in step.get("transitive_depends_on", []):
                    dep_step = next(
                        (s for s in typed_steps if s["step_index"] == dep_idx), None
                    )
                    if dep_step and any(
                        "filepath" in (a.get("artifact_type", "") or "")
                        for a in dep_step.get("produced_artifacts", [])
                    ):
                        affects_patch = True
                        break

        # affects_tool_execution
        affects_tool = (
            step_role == "tool_call"
            or step_role == "planning"
            or any(
                downstream.get("step_role") == "tool_call"
                for downstream in typed_steps
                if step_index in downstream.get("depends_on_steps", [])
            )
        )

        return affects_answer, affects_patch, affects_tool

    def _compute_recoverable(
        self,
        step: Dict[str, Any],
        typed_steps: List[Dict[str, Any]],
    ) -> bool:
        """
        Compute whether an error in this step could be recovered.

        An error is recoverable if:
        1. The step role is inherently recoverable
        2. There are retry-capable steps downstream
        3. The error would be visible (and thus correctable)
        """
        step_role = step.get("step_role", "")
        step_index = step["step_index"]

        # Base recoverability from role
        base_recoverable = self.ROLE_RECOVERABILITY.get(step_role, True)

        # Terminal steps are not recoverable
        if step.get("is_terminal_step"):
            return False

        # Extraction steps are typically not recoverable (silent propagation)
        if step_role == "extraction":
            return False

        # Check if there are similar tool calls downstream (retry pattern)
        tool_name = step.get("tool_name", "")
        if tool_name:
            for downstream in typed_steps:
                if downstream["step_index"] > step_index:
                    if downstream.get("tool_name") == tool_name:
                        return True

        return base_recoverable

    def _compute_observable(self, step: Dict[str, Any]) -> bool:
        """
        Compute whether an error in this step would be observable.

        An error is observable if:
        1. The step role is inherently observable
        2. The step produces visible output
        3. The error would cause downstream failures
        """
        step_role = step.get("step_role", "")

        # Base observability from role
        base_observable = self.ROLE_OBSERVABILITY.get(step_role, True)

        # Tool calls with output are observable
        if step.get("observation"):
            return True

        # Final response is always observable
        if step.get("produces_final_answer") or step.get("produces_patch"):
            return True

        # Extraction steps are often NOT observable (wrong value propagates silently)
        if step_role == "extraction":
            return False

        return base_observable


def get_criticality_summary(typed_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary of criticality scores for a trajectory.

    Args:
        typed_steps: List of typed steps with critical_path_score

    Returns:
        Summary dict with statistics
    """
    scores = [s.get("critical_path_score", {}).get("value", 0) for s in typed_steps]

    if not scores:
        return {"mean": 0, "max": 0, "min": 0, "high_criticality_count": 0}

    high_crit = [s for s in scores if s >= 0.7]

    return {
        "mean": round(sum(scores) / len(scores), 3),
        "max": max(scores),
        "min": min(scores),
        "high_criticality_count": len(high_crit),
        "high_criticality_ratio": round(len(high_crit) / len(scores), 3),
    }
