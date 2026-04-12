"""
ArtifactTracker: Track artifacts produced and consumed by each step.

Identifies what each step produces (search results, file paths, patches, etc.)
and what artifacts from previous steps it consumes.
"""

import re
from typing import Any, Dict, List, Set

from src.typing.schema import Artifact, ArtifactType


class ArtifactTracker:
    """
    Track artifact production and consumption across trajectory steps.

    Artifacts are typed outputs that can be consumed by later steps:
    - search_results: Results from search/query operations
    - filepath: File paths discovered or referenced
    - patch: Code changes (SWE-bench)
    - line_number: Specific line references
    - numeric_answer: Computed numeric values
    - code_snippet: Code fragments
    - diagnosis: Bug diagnoses or analysis conclusions
    - api_response: API call responses
    - text: Generic text outputs
    """

    # Read-only tools should NOT produce artifacts (they only observe)
    READ_ONLY_TOOLS = {
        "view",
        "file_view",
        "read",
        "open",
        "cat",
        "search",
        "find",
        "list",
        "ls",
        "tree",
        "grep",
        "rg",
        "ag",  # Search tools
        "head",
        "tail",
        "less",
        "more",
    }

    # Tool name patterns -> artifact type mappings
    # Note: Read-only tools are excluded; they don't produce artifacts
    TOOL_TO_ARTIFACT = {
        # Edit tools - only these produce PATCH artifacts
        "edit": ArtifactType.PATCH,
        "str_replace": ArtifactType.PATCH,
        "str_replace_editor": ArtifactType.PATCH,
        "patch": ArtifactType.PATCH,
        "write": ArtifactType.PATCH,
        "file_edit": ArtifactType.PATCH,
        # Run tools
        "bash": ArtifactType.TEXT,
        "run": ArtifactType.TEXT,
        "execute": ArtifactType.TEXT,
        "test": ArtifactType.TEXT,
    }

    def __init__(self):
        # Pattern for identifying file paths in content
        self._filepath_re = re.compile(r"(?:/[\w\.-]+)+(?:\.\w+)?")
        # Pattern for line numbers
        self._line_re = re.compile(r"line\s*(\d+)", re.IGNORECASE)
        # Pattern for numeric values
        self._numeric_re = re.compile(r"\((\d+(?:\.\d+)?)\)")

    def track_artifacts(
        self,
        typed_steps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Track artifacts for all steps in a trajectory.

        Args:
            typed_steps: List of partially typed step dictionaries

        Returns:
            Updated list with produced_artifacts and consumed_artifacts
        """
        # Map of artifact name -> step index that produced it
        artifact_registry: Dict[str, int] = {}

        for step in typed_steps:
            step_index = step["step_index"]

            # Determine produced artifacts
            produced = self._identify_produced_artifacts(step, step_index)
            step["produced_artifacts"] = produced

            # Register artifacts
            for artifact in produced:
                artifact_registry[artifact["name"]] = step_index

            # Determine consumed artifacts
            consumed = self._identify_consumed_artifacts(
                step, artifact_registry, typed_steps
            )
            step["consumed_artifacts"] = consumed

        return typed_steps

    def _identify_produced_artifacts(
        self, step: Dict[str, Any], step_index: int
    ) -> List[Dict[str, Any]]:
        """
        Identify artifacts produced by a step.

        Args:
            step: Step dictionary
            step_index: Index of the step

        Returns:
            List of Artifact dictionaries
        """
        artifacts: List[Dict[str, Any]] = []

        tool_name = (step.get("tool_name") or "").lower()
        step_role = step.get("step_role", "")
        content = step.get("raw_text", "") or ""
        tool_output = step.get("observation") or ""
        tool_args = step.get("tool_arguments") or {}

        # Skip artifact production for read-only tools
        # Read-only operations only observe data, they don't produce artifacts
        is_read_only = any(ro_tool in tool_name for ro_tool in self.READ_ONLY_TOOLS)
        if is_read_only:
            # Read-only tools don't produce artifacts (they only observe)
            return artifacts

        # 1. Check produces_patch flag first (authoritative for patch artifacts)
        if step.get("produces_patch"):
            artifacts.append(
                Artifact(
                    name=f"patch_{step_index}",
                    artifact_type=ArtifactType.PATCH.value,
                ).to_dict()
            )

        # 2. Tool-based artifact production (only for non-patch tools)
        if not artifacts:
            for pattern, artifact_type in self.TOOL_TO_ARTIFACT.items():
                if pattern in tool_name:
                    # Skip PATCH type here since we use produces_patch flag above
                    if artifact_type == ArtifactType.PATCH and not step.get(
                        "produces_patch"
                    ):
                        continue
                    artifact = Artifact(
                        name=f"{artifact_type.value}_{step_index}",
                        artifact_type=artifact_type.value,
                    )
                    artifacts.append(artifact.to_dict())
                    break

        # 3. API response for tool calls with output
        if not artifacts and step_role == "tool_call" and tool_output:
            artifacts.append(
                Artifact(
                    name=f"api_response_{step_index}",
                    artifact_type=ArtifactType.API_RESPONSE.value,
                ).to_dict()
            )

        # 4. File path extraction (only for non-read-only tools that produce files)
        if "path" in tool_args or "file" in tool_args:
            path_val = tool_args.get("path") or tool_args.get("file")
            # Only add filepath artifact if this is a write/edit operation
            if path_val and step.get("produces_patch"):
                artifacts.append(
                    Artifact(
                        name=f"filepath_{step_index}",
                        artifact_type=ArtifactType.FILEPATH.value,
                    ).to_dict()
                )

        # Line number extraction
        if self._line_re.search(content):
            artifacts.append(
                Artifact(
                    name=f"line_number_{step_index}",
                    artifact_type=ArtifactType.LINE_NUMBER.value,
                ).to_dict()
            )

        # Numeric answer (extraction steps)
        if step_role == "extraction" and step.get("extracted_value") is not None:
            val_type = step.get("value_type", "")
            if val_type in ("integer", "float"):
                artifacts.append(
                    Artifact(
                        name=f"numeric_answer_{step_index}",
                        artifact_type=ArtifactType.NUMERIC_ANSWER.value,
                    ).to_dict()
                )

        # Diagnosis (reasoning about bugs)
        if step_role in ("reasoning", "extraction"):
            diagnosis_patterns = ["bug", "issue", "problem", "error", "fix", "cause"]
            if any(p in content.lower() for p in diagnosis_patterns):
                artifacts.append(
                    Artifact(
                        name=f"diagnosis_{step_index}",
                        artifact_type=ArtifactType.DIAGNOSIS.value,
                    ).to_dict()
                )

        # Final answer production
        if step.get("produces_final_answer"):
            artifacts.append(
                Artifact(
                    name=f"final_answer_{step_index}",
                    artifact_type=ArtifactType.TEXT.value,
                ).to_dict()
            )

        # Default: if tool produced output, create generic artifact
        if not artifacts and tool_output:
            artifacts.append(
                Artifact(
                    name=f"output_{step_index}",
                    artifact_type=ArtifactType.TEXT.value,
                ).to_dict()
            )

        return artifacts

    def _identify_consumed_artifacts(
        self,
        step: Dict[str, Any],
        artifact_registry: Dict[str, int],
        all_steps: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify artifacts consumed by a step.

        Args:
            step: Step dictionary
            artifact_registry: Map of artifact name -> producing step index
            all_steps: All steps for reference lookup

        Returns:
            List of artifact names consumed
        """
        consumed: Set[str] = set()

        step_index = step["step_index"]
        content = step.get("raw_text", "") or ""
        depends_on = step.get("depends_on_steps", [])

        # 1. Consume artifacts from explicit dependencies
        for dep_index in depends_on:
            dep_step = next(
                (s for s in all_steps if s["step_index"] == dep_index), None
            )
            if dep_step:
                for artifact in dep_step.get("produced_artifacts", []):
                    consumed.add(artifact["name"])

        # 2. Check for references to specific artifact types in content

        # File path references
        if self._filepath_re.search(content):
            for name, idx in artifact_registry.items():
                if idx < step_index and "filepath" in name:
                    consumed.add(name)

        # Numeric value references
        if self._numeric_re.search(content):
            for name, idx in artifact_registry.items():
                if idx < step_index and "numeric_answer" in name:
                    consumed.add(name)

        # 3. Check source_step for extraction steps
        source_step = step.get("source_step")
        if source_step is not None:
            for name, idx in artifact_registry.items():
                if idx == source_step:
                    consumed.add(name)

        return sorted(consumed)


def get_artifact_type_from_tool(tool_name: str) -> str:
    """
    Infer artifact type from tool name.

    Args:
        tool_name: Name of the tool

    Returns:
        ArtifactType value string
    """
    tool_lower = (tool_name or "").lower()

    if any(p in tool_lower for p in ["search", "query", "find", "grep"]):
        return ArtifactType.SEARCH_RESULTS.value

    if any(p in tool_lower for p in ["view", "read", "open", "cat"]):
        return ArtifactType.CODE_SNIPPET.value

    if any(p in tool_lower for p in ["edit", "replace", "patch", "write"]):
        return ArtifactType.PATCH.value

    if "api" in tool_lower:
        return ArtifactType.API_RESPONSE.value

    return ArtifactType.TEXT.value
