"""
DependencyAnalyzer: Build direct and transitive dependency graphs between steps.

Analyzes step content and artifacts to determine:
- Direct dependencies (which steps this step directly uses)
- Dependency edges with types (uses_observation_from, uses_extracted_value_from, etc.)
- Transitive dependencies (full closure of all upstream dependencies)
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.typing.schema import DependencyEdge, DependencyType


class DependencyAnalyzer:
    """
    Analyze dependencies between trajectory steps.

    Builds both direct dependency edges (with types and reasons)
    and transitive dependency closure for each step.
    """

    # Patterns that reference previous steps
    REFERENCE_PATTERNS = [
        r"(?:previous|prior|earlier|last|above)\s+(?:step|action|result|output)",
        r"(?:step\s*\d+|based on|from the|using the)",
        r"(?:the\s+)?(?:api|function|tool)\s+(?:call|result|response|output)",
    ]

    # Patterns indicating use of extracted values
    EXTRACTION_PATTERNS = [
        r"(?:extract|found|got|obtained|retrieved)\s+(?:value|number|result)",
        r"(?:the\s+)?(?:answer|value|result)\s+(?:is|was|equals?)",
    ]

    # Patterns indicating follow-up to a plan
    PLAN_PATTERNS = [
        r"(?:as\s+)?(?:planned|intended|mentioned)",
        r"(?:next|continuing|following)",
    ]

    # Invalid file path PREFIXES - XML tag artifacts that should never be paths
    # These are filtered even with additional segments
    INVALID_PATH_PREFIXES = {
        "/function", "/parameter",  # XML tag artifacts from SWE-bench format
        "/encrypted-tbn0.gstatic.com",  # URL artifacts
        "/images", "/www", "/http", "/https",  # URL-like patterns
    }

    # Paths that are only invalid if they're the EXACT match (no additional segments)
    # e.g., /testbed is invalid but /testbed/src/main.py is valid
    INVALID_EXACT_PATHS = {
        "/testbed", "/usr/bin/env",
    }

    # Entity blocklist - parser artifacts and generic tokens
    ENTITY_BLOCKLIST = {
        # XML tag artifacts
        "/function", "/parameter", "function", "parameter",
        # Generic tool words
        "file_edit", "file_view", "str_replace_editor", "edit", "file", "view",
        "search_code", "view_file", "bash", "submit",
        # Generic paths
        "/testbed", "testbed",
        # Noise
        "raw", "content", "command", "path", "old_str", "new_str",
        "result", "response", "output", "input", "value", "data",
        "error", "success", "true", "false",
    }

    def __init__(self):
        self._reference_re = re.compile(
            "|".join(self.REFERENCE_PATTERNS), re.IGNORECASE
        )
        self._extraction_re = re.compile(
            "|".join(self.EXTRACTION_PATTERNS), re.IGNORECASE
        )
        self._plan_re = re.compile("|".join(self.PLAN_PATTERNS), re.IGNORECASE)
        # File path pattern for dependency detection
        self._filepath_re = re.compile(r"(?:/[\w\.-]+)+(?:\.\w+)?")

    def analyze_dependencies(
        self,
        typed_steps: List[Dict[str, Any]],
        trajectory_entities: Dict[int, Set[str]],
    ) -> List[Dict[str, Any]]:
        """
        Analyze dependencies for all steps in a trajectory.

        Args:
            typed_steps: List of partially typed step dictionaries
            trajectory_entities: Map of step_index -> entities for that step

        Returns:
            Updated list of typed steps with dependency fields
        """
        # First pass: build direct dependencies
        for i, step in enumerate(typed_steps):
            step_index = step["step_index"]
            direct_deps, edges = self._find_direct_dependencies(
                step, typed_steps[:i], trajectory_entities
            )
            step["depends_on_steps"] = sorted(direct_deps)
            # Convert DependencyEdge objects to dicts for JSON serialization
            step["dependency_edges"] = [e.to_dict() for e in edges]

        # Second pass: compute transitive closure
        dep_graph = self._build_dependency_graph(typed_steps)
        for step in typed_steps:
            step_index = step["step_index"]
            step["transitive_depends_on"] = sorted(
                self._compute_transitive_closure(step_index, dep_graph)
            )

        return typed_steps

    def _find_direct_dependencies(
        self,
        step: Dict[str, Any],
        previous_steps: List[Dict[str, Any]],
        trajectory_entities: Dict[int, Set[str]],
    ) -> Tuple[Set[int], List[DependencyEdge]]:
        """
        Find direct dependencies for a single step.

        Priority order (highest confidence first):
        1. Artifact flow: step consumes artifact produced by previous step
        2. Textual reference: explicit "step N", "previous result", tool mention
        3. Extracted value: step uses value extracted in previous step
        4. File/path reuse: same file path referenced in both steps
        5. Entity overlap: >=3 specific entities match (lowest confidence)

        Args:
            step: Current step
            previous_steps: All preceding steps
            trajectory_entities: Entity sets for all steps

        Returns:
            Tuple of (set of dependency indices, list of DependencyEdge objects)
        """
        direct_deps: Set[int] = set()
        edges: List[DependencyEdge] = []

        if not previous_steps:
            return direct_deps, edges

        step_index = step["step_index"]
        content = step.get("raw_text", "") or ""
        step_role = step.get("step_role", "")
        consumed = set(step.get("consumed_artifacts", []))
        step_entities = trajectory_entities.get(step_index, set())
        tool_args = step.get("tool_arguments") or {}

        # Extract file paths from current step for file reuse detection
        step_files = self._extract_file_paths(content, tool_args)

        # Check each previous step
        for prev in previous_steps:
            prev_index = prev["step_index"]

            # Skip if already found dependency to this step
            if prev_index in direct_deps:
                continue

            prev_role = prev.get("step_role", "")
            prev_artifacts = {a["name"] for a in prev.get("produced_artifacts", [])}
            prev_entities = trajectory_entities.get(prev_index, set())
            prev_content = prev.get("raw_text", "") or ""
            prev_tool_args = prev.get("tool_arguments") or {}
            prev_files = self._extract_file_paths(prev_content, prev_tool_args)

            # Priority 1: Artifact flow (highest confidence)
            matched_artifacts = consumed & prev_artifacts
            if matched_artifacts:
                artifact_name = list(matched_artifacts)[0]
                direct_deps.add(prev_index)
                edges.append(DependencyEdge(
                    type=DependencyType.USES_OBSERVATION_FROM.value,
                    source_step=prev_index,
                    reason=f"Consumes artifact from step {prev_index}",
                    evidence=f"artifact:{artifact_name}",
                ))
                continue

            # Priority 2: Textual reference
            ref_info = self._find_textual_reference(content, prev_index, prev)
            if ref_info:
                dep_type = self._infer_dependency_type(content, prev_role)
                direct_deps.add(prev_index)
                edges.append(DependencyEdge(
                    type=dep_type,
                    source_step=prev_index,
                    reason=f"Textual reference to step {prev_index}",
                    evidence=f"textual:{ref_info}",
                ))
                continue

            # Priority 3: Extracted value usage
            if prev_role == "extraction" and step.get("source_step") == prev_index:
                extracted_val = prev.get("extracted_value")
                direct_deps.add(prev_index)
                edges.append(DependencyEdge(
                    type=DependencyType.USES_EXTRACTED_VALUE_FROM.value,
                    source_step=prev_index,
                    reason=f"Uses extracted value from step {prev_index}",
                    evidence=f"extracted:{extracted_val}" if extracted_val else None,
                ))
                continue

            # Priority 4: File/path reuse
            shared_files = step_files & prev_files
            if shared_files:
                shared_file = list(shared_files)[0]
                direct_deps.add(prev_index)
                edges.append(DependencyEdge(
                    type=DependencyType.ACTS_ON_ENTITY_FROM.value,
                    source_step=prev_index,
                    reason=f"References same file as step {prev_index}",
                    evidence=f"file_path:{shared_file}",
                ))
                continue

            # Priority 5: Entity overlap (lowest confidence, strict threshold)
            # Only use domain-bearing entities, not parser artifacts
            shared_entities = step_entities & prev_entities

            # Classify entities into structured evidence
            structured_evidence = []
            for entity in shared_entities:
                entity_lower = entity.lower()

                # Skip blocklisted entities
                if entity_lower in self.ENTITY_BLOCKLIST:
                    continue

                # Skip invalid path prefixes
                if any(entity.startswith(p) for p in self.INVALID_PATH_PREFIXES):
                    continue

                # Classify as file_path if it looks like a real path
                if "/" in entity and len(entity) > 12:
                    # Must have at least 2 segments to be meaningful
                    segments = [s for s in entity.split("/") if s]
                    if len(segments) >= 2:
                        structured_evidence.append(f"file_path:{entity}")
                        continue

                # Classify as symbol if it looks like a function/class name
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', entity) and len(entity) > 5:
                    # Skip generic words
                    if entity_lower not in {"function", "class", "method", "variable"}:
                        structured_evidence.append(f"symbol:{entity}")
                        continue

            # Require at least 2 structured evidence items
            if len(structured_evidence) >= 2:
                evidence_str = "|".join(sorted(structured_evidence)[:3])
                direct_deps.add(prev_index)
                edges.append(DependencyEdge(
                    type=DependencyType.ACTS_ON_ENTITY_FROM.value,
                    source_step=prev_index,
                    reason=f"Shares {len(structured_evidence)} domain entities",
                    evidence=evidence_str,
                ))
                continue

        # NOTE: Removed sequential tool_call fallback - dependencies must be grounded

        return direct_deps, edges

    def _extract_file_paths(
        self, content: str, tool_args: Dict[str, Any]
    ) -> Set[str]:
        """
        Extract file paths from content and tool arguments.

        Filters out XML tag artifacts (/function, /parameter) and generic paths.
        """
        paths: Set[str] = set()

        def _is_valid_path(path: str) -> bool:
            """Check if a path is a valid file reference, not an XML artifact."""
            # Must be long enough to be meaningful
            if len(path) < 8:
                return False
            # Must not start with invalid prefixes (XML artifacts, URLs)
            for prefix in self.INVALID_PATH_PREFIXES:
                if path.startswith(prefix):
                    return False
            # Must not be an exact match of invalid paths
            if path in self.INVALID_EXACT_PATHS:
                return False
            # Should have at least 2 path segments (beyond root)
            # e.g., /testbed/file.py has 2 segments after splitting on /
            segments = [s for s in path.split("/") if s]
            if len(segments) < 2:
                return False
            return True

        # From tool arguments (more trusted, but still validate)
        if "path" in tool_args and isinstance(tool_args["path"], str):
            path = tool_args["path"]
            if _is_valid_path(path):
                paths.add(path)
        if "file" in tool_args and isinstance(tool_args["file"], str):
            path = tool_args["file"]
            if _is_valid_path(path):
                paths.add(path)

        # From content using regex (needs validation)
        for match in self._filepath_re.finditer(content):
            path = match.group(0)
            if _is_valid_path(path):
                paths.add(path)

        return paths

    def _find_textual_reference(
        self, content: str, prev_index: int, prev_step: Dict[str, Any]
    ) -> Optional[str]:
        """
        Find textual reference to a previous step.

        Only returns concrete evidence (step numbers, tool names, file names).
        Does NOT return generic "previous_result" - that's too noisy.

        Returns reference type string if found, None otherwise.
        """
        # Check for explicit step number reference
        if re.search(rf"step\s*{prev_index}", content, re.IGNORECASE):
            return f"step_{prev_index}"

        # Check if mentions the previous tool by name (word boundary match)
        prev_tool = prev_step.get("tool_name", "")
        if prev_tool and len(prev_tool) > 3:
            # Use word boundary to avoid substring matches
            if re.search(rf"\b{re.escape(prev_tool)}\b", content, re.IGNORECASE):
                return f"tool:{prev_tool}"

        # Check for file name reference from previous step
        prev_files = self._extract_file_paths(
            prev_step.get("raw_text", "") or "",
            prev_step.get("tool_arguments") or {}
        )
        for prev_file in prev_files:
            # Extract just the filename (last component)
            filename = prev_file.split("/")[-1]
            if filename and len(filename) > 3:
                # Check if filename appears as distinct word in content
                if re.search(rf"\b{re.escape(filename)}\b", content, re.IGNORECASE):
                    return f"file:{filename}"

        # NO FALLBACK - do not use generic "previous_result"
        # If no concrete evidence exists, return None and let other
        # priority mechanisms (artifact flow, file reuse, entity overlap) handle it
        return None

    def _infer_dependency_type(self, content: str, prev_role: str) -> str:
        """Infer the type of dependency from content and previous step role."""
        if self._extraction_re.search(content):
            return DependencyType.USES_EXTRACTED_VALUE_FROM.value

        if prev_role == "planning":
            return DependencyType.FOLLOWS_PLAN_FROM.value

        if self._plan_re.search(content):
            return DependencyType.FOLLOWS_PLAN_FROM.value

        # Default to observation dependency
        return DependencyType.USES_OBSERVATION_FROM.value

    def _build_dependency_graph(
        self, typed_steps: List[Dict[str, Any]]
    ) -> Dict[int, Set[int]]:
        """Build adjacency list representation of dependency graph."""
        graph: Dict[int, Set[int]] = defaultdict(set)

        for step in typed_steps:
            step_index = step["step_index"]
            for dep in step.get("depends_on_steps", []):
                graph[step_index].add(dep)

        return graph

    def _compute_transitive_closure(
        self, step_index: int, graph: Dict[int, Set[int]]
    ) -> Set[int]:
        """
        Compute transitive closure of dependencies for a step.

        Uses BFS to find all upstream dependencies.
        """
        visited: Set[int] = set()
        queue = list(graph.get(step_index, set()))

        while queue:
            dep = queue.pop(0)
            if dep in visited:
                continue
            visited.add(dep)
            # Add dependencies of this dependency
            queue.extend(d for d in graph.get(dep, set()) if d not in visited)

        return visited

    def get_dependency_fanout(
        self, step_index: int, typed_steps: List[Dict[str, Any]]
    ) -> int:
        """
        Count how many downstream steps depend on this step.

        Used for critical path scoring.
        """
        count = 0
        for step in typed_steps:
            if step_index in step.get("transitive_depends_on", []):
                count += 1
        return count
