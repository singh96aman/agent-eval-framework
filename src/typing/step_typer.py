"""
StepTyper: Classify step_role, terminal flags, and extraction fields.

Pass 1 component that analyzes raw step content to determine:
- step_role (planning, tool_call, observation, reasoning, extraction, decision, final_response)
- Terminal flags (is_terminal_step, produces_final_answer, produces_patch)
- Extraction fields (extracted_value, value_type, source_step, source_description)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.typing.schema import ExtractionProvenance, StepRole, ValueType


class StepTyper:
    """Classifies step roles and extracts structured information from raw steps."""

    # Tool names that indicate final response
    FINISH_TOOLS = {"finish", "submit", "give_answer", "submit_answer", "final_answer"}

    # Tool names that produce patches (SWE-bench)
    PATCH_TOOLS = {
        "str_replace_editor",
        "file_edit",
        "patch",
        "apply_patch",
        "edit_file",
        "write",
    }

    # Tool names that are read-only (never produce patches)
    READ_ONLY_TOOLS = {
        "view",
        "view_file",
        "file_view",
        "read",
        "open",
        "cat",
        "search",
        "find",
        "list",
        "ls",
        "grep",
    }

    # Commands that produce patches (for str_replace_editor)
    PATCH_COMMANDS = {"str_replace", "create", "insert"}

    # Patterns for extraction detection
    EXTRACTION_PATTERNS = [
        # Numbers in parentheses often indicate extracted values
        r"(?:answer|result|value|found|is|equals?|=)\s*[:=]?\s*\(?(\d+(?:\.\d+)?)\)?",
        # Line numbers
        r"line\s*(\d+)",
        # File paths
        r"(?:file|path)\s*[:=]?\s*['\"]?([/\w\.-]+)['\"]?",
    ]

    # Patterns that suggest planning
    PLANNING_PATTERNS = [
        r"(?:i will|let me|first|next|then|plan|approach|strategy|step\s*\d+)",
        r"(?:need to|should|must|going to)",
        r"(?:break down|decompose|divide)",
    ]

    # Patterns that suggest reasoning
    REASONING_PATTERNS = [
        r"(?:because|therefore|thus|hence|so|since|as a result)",
        r"(?:this means|this indicates|this suggests)",
        r"(?:compare|calculate|compute|determine|analyze|evaluate)",
    ]

    # Patterns that suggest decision
    DECISION_PATTERNS = [
        r"(?:i choose|i select|decided to|choosing|selecting)",
        r"(?:the best|the correct|the right|should use)",
    ]

    # Tools that indicate execution (not read/write)
    EXECUTE_TOOLS = {"bash", "run_tests", "pytest", "python", "execute"}

    def __init__(self):
        # Compile patterns for efficiency
        self._planning_re = re.compile("|".join(self.PLANNING_PATTERNS), re.IGNORECASE)
        self._reasoning_re = re.compile(
            "|".join(self.REASONING_PATTERNS), re.IGNORECASE
        )
        self._decision_re = re.compile("|".join(self.DECISION_PATTERNS), re.IGNORECASE)
        self._extraction_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.EXTRACTION_PATTERNS
        ]

    def classify_step_role(
        self,
        step: Dict[str, Any],
        step_index: int,
        total_steps: int,
        benchmark: str,
    ) -> str:
        """
        Classify the role of a step.

        Args:
            step: Raw step dictionary
            step_index: 1-indexed position in trajectory
            total_steps: Total number of steps
            benchmark: Source benchmark (toolbench, gaia, swebench)

        Returns:
            StepRole value as string
        """
        content = step.get("content", "")
        tool_name = step.get("tool_name", "")
        step_type = step.get("step_type", "")
        tool_output = step.get("tool_output")

        # Normalize tool name for comparison
        tool_name_lower = (tool_name or "").lower()

        # 1. Check for final response (Finish tool)
        if tool_name_lower in self.FINISH_TOOLS or "finish" in tool_name_lower:
            return StepRole.FINAL_RESPONSE.value

        # 2. Check for tool call
        if tool_name and tool_name_lower not in self.FINISH_TOOLS:
            return StepRole.TOOL_CALL.value

        # 3. Check if this is an observation (tool output without new action)
        # Handle both enum strings (OBSERVATION) and lowercase variants
        step_type_lower = (step_type or "").lower()
        if step_type_lower == "observation" or (tool_output and not tool_name):
            return StepRole.OBSERVATION.value
        # Also check metadata for segmented steps
        segmented_from = step.get("metadata", {}).get("segmented_from", "")
        if segmented_from == "observation":
            return StepRole.OBSERVATION.value

        # 4. For text content, analyze the thought/reasoning
        thought = step.get("metadata", {}).get("thought", "") or content

        # Check for extraction (numbers, values being noted)
        if self._is_extraction(thought, step):
            return StepRole.EXTRACTION.value

        # Check for planning
        if self._planning_re.search(thought) and step_index <= 2:
            return StepRole.PLANNING.value

        # Check for decision
        if self._decision_re.search(thought):
            return StepRole.DECISION.value

        # Check for reasoning
        if self._reasoning_re.search(thought):
            return StepRole.REASONING.value

        # 5. Default based on step type
        if step_type == "tool_execution":
            return StepRole.TOOL_CALL.value
        elif step_type == "reasoning":
            return StepRole.REASONING.value
        elif step_type == "planning":
            return StepRole.PLANNING.value

        # Default to reasoning for unclassified steps
        return StepRole.REASONING.value

    def _is_extraction(self, content: str, step: Dict[str, Any]) -> bool:
        """Check if content represents value extraction."""
        # Look for explicit extraction patterns
        for pattern in self._extraction_patterns:
            if pattern.search(content):
                return True

        # Check for parenthetical values like "(12)" or "(6)"
        if re.search(r"\((\d+(?:\.\d+)?)\)", content):
            return True

        return False

    def determine_terminal_flags(
        self,
        step: Dict[str, Any],
        step_index: int,
        total_steps: int,
        benchmark: str,
    ) -> Tuple[bool, bool, bool]:
        """
        Determine terminal flags for a step.

        Args:
            step: Raw step dictionary
            step_index: 1-indexed position
            total_steps: Total number of steps
            benchmark: Source benchmark

        Returns:
            Tuple of (is_terminal_step, produces_final_answer, produces_patch)
        """
        tool_name = step.get("tool_name", "") or ""
        tool_name_lower = tool_name.lower()
        tool_input = step.get("tool_input", {}) or {}

        is_terminal = step_index == total_steps
        produces_final_answer = False
        produces_patch = False

        # Check for Finish tool with give_answer
        if "finish" in tool_name_lower:
            is_terminal = True
            return_type = tool_input.get("return_type", "")
            if return_type == "give_answer" or "final_answer" in tool_input:
                produces_final_answer = True

        # Check for patch-producing tools (SWE-bench)
        if benchmark == "swebench":
            command = tool_input.get("command", "")
            # Extract command from raw field if not directly available
            if not command and "raw" in tool_input:
                raw = tool_input.get("raw", "")
                cmd_match = re.search(r"<parameter=command>(\w+)</parameter>", raw)
                if cmd_match:
                    command = cmd_match.group(1)

            # FIRST: View command NEVER produces patches, regardless of tool name
            if command == "view":
                produces_patch = False
            # Exclude read-only tools
            elif tool_name_lower in self.READ_ONLY_TOOLS:
                produces_patch = False
            elif tool_name_lower in self.PATCH_TOOLS or "edit" in tool_name_lower:
                # Check for str_replace command with old_str/new_str
                if "old_str" in tool_input and "new_str" in tool_input:
                    produces_patch = True
                # Check for create/insert commands that produce files
                elif command in self.PATCH_COMMANDS:
                    produces_patch = True
                # Check for bash_command that might create/edit files
                elif "bash_command" in tool_input:
                    bash_cmd = tool_input.get("bash_command", "").lower()
                    # Detect file-modifying bash commands
                    if any(
                        kw in bash_cmd
                        for kw in [">>", ">", "echo", "sed", "patch", "cp", "mv"]
                    ):
                        produces_patch = True
            # str_replace_editor produces patch only for write commands
            elif tool_name_lower == "str_replace_editor":
                if command in {"str_replace", "create", "insert"}:
                    produces_patch = True

        return is_terminal, produces_final_answer, produces_patch

    def compute_normalized_operation(
        self,
        tool_name: Optional[str],
        tool_input: Optional[Dict[str, Any]],
        step_role: str,
    ) -> str:
        """
        Compute the normalized operation type for a step.

        This provides a semantic classification independent of tool_name,
        based on the actual command/action being performed.

        Returns one of: "read", "write", "execute", "submit", "reason"
        """
        tool_input = tool_input or {}
        command = tool_input.get("command", "")
        tool_lower = (tool_name or "").lower()

        # If command not directly available, try to extract from raw field
        # This handles SWE-bench format where tool_input is {'raw': '...<parameter=command>view</parameter>...'}
        if not command and "raw" in tool_input:
            raw = tool_input.get("raw", "")
            cmd_match = re.search(r"<parameter=command>(\w+)</parameter>", raw)
            if cmd_match:
                command = cmd_match.group(1)

        # Command takes priority (most reliable signal)
        if command == "view":
            return "read"
        if command in ("str_replace", "create", "insert"):
            return "write"

        # Check for submit/finish tools
        if tool_lower in self.FINISH_TOOLS or tool_lower == "submit":
            return "submit"

        # Check for execute tools
        if tool_lower in self.EXECUTE_TOOLS:
            # But bash with file-modifying commands is "write"
            bash_cmd = tool_input.get("bash_command", "").lower()
            if bash_cmd and any(
                kw in bash_cmd for kw in [">>", ">", "echo", "sed", "patch"]
            ):
                return "write"
            return "execute"

        # Check for read-only tools
        if tool_lower in self.READ_ONLY_TOOLS:
            return "read"

        # Check for patch/edit tools
        if tool_lower in self.PATCH_TOOLS or "edit" in tool_lower:
            return "write"

        # Step role fallback for non-tool steps
        if not tool_name or step_role in (
            "planning",
            "reasoning",
            "extraction",
            "decision",
        ):
            return "reason"

        # Default based on whether it looks like a tool call
        if step_role == "tool_call":
            return "execute"

        return "reason"

    def extract_extraction_fields(
        self,
        step: Dict[str, Any],
        step_role: str,
        previous_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Extract extraction-specific fields for extraction steps.

        Includes provenance tracking and validation.
        If extraction cannot be grounded, sets is_grounded=False
        so the caller can demote to reasoning.

        Args:
            step: Raw step dictionary
            step_role: Classified step role
            previous_steps: List of preceding steps

        Returns:
            Dict with extracted_value, value_type, source_step, source_description,
            extraction_provenance, and is_grounded
        """
        result = {
            "extracted_value": None,
            "value_type": None,
            "source_step": None,
            "source_description": None,
            "extraction_provenance": None,
            "is_grounded": False,
        }

        if step_role != StepRole.EXTRACTION.value:
            return result

        content = step.get("content", "")
        thought = step.get("metadata", {}).get("thought", "") or content
        extraction_method = None
        evidence_in_content = False

        # Try to extract numeric values
        numeric_match = re.search(r"\((\d+(?:\.\d+)?)\)", content)
        if numeric_match:
            value = numeric_match.group(1)
            if "." in value:
                result["extracted_value"] = float(value)
                result["value_type"] = ValueType.FLOAT.value
            else:
                result["extracted_value"] = int(value)
                result["value_type"] = ValueType.INTEGER.value
            extraction_method = "regex_numeric"
            evidence_in_content = True

        # Try to extract line numbers
        line_match = re.search(r"line\s*(\d+)", content, re.IGNORECASE)
        if line_match and not result["extracted_value"]:
            result["extracted_value"] = int(line_match.group(1))
            result["value_type"] = ValueType.LINE_NUMBER.value
            extraction_method = "regex_line_number"
            evidence_in_content = True

        # Try to extract file paths
        path_match = re.search(
            r"(?:file|path)\s*[:=]?\s*['\"]?([/\w\.-]+\.\w+)['\"]?",
            content,
            re.IGNORECASE,
        )
        if path_match and not result["extracted_value"]:
            result["extracted_value"] = path_match.group(1)
            result["value_type"] = ValueType.FILEPATH.value
            extraction_method = "regex_filepath"
            evidence_in_content = True

        # Determine source step (previous step with tool output)
        source_tool_name = None
        if previous_steps:
            for i in range(len(previous_steps) - 1, -1, -1):
                prev = previous_steps[i]
                if prev.get("tool_output"):
                    result["source_step"] = prev.get("step_number", i + 1)
                    source_tool_name = prev.get("tool_name", "unknown")
                    result["source_description"] = (
                        f"Extracted from {source_tool_name} output"
                    )
                    break

        # Build extraction provenance
        if result["extracted_value"] is not None and extraction_method:
            # Calculate confidence based on evidence quality
            confidence = 0.3  # Base confidence
            if evidence_in_content:
                confidence += 0.3
            if result["source_step"] is not None:
                confidence += 0.2
            if result["value_type"] is not None:
                confidence += 0.2

            result["extraction_provenance"] = ExtractionProvenance(
                extraction_method=extraction_method,
                evidence_in_content=evidence_in_content,
                source_tool_name=source_tool_name,
                confidence=confidence,
            ).to_dict()

            # Grounded if we have: extracted_value + value_type + source_step
            result["is_grounded"] = (
                result["extracted_value"] is not None
                and result["value_type"] is not None
                and result["source_step"] is not None
            )

        return result

    def type_step(
        self,
        step: Dict[str, Any],
        step_index: int,
        total_steps: int,
        benchmark: str,
        previous_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Fully type a single step with role and flags.

        Args:
            step: Raw step dictionary
            step_index: 1-indexed position
            total_steps: Total number of steps
            benchmark: Source benchmark
            previous_steps: List of preceding steps

        Returns:
            Dict with typed fields ready for TypedStep
        """
        # Classify role
        step_role = self.classify_step_role(step, step_index, total_steps, benchmark)

        # Determine terminal flags
        is_terminal, produces_answer, produces_patch = self.determine_terminal_flags(
            step, step_index, total_steps, benchmark
        )

        # Extract extraction fields
        extraction = self.extract_extraction_fields(step, step_role, previous_steps)

        # Demote ungrounded extraction steps to reasoning
        # An extraction must have: extracted_value, value_type, and source_step
        if step_role == StepRole.EXTRACTION.value and not extraction["is_grounded"]:
            step_role = StepRole.REASONING.value

        # Prepare tool fields
        tool_name = step.get("tool_name")
        tool_arguments = step.get("tool_input")
        observation = step.get("tool_output")

        # Compute normalized operation (semantic classification)
        normalized_op = self.compute_normalized_operation(
            tool_name, tool_arguments, step_role
        )

        return {
            "step_index": step_index,
            "raw_text": step.get("content", ""),
            "step_role": step_role,
            "is_terminal_step": is_terminal,
            "produces_final_answer": produces_answer,
            "produces_patch": produces_patch,
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
            "observation": observation,
            "normalized_operation": normalized_op,
            "extracted_value": extraction["extracted_value"],
            "value_type": extraction["value_type"],
            "source_step": extraction["source_step"],
            "source_description": extraction["source_description"],
            "extraction_provenance": extraction["extraction_provenance"],
        }
