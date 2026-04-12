"""
SlotTyper: Identify perturbable slots with their types.

For each step, identifies what can be mutated and how:
- Tool name (tool_selection perturbation)
- Tool arguments (parameter perturbation)
- Extracted values (data_reference perturbation)
- Decision targets (structural perturbation)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.typing.schema import PerturbableSlot, ValueType


class SlotTyper:
    """
    Identify perturbable slots in trajectory steps.

    Each slot has:
    - slot: JSON path to the value
    - value_type: Type of the value
    - current_value: Current value at this slot
    - allowed_perturbation_types: What kinds of perturbations are valid
    """

    # Argument name -> value type mapping
    ARG_TYPE_MAP = {
        # File/path arguments
        "path": ValueType.FILEPATH,
        "file": ValueType.FILEPATH,
        "filepath": ValueType.FILEPATH,
        "filename": ValueType.FILEPATH,
        "directory": ValueType.FILEPATH,
        "dir": ValueType.FILEPATH,
        # Line/position arguments
        "line": ValueType.LINE_NUMBER,
        "line_number": ValueType.LINE_NUMBER,
        "start_line": ValueType.LINE_NUMBER,
        "end_line": ValueType.LINE_NUMBER,
        # Query arguments
        "query": ValueType.SEARCH_QUERY,
        "search": ValueType.SEARCH_QUERY,
        "q": ValueType.SEARCH_QUERY,
        "keyword": ValueType.SEARCH_QUERY,
        # Code arguments
        "code": ValueType.CODE_SNIPPET,
        "old_str": ValueType.CODE_SNIPPET,
        "new_str": ValueType.CODE_SNIPPET,
        "content": ValueType.CODE_SNIPPET,
        "command": ValueType.STRING,
        # URL arguments
        "url": ValueType.URL,
        "endpoint": ValueType.API_ENDPOINT,
        "api": ValueType.API_ENDPOINT,
        # Network arguments
        "ip": ValueType.IPV4,
        "ip_address": ValueType.IPV4,
        "host": ValueType.DOMAIN,
        "domain": ValueType.DOMAIN,
        "hostname": ValueType.DOMAIN,
        "email": ValueType.EMAIL,
        # Coordinate arguments
        "latitude": ValueType.LATITUDE,
        "lat": ValueType.LATITUDE,
        "longitude": ValueType.LONGITUDE,
        "lng": ValueType.LONGITUDE,
        "lon": ValueType.LONGITUDE,
        # Numeric arguments
        "count": ValueType.INTEGER,
        "limit": ValueType.INTEGER,
        "top_k": ValueType.INTEGER,
        "year": ValueType.INTEGER,
        "num": ValueType.INTEGER,
        "threshold": ValueType.FLOAT,
        "temperature": ValueType.FLOAT,
        # Identifier arguments
        "name": ValueType.IDENTIFIER,
        "id": ValueType.IDENTIFIER,
        "key": ValueType.IDENTIFIER,
        "function": ValueType.IDENTIFIER,
        "class": ValueType.IDENTIFIER,
        "method": ValueType.IDENTIFIER,
    }

    # Perturbation types by slot category
    TOOL_PERTURBATIONS = ["tool_selection"]
    PARAM_PERTURBATIONS = ["parameter", "data_reference"]
    DATA_PERTURBATIONS = ["data_reference"]
    STRUCTURAL_PERTURBATIONS = ["structural"]

    def __init__(self):
        # Pattern for detecting file paths
        self._filepath_re = re.compile(r"^[/\\]?[\w\.-]+([/\\][\w\.-]+)*\.\w{1,6}$")
        # Pattern for URLs
        self._url_re = re.compile(r"^https?://")
        # Pattern for code snippets
        self._code_re = re.compile(r"[\{\}\(\)\[\];=]|def |class |import ")
        # Pattern for IPv4 addresses (check before filepath!)
        self._ipv4_re = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
        # Pattern for domain names
        self._domain_re = re.compile(
            r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
            r"(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
        )
        # Pattern for email addresses
        self._email_re = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        # Pattern for coordinates (decimal degrees)
        self._coord_re = re.compile(r"^-?\d{1,3}\.\d+$")

    def identify_slots(self, step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify all perturbable slots in a step.

        Args:
            step: Typed step dictionary

        Returns:
            List of PerturbableSlot dictionaries
        """
        slots: List[Dict[str, Any]] = []

        step_role = step.get("step_role", "")

        # 1. Tool name slot (for tool_call steps)
        if step_role == "tool_call" and step.get("tool_name"):
            slot = PerturbableSlot(
                slot="tool_name",
                value_type=ValueType.TOOL.value,
                current_value=step["tool_name"],
                allowed_perturbation_types=self.TOOL_PERTURBATIONS,
            )
            slots.append(slot.to_dict())

        # 2. Tool argument slots
        tool_args = step.get("tool_arguments") or {}
        for arg_name, arg_value in tool_args.items():
            if arg_value is None:
                continue

            arg_slots = self._identify_argument_slots(
                arg_name, arg_value, f"tool_arguments.{arg_name}"
            )
            slots.extend(arg_slots)

        # 3. Extracted value slot (for extraction steps)
        if step_role == "extraction" and step.get("extracted_value") is not None:
            value_type = step.get("value_type") or self._infer_value_type(
                step["extracted_value"]
            )
            slot = PerturbableSlot(
                slot="extracted_value",
                value_type=value_type,
                current_value=step["extracted_value"],
                allowed_perturbation_types=self.DATA_PERTURBATIONS,
            )
            slots.append(slot.to_dict())

        # 4. Final answer slot (for final_response steps)
        if step.get("produces_final_answer"):
            final_answer = self._extract_final_answer(step)
            if final_answer:
                value_type = self._infer_value_type(final_answer)
                slot = PerturbableSlot(
                    slot="final_answer",
                    value_type=value_type,
                    current_value=final_answer,
                    allowed_perturbation_types=self.DATA_PERTURBATIONS,
                )
                slots.append(slot.to_dict())

        return slots

    def _identify_argument_slots(
        self, arg_name: str, arg_value: Any, path_prefix: str
    ) -> List[Dict[str, Any]]:
        """
        Identify slots within a tool argument.

        Args:
            arg_name: Name of the argument
            arg_value: Value of the argument
            path_prefix: JSON path prefix

        Returns:
            List of PerturbableSlot dictionaries
        """
        slots: List[Dict[str, Any]] = []

        # Handle dict arguments recursively
        if isinstance(arg_value, dict):
            for key, value in arg_value.items():
                slots.extend(
                    self._identify_argument_slots(key, value, f"{path_prefix}.{key}")
                )
            return slots

        # Handle list arguments
        if isinstance(arg_value, list):
            for i, item in enumerate(arg_value[:5]):  # Limit to first 5
                if isinstance(item, (str, int, float)):
                    value_type = self._infer_value_type(item)
                    slot = PerturbableSlot(
                        slot=f"{path_prefix}[{i}]",
                        value_type=value_type,
                        current_value=item,
                        allowed_perturbation_types=self.PARAM_PERTURBATIONS,
                    )
                    slots.append(slot.to_dict())
            return slots

        # Handle scalar values
        if isinstance(arg_value, (str, int, float, bool)):
            value_type = self._get_arg_type(arg_name, arg_value)
            perturbation_types = self._get_perturbation_types(arg_name, value_type)

            slot = PerturbableSlot(
                slot=path_prefix,
                value_type=value_type,
                current_value=arg_value,
                allowed_perturbation_types=perturbation_types,
            )
            slots.append(slot.to_dict())

        return slots

    def _get_arg_type(self, arg_name: str, arg_value: Any) -> str:
        """Get value type based on argument name and value."""
        # Check name-based mapping first
        arg_lower = arg_name.lower()
        for pattern, vtype in self.ARG_TYPE_MAP.items():
            if pattern in arg_lower:
                return vtype.value

        # Infer from value (with arg name for context)
        return self._infer_value_type(arg_value, arg_name)

    def _infer_value_type(self, value: Any, arg_name: str = "") -> str:
        """
        Infer value type from the value itself.

        Uses semantic validation to avoid mis-typing (e.g., IP as filepath).
        Priority order: network types > coordinates > filepath > url > code > string
        """
        if isinstance(value, bool):
            return ValueType.BOOLEAN.value

        if isinstance(value, int):
            return ValueType.INTEGER.value

        if isinstance(value, float):
            # Check if this could be a coordinate based on arg name
            arg_lower = arg_name.lower()
            if any(kw in arg_lower for kw in ["lat", "lng", "lon", "coord"]):
                if -90 <= value <= 90 and "lat" in arg_lower:
                    return ValueType.LATITUDE.value
                if -180 <= value <= 180 and ("lng" in arg_lower or "lon" in arg_lower):
                    return ValueType.LONGITUDE.value
            return ValueType.FLOAT.value

        if isinstance(value, str):
            # Check network types FIRST (before filepath to avoid mis-typing)
            if self._ipv4_re.match(value):
                return ValueType.IPV4.value
            if self._email_re.match(value):
                return ValueType.EMAIL.value
            if self._domain_re.match(value) and "." in value:
                # Additional check: domains have TLD, filepaths usually have / or \
                if "/" not in value and "\\" not in value:
                    return ValueType.DOMAIN.value

            # Check coordinates (only for values that look like decimal degrees)
            if self._coord_re.match(value):
                arg_lower = arg_name.lower()
                if any(kw in arg_lower for kw in ["lat", "latitude"]):
                    return ValueType.LATITUDE.value
                if any(kw in arg_lower for kw in ["lng", "lon", "longitude"]):
                    return ValueType.LONGITUDE.value

            # Check URL
            if self._url_re.match(value):
                return ValueType.URL.value

            # Check filepath (after ruling out IP/domain/email)
            if self._filepath_re.match(value):
                return ValueType.FILEPATH.value

            # Check code snippets
            if self._code_re.search(value):
                return ValueType.CODE_SNIPPET.value

            return ValueType.STRING.value

        if isinstance(value, dict):
            return ValueType.JSON_OBJECT.value

        return ValueType.STRING.value

    def _get_perturbation_types(self, arg_name: str, value_type: str) -> List[str]:
        """Determine allowed perturbation types for a slot."""
        perturbations = list(self.PARAM_PERTURBATIONS)

        # File paths and line numbers are good data_reference targets
        if value_type in (
            ValueType.FILEPATH.value,
            ValueType.LINE_NUMBER.value,
            ValueType.INTEGER.value,
        ):
            if "data_reference" not in perturbations:
                perturbations.append("data_reference")

        # Code snippets support structural perturbations
        if value_type == ValueType.CODE_SNIPPET.value:
            perturbations.append("structural")

        return perturbations

    def _extract_final_answer(self, step: Dict[str, Any]) -> Optional[Any]:
        """Extract final answer from a final_response step."""
        tool_args = step.get("tool_arguments") or {}

        # Check for explicit final_answer field
        if "final_answer" in tool_args:
            return tool_args["final_answer"]

        # Check raw text for answer pattern
        content = step.get("raw_text", "")
        answer_match = re.search(
            r"(?:answer|result)\s*[:=]\s*(.+?)(?:\n|$)",
            content,
            re.IGNORECASE,
        )
        if answer_match:
            return answer_match.group(1).strip()

        return None


def get_slot_from_path(step: Dict[str, Any], path: str) -> Optional[Any]:
    """
    Get value from a step using a JSON path.

    Args:
        step: Step dictionary
        path: JSON path like "tool_arguments.path" or "tool_arguments.items[0]"

    Returns:
        Value at the path or None
    """
    parts = path.replace("[", ".").replace("]", "").split(".")
    current = step

    for part in parts:
        if not part:
            continue

        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                return None
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None

    return current
