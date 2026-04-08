"""
EntityExtractor: Extract named entities from step content.

Uses regex patterns for domain-specific entities (file paths, line numbers,
API names, URLs, etc.). Optionally can integrate spaCy for general NER.
"""

import re
from typing import Any, Dict, List, Set


class EntityExtractor:
    """
    Extract entities from trajectory step content.

    Focuses on domain-specific entities relevant to agent trajectories:
    - File paths
    - Line numbers
    - Function/class names
    - API endpoints
    - URLs
    - Numbers (potential values)
    - Tool names
    """

    # Parser artifacts and generic tokens to filter out
    ENTITY_BLOCKLIST = {
        # XML tag artifacts from SWE-bench format
        "/function", "/parameter", "function", "parameter",
        # Generic tool words
        "file_edit", "file_view", "str_replace_editor", "edit", "file", "view",
        "search_code", "view_file", "bash", "submit",
        # Generic paths/roots
        "/testbed", "testbed",
        # Common noise words
        "raw", "content", "command", "path", "old_str", "new_str",
        "insert_line", "view_range",
        # Generic descriptors
        "result", "response", "output", "input", "value", "data",
        "error", "success", "failed", "true", "false",
    }

    # Path prefixes that are always invalid (XML artifacts, URLs)
    INVALID_PATH_PREFIXES = (
        "/function", "/parameter", "/www", "/http", "/https",
        "/encrypted", "/images",
    )

    # File path patterns
    FILE_PATH_PATTERNS = [
        # Unix-style paths
        r"(?:/[\w\.-]+)+(?:\.\w+)?",
        # Windows-style paths
        r"(?:[A-Za-z]:\\)?(?:\\[\w\.-]+)+(?:\.\w+)?",
        # Relative paths with extension
        r"[\w\.-]+(?:/[\w\.-]+)*\.\w{1,6}",
    ]

    # Line number patterns
    LINE_NUMBER_PATTERNS = [
        r"line\s*(\d+)",
        r"L(\d+)",
        r":(\d+)(?::\d+)?$",  # filename:line
    ]

    # Function/class name patterns
    CODE_ENTITY_PATTERNS = [
        # Python-style function/class names
        r"`(\w+)`",
        r"def\s+(\w+)",
        r"class\s+(\w+)",
        # Quoted identifiers
        r"'(\w+)'",
        r'"(\w+)"',
    ]

    # API/endpoint patterns
    API_PATTERNS = [
        r"(?:api|endpoint|function)[:\s]+['\"]?(\w+)['\"]?",
        r"(\w+_for_\w+)",  # ToolBench style: xxx_for_yyy
    ]

    # URL patterns
    URL_PATTERN = r"https?://[^\s<>\"')\]]+"

    # Number patterns (potential extracted values)
    NUMBER_PATTERNS = [
        r"\((\d+(?:\.\d+)?)\)",  # Numbers in parentheses
        r"=\s*(\d+(?:\.\d+)?)",   # Assignments
        r"(?:is|equals?|result)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    ]

    def __init__(self, use_spacy: bool = False):
        """
        Initialize entity extractor.

        Args:
            use_spacy: Whether to use spaCy for general NER (optional, slower)
        """
        self.use_spacy = use_spacy
        self._nlp = None

        # Compile patterns
        self._file_path_re = [re.compile(p) for p in self.FILE_PATH_PATTERNS]
        self._line_number_re = [re.compile(p, re.IGNORECASE) for p in self.LINE_NUMBER_PATTERNS]
        self._code_entity_re = [re.compile(p) for p in self.CODE_ENTITY_PATTERNS]
        self._api_re = [re.compile(p, re.IGNORECASE) for p in self.API_PATTERNS]
        self._url_re = re.compile(self.URL_PATTERN)
        self._number_re = [re.compile(p, re.IGNORECASE) for p in self.NUMBER_PATTERNS]

        if use_spacy:
            self._init_spacy()

    def _init_spacy(self):
        """Initialize spaCy model lazily."""
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            self.use_spacy = False
            self._nlp = None

    def extract_entities(self, step: Dict[str, Any]) -> List[str]:
        """
        Extract all entities from a step.

        Args:
            step: Step dictionary with content, tool_name, etc.

        Returns:
            List of unique entity strings
        """
        entities: Set[str] = set()

        content = step.get("content", "") or ""
        tool_name = step.get("tool_name", "") or ""
        tool_input = step.get("tool_input", {}) or {}
        tool_output = step.get("tool_output", "") or ""

        # Extract from main content
        entities.update(self._extract_from_text(content))

        # Extract from tool name
        if tool_name:
            entities.add(tool_name)
            # Also extract parts of tool name (e.g., "api_nutrition_data_for_edamam")
            parts = tool_name.split("_")
            for part in parts:
                if len(part) > 2 and part.lower() not in {"for", "the", "and", "api"}:
                    entities.add(part)

        # Extract from tool input
        entities.update(self._extract_from_dict(tool_input))

        # Extract from tool output (limited to avoid noise)
        if tool_output and len(tool_output) < 2000:
            entities.update(self._extract_from_text(str(tool_output), limit=20))

        # Use spaCy if available
        if self.use_spacy and self._nlp:
            entities.update(self._extract_with_spacy(content))

        # Filter and clean entities
        entities = self._clean_entities(entities)

        return sorted(entities)

    def _extract_from_text(self, text: str, limit: int = 50) -> Set[str]:
        """Extract entities from text content."""
        entities: Set[str] = set()

        # File paths
        for pattern in self._file_path_re:
            for match in pattern.finditer(text):
                path = match.group(0)
                if len(path) > 3 and "/" in path or "\\" in path or "." in path:
                    entities.add(path)
                    if len(entities) >= limit:
                        return entities

        # URLs
        for match in self._url_re.finditer(text):
            entities.add(match.group(0))

        # Line numbers (just note the line number as entity)
        for pattern in self._line_number_re:
            for match in pattern.finditer(text):
                entities.add(f"line {match.group(1)}")

        # Code entities
        for pattern in self._code_entity_re:
            for match in pattern.finditer(text):
                entities.add(match.group(1))

        # API names
        for pattern in self._api_re:
            for match in pattern.finditer(text):
                entities.add(match.group(1))

        # Numbers in context
        for pattern in self._number_re:
            for match in pattern.finditer(text):
                entities.add(match.group(1))

        return entities

    def _extract_from_dict(self, d: Dict[str, Any], depth: int = 0) -> Set[str]:
        """Extract entities from dictionary (tool arguments)."""
        if depth > 3:
            return set()

        entities: Set[str] = set()

        for key, value in d.items():
            # Key might be entity (e.g., "path", "query", "file")
            if key in {"path", "file", "filepath", "query", "url", "name"}:
                if isinstance(value, str):
                    entities.add(value)

            # Process value
            if isinstance(value, str):
                entities.update(self._extract_from_text(value, limit=10))
            elif isinstance(value, dict):
                entities.update(self._extract_from_dict(value, depth + 1))
            elif isinstance(value, list):
                for item in value[:10]:
                    if isinstance(item, str):
                        entities.update(self._extract_from_text(item, limit=5))
                    elif isinstance(item, dict):
                        entities.update(self._extract_from_dict(item, depth + 1))

        return entities

    def _extract_with_spacy(self, text: str) -> Set[str]:
        """Extract named entities using spaCy."""
        if not self._nlp:
            return set()

        entities: Set[str] = set()
        doc = self._nlp(text[:5000])  # Limit text length

        for ent in doc.ents:
            # Include relevant entity types
            if ent.label_ in {"ORG", "PRODUCT", "PERSON", "GPE", "LOC", "FAC", "WORK_OF_ART"}:
                entities.add(ent.text)

        return entities

    def _clean_entities(self, entities: Set[str]) -> Set[str]:
        """Clean and filter entities, removing parser artifacts and noise."""
        cleaned: Set[str] = set()

        for entity in entities:
            # Skip very short entities
            if len(entity) < 2:
                continue

            # Skip very long entities
            if len(entity) > 200:
                continue

            # Skip entities in blocklist (case-insensitive)
            lower = entity.lower()
            if lower in self.ENTITY_BLOCKLIST:
                continue

            # Skip invalid path prefixes (XML artifacts)
            if any(entity.startswith(prefix) for prefix in self.INVALID_PATH_PREFIXES):
                continue

            # Skip paths that are just roots without meaningful content
            if entity in {"/testbed", "/usr", "/bin", "/tmp", "/var"}:
                continue

            # Skip common stop words
            if lower in {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                         "null", "none", "this", "that", "from", "with", "will", "have"}:
                continue

            # Skip pure numbers (unless they look like line numbers or specific values)
            if entity.isdigit() and len(entity) > 6:
                continue

            # Skip single path segments that look like generic words
            if "/" not in entity and len(entity) < 4:
                continue

            cleaned.add(entity)

        return cleaned


def extract_tool_name_parts(tool_name: str) -> List[str]:
    """
    Extract meaningful parts from a tool name.

    E.g., "api_nutrition_data_for_edamam_nutrition_analysis" ->
    ["nutrition", "data", "edamam", "analysis"]
    """
    if not tool_name:
        return []

    # Split by underscores and other separators
    parts = re.split(r"[_\-\s]+", tool_name)

    # Filter out common words
    stop_words = {"api", "for", "the", "a", "an", "get", "set", "from", "to", "with"}

    return [p for p in parts if len(p) > 2 and p.lower() not in stop_words]
