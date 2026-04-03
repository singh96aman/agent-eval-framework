"""
Tool similarity matching for realistic perturbations.

This module determines which tools can plausibly be confused with each other
based on API family, name similarity, and functional purpose.
"""

import re
from typing import Dict, List, Set, Optional
from dataclasses import dataclass


@dataclass
class ToolInfo:
    """Information about a tool extracted from system prompt."""
    name: str
    api_family: str  # e.g., "greyhound_racing_uk"
    description: str
    purpose_keywords: Set[str]  # e.g., {"get", "list", "search"}


class ToolSimilarityMatcher:
    """
    Determines plausible tool substitutions for realistic perturbations.

    Strategy:
    1. Extract tools from system prompt
    2. Group by API family
    3. Within family, find tools with similar purposes but different semantics

    Example swaps:
    - racecards (future) <-> results (past)
    - latest <-> trending <-> popular
    - by_date <-> by_id
    """

    # Keywords that indicate similar but distinct purposes
    TEMPORAL_KEYWORDS = {
        "latest": {"trending", "popular", "recent"},
        "trending": {"latest", "popular"},
        "popular": {"latest", "trending"},
        "fixtures": {"results"},  # future vs past
        "results": {"fixtures"},
        "racecards": {"results"},
    }

    RETRIEVAL_KEYWORDS = {
        "get": {"list", "search", "fetch"},
        "list": {"get", "search"},
        "search": {"get", "list"},
        "by_date": {"by_id", "by_name"},
        "by_id": {"by_date", "by_name"},
    }

    def __init__(self):
        """Initialize the tool similarity matcher."""
        self.tools_by_family: Dict[str, List[ToolInfo]] = {}

    def extract_tools_from_system_prompt(self, system_prompt: str) -> List[ToolInfo]:
        """
        Extract tool definitions from ToolBench system prompt.

        Format:
        {'name': 'tool_name', 'description': '...', 'parameters': {...}}

        Args:
            system_prompt: System message with tool definitions

        Returns:
            List of ToolInfo objects
        """
        tools = []

        # Match tool definitions in the system prompt
        # Pattern: {'name': 'tool_name_for_api', 'description': ...}
        tool_pattern = r"'name':\s*'([^']+)'.*?'description':\s*'([^']*?)'"
        matches = re.finditer(tool_pattern, system_prompt, re.DOTALL)

        for match in matches:
            tool_name = match.group(1)
            description = match.group(2)

            # Skip special tools like Finish
            if tool_name in ["Finish"]:
                continue

            # Extract API family (everything before "_for_" if present)
            if "_for_" in tool_name:
                parts = tool_name.split("_for_")
                api_family = parts[-1] if len(parts) > 1 else "unknown"
                base_name = parts[0]
            else:
                api_family = "unknown"
                base_name = tool_name

            # Extract purpose keywords from base name
            purpose_keywords = self._extract_purpose_keywords(base_name)

            tools.append(ToolInfo(
                name=tool_name,
                api_family=api_family,
                description=description,
                purpose_keywords=purpose_keywords
            ))

        return tools

    def _extract_purpose_keywords(self, tool_name: str) -> Set[str]:
        """
        Extract semantic keywords from tool name.

        Examples:
        - latest_coupons → {"latest", "coupons"}
        - get_user_info → {"get", "user", "info"}
        """
        # Split on underscores and common separators
        parts = re.split(r'[_\-]', tool_name.lower())

        # Filter out common noise words
        noise_words = {"for", "the", "a", "an", "by", "with", "from", "to"}
        keywords = {p for p in parts if p and p not in noise_words}

        return keywords

    def index_tools(self, system_prompt: str):
        """
        Index tools from system prompt for similarity matching.

        Args:
            system_prompt: System message with tool definitions
        """
        tools = self.extract_tools_from_system_prompt(system_prompt)

        # Group by API family
        self.tools_by_family.clear()
        for tool in tools:
            if tool.api_family not in self.tools_by_family:
                self.tools_by_family[tool.api_family] = []
            self.tools_by_family[tool.api_family].append(tool)

    def find_plausible_substitutes(
        self,
        tool_name: str,
        max_substitutes: int = 3
    ) -> List[str]:
        """
        Find plausible tool substitutes for perturbation.

        Strategy:
        1. Find tools in same API family
        2. Prefer tools with overlapping keywords but semantic differences
        3. Rank by similarity score

        Args:
            tool_name: Original tool name
            max_substitutes: Maximum number of substitutes to return

        Returns:
            List of substitute tool names (most plausible first)
        """
        # Find the original tool
        original_tool = None
        for family_tools in self.tools_by_family.values():
            for tool in family_tools:
                if tool.name == tool_name:
                    original_tool = tool
                    break
            if original_tool:
                break

        if not original_tool:
            return []

        # Get tools from same API family
        family_tools = self.tools_by_family.get(original_tool.api_family, [])

        # Compute similarity scores
        candidates = []
        for tool in family_tools:
            if tool.name == tool_name:
                continue  # Skip self

            score = self._compute_similarity(original_tool, tool)
            if score > 0:
                candidates.append((tool.name, score))

        # Sort by score (descending) and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates[:max_substitutes]]

    def _compute_similarity(self, tool1: ToolInfo, tool2: ToolInfo) -> float:
        """
        Compute semantic similarity score between two tools.

        High score = plausible confusion but semantically different

        Factors:
        - Same API family: +1.0
        - Overlapping keywords: +0.1 per overlap
        - Known confusable pairs: +0.5

        Returns:
            Similarity score (higher = more plausible substitution)
        """
        score = 0.0

        # Must be in same API family
        if tool1.api_family != tool2.api_family:
            return 0.0

        score += 1.0  # Base score for same family

        # Keyword overlap (structural similarity)
        overlap = tool1.purpose_keywords & tool2.purpose_keywords
        score += 0.1 * len(overlap)

        # Check for known confusable pairs
        for kw1 in tool1.purpose_keywords:
            if kw1 in self.TEMPORAL_KEYWORDS:
                confusable = self.TEMPORAL_KEYWORDS[kw1]
                if any(kw2 in confusable for kw2 in tool2.purpose_keywords):
                    score += 0.5

            if kw1 in self.RETRIEVAL_KEYWORDS:
                confusable = self.RETRIEVAL_KEYWORDS[kw1]
                if any(kw2 in confusable for kw2 in tool2.purpose_keywords):
                    score += 0.3

        return score

    def get_api_families(self) -> List[str]:
        """Get list of all API families in the indexed tools."""
        return list(self.tools_by_family.keys())

    def get_tools_in_family(self, api_family: str) -> List[str]:
        """Get all tool names in a specific API family."""
        return [tool.name for tool in self.tools_by_family.get(api_family, [])]
