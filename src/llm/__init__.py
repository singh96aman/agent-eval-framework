"""LLM client module."""

from .bedrock_client import BedrockClient, get_bedrock_client

__all__ = ["BedrockClient", "get_bedrock_client"]
