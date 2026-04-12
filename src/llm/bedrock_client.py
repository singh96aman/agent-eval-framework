"""
Central Bedrock client for all LLM calls.

Usage:
    from src.llm import get_bedrock_client

    client = get_bedrock_client(log_calls=True)
    response = client.invoke(model_id, prompt, max_tokens=500)
"""

import json
import time
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError

# Global singleton
_client_instance: Optional["BedrockClient"] = None


class BedrockClient:
    """Central Bedrock client with optional logging."""

    def __init__(self, region_name: str = "us-east-1", log_calls: bool = False):
        self.region_name = region_name
        self.log_calls = log_calls
        self.call_count = 0
        self.total_tokens = 0
        self.total_time_ms = 0

        self.client = boto3.client(
            service_name="bedrock-runtime", region_name=region_name
        )

    def invoke(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Invoke Bedrock model.

        Args:
            model_id: Bedrock model ID
            prompt: User prompt
            max_tokens: Max tokens for response
            temperature: Sampling temperature

        Returns:
            Dict with: response, tokens_used, time_ms
        """
        start_time = time.time()

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            content = response_body.get("content", [])
            if not content:
                raise ValueError("Empty response from LLM")

            response_text = content[0].get("text", "")
            usage = response_body.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            tokens_used = input_tokens + output_tokens
            elapsed_ms = int((time.time() - start_time) * 1000)

            # Update stats
            self.call_count += 1
            self.total_tokens += tokens_used
            self.total_time_ms += elapsed_ms

            # Log if enabled
            if self.log_calls:
                print(
                    f"   [Bedrock] {model_id.split('/')[-1][:20]} | "
                    f"{elapsed_ms}ms | "
                    f"{input_tokens}+{output_tokens}={tokens_used} tokens"
                )

            return {
                "response": response_text,
                "tokens_used": tokens_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "time_ms": elapsed_ms,
            }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            raise Exception(f"Bedrock API error ({error_code}): {error_message}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cumulative stats."""
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": (
                self.total_time_ms // self.call_count if self.call_count else 0
            ),
        }

    def print_stats(self):
        """Print cumulative stats."""
        stats = self.get_stats()
        print(
            f"\n📊 Bedrock Stats: {stats['call_count']} calls | "
            f"{stats['total_tokens']} tokens | "
            f"{stats['total_time_ms']}ms total | "
            f"{stats['avg_time_ms']}ms avg"
        )


def get_bedrock_client(
    region_name: str = "us-east-1", log_calls: bool = False, reset: bool = False
) -> BedrockClient:
    """
    Get or create the singleton Bedrock client.

    Args:
        region_name: AWS region
        log_calls: Enable call logging
        reset: Force create new instance

    Returns:
        BedrockClient instance
    """
    global _client_instance

    if _client_instance is None or reset:
        _client_instance = BedrockClient(region_name=region_name, log_calls=log_calls)
    elif log_calls and not _client_instance.log_calls:
        # Enable logging on existing instance
        _client_instance.log_calls = True

    return _client_instance
