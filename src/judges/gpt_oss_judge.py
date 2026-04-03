"""
GPT-OSS 120B Judge implementation using AWS Bedrock.
"""

import json
import time
from typing import Dict, Any
import boto3
from botocore.exceptions import ClientError

from src.judges import Judge, parse_json_response
from src.judges.schema import JudgeOutput


class GPTOSSJudge(Judge):
    """
    Judge implementation using GPT-OSS 120B via AWS Bedrock.

    Model: openai.gpt-oss-120b-1:0
    """

    def __init__(
        self,
        model_id: str = "openai.gpt-oss-120b-1:0",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
        region_name: str = "us-east-1"
    ):
        """
        Initialize GPT-OSS judge with Bedrock client.

        Args:
            model_id: Bedrock model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: API timeout in seconds
            region_name: AWS region for Bedrock
        """
        super().__init__(
            name="gpt-oss-120b",
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )

        # Initialize Bedrock client
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )

        print(f"✓ Initialized GPT-OSS Judge: {model_id}")

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Call GPT-OSS via AWS Bedrock.

        Args:
            system_prompt: System message
            user_prompt: User message

        Returns:
            Dict with response, tokens_used, time_ms

        Raises:
            ClientError: If Bedrock API call fails
        """
        start_time = time.time()

        # Prepare request body for GPT-OSS (OpenAI-compatible format)
        request_body = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }

        try:
            # Invoke Bedrock
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )

            # Parse response
            response_body = json.loads(response['body'].read())

            # Extract text from choices (OpenAI format)
            choices = response_body.get('choices', [])
            if not choices:
                raise ValueError("Empty response from GPT-OSS")

            message = choices[0].get('message', {})
            response_text = message.get('content', '')

            # Extract token usage (OpenAI format)
            usage = response_body.get('usage', {})
            tokens_used = usage.get('total_tokens', 0)

            elapsed_ms = int((time.time() - start_time) * 1000)

            return {
                "response": response_text,
                "tokens_used": tokens_used,
                "time_ms": elapsed_ms
            }

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            raise Exception(f"Bedrock API error ({error_code}): {error_message}")

        except Exception as e:
            raise Exception(f"GPT-OSS API call failed: {e}")

    def _parse_response(
        self,
        response_text: str,
        trajectory_id: str
    ) -> JudgeOutput:
        """
        Parse GPT-OSS's JSON response into JudgeOutput.

        Args:
            response_text: Raw GPT-OSS response
            trajectory_id: Trajectory ID

        Returns:
            JudgeOutput object

        Raises:
            ValueError: If response cannot be parsed
        """
        return parse_json_response(
            response_text=response_text,
            trajectory_id=trajectory_id,
            judge_name=self.name,
            model_id=self.model_id
        )


def create_gpt_oss_judge(config: Dict[str, Any]) -> GPTOSSJudge:
    """
    Factory function to create GPT-OSS judge from config.

    Args:
        config: Judge configuration dict with keys:
            - model_id: Bedrock model ID
            - config: Dict with temperature, max_tokens
            - region_name (optional): AWS region

    Returns:
        Configured GPTOSSJudge instance

    Example config:
        {
            "name": "gpt-oss-120b",
            "provider": "aws_bedrock",
            "model_id": "openai.gpt-oss-120b-1:0",
            "config": {
                "temperature": 0.7,
                "max_tokens": 2000
            }
        }
    """
    model_id = config.get('model_id')
    judge_config = config.get('config', {})
    region_name = config.get('region_name', 'us-east-1')

    return GPTOSSJudge(
        model_id=model_id,
        temperature=judge_config.get('temperature', 0.7),
        max_tokens=judge_config.get('max_tokens', 2000),
        timeout=judge_config.get('timeout', 60),
        region_name=region_name
    )
