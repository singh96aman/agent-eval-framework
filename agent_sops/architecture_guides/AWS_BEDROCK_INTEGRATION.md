# AWS Bedrock Integration for LLM Judges

## Overview

Both judges (Claude 3.5 Sonnet and GPT-OSS 120B) are accessed via **AWS Bedrock APIs** for consistency, cost efficiency, and simplified deployment.

## Configuration

### Environment Variables

```bash
# AWS Credentials (shared for both judges)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Judge 1: Claude 3.5 Sonnet
AWS_BEDROCK_CLAUDE_3_5_SONNET=anthropic.claude-3-5-sonnet-20241022-v2:0

# Judge 2: GPT-OSS 120B
AWS_BEDROCK_GPT_OSS=your-gpt-oss-bedrock-model-id
```

## Pre-Requisite Checks with Test Calls

### What the Checker Does

For each judge, the pre-requisite checker:
1. Validates AWS credentials
2. Creates Bedrock runtime client
3. **Makes a "hello world" test call** to verify model access
4. Parses response to ensure model is working correctly
5. Reports success/failure with details

### Claude 3.5 Sonnet Test Call

```python
# Payload
{
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 10,
    "messages": [
        {
            "role": "user",
            "content": "Say 'Hello' in one word."
        }
    ]
}

# Expected Response
{
    "content": [
        {
            "type": "text",
            "text": "Hello"
        }
    ],
    "stop_reason": "end_turn",
    ...
}
```

### GPT-OSS Test Call

```python
# Payload (format may vary by model)
{
    "prompt": "Say 'Hello' in one word.",
    "max_gen_len": 10,
    "temperature": 0.1
}

# Expected Response
# (Format depends on specific GPT-OSS model on Bedrock)
```

## Benefits of Bedrock for Both Judges

1. **Unified API**: Single boto3 client for both models
2. **Cost Tracking**: All API costs in one AWS account
3. **Rate Limits**: Bedrock handles throttling consistently
4. **Security**: IAM-based access control
5. **Logging**: CloudWatch logs for all API calls
6. **Monitoring**: CloudWatch metrics for latency, errors
7. **Deployment**: No separate endpoints to manage

## Usage in Experiment Code

```python
import boto3
import json

class BedrockJudge:
    def __init__(self, model_id, region="us-east-1"):
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region
        )
        self.model_id = model_id
    
    def evaluate(self, trajectory):
        # Build prompt
        prompt = self._build_prompt(trajectory)
        
        # Call model via Bedrock
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(prompt)
        )
        
        # Parse response
        return self._parse_response(response)

# Usage
claude_judge = BedrockJudge(
    model_id=os.getenv("AWS_BEDROCK_CLAUDE_3_5_SONNET")
)

gpt_judge = BedrockJudge(
    model_id=os.getenv("AWS_BEDROCK_GPT_OSS")
)
```

## Error Handling

Common errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `NoCredentials` | AWS credentials not configured | Run `aws configure` or set env vars |
| `AccessDenied` | IAM policy missing Bedrock permissions | Add `bedrock:InvokeModel` permission |
| `ResourceNotFound` | Model ID incorrect or not enabled | Check model ID, enable in Bedrock console |
| `ThrottlingException` | Rate limit exceeded | Implement exponential backoff |
| `ValidationException` | Invalid request format | Check model-specific payload format |

## IAM Policy Required

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet*",
                "arn:aws:bedrock:*::foundation-model/YOUR_GPT_OSS_MODEL"
            ]
        }
    ]
}
```

## Cost Estimation

**POC Experiment (50 trajectories):**
- 50 perturbed trajectories × 2 judges × 3 samples = 300 evaluations
- 30 baseline trajectories × 2 judges × 1 sample = 60 evaluations
- **Total: 360 API calls**

**Estimated Costs (per call):**
- Claude 3.5 Sonnet: ~$0.10 - $0.20 (depends on trajectory length)
- GPT-OSS: ~$0.05 - $0.15 (varies by model)

**POC Total: $30-80**

## Testing Your Setup

1. Configure `.env`:
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

2. Run pre-requisite checker:
   ```bash
   python src/prereq_check.py
   ```

3. Look for:
   ```
   ✓ PASS: Claude 3.5 Sonnet (Bedrock)
     ✓ Test call successful. Model: sonnet-20241022-v2:0
   
   ✓ PASS: GPT-OSS 120B (Bedrock)
     ✓ Test call successful. Model: your-model-id
   ```

If both pass, you're ready to run experiments!
