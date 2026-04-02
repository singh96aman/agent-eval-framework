"""
Pre-requisite verification system for POC experiment.

Checks:
- MongoDB connection
- HuggingFace access (datasets library and token)
- AWS Bedrock (Claude)
- GPT-OSS endpoint
- Python dependencies
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class CheckResult:
    """Result of a pre-requisite check."""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, any] = None

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status}: {self.check_name}\n  {self.message}"


class PrerequisiteChecker:
    """Checks all pre-requisites for running the POC experiment."""

    def __init__(self, project_root: str = None):
        """Initialize checker."""
        if project_root is None:
            project_root = Path(__file__).parent.parent
        self.project_root = Path(project_root)
        self.results: List[CheckResult] = []

    def run_all_checks(self) -> bool:
        """Run all pre-requisite checks."""
        print("=" * 70)
        print("PRE-REQUISITE VERIFICATION FOR POC EXPERIMENT")
        print("=" * 70)
        print()

        self.results = []

        # Run all checks
        self.check_directory_structure()
        self.check_mongodb_connection()
        self.check_huggingface_access()
        self.check_claude_bedrock_api()
        self.check_gpt_oss_api()
        self.check_python_dependencies()

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print()

        for result in self.results:
            print(result)
            print()

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        pass_rate = (passed / total * 100) if total > 0 else 0

        print("=" * 70)
        print(f"PASSED: {passed}/{total} ({pass_rate:.0f}%)")
        print("=" * 70)

        all_passed = all(r.passed for r in self.results)
        if all_passed:
            print("\n✓ All pre-requisites satisfied. Ready to proceed!")
        else:
            print("\n✗ Fix issues above before proceeding.")

        return all_passed

    def check_directory_structure(self) -> CheckResult:
        """Check that required directories exist."""
        required_dirs = [
            "src",
            "tests",
            "data",
            "results",
            "paper",
        ]

        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            result = CheckResult(
                check_name="Directory Structure",
                passed=False,
                message=f"Missing: {', '.join(missing_dirs)}",
                details={"missing": missing_dirs}
            )
        else:
            result = CheckResult(
                check_name="Directory Structure",
                passed=True,
                message="All required directories exist"
            )

        self.results.append(result)
        return result

    def check_mongodb_connection(self) -> CheckResult:
        """Check MongoDB connection."""
        mongodb_uri = os.getenv("MONGODB_URI")

        if not mongodb_uri:
            result = CheckResult(
                check_name="MongoDB Connection",
                passed=False,
                message=(
                    "MONGODB_URI not set in environment. "
                    "Set in .env file."
                ),
                details={"env_var": "MONGODB_URI"}
            )
            self.results.append(result)
            return result

        try:
            from src.storage.mongodb import MongoDBStorage

            storage = MongoDBStorage()
            connected = storage.test_connection()
            storage.close()

            if connected:
                result = CheckResult(
                    check_name="MongoDB Connection",
                    passed=True,
                    message="Connected to MongoDB successfully",
                    details={"uri": mongodb_uri.split("@")[-1]}
                )
            else:
                result = CheckResult(
                    check_name="MongoDB Connection",
                    passed=False,
                    message="Could not connect to MongoDB",
                    details={"uri": mongodb_uri}
                )

        except ImportError:
            result = CheckResult(
                check_name="MongoDB Connection",
                passed=False,
                message="pymongo not installed. Run: pip install pymongo",
                details={"error": "ImportError"}
            )
        except Exception as e:
            result = CheckResult(
                check_name="MongoDB Connection",
                passed=False,
                message=f"MongoDB connection failed: {str(e)}",
                details={"error": str(e)}
            )

        self.results.append(result)
        return result

    def check_huggingface_access(self) -> CheckResult:
        """Check HuggingFace datasets access."""
        hf_token = os.getenv("HUGGINGFACE_TOKEN")

        if not hf_token:
            result = CheckResult(
                check_name="HuggingFace Access",
                passed=False,
                message=(
                    "HUGGINGFACE_TOKEN not set. Get token from: "
                    "https://huggingface.co/settings/tokens"
                ),
                details={"env_var": "HUGGINGFACE_TOKEN"}
            )
            self.results.append(result)
            return result

        try:
            from datasets import load_dataset

            # Try to load a public dataset to test access
            # (Small dataset to avoid downloading large data)
            try:
                # Test with a tiny public dataset
                load_dataset(
                    "squad",
                    split="train[:1]",
                    token=hf_token,
                )

                result = CheckResult(
                    check_name="HuggingFace Access",
                    passed=True,
                    message=(
                        "HuggingFace token valid and datasets accessible"
                    ),
                    details={"token_set": True}
                )
            except Exception as e:
                result = CheckResult(
                    check_name="HuggingFace Access",
                    passed=False,
                    message=f"HuggingFace access failed: {str(e)}",
                    details={"error": str(e)}
                )

        except ImportError:
            result = CheckResult(
                check_name="HuggingFace Access",
                passed=False,
                message="datasets library not installed. "
                        "Run: pip install datasets",
                details={"error": "ImportError"}
            )

        self.results.append(result)
        return result

    def check_claude_bedrock_api(self) -> CheckResult:
        """Check Claude-3.5-Sonnet on AWS Bedrock with test call."""
        try:
            import boto3
            import json
            from botocore.exceptions import ClientError

            model_id = os.getenv(
                'AWS_BEDROCK_CLAUDE_3_5_SONNET',
                'anthropic.claude-3-5-sonnet-20241022-v2:0'
            )

            # Check AWS credentials
            try:
                session = boto3.Session()
                credentials = session.get_credentials()

                if credentials is None:
                    result = CheckResult(
                        check_name="Claude 3.5 Sonnet (Bedrock)",
                        passed=False,
                        message=(
                            "AWS credentials not found. "
                            "Configure via AWS CLI."
                        ),
                        details={"error": "NoCredentials"}
                    )
                    self.results.append(result)
                    return result

            except Exception as e:
                result = CheckResult(
                    check_name="Claude 3.5 Sonnet (Bedrock)",
                    passed=False,
                    message=f"AWS credentials error: {str(e)}",
                    details={"error": str(e)}
                )
                self.results.append(result)
                return result

            # Create Bedrock client and make test call
            try:
                client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=os.getenv('AWS_REGION', 'us-east-1')
                )

                # Make "hello world" test call
                test_payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 10,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Say 'Hello' in one word."
                        }
                    ]
                }

                response = client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(test_payload)
                )

                response_body = json.loads(
                    response['body'].read().decode('utf-8')
                )

                # Check response is valid
                if 'content' in response_body:
                    result = CheckResult(
                        check_name="Claude 3.5 Sonnet (Bedrock)",
                        passed=True,
                        message=(
                            "✓ Test call successful. "
                            f"Model: {model_id.split('.')[-1]}"
                        ),
                        details={
                            "region": os.getenv('AWS_REGION', 'us-east-1'),
                            "model_id": model_id,
                            "test_response": response_body['content'][0]['text'][:50]
                        }
                    )
                else:
                    result = CheckResult(
                        check_name="Claude 3.5 Sonnet (Bedrock)",
                        passed=False,
                        message="Unexpected response format from model",
                        details={"response": str(response_body)[:200]}
                    )

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_msg = e.response.get('Error', {}).get('Message', '')

                result = CheckResult(
                    check_name="Claude 3.5 Sonnet (Bedrock)",
                    passed=False,
                    message=(
                        f"Bedrock API error: {error_code}. "
                        f"{error_msg[:100]}"
                    ),
                    details={
                        "error_code": error_code,
                        "model_id": model_id
                    }
                )

            except Exception as e:
                result = CheckResult(
                    check_name="Claude 3.5 Sonnet (Bedrock)",
                    passed=False,
                    message=f"Test call failed: {str(e)[:100]}",
                    details={"error": str(e)}
                )

        except ImportError:
            result = CheckResult(
                check_name="Claude 3.5 Sonnet (Bedrock)",
                passed=False,
                message="boto3 not installed. Run: pip install boto3",
                details={"error": "ImportError"}
            )

        self.results.append(result)
        return result

    def check_gpt_oss_api(self) -> CheckResult:
        """Check GPT-OSS via AWS Bedrock with test call."""
        try:
            import boto3
            import json
            from botocore.exceptions import ClientError

            model_id = os.getenv('AWS_BEDROCK_GPT_OSS')

            if not model_id:
                result = CheckResult(
                    check_name="GPT-OSS 120B (Bedrock)",
                    passed=False,
                    message=(
                        "AWS_BEDROCK_GPT_OSS not set in .env. "
                        "Set to Bedrock model ID for GPT-OSS."
                    ),
                    details={"env_var": "AWS_BEDROCK_GPT_OSS"}
                )
                self.results.append(result)
                return result

            # Check AWS credentials (reuse session)
            try:
                session = boto3.Session()
                credentials = session.get_credentials()

                if credentials is None:
                    result = CheckResult(
                        check_name="GPT-OSS 120B (Bedrock)",
                        passed=False,
                        message="AWS credentials not found.",
                        details={"error": "NoCredentials"}
                    )
                    self.results.append(result)
                    return result

            except Exception as e:
                result = CheckResult(
                    check_name="GPT-OSS 120B (Bedrock)",
                    passed=False,
                    message=f"AWS credentials error: {str(e)}",
                    details={"error": str(e)}
                )
                self.results.append(result)
                return result

            # Create Bedrock client and make test call
            try:
                client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=os.getenv('AWS_REGION', 'us-east-1')
                )

                # Make "hello world" test call
                # Note: Payload format may vary by model
                test_payload = {
                    "prompt": "Say 'Hello' in one word.",
                    "max_gen_len": 10,
                    "temperature": 0.1,
                }

                response = client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(test_payload)
                )

                response_body = json.loads(
                    response['body'].read().decode('utf-8')
                )

                # Check response is valid (format depends on model)
                if response_body:
                    result = CheckResult(
                        check_name="GPT-OSS 120B (Bedrock)",
                        passed=True,
                        message=(
                            "✓ Test call successful. "
                            f"Model: {model_id.split('.')[-1]}"
                        ),
                        details={
                            "region": os.getenv('AWS_REGION', 'us-east-1'),
                            "model_id": model_id,
                            "test_response": str(response_body)[:50]
                        }
                    )
                else:
                    result = CheckResult(
                        check_name="GPT-OSS 120B (Bedrock)",
                        passed=False,
                        message="Empty response from model",
                        details={"response": str(response_body)}
                    )

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_msg = e.response.get('Error', {}).get('Message', '')

                result = CheckResult(
                    check_name="GPT-OSS 120B (Bedrock)",
                    passed=False,
                    message=(
                        f"Bedrock API error: {error_code}. "
                        f"{error_msg[:100]}"
                    ),
                    details={
                        "error_code": error_code,
                        "model_id": model_id,
                        "hint": "Verify model ID is correct for GPT-OSS on Bedrock"
                    }
                )

            except Exception as e:
                result = CheckResult(
                    check_name="GPT-OSS 120B (Bedrock)",
                    passed=False,
                    message=f"Test call failed: {str(e)[:100]}",
                    details={"error": str(e)}
                )

        except ImportError:
            result = CheckResult(
                check_name="GPT-OSS 120B (Bedrock)",
                passed=False,
                message="boto3 not installed. Run: pip install boto3",
                details={"error": "ImportError"}
            )

        self.results.append(result)
        return result

    def check_python_dependencies(self) -> CheckResult:
        """Check required Python packages."""
        required_packages = [
            "boto3",
            "pandas",
            "numpy",
            "matplotlib",
            "seaborn",
            "scipy",
            "pytest",
            "datasets",
            "pymongo",
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            result = CheckResult(
                check_name="Python Dependencies",
                passed=False,
                message=(
                    f"Missing: {', '.join(missing_packages)}. "
                    "Run: pip install -r requirements.txt"
                ),
                details={"missing": missing_packages}
            )
        else:
            result = CheckResult(
                check_name="Python Dependencies",
                passed=True,
                message="All required packages installed"
            )

        self.results.append(result)
        return result

    def get_summary(self) -> Dict[str, any]:
        """Get summary of all check results."""
        return {
            "total_checks": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "all_passed": all(r.passed for r in self.results),
            "results": [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ]
        }


def main():
    """Run pre-requisite checks from command line."""
    checker = PrerequisiteChecker()
    all_passed = checker.run_all_checks()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
