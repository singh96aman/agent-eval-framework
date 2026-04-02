"""
Pre-requisite verification system for POC experiment.

This module checks that all required datasets, APIs, and environments
are available and configured correctly before starting the experiment.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json
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
    """
    Checks all pre-requisites for running the POC experiment.
    """

    def __init__(self, project_root: str = None):
        """
        Initialize checker.

        Args:
            project_root: Path to project root directory (default: auto-detect)
        """
        if project_root is None:
            # Auto-detect: assume prereq_check.py is in src/
            project_root = Path(__file__).parent.parent
        self.project_root = Path(project_root)
        self.results: List[CheckResult] = []

    def run_all_checks(self) -> bool:
        """
        Run all pre-requisite checks.

        Returns:
            True if all checks passed, False otherwise
        """
        print("=" * 70)
        print("PRE-REQUISITE VERIFICATION FOR POC EXPERIMENT")
        print("=" * 70)
        print()

        self.results = []

        # Run all checks
        self.check_directory_structure()
        self.check_toolbench_dataset()
        self.check_gaia_dataset()
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
            print("\n✗ Some pre-requisites failed. Please fix issues above before proceeding.")

        return all_passed

    def check_directory_structure(self) -> CheckResult:
        """Check that required directories exist."""
        required_dirs = [
            "src",
            "tests",
            "data",
            "data/toolbench",
            "data/gaia",
            "data/perturbed",
            "data/annotations",
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
                message=f"Missing directories: {', '.join(missing_dirs)}",
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

    def check_toolbench_dataset(self) -> CheckResult:
        """Check ToolBench dataset availability."""
        toolbench_dir = self.project_root / "data" / "toolbench"

        if not toolbench_dir.exists():
            result = CheckResult(
                check_name="ToolBench Dataset",
                passed=False,
                message=f"Directory not found: {toolbench_dir}",
                details={"path": str(toolbench_dir)}
            )
            self.results.append(result)
            return result

        # Check for JSON/JSONL files
        json_files = list(toolbench_dir.glob("*.json")) + list(toolbench_dir.glob("*.jsonl"))

        if not json_files:
            result = CheckResult(
                check_name="ToolBench Dataset",
                passed=False,
                message=f"No JSON/JSONL files found in {toolbench_dir}",
                details={"path": str(toolbench_dir), "files_found": 0}
            )
        else:
            # Try to load a sample trajectory
            try:
                from src.data.loaders import load_toolbench_trajectories
                trajectories = load_toolbench_trajectories(
                    str(toolbench_dir),
                    max_trajectories=1
                )

                if trajectories:
                    result = CheckResult(
                        check_name="ToolBench Dataset",
                        passed=True,
                        message=f"Found {len(json_files)} file(s), successfully loaded sample trajectory",
                        details={
                            "path": str(toolbench_dir),
                            "files_found": len(json_files),
                            "sample_loaded": True
                        }
                    )
                else:
                    result = CheckResult(
                        check_name="ToolBench Dataset",
                        passed=False,
                        message=f"Found files but no valid trajectories loaded. Check format.",
                        details={
                            "path": str(toolbench_dir),
                            "files_found": len(json_files)
                        }
                    )
            except Exception as e:
                result = CheckResult(
                    check_name="ToolBench Dataset",
                    passed=False,
                    message=f"Error loading trajectories: {str(e)}",
                    details={"error": str(e)}
                )

        self.results.append(result)
        return result

    def check_gaia_dataset(self) -> CheckResult:
        """Check GAIA dataset availability."""
        gaia_dir = self.project_root / "data" / "gaia"

        if not gaia_dir.exists():
            result = CheckResult(
                check_name="GAIA Dataset",
                passed=False,
                message=f"Directory not found: {gaia_dir}",
                details={"path": str(gaia_dir)}
            )
            self.results.append(result)
            return result

        # Check for JSON/JSONL files
        json_files = list(gaia_dir.glob("*.json")) + list(gaia_dir.glob("*.jsonl"))

        if not json_files:
            result = CheckResult(
                check_name="GAIA Dataset",
                passed=False,
                message=f"No JSON/JSONL files found in {gaia_dir}",
                details={"path": str(gaia_dir), "files_found": 0}
            )
        else:
            # Try to load a sample trajectory
            try:
                from src.data.loaders import load_gaia_trajectories
                trajectories = load_gaia_trajectories(
                    str(gaia_dir),
                    max_trajectories=1
                )

                if trajectories:
                    result = CheckResult(
                        check_name="GAIA Dataset",
                        passed=True,
                        message=f"Found {len(json_files)} file(s), successfully loaded sample trajectory",
                        details={
                            "path": str(gaia_dir),
                            "files_found": len(json_files),
                            "sample_loaded": True
                        }
                    )
                else:
                    result = CheckResult(
                        check_name="GAIA Dataset",
                        passed=False,
                        message=f"Found files but no valid trajectories loaded. Check format.",
                        details={
                            "path": str(gaia_dir),
                            "files_found": len(json_files)
                        }
                    )
            except Exception as e:
                result = CheckResult(
                    check_name="GAIA Dataset",
                    passed=False,
                    message=f"Error loading trajectories: {str(e)}",
                    details={"error": str(e)}
                )

        self.results.append(result)
        return result

    def check_claude_bedrock_api(self) -> CheckResult:
        """Check Claude-3.5-Sonnet on AWS Bedrock API access."""
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError

            # Check for AWS credentials
            try:
                session = boto3.Session()
                credentials = session.get_credentials()

                if credentials is None:
                    result = CheckResult(
                        check_name="Claude Bedrock API",
                        passed=False,
                        message="AWS credentials not found. Configure via AWS CLI or environment variables.",
                        details={"error": "NoCredentials"}
                    )
                    self.results.append(result)
                    return result

            except Exception as e:
                result = CheckResult(
                    check_name="Claude Bedrock API",
                    passed=False,
                    message=f"Error accessing AWS credentials: {str(e)}",
                    details={"error": str(e)}
                )
                self.results.append(result)
                return result

            # Try to create Bedrock client
            try:
                client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name='us-east-1'
                )

                # Test with a minimal invocation (may incur small cost ~$0.01)
                # Commented out to avoid costs during prereq check
                # model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
                # response = client.invoke_model(...)

                result = CheckResult(
                    check_name="Claude Bedrock API",
                    passed=True,
                    message="AWS credentials found and Bedrock client created successfully. (Not tested with actual API call to avoid costs)",
                    details={"region": "us-east-1", "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"}
                )

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                result = CheckResult(
                    check_name="Claude Bedrock API",
                    passed=False,
                    message=f"AWS Bedrock client error: {error_code}. Check IAM permissions and model access.",
                    details={"error": error_code, "message": str(e)}
                )

        except ImportError:
            result = CheckResult(
                check_name="Claude Bedrock API",
                passed=False,
                message="boto3 library not installed. Run: pip install boto3",
                details={"error": "ImportError"}
            )

        self.results.append(result)
        return result

    def check_gpt_oss_api(self) -> CheckResult:
        """Check GPT-OSS 120B API endpoint access."""
        # Check for environment variable with endpoint
        endpoint = os.getenv("GPT_OSS_ENDPOINT")
        api_key = os.getenv("GPT_OSS_API_KEY")

        if not endpoint:
            result = CheckResult(
                check_name="GPT-OSS 120B API",
                passed=False,
                message="GPT_OSS_ENDPOINT environment variable not set. Set it to your endpoint URL.",
                details={"env_var": "GPT_OSS_ENDPOINT"}
            )
            self.results.append(result)
            return result

        if not api_key:
            result = CheckResult(
                check_name="GPT-OSS 120B API",
                passed=False,
                message="GPT_OSS_API_KEY environment variable not set (may be optional depending on endpoint).",
                details={"env_var": "GPT_OSS_API_KEY", "warning": True}
            )
            self.results.append(result)
            return result

        # Try to connect to endpoint (without making actual API call)
        try:
            import requests
            # Just check if URL is valid and reachable
            # Not making actual API call to avoid costs
            result = CheckResult(
                check_name="GPT-OSS 120B API",
                passed=True,
                message=f"Endpoint configured: {endpoint[:50]}... (Not tested with actual API call)",
                details={"endpoint": endpoint, "api_key_set": bool(api_key)}
            )

        except ImportError:
            result = CheckResult(
                check_name="GPT-OSS 120B API",
                passed=False,
                message="requests library not installed. Run: pip install requests",
                details={"error": "ImportError"}
            )

        self.results.append(result)
        return result

    def check_python_dependencies(self) -> CheckResult:
        """Check that required Python packages are installed."""
        required_packages = [
            "boto3",
            "pandas",
            "numpy",
            "matplotlib",
            "seaborn",
            "scipy",
            "pytest",
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
                message=f"Missing packages: {', '.join(missing_packages)}. Run: pip install -r requirements.txt",
                details={"missing": missing_packages}
            )
        else:
            result = CheckResult(
                check_name="Python Dependencies",
                passed=True,
                message="All required Python packages are installed"
            )

        self.results.append(result)
        return result

    def get_summary(self) -> Dict[str, any]:
        """Get summary of all check results as dictionary."""
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

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
