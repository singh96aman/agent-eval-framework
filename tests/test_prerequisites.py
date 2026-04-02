"""
Tests for pre-requisite verification system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from src.prereq_check import PrerequisiteChecker, CheckResult


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create required directories
        (tmpdir / "src").mkdir()
        (tmpdir / "tests").mkdir()
        (tmpdir / "data").mkdir()
        (tmpdir / "results").mkdir()
        (tmpdir / "paper").mkdir()

        yield str(tmpdir)


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_check_result_creation(self):
        """Test creating a CheckResult."""
        result = CheckResult(
            check_name="Test Check",
            passed=True,
            message="Everything is fine"
        )
        assert result.check_name == "Test Check"
        assert result.passed is True
        assert result.message == "Everything is fine"

    def test_check_result_string(self):
        """Test string representation."""
        result = CheckResult(
            check_name="Test Check",
            passed=True,
            message="OK"
        )
        assert "✓ PASS" in str(result)
        assert "Test Check" in str(result)

        result_fail = CheckResult(
            check_name="Test Check",
            passed=False,
            message="Failed"
        )
        assert "✗ FAIL" in str(result_fail)


class TestPrerequisiteChecker:
    """Tests for PrerequisiteChecker class."""

    def test_checker_initialization(self, temp_project_dir):
        """Test checker initialization."""
        checker = PrerequisiteChecker(temp_project_dir)
        assert checker.project_root == Path(temp_project_dir)
        assert checker.results == []

    def test_check_directory_structure_pass(self, temp_project_dir):
        """Test directory structure check with all directories present."""
        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_directory_structure()

        assert result.passed is True
        assert "All required directories exist" in result.message

    def test_check_directory_structure_fail(self):
        """Test directory structure check with missing directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = PrerequisiteChecker(tmpdir)
            result = checker.check_directory_structure()

            assert result.passed is False
            assert "Missing" in result.message
            assert result.details is not None
            assert "missing" in result.details

    @patch.dict('os.environ', {}, clear=True)
    def test_check_mongodb_connection_no_uri(self, temp_project_dir):
        """Test MongoDB check with no URI configured."""
        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_mongodb_connection()

        assert result.passed is False
        assert "MONGODB_URI" in result.message

    @patch.dict('os.environ', {}, clear=True)
    def test_check_huggingface_no_token(self, temp_project_dir):
        """Test HuggingFace check with no token."""
        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_huggingface_access()

        assert result.passed is False
        assert "HUGGINGFACE_TOKEN" in result.message

    @patch('boto3.Session')
    def test_check_claude_bedrock_no_credentials(self, mock_session, temp_project_dir):
        """Test Claude Bedrock check with no credentials."""
        mock_session.return_value.get_credentials.return_value = None

        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_claude_bedrock_api()

        assert result.passed is False
        assert "credentials not found" in result.message.lower()

    @patch('boto3.client')
    @patch('boto3.Session')
    def test_check_claude_bedrock_success(self, mock_session, mock_client, temp_project_dir):
        """Test Claude Bedrock check with successful API call."""
        # Mock credentials
        mock_credentials = MagicMock()
        mock_session.return_value.get_credentials.return_value = mock_credentials

        # Mock Bedrock client response
        mock_bedrock_runtime = MagicMock()
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = b'{"content": [{"text": "Hello"}]}'
        mock_bedrock_runtime.invoke_model.return_value = mock_response
        mock_client.return_value = mock_bedrock_runtime

        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_claude_bedrock_api()

        assert result.passed is True
        assert "successful" in result.message.lower()

    @patch.dict('os.environ', {}, clear=True)
    def test_check_gpt_oss_no_model_id(self, temp_project_dir):
        """Test GPT-OSS check with no model ID configured."""
        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_gpt_oss_api()

        assert result.passed is False
        assert "AWS_BEDROCK_GPT_OSS" in result.message

    @patch.dict('os.environ', {'AWS_BEDROCK_GPT_OSS': 'test-model-id'})
    @patch('boto3.client')
    @patch('boto3.Session')
    def test_check_gpt_oss_with_model(self, mock_session, mock_client, temp_project_dir):
        """Test GPT-OSS check with model ID configured."""
        # Mock credentials
        mock_credentials = MagicMock()
        mock_session.return_value.get_credentials.return_value = mock_credentials

        # Mock Bedrock client response
        mock_bedrock_runtime = MagicMock()
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = b'{"generation": "Hello"}'
        mock_bedrock_runtime.invoke_model.return_value = mock_response
        mock_client.return_value = mock_bedrock_runtime

        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_gpt_oss_api()

        assert result.passed is True
        assert "successful" in result.message.lower()

    def test_check_python_dependencies(self, temp_project_dir):
        """Test Python dependencies check."""
        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_python_dependencies()

        # At least pytest should be installed (since we're running tests)
        # But some packages might be missing, so we just check the result format
        assert result.check_name == "Python Dependencies"
        assert isinstance(result.passed, bool)

    def test_get_summary(self, temp_project_dir):
        """Test getting summary of check results."""
        checker = PrerequisiteChecker(temp_project_dir)

        # Run a few checks
        checker.check_directory_structure()
        checker.check_python_dependencies()

        summary = checker.get_summary()

        assert "total_checks" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "all_passed" in summary
        assert "results" in summary
        assert isinstance(summary["results"], list)
        assert len(summary["results"]) == 2

    @patch.dict('os.environ', {}, clear=True)
    @patch('sys.stdout')
    def test_run_all_checks(self, mock_stdout, temp_project_dir):
        """Test running all checks."""
        checker = PrerequisiteChecker(temp_project_dir)
        all_passed = checker.run_all_checks()

        assert isinstance(all_passed, bool)
        assert len(checker.results) > 0

        # Verify all expected checks were run
        check_names = [r.check_name for r in checker.results]
        assert "Directory Structure" in check_names
        assert "MongoDB Connection" in check_names
        assert "HuggingFace Access" in check_names
        assert "Claude 3.5 Sonnet (Bedrock)" in check_names
        assert "GPT-OSS 120B (Bedrock)" in check_names
        assert "Python Dependencies" in check_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
