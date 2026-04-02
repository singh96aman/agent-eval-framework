"""
Tests for pre-requisite verification system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
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
        (tmpdir / "data" / "toolbench").mkdir()
        (tmpdir / "data" / "gaia").mkdir()
        (tmpdir / "data" / "perturbed").mkdir()
        (tmpdir / "data" / "annotations").mkdir()
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
            assert "Missing directories" in result.message
            assert result.details is not None
            assert "missing" in result.details

    def test_check_toolbench_dataset_missing(self, temp_project_dir):
        """Test ToolBench check with no files."""
        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_toolbench_dataset()

        assert result.passed is False
        assert "No JSON/JSONL files found" in result.message

    def test_check_gaia_dataset_missing(self, temp_project_dir):
        """Test GAIA check with no files."""
        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_gaia_dataset()

        assert result.passed is False
        assert "No JSON/JSONL files found" in result.message

    @patch('boto3.Session')
    def test_check_claude_bedrock_no_credentials(self, mock_session, temp_project_dir):
        """Test Claude Bedrock check with no credentials."""
        mock_session.return_value.get_credentials.return_value = None

        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_claude_bedrock_api()

        assert result.passed is False
        assert "credentials not found" in result.message.lower()

    @patch('boto3.Session')
    @patch('boto3.client')
    def test_check_claude_bedrock_success(self, mock_client, mock_session, temp_project_dir):
        """Test Claude Bedrock check with valid credentials."""
        # Mock credentials
        mock_credentials = MagicMock()
        mock_session.return_value.get_credentials.return_value = mock_credentials

        # Mock Bedrock client
        mock_bedrock_client = MagicMock()
        mock_client.return_value = mock_bedrock_client

        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_claude_bedrock_api()

        assert result.passed is True
        assert "credentials found" in result.message.lower()

    @patch.dict('os.environ', {}, clear=True)
    def test_check_gpt_oss_no_endpoint(self, temp_project_dir):
        """Test GPT-OSS check with no endpoint configured."""
        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_gpt_oss_api()

        assert result.passed is False
        assert "GPT_OSS_ENDPOINT" in result.message

    @patch.dict('os.environ', {'GPT_OSS_ENDPOINT': 'http://localhost:8000', 'GPT_OSS_API_KEY': 'test_key'})
    def test_check_gpt_oss_with_endpoint(self, temp_project_dir):
        """Test GPT-OSS check with endpoint configured."""
        checker = PrerequisiteChecker(temp_project_dir)
        result = checker.check_gpt_oss_api()

        assert result.passed is True
        assert "Endpoint configured" in result.message

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

    def test_run_all_checks(self, temp_project_dir):
        """Test running all checks."""
        checker = PrerequisiteChecker(temp_project_dir)

        with patch('sys.stdout'):  # Suppress output
            all_passed = checker.run_all_checks()

        assert isinstance(all_passed, bool)
        assert len(checker.results) > 0

        # Verify all expected checks were run
        check_names = [r.check_name for r in checker.results]
        assert "Directory Structure" in check_names
        assert "ToolBench Dataset" in check_names
        assert "GAIA Dataset" in check_names
        assert "Claude Bedrock API" in check_names
        assert "GPT-OSS 120B API" in check_names
        assert "Python Dependencies" in check_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
