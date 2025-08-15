"""
Test module for admin dashboard functions.
Tests the is_admin, run_repair, and git_commit_and_push functions.
"""

import sys
import os
from unittest.mock import patch, MagicMock
import subprocess

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest

# Import functions from pages.admin_dashboard
from pages.admin_dashboard import is_admin, run_repair, git_commit_and_push


class TestAdminDashboard:
    """Test class for admin dashboard functions."""

    def test_is_admin_correct_password(self):
        """Test is_admin function with correct password."""
        result = is_admin("4250Galt")
        assert result is True

    def test_is_admin_incorrect_password(self):
        """Test is_admin function with incorrect password."""
        result = is_admin("wrong_password")
        assert result is False

    def test_is_admin_empty_password(self):
        """Test is_admin function with empty password."""
        result = is_admin("")
        assert result is False

    @patch('pages.admin_dashboard.subprocess.run')
    def test_run_repair_success(self, mock_subprocess_run):
        """Test run_repair function when repair script succeeds."""
        # Mock successful subprocess call
        mock_result = MagicMock()
        mock_result.stdout = "Repair completed successfully"
        mock_subprocess_run.return_value = mock_result
        
        result = run_repair()
        
        assert result == "Repair completed successfully"
        mock_subprocess_run.assert_called_once_with(
            ["python", "auto_debug_repair.py"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch('pages.admin_dashboard.subprocess.run')
    def test_run_repair_failure(self, mock_subprocess_run):
        """Test run_repair function when repair script fails."""
        # Mock failed subprocess call
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "python", stderr="Error in repair script"
        )
        
        result = run_repair()
        
        assert "Repair script failed with error:" in result
        assert "Error in repair script" in result

    @patch('pages.admin_dashboard.subprocess.run')
    def test_run_repair_unexpected_error(self, mock_subprocess_run):
        """Test run_repair function when unexpected error occurs."""
        # Mock unexpected error
        mock_subprocess_run.side_effect = Exception("Unexpected error")
        
        result = run_repair()
        
        assert "Unexpected error occurred:" in result
        assert "Unexpected error" in result

    @patch('pages.admin_dashboard.subprocess.run')
    @patch('pages.admin_dashboard.st')
    def test_git_commit_and_push_success(self, mock_st, mock_subprocess_run):
        """Test git_commit_and_push function when git commands succeed."""
        # Mock successful subprocess calls
        mock_subprocess_run.return_value = MagicMock()
        
        result = git_commit_and_push()
        
        assert result is True
        # Verify all three git commands were called
        expected_calls = [
            (["git", "add", "."], {'check': True, 'capture_output': True}),
            (["git", "commit", "-m", "Automated debug repairs"], {'check': True, 'capture_output': True}),
            (["git", "push"], {'check': True, 'capture_output': True}),
        ]
        
        actual_calls = [call[0] for call in mock_subprocess_run.call_args_list]
        assert len(actual_calls) == 3

    @patch('pages.admin_dashboard.subprocess.run')
    @patch('pages.admin_dashboard.st')
    def test_git_commit_and_push_failure(self, mock_st, mock_subprocess_run):
        """Test git_commit_and_push function when git command fails."""
        # Mock failed subprocess call
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "git", stderr="Git command failed"
        )
        
        result = git_commit_and_push()
        
        assert result is False
        mock_st.error.assert_called_once()

    @patch('pages.admin_dashboard.subprocess.run')
    @patch('pages.admin_dashboard.st')
    def test_git_commit_and_push_unexpected_error(self, mock_st, mock_subprocess_run):
        """Test git_commit_and_push function when unexpected error occurs."""
        # Mock unexpected error
        mock_subprocess_run.side_effect = Exception("Unexpected error")
        
        result = git_commit_and_push()
        
        assert result is False
        mock_st.error.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])